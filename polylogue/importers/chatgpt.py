from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Tuple

from ..render import AttachmentInfo, build_markdown_from_chunks
from ..document_store import persist_document
from ..util import assign_conversation_slug
from .base import ImportResult
from .utils import estimate_token_count, store_large_text


def _load_export(path: Path) -> Tuple[Path, Optional[TemporaryDirectory]]:
    if path.is_dir():
        return path, None
    if path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Unsupported ChatGPT export: {path}")
    tmp = TemporaryDirectory(prefix="chatgpt-export-")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp.name)
    return Path(tmp.name), tmp


def _render_parts(parts: Iterable) -> str:
    fragments: List[str] = []
    for part in parts:
        if isinstance(part, str):
            fragments.append(part)
            continue
        if not isinstance(part, dict):
            continue
        content_type = part.get("content_type") or part.get("type")
        if content_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                fragments.append(text)
        elif content_type == "code":
            code = part.get("text") or part.get("code") or ""
            language = part.get("language") or part.get("mime_type") or ""
            fragments.append(f"```{language}\n{code}\n```".strip())
        elif content_type == "table":
            rows = part.get("rows") or []
            if rows:
                header = rows[0]
                fragments.append(_format_table(rows, header))
        elif content_type in {"image_file", "attachment"}:
            name = part.get("name") or part.get("filename") or part.get("file_id")
            if name:
                fragments.append(f"![{name}](attachment://{name})")
        elif content_type == "link":
            url = part.get("url")
            text = part.get("text") or url
            if url and text:
                fragments.append(f"[{text}]({url})")
        elif content_type in {"tool_calls", "tool_call"}:
            name = part.get("name") or part.get("id") or "tool"
            payload = part.get("input") or part.get("arguments") or {}
            if isinstance(payload, str):
                payload_text = payload
            else:
                payload_text = json.dumps(payload, indent=2, ensure_ascii=False)
            fragments.append(f"Tool call `{name}`\n```json\n{payload_text}\n```")
        elif content_type in {"tool_result", "tool_results"}:
            output_text = part.get("text") or part.get("output") or ""
            fragments.append(f"Tool result\n````\n{output_text}\n````")
        elif content_type in {"system_message", "observation"}:
            text = part.get("text")
            if text:
                fragments.append(text)
    return "\n\n".join([frag for frag in fragments if frag])


def _format_table(rows: List, header: Iterable) -> str:
    def fmt_row(row: Iterable) -> List[str]:
        result: List[str] = []
        for cell in row:
            if isinstance(cell, str):
                result.append(cell)
            else:
                result.append(json.dumps(cell))
        return result

    header_cells = fmt_row(header)
    lines = ["| " + " | ".join(header_cells) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(fmt_row(row)) + " |")
    return "\n".join(lines)


def _gather_messages(conv: Dict) -> List[Dict]:
    mapping = conv.get("mapping", {})
    messages: List[Dict] = []
    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        author = (msg.get("author") or {}).get("role")
        if author == "system" and not msg.get("content"):
            continue
        created = msg.get("create_time") or node.get("create_time")
        messages.append(
            {
                "id": msg.get("id") or node.get("id"),
                "author": author,
                "content": msg.get("content"),
                "metadata": msg.get("metadata") or {},
                "create_time": created,
            }
        )
    messages.sort(key=lambda m: (m.get("create_time") or 0, m.get("id") or ""))
    return messages


def import_chatgpt_export(
    export_path: Path,
    *,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    selected_ids: Optional[List[str]] = None,
) -> List[ImportResult]:
    base_path, tmp = _load_export(export_path)
    try:
        convo_path = base_path / "conversations.json"
        if not convo_path.exists():
            raise FileNotFoundError("conversations.json missing in export")
        conversations = json.loads(convo_path.read_text(encoding="utf-8"))
        if not isinstance(conversations, list):
            raise ValueError("Unexpected ChatGPT export format")

        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[ImportResult] = []
        for conv in conversations:
            conv_id = conv.get("id") or conv.get("conversation_id")
            if selected_ids and conv_id not in selected_ids:
                continue
            results.append(
                _render_chatgpt_conversation(
                    conv,
                    base_path,
                    output_dir,
                    collapse_threshold=collapse_threshold,
                    html=html,
                    html_theme=html_theme,
                )
            )
        return results
    finally:
        if tmp is not None:
            tmp.cleanup()


def list_chatgpt_conversations(export_path: Path) -> List[Dict[str, Optional[str]]]:
    base_path, tmp = _load_export(export_path)
    try:
        convo_path = base_path / "conversations.json"
        if not convo_path.exists():
            raise FileNotFoundError("conversations.json missing in export")
        conversations = json.loads(convo_path.read_text(encoding="utf-8"))
        if not isinstance(conversations, list):
            raise ValueError("Unexpected ChatGPT export format")
        results: List[Dict[str, Optional[str]]] = []
        for conv in conversations:
            results.append(
                {
                    "id": conv.get("id") or conv.get("conversation_id"),
                    "title": conv.get("title"),
                    "update_time": conv.get("update_time"),
                    "create_time": conv.get("create_time"),
                }
            )
        return results
    finally:
        if tmp is not None:
            tmp.cleanup()




def _render_chatgpt_conversation(
    conv: Dict,
    export_root: Path,
    output_dir: Path,
    *,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
) -> ImportResult:
    title = conv.get("title") or "chatgpt-conversation"
    conv_id = conv.get("id") or conv.get("conversation_id") or "chat"
    slug = assign_conversation_slug("chatgpt", conv_id, title, id_hint=(conv_id or "")[:8])
    markdown_path = output_dir / f"{slug}.md"
    attachments_dir = markdown_path.parent / f"{slug}_attachments"

    chunks: List[Dict] = []
    attachments: List[AttachmentInfo] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}

    files_dir = export_root / "files"
    file_index = _build_file_index(files_dir)

    model_slug = (
        conv.get("model_slug")
        or conv.get("model")
        or conv.get("default_model_slug")
        or conv.get("current_model")
    )

    messages = _gather_messages(conv)
    for idx, msg in enumerate(messages):
        role = msg.get("author") or "assistant"
        content = msg.get("content") or {}
        text_block = _extract_text(role, content)
        metadata = msg.get("metadata") or {}
        canonical_role = _normalise_role(role)
        chunk = {
            "role": canonical_role,
            "text": text_block,
            "tokenCount": estimate_token_count(text_block, model=model_slug),
        }
        if msg.get("create_time"):
            chunk["timestamp"] = msg["create_time"]
        chunks.append(chunk)
        attachment_refs = metadata.get("attachments") or []
        for ref in attachment_refs:
            _copy_attachment(
                ref,
                file_index,
                attachments_dir,
                markdown_path.parent,
                attachments,
                per_chunk_links,
                idx,
            )
        if text_block:
            preview = store_large_text(
                text_block,
                chunk_index=idx,
                attachments_dir=attachments_dir,
                markdown_dir=markdown_path.parent,
                attachments=attachments,
                per_chunk_links=per_chunk_links,
                prefix="chatgpt",
            )
            if preview != text_block:
                chunks[idx]["text"] = preview

    conversation_id = conv.get("id") or conv.get("conversation_id")
    extra_yaml = {
        "sourcePlatform": "chatgpt",
        "conversationId": conversation_id,
        "sourceModel": model_slug,
        "sourceExportPath": str(export_root),
    }

    document = build_markdown_from_chunks(
        chunks,
        per_chunk_links,
        title=title,
        source_file_id=conversation_id,
        modified_time=conv.get("update_time") or conv.get("modified_time"),
        created_time=conv.get("create_time"),
        run_settings=None,
        citations=None,
        source_mime="application/json",
        source_size=None,
        collapse_threshold=collapse_threshold,
        extra_yaml=extra_yaml,
        attachments=attachments,
    )

    persist_result = persist_document(
        provider="chatgpt",
        conversation_id=conversation_id,
        title=title,
        document=document,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        attachments=attachments,
        updated_at=conv.get("update_time") or conv.get("modified_time"),
        created_at=conv.get("create_time"),
        html=html,
        html_theme=html_theme,
        attachment_policy=None,
        extra_state={
            "sourceModel": model_slug,
            "sourceExportPath": str(export_root),
        },
        slug_hint=slug,
        id_hint=(conv_id or "")[:8],
    )

    if persist_result.skipped:
        return ImportResult(
            markdown_path=persist_result.markdown_path,
            html_path=persist_result.html_path,
            attachments_dir=persist_result.attachments_dir,
            document=None,
            skipped=True,
            skip_reason=persist_result.skip_reason,
            dirty=persist_result.dirty,
            content_hash=persist_result.content_hash,
        )

    return ImportResult(
        markdown_path=persist_result.markdown_path,
        html_path=persist_result.html_path,
        attachments_dir=persist_result.attachments_dir,
        document=persist_result.document or document,
        dirty=persist_result.dirty,
        content_hash=persist_result.content_hash,
    )

def _normalise_role(role: Optional[str]) -> str:
    if not role:
        return "assistant"
    if role in {"user", "assistant", "tool", "system"}:
        if role == "assistant":
            return "model"
        return role
    if role == "function":
        return "tool"
    return "assistant"


def _extract_text(role: Optional[str], content: Dict) -> str:
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if isinstance(parts, list):
        return _render_parts(parts)
    if content.get("content_type") == "text" and isinstance(parts, list):
        return _render_parts(parts)
    if isinstance(content.get("text"), str):
        return content["text"]
    return ""


def _build_file_index(files_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not files_dir.exists():
        return index
    for path in files_dir.rglob("*"):
        if path.is_file():
            index[path.name] = path
            index[path.stem] = path
    return index


def _copy_attachment(
    ref: Dict,
    file_index: Dict[str, Path],
    attachments_dir: Path,
    markdown_root: Path,
    attachments: List[AttachmentInfo],
    per_chunk_links: Dict[int, List[Tuple[str, Path]]],
    chunk_idx: int,
) -> None:
    file_id = ref.get("id") or ref.get("file_id")
    name = ref.get("name") or ref.get("filename") or file_id
    if not name:
        return
    src = None
    if file_id and file_id in file_index:
        src = file_index[file_id]
    elif name in file_index:
        src = file_index[name]
    if src is None:
        attachments.append(
            AttachmentInfo(
                name=name,
                link=ref.get("url") or f"attachment://{name}",
                local_path=None,
                size_bytes=ref.get("size"),
                remote=True,
            )
        )
        per_chunk_links.setdefault(chunk_idx, []).append((name, ref.get("url") or f"attachment://{name}"))
        return

    attachments_dir.mkdir(parents=True, exist_ok=True)
    target = attachments_dir / name
    counter = 1
    while target.exists():
        target = attachments_dir / f"{target.stem}_{counter}{target.suffix}"
        counter += 1
    target.write_bytes(src.read_bytes())
    try:
        rel = target.relative_to(markdown_root)
    except ValueError:
        rel = target
    attachments.append(
        AttachmentInfo(
            name=target.name,
            link=str(rel),
            local_path=rel,
            size_bytes=target.stat().st_size,
            remote=False,
        )
    )
    per_chunk_links.setdefault(chunk_idx, []).append((target.name, rel))
