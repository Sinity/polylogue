from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Tuple

from ..render import AttachmentInfo, MarkdownDocument, build_markdown_from_chunks
from ..util import sanitize_filename
from .base import ImportResult
from .utils import estimate_token_count, store_large_text


@dataclass
class _Conversation:
    data: Dict
    path: Path


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
    safe_name = sanitize_filename(f"{title}-{conv_id[:8]}")
    markdown_path = output_dir / f"{safe_name}.md"
    attachments_dir = markdown_path.parent / f"{markdown_path.stem}_attachments"

    chunks: List[Dict] = []
    attachments: List[AttachmentInfo] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}

    files_dir = export_root / "files"
    file_index = _build_file_index(files_dir)

    messages = _gather_messages(conv)
    for idx, msg in enumerate(messages):
        role = msg.get("author") or "assistant"
        content = msg.get("content") or {}
        text = _extract_text(role, content)
        metadata = msg.get("metadata") or {}
        canonical_role = _normalise_role(role)
        chunk = {
            "role": canonical_role,
            "text": text,
            "tokenCount": estimate_token_count(text),
        }
        if msg.get("create_time"):
            chunk["timestamp"] = msg["create_time"]
        chunks.append(chunk)
        attachment_refs = metadata.get("attachments") or []
        for ref in attachment_refs:
            _copy_attachment(ref, file_index, attachments_dir, markdown_path.parent, attachments, per_chunk_links, idx)
        if text:
            preview = store_large_text(
                text,
                chunk_index=idx,
                attachments_dir=attachments_dir,
                markdown_dir=markdown_path.parent,
                attachments=attachments,
                per_chunk_links=per_chunk_links,
                prefix="chatgpt",
            )
            if preview != text:
                chunks[idx]["text"] = preview

    metadata = {
        "title": conv.get("title") or safe_name,
        "sourcePlatform": "chatgpt",
        "conversationId": conv.get("id") or conv.get("conversation_id"),
    }
    model_slug = (
        conv.get("model_slug")
        or conv.get("model")
        or conv.get("default_model_slug")
        or conv.get("current_model")
    )
    if model_slug:
        metadata["model"] = model_slug

    document = build_markdown_from_chunks(
        chunks,
        per_chunk_links,
        title=metadata["title"],
        source_file_id=metadata.get("conversationId"),
        modified_time=conv.get("update_time"),
        created_time=conv.get("create_time"),
        run_settings=None,
        citations=None,
        source_mime="application/json",
        source_size=None,
        collapse_threshold=collapse_threshold,
        extra_yaml={
            "sourcePlatform": "chatgpt",
            "conversationId": metadata.get("conversationId"),
            "sourceModel": model_slug,
        },
        attachments=attachments,
    )
    markdown_path.write_text(document.to_markdown(), encoding="utf-8")
    html_path: Optional[Path] = None
    if html:
        from ..html import write_html

        html_path = markdown_path.with_suffix(".html")
        write_html(document, html_path, html_theme)

    if not attachments:
        try:
            attachments_dir.rmdir()
            attachments_dir = None
        except OSError:
            pass

    return ImportResult(
        markdown_path=markdown_path,
        html_path=html_path,
        attachments_dir=attachments_dir,
        document=document,
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
