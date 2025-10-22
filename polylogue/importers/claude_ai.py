from __future__ import annotations

import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

from ..render import AttachmentInfo, build_markdown_from_chunks
from ..document_store import persist_document
from ..util import assign_conversation_slug
from .base import ImportResult
from .utils import estimate_token_count, store_large_text


def _load_bundle(path: Path) -> Tuple[Path, Optional[TemporaryDirectory]]:
    if path.is_dir():
        return path, None
    if path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Unsupported Claude export: {path}")
    tmp = TemporaryDirectory(prefix="claude-export-")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp.name)
    return Path(tmp.name), tmp


def import_claude_export(
    export_path: Path,
    *,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    selected_ids: Optional[List[str]] = None,
) -> List[ImportResult]:
    root, tmp = _load_bundle(export_path)
    try:
        convo_path = root / "conversations.json"
        if not convo_path.exists():
            raise FileNotFoundError("conversations.json missing in Claude export")
        payload = json.loads(convo_path.read_text(encoding="utf-8"))
        conversations = payload.get("conversations") if isinstance(payload, dict) else payload
        if not isinstance(conversations, list):
            raise ValueError("Unexpected Claude export format")

        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[ImportResult] = []
        for conv in conversations:
            conv_id = conv.get("uuid") or conv.get("id")
            if selected_ids and conv_id not in selected_ids:
                continue
            results.append(
                _render_claude_conversation(
                    conv,
                    root,
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


def list_claude_conversations(export_path: Path) -> List[Dict[str, Optional[str]]]:
    root, tmp = _load_bundle(export_path)
    try:
        convo_path = root / "conversations.json"
        if not convo_path.exists():
            raise FileNotFoundError("conversations.json missing in Claude export")
        payload = json.loads(convo_path.read_text(encoding="utf-8"))
        conversations = payload.get("conversations") if isinstance(payload, dict) else payload
        if not isinstance(conversations, list):
            raise ValueError("Unexpected Claude export format")
        info: List[Dict[str, Optional[str]]] = []
        for conv in conversations:
            info.append(
                {
                    "id": conv.get("uuid") or conv.get("id"),
                    "title": conv.get("name") or conv.get("title"),
                    "updated_at": conv.get("updated_at"),
                    "created_at": conv.get("created_at"),
                }
            )
        return info
    finally:
        if tmp is not None:
            tmp.cleanup()




def _render_claude_conversation(
    conv: Dict,
    export_root: Path,
    output_dir: Path,
    *,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
) -> ImportResult:
    title = conv.get("name") or conv.get("title") or "claude-chat"
    conv_id = conv.get("uuid") or conv.get("id") or "claude"
    slug = assign_conversation_slug("claude.ai", conv_id, title, id_hint=(conv_id or "")[:8])
    markdown_path = output_dir / f"{slug}.md"
    attachments_dir = markdown_path.parent / f"{slug}_attachments"

    attachments: List[AttachmentInfo] = []
    file_index = _index_export_files(export_root)

    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    chunks: List[Dict] = []

    model_id = conv.get("model") or conv.get("model_id")

    for idx, message in enumerate(conv.get("chat_messages", [])):
        sender = message.get("sender") or "assistant"
        timestamp = message.get("created_at") or message.get("updated_at")
        blocks = message.get("content") or []
        text_block, links = _render_content_blocks(
            blocks,
            attachments,
            attachments_dir,
            markdown_path.parent,
            file_index,
        )
        canonical_role = "user" if sender == "human" else "model"
        chunk = {
            "role": canonical_role,
            "text": text_block,
            "tokenCount": estimate_token_count(text_block, model=model_id),
        }
        if timestamp:
            chunk["timestamp"] = timestamp
        per_chunk_links[idx] = links
        chunks.append(chunk)
        if text_block:
            preview = store_large_text(
                text_block,
                chunk_index=idx,
                attachments_dir=attachments_dir,
                markdown_dir=markdown_path.parent,
                attachments=attachments,
                per_chunk_links=per_chunk_links,
                prefix="claude",
            )
            if preview != text_block:
                chunks[idx]["text"] = preview

    extra_yaml = {
        "sourcePlatform": "claude.ai",
        "conversationId": conv_id,
        "sourceExportPath": str(export_root),
    }
    if model_id:
        extra_yaml["sourceModel"] = model_id

    document = build_markdown_from_chunks(
        chunks,
        per_chunk_links,
        title=title,
        source_file_id=conv_id,
        modified_time=conv.get("updated_at") or conv.get("modified_at"),
        created_time=conv.get("created_at"),
        run_settings=None,
        citations=None,
        source_mime="application/json",
        source_size=None,
        collapse_threshold=collapse_threshold,
        extra_yaml=extra_yaml,
        attachments=attachments,
    )

    persist_result = persist_document(
        provider="claude.ai",
        conversation_id=conv_id,
        title=title,
        document=document,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        attachments=attachments,
        updated_at=conv.get("updated_at") or conv.get("modified_at"),
        created_at=conv.get("created_at"),
        html=html,
        html_theme=html_theme,
        attachment_policy=None,
        extra_state={
            "sourceModel": model_id,
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


def _index_export_files(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file():
            index[path.name] = path
    return index



def _render_content_blocks(
    blocks: List[Dict],
    attachments: List[AttachmentInfo],
    attachments_dir: Path,
    markdown_root: Path,
    file_index: Dict[str, Path],
) -> Tuple[str, List[Tuple[str, Path]]]:
    fragments: List[str] = []
    chunk_links: List[Tuple[str, Path]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text") or ""
            fragments.append(text)
        elif block_type == "tool_use":
            name = block.get("name") or block.get("tool_name") or "tool"
            input_data = block.get("input")
            fragments.append(f"Tool call `{name}`\n```json\n{json.dumps(input_data, indent=2, ensure_ascii=False)}\n```")
        elif block_type == "tool_result":
            result_text = block.get("text") or block.get("result") or ""
            fragments.append(f"Tool result\n````\n{result_text}\n````")
        elif block_type == "thinking":
            thought = block.get("thinking") or block.get("text") or ""
            if thought:
                fragments.append(f"_(internal thought)_\n{thought}")
        elif block_type in {"image", "file"}:
            file_id = block.get("file_id") or block.get("asset_pointer")
            name = block.get("file_name") or file_id
            if name:
                copied = _copy_file(file_id, name, file_index, attachments_dir, markdown_root)
                if copied:
                    info, rel = copied
                    attachments.append(info)
                    chunk_links.append((info.name, rel))
                else:
                    fragments.append(f"[{name}](attachment://{file_id})")
    text = "\n\n".join([frag for frag in fragments if frag])
    return text, chunk_links


def _copy_file(
    file_id: Optional[str],
    name: Optional[str],
    file_index: Dict[str, Path],
    attachments_dir: Path,
    markdown_root: Path,
) -> Optional[Tuple[AttachmentInfo, Path]]:
    if not name:
        return None
    source = None
    if file_id and file_id in file_index:
        source = file_index[file_id]
    elif name in file_index:
        source = file_index[name]
    if source is None:
        return None
    attachments_dir.mkdir(parents=True, exist_ok=True)
    target = attachments_dir / name
    counter = 1
    while target.exists():
        target = attachments_dir / f"{target.stem}_{counter}{target.suffix}"
        counter += 1
    target.write_bytes(source.read_bytes())
    try:
        rel = target.relative_to(markdown_root)
    except ValueError:
        rel = target
    info = AttachmentInfo(
        name=target.name,
        link=str(rel),
        local_path=rel,
        size_bytes=target.stat().st_size,
        remote=False,
    )
    return info, rel


def _index_files(export_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for folder in ("attachments", "files", "assets"):
        path = export_root / folder
        if not path.exists():
            continue
        for file in path.rglob("*"):
            if file.is_file():
                index[file.name] = file
                index[file.stem] = file
    return index


def _normalise_sender(sender: Optional[str]) -> str:
    if not sender:
        return "model"
    sender = sender.lower()
    if sender in {"user", "assistant", "system"}:
        if sender == "assistant":
            return "model"
        return sender
    if sender in {"tool", "function"}:
        return "tool"
    return "model"
