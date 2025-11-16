from __future__ import annotations

import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set, Tuple

from ..render import AttachmentInfo
from ..util import assign_conversation_slug, sanitize_filename
from ..conversation import process_conversation
from ..branching import MessageRecord
from ..services.conversation_registrar import ConversationRegistrar, create_default_registrar
from .base import ImportResult
from .normalizer import build_message_record
from .utils import (
    estimate_token_count,
    normalise_inline_footnotes,
    safe_extractall,
    store_large_text,
)


def _load_bundle(path: Path) -> Tuple[Path, Optional[TemporaryDirectory]]:
    if path.is_dir():
        return path, None
    if path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Unsupported Claude export: {path}")
    tmp = TemporaryDirectory(prefix="claude-export-")
    with zipfile.ZipFile(path) as zf:
        safe_extractall(zf, Path(tmp.name))
    return Path(tmp.name), tmp


def import_claude_export(
    export_path: Path,
    *,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    selected_ids: Optional[List[str]] = None,
    force: bool = False,
    allow_dirty: bool = False,
    registrar: Optional[ConversationRegistrar] = None,
) -> List[ImportResult]:
    registrar = registrar or create_default_registrar()
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
                    force=force,
                    allow_dirty=allow_dirty,
                    registrar=registrar,
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
    force: bool,
    allow_dirty: bool,
    registrar: Optional[ConversationRegistrar],
) -> ImportResult:
    title = conv.get("name") or conv.get("title") or "claude-chat"
    conv_id = conv.get("uuid") or conv.get("id") or "claude"
    slug = assign_conversation_slug("claude.ai", conv_id, title, id_hint=(conv_id or "")[:8])
    conversation_dir = output_dir / slug
    conversation_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = conversation_dir / "conversation.md"
    attachments_dir = conversation_dir / "attachments"

    attachments: List[AttachmentInfo] = []
    file_index = _index_export_files(export_root)

    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    chunks: List[Dict] = []
    message_records: List[MessageRecord] = []
    seen_message_ids: Set[str] = set()

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
        message_id = (
            message.get("uuid")
            or message.get("id")
            or message.get("message_id")
            or message.get("messageId")
        )
        parent_id = (
            message.get("parent_id")
            or message.get("parentUuid")
            or message.get("parent_message_id")
            or message.get("parentMessageId")
        )
        chunk = {
            "role": canonical_role,
            "text": text_block,
            "tokenCount": estimate_token_count(text_block, model=model_id),
        }
        if message_id:
            chunk["messageId"] = message_id
        if parent_id:
            chunk["parentId"] = parent_id
            chunk["branchParent"] = parent_id
        elif message_records:
            fallback_parent = message_records[-1].message_id
            chunk["parentId"] = fallback_parent
            chunk["branchParent"] = fallback_parent
            parent_id = fallback_parent
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

        links = list(per_chunk_links.get(idx, []))
        message_records.append(
            build_message_record(
                provider="claude.ai",
                conversation_id=conv_id,
                chunk_index=idx,
                chunk=chunks[idx],
                raw_metadata=message,
                attachments=links,
                tool_calls=_extract_tool_metadata(blocks if isinstance(blocks, list) else []),
                seen_ids=seen_message_ids,
                fallback_prefix=slug,
            )
        )

    extra_yaml = {
        "sourcePlatform": "claude.ai",
        "conversationId": conv_id,
        "sourceExportPath": str(export_root),
    }
    if model_id:
        extra_yaml["sourceModel"] = model_id

    canonical_leaf_id = message_records[-1].message_id if message_records else None

    return process_conversation(
        provider="claude.ai",
        conversation_id=conv_id,
        slug=slug,
        title=title,
        message_records=message_records,
        attachments=attachments,
        canonical_leaf_id=canonical_leaf_id,
        collapse_threshold=collapse_threshold,
        html=html,
        html_theme=html_theme,
        output_dir=output_dir,
        extra_yaml=extra_yaml,
        extra_state={
            "sourceModel": model_id,
            "sourceExportPath": str(export_root),
        },
        source_file_id=conv_id,
        modified_time=conv.get("updated_at") or conv.get("modified_at"),
        created_time=conv.get("created_at"),
        run_settings=None,
        source_mime="application/json",
        source_size=None,
        attachment_policy=None,
        force=force,
        allow_dirty=allow_dirty,
        registrar=registrar,
    )


def _index_export_files(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file():
            index[path.name] = path
    return index



def _extract_tool_metadata(blocks: List[Dict]) -> List[Dict[str, object]]:
    details: List[Dict[str, object]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            details.append(
                {
                    "type": block_type,
                    "name": block.get("name") or block.get("tool_name"),
                    "id": block.get("id") or block.get("tool_use_id"),
                    "input": block.get("input"),
                    "status": block.get("status"),
                }
            )
        elif block_type == "tool_result":
            details.append(
                {
                    "type": block_type,
                    "name": block.get("name"),
                    "id": block.get("id"),
                    "output": block.get("text") or block.get("result"),
                }
            )
    return details



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
    return normalise_inline_footnotes(text), chunk_links


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
    safe_name = sanitize_filename(name)
    if not safe_name:
        safe_name = sanitize_filename(file_id or "attachment")
    if not safe_name:
        safe_name = "attachment"
    target = attachments_dir / safe_name
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
