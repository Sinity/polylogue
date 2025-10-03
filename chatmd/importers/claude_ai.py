from __future__ import annotations

import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

from ..render import AttachmentInfo, build_markdown_from_chunks
from ..util import sanitize_filename
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
    safe_name = sanitize_filename(f"{title}-{conv_id[:8]}")
    markdown_path = output_dir / f"{safe_name}.md"
    attachments_dir = markdown_path.parent / f"{markdown_path.stem}_attachments"

    messages = conv.get("chat_messages") or conv.get("messages") or []
    attachments: List[AttachmentInfo] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    chunks: List[Dict] = []

    # Files are stored under attachments/ or files/
    file_index = _index_files(export_root)

    for idx, message in enumerate(messages):
        sender = message.get("sender") or message.get("role")
        role = _normalise_sender(sender)
        content_blocks = message.get("content") or []
        text, block_links = _render_content_blocks(
            content_blocks,
            attachments,
            attachments_dir,
            markdown_path.parent,
            file_index,
        )
        chunk = {
            "role": role,
            "text": text,
            "tokenCount": estimate_token_count(text),
        }
        if message.get("created_at"):
            chunk["timestamp"] = message["created_at"]
        chunks.append(chunk)

        if block_links:
            per_chunk_links.setdefault(idx, []).extend(block_links)
        preview = store_large_text(
            text,
            chunk_index=idx,
            attachments_dir=attachments_dir,
            markdown_dir=markdown_path.parent,
            attachments=attachments,
            per_chunk_links=per_chunk_links,
            prefix="claude",
        )
        if preview != text:
            chunks[idx]["text"] = preview

    metadata = {
        "title": title,
        "sourcePlatform": "claude.ai",
        "conversationId": conv_id,
    }

    frontmatter = {
        "sourcePlatform": "claude.ai",
        "conversationId": conv_id,
    }

    model_id = conv.get("model") or conv.get("model_id")
    if model_id:
        frontmatter["sourceModel"] = model_id
        metadata["model"] = model_id

    document = build_markdown_from_chunks(
        chunks,
        per_chunk_links,
        title=metadata["title"],
        source_file_id=conv_id,
        modified_time=conv.get("updated_at"),
        created_time=conv.get("created_at"),
        run_settings=None,
        citations=None,
        source_mime="application/json",
        source_size=None,
        collapse_threshold=collapse_threshold,
        extra_yaml=frontmatter,
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
