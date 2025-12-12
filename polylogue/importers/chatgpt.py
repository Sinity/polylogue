from __future__ import annotations

import json
import re
import urllib.parse
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import ijson

from ..render import AttachmentInfo
from ..util import assign_conversation_slug, sanitize_filename
from ..conversation import process_conversation
from ..branching import MessageRecord
from ..services.conversation_registrar import ConversationRegistrar, create_default_registrar
from .base import ImportResult
from .normalizer import build_message_record
from .utils import (
    CHAR_THRESHOLD,
    LINE_THRESHOLD,
    PREVIEW_LINES,
    estimate_token_count,
    normalise_inline_footnotes,
    safe_extractall,
    store_large_text,
)
from .raw_storage import compute_hash, store_raw_import, mark_parse_success, mark_parse_failed


def _load_export(path: Path) -> Tuple[Path, Optional[TemporaryDirectory]]:
    if path.is_dir():
        return path, None
    if path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Unsupported ChatGPT export: {path}")
    tmp = TemporaryDirectory(prefix="chatgpt-export-")
    with zipfile.ZipFile(path) as zf:
        safe_extractall(zf, Path(tmp.name))
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


def _extract_tool_calls(content: Dict[str, Any]) -> List[Dict[str, Any]]:
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return []
    calls: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        content_type = part.get("content_type") or part.get("type")
        if content_type in {"tool_calls", "tool_call"}:
            payload = part.get("input") or part.get("arguments")
            call = {
                "type": content_type,
                "name": part.get("name") or part.get("id") or "tool",
            }
            if part.get("id"):
                call["id"] = part["id"]
            if payload is not None:
                call["arguments"] = payload
            calls.append(call)
        elif content_type in {"tool_result", "tool_results"}:
            call_result = {
                "type": content_type,
                "id": part.get("id"),
                "output": part.get("text") or part.get("output"),
            }
            calls.append(call_result)
    return calls


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
                "node_id": node.get("id"),
                "parent": node.get("parent"),
            }
        )
    messages.sort(key=lambda m: (m.get("create_time") or 0, m.get("id") or ""))
    return messages


def import_chatgpt_export(
    export_path: Path,
    *,
    output_dir: Path,
    collapse_threshold: int,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool,
    html_theme: str,
    selected_ids: Optional[List[str]] = None,
    force: bool = False,
    allow_dirty: bool = False,
    registrar: Optional[ConversationRegistrar] = None,
    attachment_ocr: bool = False,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
) -> List[ImportResult]:
    """Import ChatGPT export to database.

    Database-first: All conversation data is written to SQLite.
    Use 'polylogue render --force' to generate markdown files.
    Stores per-conversation raw blobs to avoid re-ingesting full exports repeatedly.
    """
    registrar = registrar or create_default_registrar()
    base_path, tmp = _load_export(export_path)
    bundle_hash: Optional[str] = None
    try:
        convo_path = base_path / "conversations.json"
        if not convo_path.exists():
            raise FileNotFoundError("conversations.json missing in export")
        # Hash the bundle content to annotate per-conversation raws
        try:
            bundle_hash = compute_hash(convo_path.read_bytes())
        except Exception:
            bundle_hash = None

        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[ImportResult] = []
        with convo_path.open("rb") as fh:
            try:
                for conv in ijson.items(fh, "item"):
                    if not isinstance(conv, dict):
                        raise ValueError
                    conv_id = conv.get("id") or conv.get("conversation_id")
                    if selected_ids and conv_id not in selected_ids:
                        continue
                    # Store per-conversation raw snapshot (deduped by hash/version)
                    raw_bytes = json.dumps(conv, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    raw_hash = store_raw_import(
                        data=raw_bytes,
                        provider="chatgpt",
                        conversation_id=conv_id or "conversation",
                        source_path=export_path,
                        metadata={
                            "bundle_hash": bundle_hash,
                            "bundle_path": str(export_path),
                            "export_root": str(base_path),
                        },
                    )
                    try:
                        results.append(
                            _render_chatgpt_conversation(
                                conv,
                                base_path,
                                output_dir,
                                collapse_threshold=collapse_threshold,
                                collapse_thresholds=collapse_thresholds,
                                html=html,
                                html_theme=html_theme,
                                force=force,
                                allow_dirty=allow_dirty,
                                registrar=registrar,
                                attachment_ocr=attachment_ocr,
                                sanitize_html=sanitize_html,
                                meta=meta,
                            )
                        )
                        if raw_hash:
                            mark_parse_success(raw_hash)
                    except Exception:
                        if raw_hash:
                            import traceback

                            mark_parse_failed(raw_hash, traceback.format_exc())
                        raise
            except Exception as exc:
                raise ValueError(
                    "Unexpected ChatGPT export format: conversations.json must contain a list. "
                    "Make sure you're using a valid ChatGPT export from the official export feature."
                ) from exc

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
        results: List[Dict[str, Optional[str]]] = []
        with convo_path.open("rb") as fh:
            try:
                for conv in ijson.items(fh, "item"):
                    if not isinstance(conv, dict):
                        raise ValueError
                    results.append(
                        {
                            "id": conv.get("id") or conv.get("conversation_id"),
                            "title": conv.get("title"),
                            "update_time": conv.get("update_time"),
                            "create_time": conv.get("create_time"),
                        }
                    )
            except Exception as exc:
                raise ValueError(
                    "Unexpected ChatGPT export format: ZIP archive must contain a valid conversations.json file. "
                    "Make sure you're using an official ChatGPT export."
                ) from exc
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
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool,
    html_theme: str,
    force: bool,
    allow_dirty: bool,
    registrar: Optional[ConversationRegistrar],
    attachment_ocr: bool = False,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
) -> ImportResult:
    title = conv.get("title") or "chatgpt-conversation"
    conv_id = conv.get("id") or conv.get("conversation_id") or "chat"
    slug = assign_conversation_slug("chatgpt", conv_id, title, id_hint=(conv_id or "")[:8])
    conversation_dir = output_dir / slug
    conversation_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = conversation_dir / "conversation.md"
    attachments_dir = conversation_dir / "attachments"

    chunks: List[Dict] = []
    attachments: List[AttachmentInfo] = []
    attachment_lookup: Dict[str, AttachmentInfo] = {}
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    message_records: List[MessageRecord] = []
    seen_message_ids: Set[str] = set()
    routing_stats: Dict[str, int] = {"routed": 0, "skipped": 0}

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
        text_block = normalise_inline_footnotes(text_block)
        metadata = msg.get("metadata") or {}
        canonical_role = _normalise_role(role)
        chunk = {
            "role": canonical_role,
            "text": text_block,
            "tokenCount": estimate_token_count(text_block, model=model_slug),
        }
        message_id = msg.get("id") or msg.get("node_id")
        parent_id = msg.get("parent")
        if message_id:
            chunk["messageId"] = message_id
            chunk["nodeId"] = msg.get("node_id") or message_id
        if parent_id:
            chunk["parentId"] = parent_id
            chunk["branchParent"] = parent_id
        elif message_records:
            fallback_parent = message_records[-1].message_id
            chunk["parentId"] = fallback_parent
            chunk["branchParent"] = fallback_parent
            parent_id = fallback_parent
        if msg.get("create_time"):
            chunk["timestamp"] = msg["create_time"]
        chunks.append(chunk)
        attachment_refs = metadata.get("attachments") or []
        for ref in attachment_refs:
            info = _copy_attachment(
                ref,
                file_index,
                attachments_dir,
                markdown_path.parent,
                attachments,
                per_chunk_links,
                idx,
            )
            if info is not None:
                file_id = ref.get("id") or ref.get("file_id")
                if file_id:
                    attachment_lookup[file_id] = info
        chunks[idx]["text"] = _inject_citations(chunks[idx]["text"], metadata, attachment_lookup)
        current_text = chunks[idx]["text"]
        if current_text:
            current_text = normalise_inline_footnotes(current_text)
            preview = store_large_text(
                current_text,
                chunk_index=idx,
                attachments_dir=attachments_dir,
                markdown_dir=markdown_path.parent,
                attachments=attachments,
                per_chunk_links=per_chunk_links,
                prefix="chatgpt",
                routing_stats=routing_stats,
            )
            if preview != current_text:
                chunks[idx]["text"] = preview

        links = list(per_chunk_links.get(idx, []))
        message_records.append(
            build_message_record(
                provider="chatgpt",
                conversation_id=conv_id,
                chunk_index=idx,
                chunk=chunks[idx],
                raw_metadata=metadata,
                attachments=links,
                tool_calls=_extract_tool_calls(content if isinstance(content, dict) else {}),
                seen_ids=seen_message_ids,
                fallback_prefix=slug,
            )
        )

    conversation_id = conv.get("id") or conv.get("conversation_id")
    extra_yaml = {
        "sourcePlatform": "chatgpt",
        "conversationId": conversation_id,
        "sourceModel": model_slug,
        "sourceExportPath": str(export_root),
    }

    thresholds = collapse_thresholds or {"message": collapse_threshold, "tool": collapse_threshold}
    attachment_policy = {
        "previewLines": PREVIEW_LINES,
        "lineThreshold": LINE_THRESHOLD,
        "charThreshold": CHAR_THRESHOLD,
        "routing": routing_stats,
    }

    return process_conversation(
        provider="chatgpt",
        conversation_id=conversation_id,
        slug=slug,
        title=title,
        message_records=message_records,
        attachments=attachments,
        canonical_leaf_id=conv.get("current_node"),
        collapse_threshold=collapse_threshold,
        collapse_thresholds=thresholds,
        html=html,
        html_theme=html_theme,
        output_dir=output_dir,
        extra_yaml=extra_yaml,
        extra_state={
            "sourceModel": model_slug,
            "sourceExportPath": str(export_root),
            "cliMeta": dict(meta) if meta else None,
        },
        source_file_id=conversation_id,
        modified_time=conv.get("update_time") or conv.get("modified_time"),
        created_time=conv.get("create_time"),
        run_settings=None,
        source_mime="application/json",
        source_size=None,
        attachment_policy=attachment_policy,
        force=force,
        allow_dirty=allow_dirty,
        attachment_ocr=attachment_ocr,
        sanitize_html=sanitize_html,
        registrar=registrar,
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
) -> Optional[AttachmentInfo]:
    file_id = ref.get("id") or ref.get("file_id")
    raw_name = ref.get("name") or ref.get("filename") or file_id
    if not raw_name:
        return None
    safe_name = sanitize_filename(raw_name)
    if not safe_name:
        safe_name = sanitize_filename(file_id or "attachment")
    if not safe_name:
        safe_name = "attachment"
    src = None
    if file_id and file_id in file_index:
        src = file_index[file_id]
    elif raw_name in file_index:
        src = file_index[raw_name]
    if src is None:
        info = AttachmentInfo(
            name=raw_name,
            link=ref.get("url") or f"attachment://{raw_name}",
            local_path=None,
            size_bytes=ref.get("size"),
            remote=True,
        )
        attachments.append(info)
        per_chunk_links.setdefault(chunk_idx, []).append(
            (raw_name, ref.get("url") or f"attachment://{raw_name}")
        )
        return info

    attachments_dir.mkdir(parents=True, exist_ok=True)
    target = attachments_dir / safe_name
    counter = 1
    while target.exists():
        target = attachments_dir / f"{target.stem}_{counter}{target.suffix}"
        counter += 1
    target.write_bytes(src.read_bytes())
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
    attachments.append(info)
    per_chunk_links.setdefault(chunk_idx, []).append((target.name, rel))
    return info


_CITE_TOKEN_RE = re.compile(r"\uE200(?P<tag>[^\uE200\uE201]+)((?:\uE202[^\uE200\uE201]+)*)\uE201")


def _inject_citations(text: str, metadata: Dict[str, Any], attachments_by_id: Dict[str, AttachmentInfo]) -> str:
    if not text:
        return text
    citations = metadata.get("citations") or []
    ref_map = {
        ref.get("matched_text"): ref
        for ref in metadata.get("content_references") or []
        if isinstance(ref, dict) and ref.get("matched_text")
    }
    if not citations and not ref_map:
        return text

    labels: Dict[Tuple[str, Optional[str]], str] = {}
    footnotes: Dict[str, str] = {}
    counter = 1
    cite_index = 0
    total_cites = len(citations)

    def ensure_label(key: Tuple[str, Optional[str]], description_provider) -> str:
        nonlocal counter
        label = labels.get(key)
        if label is None:
            label = f"cite{counter}"
            counter += 1
            description = description_provider()
            if description:
                footnotes[label] = description
            labels[key] = label
        return label

    def replacer(match: re.Match[str]) -> str:
        nonlocal cite_index
        if cite_index < total_cites:
            citation = citations[cite_index]
            cite_index += 1
            key = ("citation",) + _citation_key(citation)
            label = ensure_label(key, lambda: _format_citation_description(citation, attachments_by_id))
            return f"[^{label}]"

        ref = ref_map.get(match.group(0))
        if ref:
            ref_key = ("reference", ref.get("matched_text"))
            label = ensure_label(ref_key, lambda: _format_reference_description(ref, attachments_by_id))
            return f"[^{label}]"
        return ""

    rendered = _CITE_TOKEN_RE.sub(replacer, text)
    if not footnotes:
        return rendered
    footnote_lines = [f"[^{label}]: {desc}" for label, desc in footnotes.items()]
    return rendered.rstrip() + "\n\n" + "\n".join(footnote_lines)


def _citation_key(citation: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    meta = citation.get("metadata") or {}
    ctype = meta.get("type") or "unknown"
    if ctype == "file":
        return (ctype, meta.get("id") or meta.get("name"))
    if ctype == "webpage":
        return (ctype, meta.get("url"))
    if ctype == "image_inline":
        links = meta.get("asset_pointer_links") or []
        return (ctype, links[0] if links else meta.get("clicked_from_url"))
    return (ctype, None)


def _format_citation_description(citation: Dict[str, Any], attachments_by_id: Dict[str, AttachmentInfo]) -> str:
    meta = citation.get("metadata") or {}
    ctype = meta.get("type")
    if ctype == "file":
        file_id = meta.get("id")
        info = attachments_by_id.get(file_id)
        name = meta.get("name") or (info.name if info else file_id) or "attachment"
        if info:
            if info.local_path:
                target = urllib.parse.quote(info.local_path.as_posix())
            else:
                target = info.link
            return f"[{name}]({target})"
        link = meta.get("url") or f"attachment://{name}"
        return f"[{name}]({link})"
    if ctype == "webpage":
        title = meta.get("title") or meta.get("url") or "Source"
        url = meta.get("url")
        return f"[{title}]({url})" if url else title
    if ctype == "image_inline":
        title = meta.get("clicked_from_title") or "Image reference"
        url = meta.get("clicked_from_url")
        if not url:
            links = meta.get("asset_pointer_links") or []
            url = links[0] if links else None
        return f"{title} — {url}" if url else title
    text = meta.get("text")
    if text:
        return text
    name = meta.get("name")
    if name:
        return name
    return "Citation"


def _format_reference_description(reference: Dict[str, Any], attachments_by_id: Dict[str, AttachmentInfo]) -> str:
    alt = reference.get("alt")
    if isinstance(alt, str):
        alt = alt.strip()
        if alt.startswith("(") and alt.endswith(")"):
            alt = alt[1:-1].strip()
        if alt:
            return alt
    urls = reference.get("safe_urls") or []
    if urls:
        title = reference.get("title") or reference.get("matched_text") or "Reference"
        return f"[{title}]({urls[0]})"
    refs = reference.get("refs") or []
    if refs:
        ref_id = refs[0]
        if isinstance(ref_id, dict):
            ref_id = ref_id.get("ref_type") or ref_id.get("id") or "internal"
        if isinstance(ref_id, str) and ref_id.startswith("hidden"):
            return "Live market quote (tool result)"
        return f"Internal reference {ref_id}"
    matched = reference.get("matched_text")
    if matched:
        return f"Reference {matched.strip()}"
    return "Reference"
