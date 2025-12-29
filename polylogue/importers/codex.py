from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..render import AttachmentInfo
from ..util import CODEX_SESSIONS_ROOT, assign_conversation_slug
from ..conversation import process_conversation
from ..branching import MessageRecord
from .base import ImportResult
from .utils import (
    CHAR_THRESHOLD,
    LINE_THRESHOLD,
    PREVIEW_LINES,
    estimate_token_count,
    normalise_inline_footnotes,
    store_large_text,
)
from .normalizer import build_message_record
from ..services.conversation_registrar import ConversationRegistrar, create_default_registrar

_DEFAULT_BASE = CODEX_SESSIONS_ROOT


def import_codex_session(
    session_id: str,
    *,
    base_dir: Path = _DEFAULT_BASE,
    output_dir: Path,
    collapse_threshold: int = 24,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool = False,
    html_theme: str = "light",
    force: bool = False,
    allow_dirty: bool = False,
    registrar: Optional[ConversationRegistrar] = None,
    attachment_ocr: bool = True,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
    conversation_id_override: Optional[str] = None,
) -> ImportResult:
    from .raw_storage import store_raw_import, mark_parse_success, mark_parse_failed

    cli_meta = dict(meta) if meta else None
    registrar = registrar or create_default_registrar()
    base_dir = base_dir.expanduser()
    conversation_id = conversation_id_override or session_id
    candidate = Path(session_id)
    if candidate.exists() and candidate.is_file():
        session_path = candidate
    else:
        matches = list(base_dir.rglob(f"*{session_id}.jsonl"))
        if not matches:
            raise FileNotFoundError(f"Could not locate session {session_id} under {base_dir}")
        session_path = max(
            matches,
            key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, str(p)),
        )

    # Store raw session data before parsing
    data_hash = None
    if session_path.exists():
        raw_data = session_path.read_bytes()
        # Use the resolved conversation ID for raw import tracking.
        data_hash = store_raw_import(
            data=raw_data,
            provider="codex",
            conversation_id=conversation_id,
            source_path=session_path,
        )

    attachments: List[AttachmentInfo] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    chunks: List[Dict[str, str]] = []
    chunk_metadata: List[Dict[str, object]] = []
    call_index: Dict[str, int] = {}
    message_records: List[MessageRecord] = []
    seen_message_ids: Set[str] = set()
    routing_stats: Dict[str, int] = {"routed": 0, "skipped": 0}

    output_dir.mkdir(parents=True, exist_ok=True)
    title = session_path.stem
    slug = assign_conversation_slug("codex", conversation_id, title, id_hint=title[:8])
    conversation_dir = output_dir / slug
    conversation_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = conversation_dir / "conversation.md"
    attachments_dir = conversation_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    def _format_json(value: Any) -> str:
        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    def _normalise_output(value: Any) -> str:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return normalise_inline_footnotes(value)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("output"), str):
                    return normalise_inline_footnotes(parsed["output"])
                return normalise_inline_footnotes(_format_json(parsed))
            if isinstance(parsed, list):
                return normalise_inline_footnotes(_format_json(parsed))
            return normalise_inline_footnotes(str(parsed))
        return normalise_inline_footnotes(_format_json(value))

    with session_path.open(encoding="utf-8") as handle:
        for line_idx, raw in enumerate(handle, start=1):
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                fallback = raw.strip()
                if not fallback:
                    continue
                chunk_text = f"Unparsed Codex log line\n```\n{fallback}```"
                chunks.append(
                    {
                        "role": "model",
                        "text": chunk_text,
                        "tokenCount": estimate_token_count(chunk_text, model="gpt-5-codex"),
                    }
                )
                chunk_metadata.append({"raw": {"type": "unparsed", "payload": fallback}})
                continue
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("type")
            if entry_type in {None, "session_meta"}:
                continue
            payload = entry.get("payload") if entry_type == "response_item" else entry
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type") or entry_type
            if not isinstance(ptype, str):
                continue
            message_id = (
                payload.get("message_id")
                or payload.get("id")
                or entry.get("message_id")
                or entry.get("id")
                or payload.get("call_id")
                or payload.get("callId")
                or f"item-{line_idx}"
            )
            parent_id = (
                payload.get("parent_id")
                or payload.get("parentId")
                or entry.get("parent_id")
                or entry.get("parentId")
            )

            if ptype == "message":
                role = payload.get("role") or "assistant"
                canonical_role = "model" if role in {"assistant", "model"} else "user"
                parts: List[str] = []
                content_segments = payload.get("content") or []
                if not isinstance(content_segments, list):
                    content_segments = []
                for seg in content_segments:
                    if not isinstance(seg, dict):
                        continue
                    text = seg.get("text") or ""
                    if text.startswith("<user_instructions>") or text.startswith("<environment_context>"):
                        parts = []
                        break
                    parts.append(text)
                if not parts:
                    continue
                text = normalise_inline_footnotes("\n".join(parts))
                chunks.append(
                    {
                        "role": canonical_role,
                        "text": text,
                        "tokenCount": estimate_token_count(text, model="gpt-5-codex"),
                        "messageId": message_id,
                    }
                )
                chunk_metadata.append({"raw": payload})
                if parent_id:
                    chunks[-1]["parentId"] = parent_id
                    chunks[-1]["branchParent"] = parent_id

            elif ptype in {"function_call", "local_shell_call"}:
                if ptype == "local_shell_call":
                    action = payload.get("action") or {}
                    name = action.get("type") or "shell"
                    args_obj: Any = {
                        key: value
                        for key, value in action.items()
                        if key not in {"type"}
                    }
                    if payload.get("status"):
                        args_obj["status"] = payload["status"]
                else:
                    name = payload.get("name") or "tool"
                    args_obj = payload.get("arguments") or {}
                args_json = args_obj if isinstance(args_obj, str) else _format_json(args_obj)
                chunk_text = f"Tool call `{name}`\n```json\n{args_json}\n```"
                chunks.append(
                    {
                        "role": "model",
                        "text": chunk_text,
                        "tokenCount": estimate_token_count(chunk_text, model="gpt-5-codex"),
                        "messageId": payload.get("call_id") or payload.get("callId") or message_id,
                    }
                )
                call_meta = {
                    "type": ptype,
                    "name": name,
                    "id": payload.get("call_id") or payload.get("callId") or entry.get("call_id"),
                    "arguments": args_obj,
                }
                chunk_metadata.append({"raw": payload, "tool_calls": [call_meta]})
                call_id = (
                    payload.get("call_id")
                    or payload.get("callId")
                    or entry.get("call_id")
                    or ""
                )
                call_index[call_id] = len(chunks) - 1
                if parent_id:
                    chunks[-1]["parentId"] = parent_id
                    chunks[-1]["branchParent"] = parent_id

            elif ptype == "function_call_output":
                output_text = _normalise_output(payload.get("output"))
                call_id = (
                    payload.get("call_id")
                    or payload.get("callId")
                    or entry.get("call_id")
                    or ""
                )
                idx = call_index.get(call_id)
                if idx is not None:
                    chunks[idx]["text"] += f"\n\nOutput:\n{output_text}"
                    chunks[idx]["tokenCount"] = estimate_token_count(
                        chunks[idx]["text"], model="gpt-5-codex"
                    )
                    if parent_id and "parentId" not in chunks[idx]:
                        chunks[idx]["parentId"] = parent_id
                        chunks[idx]["branchParent"] = parent_id
                    chunk_meta = chunk_metadata[idx]
                    calls = chunk_meta.setdefault("tool_calls", [])
                    if calls:
                        calls[0]["output"] = output_text
                    else:
                        calls.append({"type": "function_call_output", "output": output_text, "id": call_id})
                else:
                    chunks.append(
                        {
                            "role": "tool",
                            "text": output_text,
                            "tokenCount": estimate_token_count(output_text, model="gpt-5-codex"),
                            "messageId": message_id,
                        }
                    )
                    chunk_metadata.append(
                        {
                            "raw": payload,
                            "tool_calls": [
                                {
                                    "type": "function_call_output",
                                    "id": call_id,
                                    "output": output_text,
                                }
                            ],
                        }
                    )
                    parent_ref = parent_id or call_id
                    if parent_ref:
                        chunks[-1]["parentId"] = parent_ref
                        chunks[-1]["branchParent"] = parent_ref

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "") or ""
        preview = store_large_text(
            text,
            chunk_index=idx,
            attachments_dir=attachments_dir,
            markdown_dir=markdown_path.parent,
            attachments=attachments,
            per_chunk_links=per_chunk_links,
            prefix="chunk",
            routing_stats=routing_stats,
        )
        if preview != text:
            chunk["text"] = preview

    for idx, chunk in enumerate(chunks):
        parent_id = chunk.get("parentId") or (message_records[-1].message_id if message_records else None)
        if parent_id and "parentId" not in chunk:
            chunk["parentId"] = parent_id
            chunk["branchParent"] = parent_id
        chunk_meta = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
        links = list(per_chunk_links.get(idx, []))
        message_records.append(
            build_message_record(
                provider="codex",
                conversation_id=conversation_id,
                chunk_index=idx,
                chunk=chunk,
                raw_metadata=chunk_meta.get("raw") if isinstance(chunk_meta, dict) else None,
                attachments=links,
                tool_calls=chunk_meta.get("tool_calls") if isinstance(chunk_meta, dict) else None,
                seen_ids=seen_message_ids,
                fallback_prefix=slug,
            )
        )

    metadata = {"model": "gpt-5-codex"}
    extra_yaml = {
        "sourcePlatform": "codex",
        "sessionPath": str(session_path),
    }

    try:
        result = process_conversation(
            provider="codex",
            conversation_id=conversation_id,
            slug=slug,
            title=title,
            message_records=message_records,
            attachments=attachments,
            canonical_leaf_id=message_records[-1].message_id if message_records else None,
            collapse_threshold=collapse_threshold,
            collapse_thresholds=collapse_thresholds,
            html=html,
            html_theme=html_theme,
            output_dir=output_dir,
            extra_yaml=extra_yaml,
            extra_state={
                "sessionPath": str(session_path),
                "cliMeta": cli_meta,
            },
            source_file_id=session_id,
            modified_time=None,
            created_time=None,
            run_settings=metadata,
            source_mime="application/jsonl",
            source_size=session_path.stat().st_size,
            attachment_policy={
                "previewLines": PREVIEW_LINES,
                "lineThreshold": LINE_THRESHOLD,
                "charThreshold": CHAR_THRESHOLD,
                "routing": routing_stats,
            },
            force=force,
            allow_dirty=allow_dirty,
            attachment_ocr=attachment_ocr,
            sanitize_html=sanitize_html,
            registrar=registrar,
        )

        # Mark parse success if we stored raw data
        if data_hash:
            mark_parse_success(data_hash)

        return result
    except Exception:
        # Mark parse failure if we stored raw data
        if data_hash:
            import traceback
            mark_parse_failed(data_hash, traceback.format_exc())
        raise
