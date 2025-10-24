from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from ..render import AttachmentInfo
from ..util import assign_conversation_slug
from ..conversation import process_conversation
from ..branching import MessageRecord
from .base import ImportResult
from .utils import CHAR_THRESHOLD, LINE_THRESHOLD, PREVIEW_LINES, estimate_token_count, normalise_inline_footnotes

_DEFAULT_BASE = Path.home() / ".codex" / "sessions"

def _truncate_with_preview(text: str, attachment_name: str) -> str:
    lines = text.splitlines()
    if len(lines) <= PREVIEW_LINES * 2 + 1:
        return text
    head = lines[:PREVIEW_LINES]
    tail = lines[-PREVIEW_LINES:]
    preview = "\n".join(head + ["…", "", f"(Full content saved to {attachment_name})", ""] + tail)
    return preview


def import_codex_session(
    session_id: str,
    *,
    base_dir: Path = _DEFAULT_BASE,
    output_dir: Path,
    collapse_threshold: int = 24,
    html: bool = False,
    html_theme: str = "light",
    force: bool = False,
) -> ImportResult:
    candidate = Path(session_id)
    if candidate.exists() and candidate.is_file():
        session_path = candidate
    else:
        session_glob = sorted(base_dir.rglob(f"*{session_id}.jsonl"))
        if not session_glob:
            raise FileNotFoundError(f"Could not locate session {session_id} under {base_dir}")
        session_path = session_glob[0]

    attachments: List[AttachmentInfo] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    chunks: List[Dict[str, str]] = []
    call_index: Dict[str, int] = {}
    message_records: List[MessageRecord] = []
    seen_message_ids: Set[str] = set()

    output_dir.mkdir(parents=True, exist_ok=True)
    title = session_path.stem
    slug = assign_conversation_slug("codex", session_id, title, id_hint=title[:8])
    markdown_path = output_dir / f"{slug}.md"
    attachments_dir = markdown_path.parent / f"{slug}_attachments"
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
                chunk_text = "Unparsed Codex log line\n```\n{}```".format(fallback)
                chunks.append(
                    {
                        "role": "model",
                        "text": chunk_text,
                        "tokenCount": estimate_token_count(chunk_text, model="gpt-5-codex"),
                    }
                )
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
                else:
                    chunks.append(
                        {
                            "role": "tool",
                            "text": output_text,
                            "tokenCount": estimate_token_count(output_text, model="gpt-5-codex"),
                            "messageId": message_id,
                        }
                    )
                    parent_ref = parent_id or call_id
                    if parent_ref:
                        chunks[-1]["parentId"] = parent_ref
                        chunks[-1]["branchParent"] = parent_ref

    # attachment extraction pass
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        lines = text.count("\n") + 1
        if lines <= LINE_THRESHOLD and len(text) <= CHAR_THRESHOLD:
            continue
        attachment_name = f"chunk{idx:03d}.txt"
        attachment_path = attachments_dir / attachment_name
        attachment_path.write_text(text, encoding="utf-8")
        rel = attachment_path.relative_to(markdown_path.parent)
        attachments.append(AttachmentInfo(attachment_name, str(rel), rel, attachment_path.stat().st_size, False))
        per_chunk_links.setdefault(idx, []).append((attachment_name, rel))
        chunk["text"] = _truncate_with_preview(text, attachment_name)

    for idx, chunk in enumerate(chunks):
        message_id = chunk.get("messageId") or f"{session_id}-msg-{idx}"
        while message_id in seen_message_ids:
            message_id = f"{message_id}-dup"
        seen_message_ids.add(message_id)
        parent_id = chunk.get("parentId") or (message_records[-1].message_id if message_records else None)
        if parent_id and "parentId" not in chunk:
            chunk["parentId"] = parent_id
            chunk["branchParent"] = parent_id
        message_records.append(
            MessageRecord(
                message_id=message_id,
                parent_id=parent_id,
                role=chunk.get("role", "model"),
                text=chunk.get("text", ""),
                token_count=int(chunk.get("tokenCount", 0) or 0),
                word_count=len((chunk.get("text", "") or "").split()),
                timestamp=chunk.get("timestamp"),
                attachments=len(per_chunk_links.get(idx, [])),
                chunk=chunk,
                links=list(per_chunk_links.get(idx, [])),
                metadata={"raw": {}},
            )
        )

    metadata = {"model": "gpt-5-codex"}
    extra_yaml = {
        "sourcePlatform": "codex",
        "sessionPath": str(session_path),
    }

    return process_conversation(
        provider="codex",
        conversation_id=session_id,
        slug=slug,
        title=title,
        message_records=message_records,
        attachments=attachments,
        canonical_leaf_id=message_records[-1].message_id if message_records else None,
        collapse_threshold=collapse_threshold,
        html=html,
        html_theme=html_theme,
        output_dir=output_dir,
        extra_yaml=extra_yaml,
        extra_state={
            "sessionPath": str(session_path),
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
        },
        force=force,
    )
