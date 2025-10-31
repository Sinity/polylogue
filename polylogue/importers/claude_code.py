from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..render import AttachmentInfo
from ..util import CLAUDE_CODE_PROJECT_ROOT, assign_conversation_slug
from ..conversation import process_conversation
from ..branching import MessageRecord
from .base import ImportResult
from .utils import estimate_token_count, normalise_inline_footnotes, store_large_text


DEFAULT_PROJECT_ROOT = CLAUDE_CODE_PROJECT_ROOT


def import_claude_code_session(
    session_id: str,
    *,
    base_dir: Path = DEFAULT_PROJECT_ROOT,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    force: bool = False,
) -> ImportResult:
    base_dir = base_dir.expanduser()
    session_path = _locate_session_file(session_id, base_dir)
    if session_path is None:
        raise FileNotFoundError(f"Claude Code session {session_id} not found under {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    title = session_path.stem
    slug = assign_conversation_slug("claude-code", session_id, title, id_hint=title[:8])
    conversation_dir = output_dir / slug
    conversation_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = conversation_dir / "conversation.md"
    attachments_dir = conversation_dir / "attachments"

    chunks: List[Dict] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    attachments: List[AttachmentInfo] = []
    message_records: List[MessageRecord] = []
    seen_message_ids: Set[str] = set()
    summaries: List[str] = []
    tool_results: Dict[str, int] = {}

    with session_path.open(encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                fallback = line.strip()
                if not fallback:
                    continue
                chunk_text = "Unparsed Claude Code log line\n```\n{}```".format(fallback)
                chunks.append(
                    {
                        "role": "model",
                        "text": chunk_text,
                        "tokenCount": estimate_token_count(chunk_text, model="claude-code"),
                    }
                )
                continue
            if not isinstance(entry, dict):
                chunk_text = "Unparsed Claude Code entry\n```\n{}```".format(entry)
                chunks.append(
                    {
                        "role": "model",
                        "text": chunk_text,
                        "tokenCount": estimate_token_count(chunk_text, model="claude-code"),
                    }
                )
                continue
            etype = entry.get("type")
            entry_id = (
                entry.get("id")
                or entry.get("uuid")
                or entry.get("tool_use_id")
                or entry.get("toolUseId")
                or entry.get("message", {}).get("id")
                or entry.get("message", {}).get("uuid")
                or f"line-{line_idx}"
            )
            parent_id = (
                entry.get("parentUuid")
                or entry.get("parent_uuid")
                or entry.get("parentId")
                or entry.get("parent_id")
                or entry.get("message", {}).get("parentUuid")
                or entry.get("message", {}).get("parent_id")
            )
            if etype == "summary":
                text = entry.get("summary")
                if text:
                    summaries.append(normalise_inline_footnotes(text))
                continue
            if etype == "user":
                text = _extract_text(entry)
                chunks.append(
                    {
                        "role": "user",
                        "text": text or "",
                        "tokenCount": estimate_token_count(text or "", model="claude-code"),
                        "messageId": entry_id,
                    }
                )
                if parent_id:
                    chunks[-1]["parentId"] = parent_id
                    chunks[-1]["branchParent"] = parent_id
            elif etype == "assistant":
                text = _extract_text(entry)
                chunks.append(
                    {
                        "role": "model",
                        "text": text or "",
                        "tokenCount": estimate_token_count(text or "", model="claude-code"),
                        "messageId": entry_id,
                    }
                )
                if parent_id:
                    chunks[-1]["parentId"] = parent_id
                    chunks[-1]["branchParent"] = parent_id
            elif etype == "tool_use":
                name = entry.get("name") or entry.get("toolName") or "tool"
                input_payload = entry.get("input") or entry.get("args") or {}
                text = f"Tool call `{name}`\n```json\n{json.dumps(input_payload, indent=2, ensure_ascii=False)}\n```"
                chunks.append(
                    {
                        "role": "model",
                        "text": text,
                        "tokenCount": estimate_token_count(text, model="claude-code"),
                        "messageId": entry_id,
                    }
                )
                tool_results[entry.get("id") or entry.get("toolUseId") or ""] = len(chunks) - 1
                if parent_id:
                    chunks[-1]["parentId"] = parent_id
                    chunks[-1]["branchParent"] = parent_id
            elif etype == "tool_result":
                result_text = _extract_tool_result(entry)
                tool_id = entry.get("tool_use_id") or entry.get("toolUseId") or entry.get("id")
                idx = tool_results.get(tool_id)
                if idx is not None:
                    original = chunks[idx]["text"]
                    combined = f"{original}\n\nResult:\n{result_text}"
                    chunks[idx]["text"] = combined
                    chunks[idx]["tokenCount"] = estimate_token_count(combined, model="claude-code")
                    if parent_id and "parentId" not in chunks[idx]:
                        chunks[idx]["parentId"] = parent_id
                        chunks[idx]["branchParent"] = parent_id
                else:
                    chunks.append(
                        {
                            "role": "tool",
                            "text": result_text,
                            "tokenCount": estimate_token_count(result_text, model="claude-code"),
                            "messageId": entry_id,
                        }
                    )
                    parent_ref = parent_id or tool_id
                    if parent_ref:
                        chunks[-1]["parentId"] = parent_ref
                        chunks[-1]["branchParent"] = parent_ref

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text") or ""
        preview = store_large_text(
            text,
            chunk_index=idx,
            attachments_dir=attachments_dir,
            markdown_dir=markdown_path.parent,
            attachments=attachments,
            per_chunk_links=per_chunk_links,
            prefix="claudecode",
        )
        if preview != text:
            chunk["text"] = preview
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
                metadata={},
            )
        )

    extra_yaml = {"sourcePlatform": "claude-code", "sessionFile": str(session_path)}
    if summaries:
        extra_yaml["summaries"] = summaries

    canonical_leaf_id = message_records[-1].message_id if message_records else None

    extra_state = {
        "sessionFile": str(session_path),
        "workspace": session_path.parent.name,
    }

    return process_conversation(
        provider="claude-code",
        conversation_id=session_id,
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
        extra_state=extra_state,
        source_file_id=session_path.name,
        modified_time=None,
        created_time=None,
        run_settings=None,
        source_mime="application/jsonl",
        source_size=session_path.stat().st_size,
        attachment_policy=None,
        force=force,
    )


def list_claude_code_sessions(base_dir: Path = DEFAULT_PROJECT_ROOT) -> List[Dict[str, str]]:
    sessions: List[Dict[str, str]] = []
    if not base_dir.exists():
        return sessions
    for path in base_dir.rglob("*.jsonl"):
        sessions.append(
            {
                "path": str(path),
                "name": path.stem,
                "workspace": path.parent.name,
            }
        )
    sessions.sort(key=lambda s: s["path"])
    return sessions


def _locate_session_file(session_id: str, base_dir: Path) -> Optional[Path]:
    candidate = Path(session_id)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if session_id.endswith(".jsonl"):
        path = base_dir / session_id
        if path.exists():
            return path
    if (base_dir / session_id).exists():
        return base_dir / session_id
    pattern = f"*{session_id}*.jsonl"
    matches = list(base_dir.rglob(pattern))
    if matches:
        return matches[0]
    return None


def _extract_text(entry: Dict) -> str:
    message = entry.get("message") or {}
    content = message.get("content") or []
    fragments: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    fragments.append(text)
            elif block.get("type") == "command_output":
                stdout = block.get("text") or block.get("output")
                if stdout:
                    fragments.append(f"````\n{stdout}\n````")
        elif isinstance(block, str):
            fragments.append(block)
    if not fragments and isinstance(message.get("text"), str):
        fragments.append(message["text"])
    return normalise_inline_footnotes("\n\n".join(fragments))


def _extract_tool_result(entry: Dict) -> str:
    text = entry.get("text") or entry.get("result") or ""
    if isinstance(text, list):
        fragments: List[str] = []
        for block in text:
            if isinstance(block, str):
                fragments.append(block)
            elif isinstance(block, dict) and block.get("text"):
                fragments.append(block["text"])
        text = "\n".join(fragments)
    return normalise_inline_footnotes(text)
