from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..render import AttachmentInfo
from ..util import CLAUDE_CODE_PROJECT_ROOT, assign_conversation_slug, path_order_key
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
    store_large_text,
)


DEFAULT_PROJECT_ROOT = CLAUDE_CODE_PROJECT_ROOT


def import_claude_code_session(
    session_id: str,
    *,
    base_dir: Path = DEFAULT_PROJECT_ROOT,
    output_dir: Path,
    collapse_threshold: int,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool,
    html_theme: str,
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
    session_paths, resolved_session_id, primary_path = _collect_session_files(session_id, base_dir)
    if not session_paths:
        raise FileNotFoundError(f"Claude Code session {session_id} not found under {base_dir}")

    conversation_id = conversation_id_override or resolved_session_id or session_id
    primary_path = primary_path or session_paths[0]

    # Store raw session data before parsing
    data_hashes: List[str] = []
    for session_path in session_paths:
        if not session_path.exists():
            continue
        raw_data = session_path.read_bytes()
        data_hash = store_raw_import(
            data=raw_data,
            provider="claude-code",
            conversation_id=conversation_id,
            source_path=session_path,
            metadata={"sessionFile": str(session_path)},
        )
        data_hashes.append(data_hash)

    output_dir.mkdir(parents=True, exist_ok=True)
    title = resolved_session_id or primary_path.stem
    slug = assign_conversation_slug("claude-code", conversation_id, title, id_hint=title[:8])
    conversation_dir = output_dir / slug
    conversation_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = conversation_dir / "conversation.md"
    attachments_dir = conversation_dir / "attachments"

    chunks: List[Dict] = []
    chunk_metadata: List[Dict[str, object]] = []
    per_chunk_links: Dict[int, List[Tuple[str, Path]]] = {}
    attachments: List[AttachmentInfo] = []
    message_records: List[MessageRecord] = []
    seen_message_ids: Set[str] = set()
    summaries: List[str] = []
    leaf_summaries: Dict[str, str] = {}
    tool_results: Dict[str, int] = {}
    routing_stats: Dict[str, int] = {"routed": 0, "skipped": 0}

    entries, min_ts, max_ts, agent_ids = _load_session_entries(session_paths)
    file_prefixes = {path: path.name for path in session_paths}
    has_non_summary = False
    for entry_item in entries:
        path = entry_item["path"]
        line_idx = entry_item["line_idx"]
        raw = entry_item.get("raw")
        entry = entry_item.get("entry")
        if entry is None:
            if not raw:
                continue
            chunk_text = f"Unparsed Claude Code log line\n```\n{raw}```"
            chunks.append(
                {
                    "role": "model",
                    "text": chunk_text,
                    "tokenCount": estimate_token_count(chunk_text, model="claude-code"),
                }
            )
            chunk_metadata.append({"raw": {"type": "unparsed", "payload": raw}})
            has_non_summary = True
            continue
        if not isinstance(entry, dict):
            chunk_text = f"Unparsed Claude Code entry\n```\n{entry}```"
            chunks.append(
                {
                    "role": "model",
                    "text": chunk_text,
                    "tokenCount": estimate_token_count(chunk_text, model="claude-code"),
                }
            )
            chunk_metadata.append({"raw": {"type": "unparsed", "payload": entry}})
            has_non_summary = True
            continue
        etype = entry.get("type")
        entry_id = (
            entry.get("id")
            or entry.get("uuid")
            or entry.get("tool_use_id")
            or entry.get("toolUseId")
            or entry.get("message", {}).get("id")
            or entry.get("message", {}).get("uuid")
            or f"{file_prefixes.get(path, path.name)}-line-{line_idx}"
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
                summary_text = normalise_inline_footnotes(text)
                summaries.append(summary_text)
                leaf_uuid = entry.get("leafUuid") or entry.get("leaf_uuid")
                if isinstance(leaf_uuid, str) and leaf_uuid not in leaf_summaries:
                    leaf_summaries[leaf_uuid] = summary_text
            continue
        has_non_summary = True
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
            chunk_metadata.append({"raw": entry})
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
            chunk_metadata.append({"raw": entry})
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
            chunk_metadata.append(
                {
                    "raw": entry,
                    "tool_calls": [
                        {
                            "type": "tool_use",
                            "name": name,
                            "id": entry.get("id") or entry.get("toolUseId"),
                            "input": input_payload,
                        }
                    ],
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
                chunk_meta = chunk_metadata[idx]
                calls = chunk_meta.setdefault("tool_calls", [])
                if calls:
                    calls[0]["output"] = result_text
                else:
                    calls.append({"type": "tool_result", "id": tool_id, "output": result_text})
            else:
                chunks.append(
                    {
                        "role": "tool",
                        "text": result_text,
                        "tokenCount": estimate_token_count(result_text, model="claude-code"),
                        "messageId": entry_id,
                    }
                )
                chunk_metadata.append(
                    {
                        "raw": entry,
                        "tool_calls": [
                            {
                                "type": "tool_result",
                                "id": tool_id,
                                "output": result_text,
                            }
                        ],
                    }
                )
                parent_ref = parent_id or tool_id
                if parent_ref:
                    chunks[-1]["parentId"] = parent_ref
                    chunks[-1]["branchParent"] = parent_ref

    if not has_non_summary and summaries:
        for data_hash in data_hashes:
            mark_parse_success(data_hash)
        return ImportResult(
            markdown_path=markdown_path,
            html_path=None,
            attachments_dir=None,
            document=None,
            slug=slug,
            skipped=True,
            skip_reason="summary-only log",
        )

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
            routing_stats=routing_stats,
        )
        if preview != text:
            chunk["text"] = preview
        parent_id = chunk.get("parentId") or (message_records[-1].message_id if message_records else None)
        if parent_id and "parentId" not in chunk:
            chunk["parentId"] = parent_id
            chunk["branchParent"] = parent_id
        chunk_meta = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
        links = list(per_chunk_links.get(idx, []))
        message_records.append(
            build_message_record(
                provider="claude-code",
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

    if message_records:
        workspace_index = _load_workspace_leaf_summaries(primary_path.parent)
        if workspace_index:
            for record in message_records:
                if record.message_id not in leaf_summaries:
                    summary = workspace_index.get(record.message_id)
                    if summary:
                        leaf_summaries[record.message_id] = summary
        present_ids = {record.message_id for record in message_records}
        leaf_summaries = {
            message_id: summary
            for message_id, summary in leaf_summaries.items()
            if message_id in present_ids
        }
        if leaf_summaries:
            for record in message_records:
                summary = leaf_summaries.get(record.message_id)
                if summary:
                    record.metadata["leaf_summary"] = summary

    extra_yaml = {"sourcePlatform": "claude-code", "sessionFile": str(primary_path)}
    if resolved_session_id:
        extra_yaml["sessionId"] = resolved_session_id
    if len(session_paths) > 1:
        extra_yaml["sessionFiles"] = [str(path) for path in session_paths]
    if agent_ids:
        extra_yaml["agentIds"] = sorted(agent_ids)
    if summaries:
        extra_yaml["summaries"] = summaries
    if leaf_summaries:
        extra_yaml["leafSummaries"] = leaf_summaries

    canonical_leaf_id = message_records[-1].message_id if message_records else None
    attachment_policy = {
        "previewLines": PREVIEW_LINES,
        "lineThreshold": LINE_THRESHOLD,
        "charThreshold": CHAR_THRESHOLD,
        "routing": routing_stats,
    }

    extra_state = {
        "sessionFile": str(primary_path),
        "workspace": primary_path.parent.name,
        "agentIds": sorted(agent_ids),
        "cliMeta": cli_meta,
    }
    if len(session_paths) > 1:
        extra_state["sessionFiles"] = [str(path) for path in session_paths]

    try:
        result = process_conversation(
            provider="claude-code",
            conversation_id=conversation_id,
            slug=slug,
            title=title,
            message_records=message_records,
            attachments=attachments,
            canonical_leaf_id=canonical_leaf_id,
            collapse_threshold=collapse_threshold,
            collapse_thresholds=collapse_thresholds,
            html=html,
            html_theme=html_theme,
            output_dir=output_dir,
            extra_yaml=extra_yaml,
            extra_state=extra_state,
            source_file_id=primary_path.name,
            modified_time=max_ts,
            created_time=min_ts,
            run_settings=None,
            source_mime="application/jsonl",
            source_size=sum(path.stat().st_size for path in session_paths),
            attachment_policy=attachment_policy,
            force=force,
            allow_dirty=allow_dirty,
            attachment_ocr=attachment_ocr,
            sanitize_html=sanitize_html,
            registrar=registrar,
        )

        # Mark parse success if we stored raw data
        for data_hash in data_hashes:
            mark_parse_success(data_hash)

        return result
    except Exception:
        # Mark parse failure if we stored raw data
        import traceback
        error = traceback.format_exc()
        for data_hash in data_hashes:
            mark_parse_failed(data_hash, error)
        raise


def list_claude_code_sessions(base_dir: Path = DEFAULT_PROJECT_ROOT) -> List[Dict[str, str]]:
    base_dir = base_dir.expanduser()
    sessions: List[Dict[str, str]] = []
    if not base_dir.exists():
        return sessions
    grouped: Dict[str, Dict[str, object]] = {}
    for path in sorted(base_dir.rglob("*.jsonl"), key=path_order_key, reverse=True):
        if _load_summary_only_file(path):
            continue
        session_key = _extract_session_id_from_file(path) or path.stem
        group = grouped.setdefault(
            session_key,
            {
                "paths": [],
                "primary": None,
                "workspace": None,
                "last_mtime": 0.0,
            },
        )
        group["paths"].append(path)
        if group["primary"] is None or _prefer_primary_path(path, group["primary"], session_key):
            group["primary"] = path
            group["workspace"] = path.parent.name
        try:
            group["last_mtime"] = max(group["last_mtime"], path.stat().st_mtime)
        except OSError:
            pass
    for session_key, group in grouped.items():
        primary = group["primary"]
        if primary is None:
            continue
        sessions.append(
            {
                "path": str(primary),
                "name": session_key,
                "workspace": group["workspace"] or primary.parent.name,
            }
        )
    sessions.sort(key=lambda entry: grouped.get(entry["name"], {}).get("last_mtime", 0.0), reverse=True)
    return sessions


def extract_claude_code_session_id(path: Path) -> Optional[str]:
    return _extract_session_id_from_file(path)


def _collect_session_files(session_id: str, base_dir: Path) -> Tuple[List[Path], Optional[str], Optional[Path]]:
    candidate = Path(session_id)
    primary: Optional[Path] = None
    if candidate.is_absolute() and candidate.exists():
        primary = candidate
    elif session_id.endswith(".jsonl"):
        path = base_dir / session_id
        if path.exists():
            primary = path
    elif (base_dir / session_id).exists():
        primary = base_dir / session_id
    else:
        pattern = f"*{session_id}*.jsonl"
        matches = list(base_dir.rglob(pattern))
        if matches:
            primary = max(matches, key=path_order_key)

    resolved_session_id = _extract_session_id_from_file(primary) if primary else None
    session_paths: List[Path] = []

    if primary is None:
        for path in base_dir.rglob("agent-*.jsonl"):
            if _extract_session_id_from_file(path) == session_id:
                session_paths.append(path)
        if session_paths:
            primary = max(session_paths, key=path_order_key)
            return _dedupe_paths(session_paths), session_id, primary
        return [], None, None

    session_paths.append(primary)
    lookup_id = resolved_session_id or session_id
    for path in base_dir.rglob("agent-*.jsonl"):
        if _extract_session_id_from_file(path) == lookup_id:
            session_paths.append(path)
    return _dedupe_paths(session_paths), resolved_session_id, primary


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


def _parse_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value > 1e12:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        ts = value.strip()
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return None


def _extract_session_id_from_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                session_id = entry.get("sessionId") or entry.get("session_id")
                if session_id:
                    return str(session_id)
    except OSError:
        return None
    return None


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    ordered: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _load_summary_only_file(path: Path) -> Optional[Dict[str, str]]:
    summaries: Dict[str, str] = {}
    saw_summary = False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if not isinstance(entry, dict):
                    return None
                if entry.get("type") != "summary":
                    return None
                saw_summary = True
                leaf_uuid = entry.get("leafUuid") or entry.get("leaf_uuid")
                summary = entry.get("summary")
                if isinstance(leaf_uuid, str) and isinstance(summary, str):
                    if leaf_uuid not in summaries:
                        summaries[leaf_uuid] = normalise_inline_footnotes(summary)
    except OSError:
        return None
    return summaries if saw_summary else None


def _load_workspace_leaf_summaries(workspace_dir: Path) -> Dict[str, str]:
    summaries: Dict[str, str] = {}
    try:
        candidates = sorted(workspace_dir.glob("*.jsonl"), key=path_order_key, reverse=True)
    except OSError:
        return summaries
    for path in candidates:
        if path.name.startswith("agent-"):
            continue
        summary_map = _load_summary_only_file(path)
        if not summary_map:
            continue
        for leaf_uuid, summary in summary_map.items():
            summaries.setdefault(leaf_uuid, summary)
    return summaries


def _prefer_primary_path(candidate: Path, current: Optional[Path], session_key: str) -> bool:
    if current is None:
        return True
    if candidate.stem == session_key and current.stem != session_key:
        return True
    if candidate.stem != session_key and current.stem == session_key:
        return False
    return path_order_key(candidate) > path_order_key(current)


def _load_session_entries(
    session_paths: List[Path],
) -> Tuple[List[Dict[str, object]], Optional[str], Optional[str], Set[str]]:
    entries: List[Dict[str, object]] = []
    min_ts: Optional[datetime] = None
    max_ts: Optional[datetime] = None
    agent_ids: Set[str] = set()

    path_index = {path: idx for idx, path in enumerate(session_paths)}
    for path in session_paths:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line_idx, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        entries.append(
                            {
                                "path": path,
                                "line_idx": line_idx,
                                "raw": line.strip(),
                                "entry": None,
                            }
                        )
                        continue
                    if isinstance(entry, dict):
                        agent_id = entry.get("agentId") or entry.get("agent_id")
                        if agent_id:
                            agent_ids.add(str(agent_id))
                        ts = _parse_timestamp(entry.get("timestamp"))
                        if ts:
                            min_ts = ts if min_ts is None or ts < min_ts else min_ts
                            max_ts = ts if max_ts is None or ts > max_ts else max_ts
                        entries.append(
                            {
                                "path": path,
                                "line_idx": line_idx,
                                "entry": entry,
                                "raw": None,
                                "timestamp": ts,
                            }
                        )
                    else:
                        entries.append(
                            {
                                "path": path,
                                "line_idx": line_idx,
                                "entry": entry,
                                "raw": None,
                                "timestamp": None,
                            }
                        )
        except OSError:
            continue

    if len(session_paths) > 1:
        entries.sort(
            key=lambda item: (
                1 if item.get("timestamp") is None else 0,
                item.get("timestamp") or datetime.max.replace(tzinfo=timezone.utc),
                path_index.get(item["path"], 0),
                item["line_idx"],
            )
        )

    min_ts_str = min_ts.isoformat() if min_ts else None
    max_ts_str = max_ts.isoformat() if max_ts else None
    return entries, min_ts_str, max_ts_str, agent_ids
