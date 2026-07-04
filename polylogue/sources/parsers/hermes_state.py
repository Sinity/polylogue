"""Hermes ``state.db`` parser.

Hermes's durable session source is the SQLite database under
``~/.hermes/state.db``.  The older JSON document parser in
``local_agent.py`` remains useful for exported snapshots, but this parser owns
the authoritative live state shape.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.core.json import JSONDocument, json_document

from .base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from .local_agent import _content_blocks_from_content, _content_text, _tool_use_block

HERMES_STATE_DB_MARKER = "hermes_state_db"
_CONTENT_JSON_PREFIX = "\x00json:"
_SESSION_COLUMNS = frozenset(
    {
        "id",
        "model",
        "model_config",
        "system_prompt",
        "parent_session_id",
        "started_at",
        "ended_at",
        "end_reason",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "cwd",
        "git_branch",
        "git_repo_root",
        "estimated_cost_usd",
        "actual_cost_usd",
        "title",
        "api_call_count",
        "rewind_count",
        "archived",
    }
)
_MESSAGE_COLUMNS = frozenset(
    {
        "id",
        "session_id",
        "role",
        "content",
        "tool_call_id",
        "tool_calls",
        "tool_name",
        "timestamp",
        "token_count",
        "finish_reason",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
        "platform_message_id",
        "observed",
        "active",
        "compacted",
    }
)


def marker_payload(path: Path) -> JSONDocument:
    """Return the JSON marker that routes a raw SQLite blob to this parser."""
    return {"polylogue_artifact": HERMES_STATE_DB_MARKER, "state_db_path": str(path)}


def looks_like_state_db_payload(payload: JSONDocument) -> bool:
    return payload.get("polylogue_artifact") == HERMES_STATE_DB_MARKER and isinstance(payload.get("state_db_path"), str)


def looks_like_state_db_path(path: Path) -> bool:
    """Return true when *path* is a readable Hermes state database."""
    try:
        with _connect_readonly(path) as conn:
            return _has_required_tables(conn)
    except sqlite3.Error:
        return False


def parse_state_db_payload(payload: JSONDocument, fallback_id: str) -> list[ParsedSession]:
    path_value = payload.get("state_db_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Hermes state.db marker is missing state_db_path")
    return parse_state_db(Path(path_value), fallback_id=fallback_id)


def parse_state_db(path: Path, *, fallback_id: str | None = None) -> list[ParsedSession]:
    """Parse all active sessions from a Hermes ``state.db`` file."""
    del fallback_id
    with _connect_readonly(path) as conn:
        if not _has_required_tables(conn):
            raise ValueError(f"{path} is not a Hermes state.db file")
        sessions = [
            _parse_session_row(conn, row)
            for row in conn.execute(
                """
                SELECT *
                FROM sessions
                ORDER BY COALESCE(started_at, 0), id
                """
            ).fetchall()
        ]
    return [session for session in sessions if session.messages or session.instructions_text]


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _has_required_tables(conn: sqlite3.Connection) -> bool:
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'messages')"
        ).fetchall()
    }
    if tables != {"sessions", "messages"}:
        return False
    return _SESSION_COLUMNS.issubset(_columns(conn, "sessions")) and _MESSAGE_COLUMNS.issubset(
        _columns(conn, "messages")
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _parse_session_row(conn: sqlite3.Connection, row: sqlite3.Row) -> ParsedSession:
    session_id = str(row["id"])
    messages: list[ParsedMessage] = []
    system_prompt = _optional_text(row["system_prompt"])
    model_name = _optional_text(row["model"])
    if system_prompt:
        messages.append(
            ParsedMessage(
                provider_message_id=f"{session_id}:system",
                role=Role.SYSTEM,
                text=system_prompt,
                timestamp=_epoch_iso(row["started_at"]),
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=system_prompt)],
                position=0,
                variant_index=0,
                is_active_path=True,
                model_name=model_name,
            )
        )
    for index, message_row in enumerate(
        conn.execute(
            """
            SELECT *
            FROM messages
            WHERE session_id = ?
              AND COALESCE(active, 1) = 1
            ORDER BY id
            """,
            (session_id,),
        ).fetchall(),
        start=1,
    ):
        parsed = _parse_message_row(message_row, index=index, position=len(messages), fallback_model=model_name)
        if parsed is not None:
            messages.append(parsed)
    messages = _mark_active_leaf(messages)
    parent_id = _optional_text(row["parent_session_id"])
    return ParsedSession(
        source_name=Provider.HERMES,
        provider_session_id=session_id,
        title=_optional_text(row["title"]) or session_id,
        created_at=_epoch_iso(row["started_at"]),
        updated_at=_epoch_iso(row["ended_at"]) or _latest_message_timestamp(messages),
        messages=messages,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
        session_events=_usage_and_lifecycle_events(row, messages),
        parent_session_provider_id=parent_id,
        branch_type=_branch_type(row) if parent_id else None,
        instructions_text=system_prompt,
        reported_cost_usd=_reported_cost(row),
        models_used=[model_name] if model_name else [],
        working_directories=[cwd] if (cwd := _optional_text(row["cwd"])) else [],
        git_branch=_optional_text(row["git_branch"]),
        git_repository_url=_optional_text(row["git_repo_root"]),
        ingest_flags=["hermes:state-db"],
    )


def _parse_message_row(
    row: sqlite3.Row,
    *,
    index: int,
    position: int,
    fallback_model: str | None,
) -> ParsedMessage | None:
    content = _decode_content(row["content"])
    text = _content_text(content)
    blocks = _content_blocks_from_content(content)
    reasoning = _optional_text(row["reasoning_content"]) or _optional_text(row["reasoning"])
    if reasoning:
        metadata = _reasoning_metadata(row)
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=reasoning, metadata=metadata or None))
    for tool_index, tool_call in enumerate(_json_list(row["tool_calls"]), start=1):
        tool_record = json_document(tool_call)
        if tool_record:
            blocks.append(_tool_use_block(tool_record, fallback_id=f"tool-{row['id']}-{tool_index}"))
    role = Role.normalize(_optional_text(row["role"]) or "unknown")
    tool_call_id = _optional_text(row["tool_call_id"])
    if role is Role.TOOL and text:
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_call_id,
                tool_name=_optional_text(row["tool_name"]),
                text=text,
            )
        )
    if not text and not blocks:
        return None
    token_count = _non_negative_int(row["token_count"]) or 0
    return ParsedMessage(
        provider_message_id=_message_provider_id(row),
        role=role,
        text=text,
        timestamp=_epoch_iso(row["timestamp"]),
        blocks=blocks or [ParsedContentBlock(type=BlockType.TEXT, text=text)],
        position=position,
        variant_index=0,
        is_active_path=True,
        model_name=fallback_model,
        output_tokens=token_count if role is Role.ASSISTANT else 0,
        input_tokens=token_count if role is Role.USER else 0,
        end_turn=_optional_bool(row["finish_reason"] != "tool_calls"),
    )


def _message_provider_id(row: sqlite3.Row) -> str:
    platform_id = _optional_text(row["platform_message_id"])
    if platform_id:
        return platform_id
    return f"{row['session_id']}:message:{row['id']}"


def _usage_and_lifecycle_events(row: sqlite3.Row, messages: list[ParsedMessage]) -> list[ParsedSessionEvent]:
    events: list[ParsedSessionEvent] = []
    total_usage: dict[str, int] = {
        "input_tokens": _non_negative_int(row["input_tokens"]) or 0,
        "output_tokens": _non_negative_int(row["output_tokens"]) or 0,
        "cached_input_tokens": _non_negative_int(row["cache_read_tokens"]) or 0,
        "cache_write_tokens": _non_negative_int(row["cache_write_tokens"]) or 0,
        "reasoning_output_tokens": _non_negative_int(row["reasoning_tokens"]) or 0,
    }
    total_usage["total_tokens"] = sum(total_usage.values())
    if any(total_usage.values()):
        events.append(
            ParsedSessionEvent(
                event_type="token_count",
                timestamp=_epoch_iso(row["ended_at"]) or _latest_message_timestamp(messages),
                payload={
                    "type": "token_count",
                    "model": _optional_text(row["model"]),
                    "total_token_usage": total_usage,
                    "api_call_count": _non_negative_int(row["api_call_count"]) or 0,
                    "estimated_cost_usd": _optional_float(row["estimated_cost_usd"]),
                    "actual_cost_usd": _optional_float(row["actual_cost_usd"]),
                },
            )
        )
    end_reason = _optional_text(row["end_reason"])
    if end_reason in {"compression", "compaction"}:
        events.append(
            ParsedSessionEvent(
                event_type="compaction",
                timestamp=_epoch_iso(row["ended_at"]),
                payload={"summary": f"Hermes session ended via {end_reason}", "end_reason": end_reason},
            )
        )
    if _non_negative_int(row["rewind_count"]):
        events.append(
            ParsedSessionEvent(
                event_type="rewind",
                timestamp=_epoch_iso(row["ended_at"]),
                payload={
                    "summary": "Hermes session was rewound",
                    "rewind_count": _non_negative_int(row["rewind_count"]),
                },
            )
        )
    return events


def _branch_type(row: sqlite3.Row) -> BranchType | None:
    config = _json_mapping(row["model_config"])
    if config.get("_branched_from") is not None:
        return BranchType.FORK
    if config.get("_delegate_from") is not None:
        return BranchType.SUBAGENT
    if _optional_text(row["end_reason"]) == "compression":
        return BranchType.CONTINUATION
    return BranchType.CONTINUATION


def _reported_cost(row: sqlite3.Row) -> float | None:
    actual = _optional_float(row["actual_cost_usd"])
    estimated = _optional_float(row["estimated_cost_usd"])
    return actual if actual is not None else estimated


def _reasoning_metadata(row: sqlite3.Row) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("reasoning_details", "codex_reasoning_items", "codex_message_items"):
        parsed = _json_value(row[key])
        if parsed is not None:
            metadata[key] = parsed
    return metadata


def _decode_content(value: object) -> object:
    if isinstance(value, str) and value.startswith(_CONTENT_JSON_PREFIX):
        try:
            return json.loads(value[len(_CONTENT_JSON_PREFIX) :])
        except json.JSONDecodeError:
            return value
    return value


def _json_mapping(value: object) -> dict[str, object]:
    parsed = _json_value(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _json_list(value: object) -> list[object]:
    parsed = _json_value(value)
    return parsed if isinstance(parsed, list) else []


def _json_value(value: object) -> object | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return cast(object, json.loads(value))
    except json.JSONDecodeError:
        return None


def _epoch_iso(value: object) -> str | None:
    seconds = _optional_float(value)
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def _latest_message_timestamp(messages: list[ParsedMessage]) -> str | None:
    for message in reversed(messages):
        if message.timestamp:
            return message.timestamp
    return None


def _mark_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    if not messages:
        return messages
    leaf = messages[-1].provider_message_id
    return [message.model_copy(update={"is_active_leaf": message.provider_message_id == leaf}) for message in messages]


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        parsed = int(value)
        return parsed if parsed >= 0 else None
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed >= 0 else None
    return None


__all__ = [
    "HERMES_STATE_DB_MARKER",
    "looks_like_state_db_path",
    "looks_like_state_db_payload",
    "marker_payload",
    "parse_state_db",
    "parse_state_db_payload",
]
