"""Hermes ``state.db`` parser.

Hermes's durable session source is the SQLite database under
``~/.hermes/state.db``.  The older JSON document parser in
``local_agent.py`` remains useful for exported snapshots, but this parser owns
the authoritative live state shape.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONDocument, json_document

from .base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from .local_agent import _content_blocks_from_content, _content_text, _tool_use_block

HERMES_STATE_DB_MARKER = "hermes_state_db"
_CONTENT_JSON_PREFIX = "\x00json:"
_REQUIRED_SESSION_COLUMNS = frozenset(
    {
        "id",
        "started_at",
    }
)
_REQUIRED_MESSAGE_COLUMNS = frozenset(
    {
        "id",
        "session_id",
        "role",
        "content",
        "timestamp",
    }
)
_HERMES_SIGNATURE_SESSION_COLUMNS = frozenset({"source", "model_config", "parent_session_id"})
_HERMES_SIGNATURE_MESSAGE_COLUMNS = frozenset({"tool_calls", "observed", "active", "compacted"})

_SESSION_CAPABILITIES: dict[str, frozenset[str]] = {
    "model": frozenset({"model", "model_config", "system_prompt"}),
    "lineage": frozenset({"parent_session_id", "end_reason"}),
    "usage": frozenset(
        {
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "api_call_count",
        }
    ),
    "cost_provenance": frozenset(
        {
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "pricing_version",
        }
    ),
    "repository": frozenset({"cwd", "git_branch", "git_repo_root"}),
    "lifecycle": frozenset({"ended_at", "end_reason", "rewind_count", "archived"}),
    "handoff": frozenset({"handoff_state", "handoff_platform", "handoff_error"}),
    "source_identity": frozenset({"source", "user_id"}),
}
_MESSAGE_CAPABILITIES: dict[str, frozenset[str]] = {
    "tooling": frozenset({"tool_call_id", "tool_calls", "tool_name", "finish_reason"}),
    "reasoning": frozenset(
        {
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
        }
    ),
    "provider_identity": frozenset({"platform_message_id"}),
    "message_state": frozenset({"observed", "active", "compacted"}),
    "usage": frozenset({"token_count"}),
}

_COST_FIELDS = (
    "estimated_cost_usd",
    "actual_cost_usd",
    "cost_status",
    "cost_source",
    "pricing_version",
    "billing_provider",
    "billing_base_url",
    "billing_mode",
)
_SESSION_METADATA_FIELDS = (
    "source",
    "user_id",
    "handoff_state",
    "handoff_platform",
    "handoff_error",
    "archived",
)


def marker_payload(path: Path, *, profile_root: Path | None = None) -> JSONDocument:
    """Return the JSON marker that routes a raw SQLite blob to this parser."""
    payload: JSONDocument = {
        "polylogue_artifact": HERMES_STATE_DB_MARKER,
        "state_db_path": str(path),
    }
    if profile_root is not None:
        payload["profile_root"] = str(profile_root)
    return payload


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
    profile_value = payload.get("profile_root")
    profile_root = Path(profile_value) if isinstance(profile_value, str) and profile_value else None
    return parse_state_db(Path(path_value), fallback_id=fallback_id, profile_root=profile_root)


def parse_state_db(
    path: Path,
    *,
    fallback_id: str | None = None,
    profile_root: Path | None = None,
) -> list[ParsedSession]:
    """Parse every session revision from a Hermes ``state.db`` file."""
    del fallback_id
    with _connect_readonly(path) as conn:
        if not _has_required_tables(conn):
            raise ValueError(f"{path} is not a Hermes state.db file")
        session_columns = _columns(conn, "sessions")
        message_columns = _columns(conn, "messages")
        schema_version = _schema_version(conn)
        resolved_profile_root = profile_root or path.parent
        sessions = [
            _parse_session_row(
                conn,
                row,
                profile_root=resolved_profile_root,
                schema_version=schema_version,
                session_columns=session_columns,
                message_columns=message_columns,
            )
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
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('schema_version', 'sessions', 'messages')"
        ).fetchall()
    }
    if tables != {"schema_version", "sessions", "messages"}:
        return False
    session_columns = _columns(conn, "sessions")
    message_columns = _columns(conn, "messages")
    return (
        _REQUIRED_SESSION_COLUMNS.issubset(session_columns)
        and _REQUIRED_MESSAGE_COLUMNS.issubset(message_columns)
        and _HERMES_SIGNATURE_SESSION_COLUMNS.issubset(session_columns)
        and _HERMES_SIGNATURE_MESSAGE_COLUMNS.issubset(message_columns)
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _schema_version(conn: sqlite3.Connection) -> int | None:
    if "schema_version" not in {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }:
        return None
    row = conn.execute("SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1").fetchone()
    return _non_negative_int(row[0]) if row else None


def _capabilities(columns: set[str], capability_map: Mapping[str, frozenset[str]]) -> list[str]:
    return sorted(name for name, fields in capability_map.items() if fields.issubset(columns))


def _profile_key(profile_root: Path) -> str:
    normalized = str(profile_root.expanduser().resolve(strict=False))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _qualified_session_id(raw_session_id: str, profile_key: str) -> str:
    return f"{raw_session_id}@profile-{profile_key}"


def _identity_event(
    *,
    raw_session_id: str,
    profile_key: str,
    schema_version: int | None,
    session_columns: set[str],
    message_columns: set[str],
) -> ParsedSessionEvent:
    return ParsedSessionEvent(
        event_type="hermes_identity",
        payload={
            "raw_session_id": raw_session_id,
            "profile_key": profile_key,
            "schema_version": schema_version,
            "session_capabilities": _capabilities(session_columns, _SESSION_CAPABILITIES),
            "message_capabilities": _capabilities(message_columns, _MESSAGE_CAPABILITIES),
        },
    )


def _session_metadata_events(
    row: sqlite3.Row,
    *,
    session_columns: set[str],
) -> list[ParsedSessionEvent]:
    payload = {field: _row_value(row, field) for field in _SESSION_METADATA_FIELDS if field in session_columns}
    if not payload:
        return []
    return [
        ParsedSessionEvent(
            event_type="hermes_session_metadata",
            timestamp=_epoch_iso(_row_value(row, "ended_at")) or _epoch_iso(_row_value(row, "started_at")),
            payload=payload,
        )
    ]


def _message_state_event(
    row: sqlite3.Row,
    message: ParsedMessage,
    *,
    message_columns: set[str],
) -> ParsedSessionEvent:
    active = _sqlite_bool(_row_value(row, "active"), default=True)
    observed = _sqlite_bool(_row_value(row, "observed"), default=False)
    compacted = _sqlite_bool(_row_value(row, "compacted"), default=False)
    if compacted:
        state = "compacted"
    elif not active:
        state = "rewound"
    elif observed:
        state = "observed"
    else:
        state = "active"
    return ParsedSessionEvent(
        event_type="hermes_message_state",
        timestamp=message.timestamp,
        source_message_provider_id=message.provider_message_id,
        payload={
            "state": state,
            "active": active,
            "observed": observed,
            "compacted": compacted,
            "capability_present": "message_state" in _capabilities(message_columns, _MESSAGE_CAPABILITIES),
        },
    )


def _material_origin(role: Role, *, observed: bool) -> MaterialOrigin:
    if observed:
        return MaterialOrigin.RUNTIME_CONTEXT
    if role is Role.USER:
        return MaterialOrigin.HUMAN_AUTHORED
    if role is Role.ASSISTANT:
        return MaterialOrigin.ASSISTANT_AUTHORED
    if role is Role.TOOL:
        return MaterialOrigin.TOOL_RESULT
    if role is Role.SYSTEM:
        return MaterialOrigin.RUNTIME_PROTOCOL
    return MaterialOrigin.UNKNOWN


def _row_value(row: sqlite3.Row, key: str) -> object | None:
    try:
        return cast(object, row[key])
    except IndexError:
        return None


def _sqlite_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return default


def _parse_session_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    profile_root: Path,
    schema_version: int | None,
    session_columns: set[str],
    message_columns: set[str],
) -> ParsedSession:
    raw_session_id = str(row["id"])
    profile_key = _profile_key(profile_root)
    session_id = _qualified_session_id(raw_session_id, profile_key)
    messages: list[ParsedMessage] = []
    state_events: list[ParsedSessionEvent] = []
    system_prompt = _optional_text(_row_value(row, "system_prompt"))
    model_name = _optional_text(_row_value(row, "model"))
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
    for message_row in conn.execute(
        """
            SELECT *
            FROM messages
            WHERE session_id = ?
            ORDER BY id
            """,
        (raw_session_id,),
    ).fetchall():
        parsed = _parse_message_row(message_row, position=len(messages), fallback_model=model_name)
        if parsed is not None:
            messages.append(parsed)
            state_events.append(_message_state_event(message_row, parsed, message_columns=message_columns))
    messages = _mark_active_leaf(messages)
    parent_raw_id = _optional_text(_row_value(row, "parent_session_id"))
    parent_id = _qualified_session_id(parent_raw_id, profile_key) if parent_raw_id else None
    session_events = [
        _identity_event(
            raw_session_id=raw_session_id,
            profile_key=profile_key,
            schema_version=schema_version,
            session_columns=session_columns,
            message_columns=message_columns,
        ),
        *_usage_and_lifecycle_events(row, messages, session_columns=session_columns),
        *_session_metadata_events(row, session_columns=session_columns),
        *state_events,
    ]
    active_leaf_id = next(
        (message.provider_message_id for message in reversed(messages) if message.is_active_path),
        None,
    )
    return ParsedSession(
        source_name=Provider.HERMES,
        provider_session_id=session_id,
        title=_optional_text(_row_value(row, "title")) or raw_session_id,
        created_at=_epoch_iso(row["started_at"]),
        updated_at=_epoch_iso(_row_value(row, "ended_at")) or _latest_message_timestamp(messages),
        messages=messages,
        active_leaf_message_provider_id=active_leaf_id,
        session_events=session_events,
        parent_session_provider_id=parent_id,
        branch_type=_branch_type(row) if parent_id else None,
        instructions_text=system_prompt,
        reported_cost_usd=_reported_cost(row),
        models_used=[model_name] if model_name else [],
        working_directories=[cwd] if (cwd := _optional_text(_row_value(row, "cwd"))) else [],
        git_branch=_optional_text(_row_value(row, "git_branch")),
        git_repository_url=_optional_text(_row_value(row, "git_repo_root")),
        ingest_flags=["hermes:state-db", f"hermes:schema-v{schema_version or 'unknown'}"],
    )


def _parse_message_row(
    row: sqlite3.Row,
    *,
    position: int,
    fallback_model: str | None,
) -> ParsedMessage | None:
    content = _decode_content(row["content"])
    text = _content_text(content)
    blocks = _content_blocks_from_content(content)
    reasoning = _optional_text(_row_value(row, "reasoning_content")) or _optional_text(_row_value(row, "reasoning"))
    if reasoning:
        metadata = _reasoning_metadata(row)
        blocks.append(ParsedContentBlock(type=BlockType.THINKING, text=reasoning, metadata=metadata or None))
    for tool_index, tool_call in enumerate(_json_list(_row_value(row, "tool_calls")), start=1):
        tool_record = json_document(tool_call)
        if tool_record:
            blocks.append(_tool_use_block(tool_record, fallback_id=f"tool-{row['id']}-{tool_index}"))
    role = Role.normalize(_optional_text(row["role"]) or "unknown")
    tool_call_id = _optional_text(_row_value(row, "tool_call_id"))
    if role is Role.TOOL and text:
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_call_id,
                tool_name=_optional_text(_row_value(row, "tool_name")),
                text=text,
            )
        )
    if not text and not blocks:
        return None
    token_count = _non_negative_int(_row_value(row, "token_count")) or 0
    observed = _sqlite_bool(_row_value(row, "observed"), default=False)
    active = _sqlite_bool(_row_value(row, "active"), default=True)
    return ParsedMessage(
        provider_message_id=_message_provider_id(row),
        role=role,
        text=text,
        timestamp=_epoch_iso(row["timestamp"]),
        blocks=blocks or [ParsedContentBlock(type=BlockType.TEXT, text=text)],
        position=position,
        variant_index=0,
        is_active_path=active,
        material_origin=_material_origin(role, observed=observed),
        model_name=fallback_model,
        output_tokens=token_count if role is Role.ASSISTANT else 0,
        input_tokens=token_count if role is Role.USER else 0,
        end_turn=_optional_bool(_row_value(row, "finish_reason") != "tool_calls"),
    )


def _message_provider_id(row: sqlite3.Row) -> str:
    platform_id = _optional_text(_row_value(row, "platform_message_id"))
    if platform_id:
        return platform_id
    return f"{row['session_id']}:message:{row['id']}"


def _usage_and_lifecycle_events(
    row: sqlite3.Row,
    messages: list[ParsedMessage],
    *,
    session_columns: set[str],
) -> list[ParsedSessionEvent]:
    events: list[ParsedSessionEvent] = []
    total_usage: dict[str, int] = {
        "input_tokens": _non_negative_int(_row_value(row, "input_tokens")) or 0,
        "output_tokens": _non_negative_int(_row_value(row, "output_tokens")) or 0,
        "cached_input_tokens": _non_negative_int(_row_value(row, "cache_read_tokens")) or 0,
        "cache_write_tokens": _non_negative_int(_row_value(row, "cache_write_tokens")) or 0,
        "reasoning_output_tokens": _non_negative_int(_row_value(row, "reasoning_tokens")) or 0,
    }
    total_usage["total_tokens"] = sum(total_usage.values())
    has_cost_evidence = any(field in session_columns for field in _COST_FIELDS)
    if any(total_usage.values()) or has_cost_evidence:
        cost_payload = {field: _row_value(row, field) for field in _COST_FIELDS if field in session_columns}
        events.append(
            ParsedSessionEvent(
                event_type="token_count",
                timestamp=_epoch_iso(_row_value(row, "ended_at")) or _latest_message_timestamp(messages),
                payload={
                    "type": "token_count",
                    "model": _optional_text(_row_value(row, "model")),
                    "total_token_usage": total_usage,
                    "api_call_count": _non_negative_int(_row_value(row, "api_call_count")) or 0,
                    **cost_payload,
                },
            )
        )
    end_reason = _optional_text(_row_value(row, "end_reason"))
    if end_reason in {"compression", "compaction"}:
        events.append(
            ParsedSessionEvent(
                event_type="compaction",
                timestamp=_epoch_iso(_row_value(row, "ended_at")),
                payload={"summary": f"Hermes session ended via {end_reason}", "end_reason": end_reason},
            )
        )
    if _non_negative_int(_row_value(row, "rewind_count")):
        events.append(
            ParsedSessionEvent(
                event_type="rewind",
                timestamp=_epoch_iso(_row_value(row, "ended_at")),
                payload={
                    "summary": "Hermes session was rewound",
                    "rewind_count": _non_negative_int(_row_value(row, "rewind_count")),
                },
            )
        )
    return events


def _branch_type(row: sqlite3.Row) -> BranchType | None:
    config = _json_mapping(_row_value(row, "model_config"))
    if config.get("_branched_from") is not None:
        return BranchType.FORK
    if config.get("_delegate_from") is not None:
        return BranchType.SUBAGENT
    if _optional_text(_row_value(row, "end_reason")) == "compression":
        return BranchType.CONTINUATION
    return BranchType.CONTINUATION


def _reported_cost(row: sqlite3.Row) -> float | None:
    actual = _optional_float(_row_value(row, "actual_cost_usd"))
    estimated = _optional_float(_row_value(row, "estimated_cost_usd"))
    return actual if actual is not None else estimated


def _reasoning_metadata(row: sqlite3.Row) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ("reasoning_details", "codex_reasoning_items", "codex_message_items"):
        parsed = _json_value(_row_value(row, key))
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
    leaf = next(
        (message.provider_message_id for message in reversed(messages) if message.is_active_path),
        None,
    )
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
