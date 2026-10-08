"""The orchestration evidence read shared by every public surface.

Each reader is a keyset stream over the session's own rows, so the evidence
builder holds one page per relation at a time instead of the whole session.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING

from polylogue.archive.message.models import Message
from polylogue.archive.session.events import SessionEvent
from polylogue.core.async_bridge import complete_without_suspension
from polylogue.core.identity_law import transcript_order_sql
from polylogue.storage.hydrators import message_from_record, session_event_from_record
from polylogue.storage.runtime import BlockRecord
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import MESSAGES_SPEC
from polylogue.storage.sqlite.queries.mappers import _row_to_content_block
from polylogue.storage.sqlite.queries.mappers_archive import bind_message_row_mapper
from polylogue.storage.sqlite.queries.session_events import (
    _row_to_session_event,
    hydrate_session_event_array_items,
)

if TYPE_CHECKING:
    from polylogue.analysis.orchestration_evidence import SessionOrchestrationEvidence
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_PAGE_SIZE = 200
_MESSAGE_SELECT = MESSAGES_SPEC.record_select_column_names("m")
_TRANSCRIPT_ORDER = transcript_order_sql("m")


def iter_orchestration_messages(conn: sqlite3.Connection, session_id: str) -> Iterator[Message]:
    """Stream the session's own messages in transcript order with their tool-use blocks.

    Inherited lineage prefixes are not composed: the projection excludes
    inherited dialogue, and the rows stored under ``session_id`` are exactly
    the session's own divergent tail.
    """
    origin_row = conn.execute("SELECT origin FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    origin = str(origin_row["origin"]) if origin_row is not None else None
    cursor_sql = ""
    cursor_params: tuple[object, ...] = ()
    while True:
        cursor = conn.execute(
            f"""
            SELECT {_MESSAGE_SELECT}
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            WHERE m.session_id = ?{cursor_sql}
            ORDER BY {_TRANSCRIPT_ORDER}
            LIMIT ?
            """,
            (session_id, *cursor_params, _PAGE_SIZE),
        )
        rows = cursor.fetchall()
        if not rows:
            return
        decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
        records = [decode(row) for row in rows]
        blocks = _tool_use_blocks(conn, [str(record.message_id) for record in records])
        for record in records:
            yield message_from_record(
                record.model_copy(update={"blocks": blocks.get(str(record.message_id), [])}),
                [],
                origin=origin,
            )
        last = rows[-1]
        cursor_sql = " AND (m.position > ? OR (m.position = ? AND m.variant_index > ?))"
        cursor_params = (int(last["position"]), int(last["position"]), int(last["branch_index"]))
        if len(rows) < _PAGE_SIZE:
            return


def _tool_use_blocks(conn: sqlite3.Connection, message_ids: list[str]) -> dict[str, list[BlockRecord]]:
    placeholders = ",".join("?" for _ in message_ids)
    rows = conn.execute(
        f"""
        SELECT
            block_id,
            message_id,
            session_id,
            position AS block_index,
            block_type AS type,
            text,
            tool_name,
            tool_id,
            tool_input,
            NULL AS metadata,
            semantic_type,
            tool_result_is_error,
            tool_result_exit_code,
            tool_outcome,
            tool_result_outcome_unknown_reason,
            signature
        FROM blocks
        WHERE message_id IN ({placeholders}) AND block_type = 'tool_use'
        ORDER BY message_id, position
        """,
        message_ids,
    ).fetchall()
    result: dict[str, list[BlockRecord]] = {}
    for row in rows:
        result.setdefault(str(row["message_id"]), []).append(_row_to_content_block(row))
    return result


def iter_orchestration_events(conn: sqlite3.Connection, session_id: str) -> Iterator[SessionEvent]:
    """Stream the session's own timeline events in ``event_index`` order."""
    after: tuple[object, ...] = ()
    while True:
        rows = conn.execute(
            f"""
            SELECT se.*, s.origin
            FROM session_events se
            JOIN sessions s ON s.session_id = se.session_id
            WHERE se.session_id = ?{" AND se.position > ?" if after else ""}
            ORDER BY se.position
            LIMIT ?
            """,
            (session_id, *after, _PAGE_SIZE),
        ).fetchall()
        records = [_row_to_session_event(row) for row in rows]
        hydrate_session_event_array_items(conn, records)
        for record in records:
            yield session_event_from_record(record)
        if len(rows) < _PAGE_SIZE:
            return
        after = (int(rows[-1]["position"]),)


def iter_orchestration_usage(conn: sqlite3.Connection, session_id: str) -> Iterator[dict[str, object]]:
    """Stream native counter columns; missing wire fields are not reconstructed."""
    after: tuple[object, ...] = ()
    while True:
        rows = conn.execute(
            f"""
            SELECT session_id, position, source_message_id, provider_event_type,
                   model_name, occurred_at_ms,
                   last_input_tokens, last_output_tokens, last_cached_input_tokens,
                   last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
                   total_input_tokens, total_output_tokens, total_cached_input_tokens,
                   total_cache_write_tokens, total_reasoning_output_tokens, total_tokens
            FROM session_provider_usage_events
            WHERE session_id = ?{" AND position > ?" if after else ""}
            ORDER BY position
            LIMIT ?
            """,
            (session_id, *after, _PAGE_SIZE),
        ).fetchall()
        for row in rows:
            yield dict(row)
        if len(rows) < _PAGE_SIZE:
            return
        after = (int(rows[-1]["position"]),)


def _not_aborted() -> None:
    return None


def read_session_orchestration(
    archive: ArchiveStore,
    session_ref: str,
    *,
    raise_if_aborted: Callable[[], None] = _not_aborted,
) -> SessionOrchestrationEvidence | None:
    """Project one session's orchestration evidence from a pinned archive.

    The Python API, MCP ``get(projection="orchestration")`` and the CLI
    ``read --view orchestration`` operation all read through here, so the
    surfaces cannot disagree about one stored session. ``None`` means the
    reference names no session.
    """

    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.archive.query.predicate import QueryFieldPredicate, QueryFieldRef
    from polylogue.operations.read_view_lineage import _TopologySnapshot
    from polylogue.storage.derived.topology import derive_session_topology_async

    try:
        session_id = archive.resolve_session_id(session_ref)
    except KeyError:
        return None
    topology = complete_without_suspension(
        derive_session_topology_async(_TopologySnapshot(archive, raise_if_aborted), session_id)
    )
    artifacts, _ = archive.raw_artifacts_for_session(session_id, limit=1, offset=0)
    predicate = QueryFieldPredicate(field="session.id", values=(session_id,), op="=").with_field_ref(
        QueryFieldRef(scope="session", name="id", source_name="session.id")
    )
    return build_session_orchestration(
        session_id,
        topology,
        messages=iter_orchestration_messages(archive._conn, session_id),
        events=iter_orchestration_events(archive._conn, session_id),
        acquisition=artifacts[0] if artifacts else None,
        delegations=archive.query_delegations(predicate, limit=1001),
        usage_rows=iter_orchestration_usage(archive._conn, session_id),
    )


def execute_orchestration_read(
    payload: Mapping[str, object], *, archive: ArchiveStore, raise_if_aborted: Callable[[], None] = _not_aborted
) -> dict[str, object]:
    """Serve ``read.orchestration`` from the operation's pinned archive."""

    session_ref = str(payload["session_id"])
    evidence = read_session_orchestration(archive, session_ref, raise_if_aborted=raise_if_aborted)
    if evidence is None:
        raise KeyError(f"Session not found: {session_ref}")
    return {"view": "orchestration", "payload": evidence.model_dump(mode="json")}
