"""The orchestration evidence read shared by every public surface."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.analysis.orchestration_evidence import SessionOrchestrationEvidence
    from polylogue.storage.runtime import BlockRecord
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def read_orchestration_usage(conn: sqlite3.Connection, session_id: str) -> list[dict[str, object]]:
    """Read native counter columns; missing wire fields are not reconstructed."""
    rows = conn.execute(
        """
        SELECT session_id, position, source_message_id, provider_event_type,
               model_name, occurred_at_ms,
               last_input_tokens, last_output_tokens, last_cached_input_tokens,
               last_cache_write_tokens, last_reasoning_output_tokens, last_total_tokens,
               total_input_tokens, total_output_tokens, total_cached_input_tokens,
               total_cache_write_tokens, total_reasoning_output_tokens, total_tokens
        FROM session_provider_usage_events
        WHERE session_id = ?
        ORDER BY position
        """,
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


#: The block columns ``queries.attachment_blocks.get_blocks`` selects, read by
#: session rather than by message-id batch.
_SESSION_BLOCKS_SELECT = """
    SELECT block_id, message_id, session_id, position AS block_index, block_type AS type, text,
           tool_name, tool_id, tool_input, NULL AS metadata, semantic_type, tool_result_is_error,
           tool_result_exit_code, tool_outcome, tool_result_outcome_unknown_reason, signature
    FROM blocks
    WHERE session_id = ?
    ORDER BY message_id, position
"""


def read_session_orchestration(archive: ArchiveStore, session_ref: str) -> SessionOrchestrationEvidence | None:
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
    from polylogue.storage.hydrators import session_from_records
    from polylogue.storage.sqlite.queries.mappers import _row_to_content_block, _row_to_session
    from polylogue.storage.sqlite.queries.mappers_archive import bind_message_row_mapper
    from polylogue.storage.sqlite.queries.message_query_reads import _MESSAGE_RECORD_SELECT, _TRANSCRIPT_ORDER
    from polylogue.storage.sqlite.queries.session_events import sync_session_events_batch
    from polylogue.storage.sqlite.queries.sessions_reads import _SESSION_RECORD_SELECT

    try:
        session_id = archive.resolve_session_id(session_ref)
    except KeyError:
        return None
    conn = archive._conn
    session_row = conn.execute(
        f"SELECT {_SESSION_RECORD_SELECT} FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if session_row is None:
        return None
    # The projection reads only the session's own records (inherited dialogue
    # is excluded), hydrated from the same record rows the repository uses:
    # recorded models, launches, quota windows and configured models live on
    # message rows and timeline events, which the composed session envelope
    # does not carry.
    cursor = conn.execute(
        f"SELECT {_MESSAGE_RECORD_SELECT} FROM messages m JOIN sessions s ON s.session_id = m.session_id "
        f"WHERE m.session_id = ? ORDER BY {_TRANSCRIPT_ORDER}",
        (session_id,),
    )
    decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
    messages = [decode(row) for row in cursor.fetchall()]
    blocks: dict[str, list[BlockRecord]] = {}
    for row in conn.execute(_SESSION_BLOCKS_SELECT, (session_id,)).fetchall():
        blocks.setdefault(str(row["message_id"]), []).append(_row_to_content_block(row))
    for message in messages:
        message.blocks = blocks.get(message.message_id, [])
        if not message.text:
            message.text = "\n".join(block.text for block in message.blocks if block.text) or message.text
    session = session_from_records(
        _row_to_session(session_row),
        messages,
        [],
        sync_session_events_batch(conn, [session_id]).get(session_id, []),
    )
    topology = asyncio.run(derive_session_topology_async(_TopologySnapshot(archive), session_id))
    artifacts, _ = archive.raw_artifacts_for_session(session_id, limit=1, offset=0)
    predicate = QueryFieldPredicate(field="session.id", values=(session_id,), op="=").with_field_ref(
        QueryFieldRef(scope="session", name="id", source_name="session.id")
    )
    return build_session_orchestration(
        session,
        topology,
        acquisition=artifacts[0] if artifacts else None,
        delegations=archive.query_delegations(predicate, limit=1001),
        usage_rows=read_orchestration_usage(archive._conn, session_id),
    )


def execute_orchestration_read(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    """Serve ``read.orchestration`` from the operation's pinned archive."""

    session_ref = str(payload["session_id"])
    evidence = read_session_orchestration(archive, session_ref)
    if evidence is None:
        raise KeyError(f"Session not found: {session_ref}")
    return {"view": "orchestration", "payload": evidence.model_dump(mode="json")}


__all__ = ["execute_orchestration_read", "read_orchestration_usage", "read_session_orchestration"]
