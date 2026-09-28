"""The orchestration evidence read shared by every public surface."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.analysis.orchestration_evidence import SessionOrchestrationEvidence
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


def read_session_orchestration(archive: ArchiveStore, session_ref: str) -> SessionOrchestrationEvidence | None:
    """Project one session's orchestration evidence from a pinned archive.

    The Python API, MCP ``get(projection="orchestration")`` and the CLI
    ``read --view orchestration`` operation all read through here, so the
    surfaces cannot disagree about one stored session. ``None`` means the
    reference names no session.
    """

    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.archive.hydration import archive_envelope_to_session
    from polylogue.archive.query.predicate import QueryFieldPredicate, QueryFieldRef
    from polylogue.operations.read_view_lineage import _TopologySnapshot
    from polylogue.storage.derived.topology import derive_session_topology_async

    try:
        session_id = archive.resolve_session_id(session_ref)
    except KeyError:
        return None
    summary = archive.read_summary(session_id)
    session = archive_envelope_to_session(
        archive.read_session(session_id),
        display_label=summary.display_label,
        display_label_source=summary.display_label_source,
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
