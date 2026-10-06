"""Pinned product reads for the compact lineage and session topology views."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from polylogue.core.async_bridge import complete_without_suspension
from polylogue.storage.runtime import SessionRecord
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.outcome import lineage_page_outcome


def _int_field(payload: Mapping[str, object], name: str, default: int | None) -> int | None:
    value = payload.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def execute_lineage_read(payload: Mapping[str, object], *, archive: ArchiveStore) -> dict[str, object]:
    graph = archive.read_compact_lineage(
        str(payload["session_id"]),
        node_offset=_int_field(payload, "node_offset", 0) or 0,
        node_limit=_int_field(payload, "node_limit", None),
        edge_offset=_int_field(payload, "edge_offset", 0) or 0,
        edge_limit=_int_field(payload, "edge_limit", None),
    )
    if graph is None:
        raise KeyError(f"Session not found: {payload['session_id']}")
    result = graph.model_dump(mode="json")
    # A page past offset 0 omits earlier rows just as a page with more after it
    # omits later ones; either way this answer is not the whole graph.
    nodes_partial = graph.node_page.has_more or (_int_field(payload, "node_offset", 0) or 0) > 0
    edges_partial = graph.edge_page.has_more or (_int_field(payload, "edge_offset", 0) or 0) > 0
    truncated = nodes_partial or edges_partial
    reason = "nodes" if nodes_partial else "edges" if edges_partial else "cycle" if graph.cycle_detected else None
    outcome = lineage_page_outcome(
        matched=len(graph.nodes), complete=not truncated and not graph.cycle_detected, truncation_reason=reason
    )
    result["outcome"] = outcome.to_dict()
    return {"view": "lineage", "payload": result}


class _TopologySnapshot:
    def __init__(self, archive: ArchiveStore, raise_if_aborted: Callable[[], None]) -> None:
        connection = archive.index_connection
        if connection is None:
            raise ValueError("topology requires an index snapshot")
        self.connection = connection
        self.raise_if_aborted = raise_if_aborted

    def check_cancelled(self) -> None:
        # Raises the shared controller's typed QueryCancelledError /
        # QueryTimeoutError so the operation envelope reports the real reason.
        self.raise_if_aborted()

    async def get_session(self, session_id: str) -> SessionRecord | None:
        self.check_cancelled()
        from polylogue.storage.sqlite.queries.mappers import _row_to_session
        from polylogue.storage.sqlite.queries.sessions_reads import _SESSION_RECORD_SELECT

        row = self.connection.execute(
            f"SELECT {_SESSION_RECORD_SELECT} FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row) if row is not None else None

    async def list_session_links_for_session(
        self, session_id: str, *, limit: int | None = None
    ) -> list[dict[str, object]]:
        self.check_cancelled()
        from polylogue.storage.sqlite.queries.session_links import SESSION_LINK_COLUMNS

        bound = "" if limit is None else " LIMIT ?"
        args: tuple[object, ...] = (session_id,) if limit is None else (session_id, limit)
        cursor = self.connection.execute(
            f"SELECT {SESSION_LINK_COLUMNS} FROM session_links WHERE src_session_id = ? "
            "ORDER BY link_type, dst_origin, dst_native_id" + bound,
            args,
        )
        return [dict(row) for row in cursor.fetchall()]

    async def list_session_links_to_session(self, session_id: str, *, limit: int | None) -> list[dict[str, object]]:
        self.check_cancelled()
        from polylogue.storage.sqlite.queries.session_links import SESSION_LINK_COLUMNS

        bound = "" if limit is None else " LIMIT ?"
        args: tuple[object, ...] = (session_id,) if limit is None else (session_id, limit)
        cursor = self.connection.execute(
            f"SELECT {SESSION_LINK_COLUMNS} FROM session_links WHERE resolved_dst_session_id = ? "
            "ORDER BY src_session_id, dst_origin, dst_native_id, link_type" + bound,
            args,
        )
        return [dict(row) for row in cursor.fetchall()]


def execute_topology_read(
    payload: Mapping[str, object], *, archive: ArchiveStore, raise_if_aborted: Callable[[], None]
) -> dict[str, object]:
    from polylogue.operations.topology_envelope import topology_public_envelope
    from polylogue.storage.derived.topology import derive_session_topology_async

    session_id = str(payload["session_id"])
    try:
        resolved = archive.resolve_session_id(session_id)
    except KeyError as exc:
        raise KeyError(f"Session not found: {session_id}") from exc
    snapshot = _TopologySnapshot(archive, raise_if_aborted)
    topology = complete_without_suspension(
        derive_session_topology_async(
            snapshot,
            resolved,
            node_offset=_int_field(payload, "node_offset", 0) or 0,
            node_limit=_int_field(payload, "node_limit", 200) or 200,
            edge_limit=_int_field(payload, "edge_limit", 500) or 500,
        )
    )
    if topology is None:
        raise KeyError(f"Session not found: {session_id}")
    return {
        "view": "topology",
        "payload": topology_public_envelope(
            topology,
            session_id=resolved,
            node_offset=_int_field(payload, "node_offset", 0) or 0,
        ),
    }


__all__ = ["execute_lineage_read", "execute_topology_read"]
