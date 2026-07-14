"""Topology cycle rejection and quarantine (#1260 / #866 slice C).

When resolving a topology edge would create a cycle in
``sessions.parent_session_id`` (A → B → A, longer cycles, or
the self-cycle A → A), the resolver must:

- Refuse to backfill ``parent_session_id`` (the fast-path graph
  must never enter the cycle).
- Refuse to resolve the link; instead, mark it
  ``status='quarantined'`` with an ``evidence_json`` document that
  records the detected cycle path so the operator can audit which
  session chain was rejected.
- Leave non-cyclic edges in the same resolver batch untouched (a cycle
  on one child does not poison legitimate siblings).
- Stay idempotent: re-running the resolver on an archive that already
  contains quarantined edges produces no further changes.

These tests exercise both resolver entry points
(:func:`resolve_session_links_for_session` and
:func:`resolve_unresolved_links_for_child`) and also assert the
no-false-positive case on a diamond DAG, which is a legitimate shape
(B→D, C→D both pointing at D) that some prior implementations confuse
with a cycle.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.archive.topology.edge import (
    TopologyEdgeRecord,
    TopologyEdgeStatus,
    TopologyEdgeType,
)
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.storage.sqlite.queries.session_links import (
    count_quarantined_session_links,
    resolve_session_links_for_session,
    resolve_unresolved_links_for_child,
    upsert_session_links,
)
from polylogue.storage.sqlite.schema import SCHEMA_DDL, SCHEMA_VERSION
from tests.infra.frozen_clock import fixed_now


def _now() -> str:
    return fixed_now().isoformat()


def _sid(value: str) -> str:
    if value.startswith("codex-session:"):
        return value
    if value.startswith("conv-"):
        return f"codex-session:native-{value.removeprefix('conv-')}"
    return value


def _hash_blob(value: str) -> bytes:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).digest()


def _bootstrap_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_DDL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()


def _insert_session(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    source_name: str,
    provider_session_id: str,
    parent_session_id: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sessions (
            native_id, origin, title, parent_session_id,
            content_hash, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            provider_session_id,
            Origin.CODEX_SESSION.value,
            f"conv {session_id}",
            _sid(parent_session_id) if parent_session_id else None,
            _hash_blob(session_id),
            1,
            1,
        ),
    )
    conn.commit()


class _AsyncSqliteAdapter:
    """Minimal aiosqlite-compatible wrapper around stdlib sqlite3.

    The production query helpers are written against ``aiosqlite``; the
    only async operations they perform are ``execute``/``fetchone``/
    ``fetchall``. This adapter lets the unit tests drive them against a
    plain sqlite3 connection without spinning up the full async runtime.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn

    async def execute(self, sql: str, params: tuple[object, ...] = ()) -> _AsyncCursorAdapter:
        cursor = self._conn.execute(sql, params)
        return _AsyncCursorAdapter(cursor)

    def commit(self) -> None:
        self._conn.commit()


class _AsyncCursorAdapter:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchall(self) -> list[sqlite3.Row]:
        return list(self._cursor.fetchall())

    async def fetchone(self) -> sqlite3.Row | None:
        result: sqlite3.Row | None = self._cursor.fetchone()
        return result

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


@pytest.fixture
def cycle_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "cycle.sqlite"
    _bootstrap_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _seed_edge(
    conn: sqlite3.Connection,
    *,
    src_session_id: str,
    dst_native_id: str,
    dst_origin: Origin = Origin.CODEX_SESSION,
    link_type: TopologyEdgeType = TopologyEdgeType.CONTINUATION,
) -> None:
    """Insert one unresolved session link through the production upsert."""

    import asyncio

    edge = TopologyEdgeRecord(
        src_session_id=SessionId(_sid(src_session_id)),
        dst_origin=dst_origin,
        dst_native_id=dst_native_id,
        link_type=link_type,
        status=TopologyEdgeStatus.UNRESOLVED,
    )
    asyncio.run(upsert_session_links(_AsyncSqliteAdapter(conn), [edge]))  # type: ignore[arg-type]


def _run_resolve_for_parent(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin: str,
    native_id: str,
) -> int:
    import asyncio

    return asyncio.run(
        resolve_session_links_for_session(
            _AsyncSqliteAdapter(conn),  # type: ignore[arg-type]
            session_id=_sid(session_id),
            origin=origin,
            native_id=native_id,
            resolved_at=_now(),
        )
    )


def _run_resolve_for_child(conn: sqlite3.Connection, *, src_session_id: str) -> int:
    import asyncio

    return asyncio.run(
        resolve_unresolved_links_for_child(
            _AsyncSqliteAdapter(conn),  # type: ignore[arg-type]
            src_session_id=_sid(src_session_id),
            resolved_at=_now(),
        )
    )


def _fetch_edge_status(conn: sqlite3.Connection, src_session_id: str) -> tuple[str, str | None]:
    row = conn.execute(
        "SELECT status, evidence_json FROM session_links WHERE src_session_id = ?",
        (_sid(src_session_id),),
    ).fetchone()
    assert row is not None
    status = row["status"]
    if status is None and row["evidence_json"] == "[]":
        status = (
            TopologyEdgeStatus.RESOLVED.value
            if _fetch_parent(conn, src_session_id)
            else TopologyEdgeStatus.UNRESOLVED.value
        )
    return str(status), row["evidence_json"]


def _fetch_parent(conn: sqlite3.Connection, session_id: str) -> str | None:
    row = conn.execute(
        "SELECT parent_session_id FROM sessions WHERE session_id = ?",
        (_sid(session_id),),
    ).fetchone()
    return None if row is None else row["parent_session_id"]


class TestTwoNodeCycle:
    """A → B → A (B was already saved with A as parent; A is now landing
    with a topology edge that would make B its parent)."""

    def test_two_node_cycle_quarantines_edge_and_leaves_parent_null(self, cycle_db: sqlite3.Connection) -> None:
        # Topology: A's parent_session_id will eventually be set to B,
        # but B already has A as its parent. Resolving A→B is the cycle.
        _insert_session(
            cycle_db,
            session_id="conv-A",
            source_name="codex",
            provider_session_id="native-A",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-B",
            source_name="codex",
            provider_session_id="native-B",
            parent_session_id="conv-A",  # B → A already.
        )
        # A asserts an unresolved edge to B (this would normally flip to
        # resolved when B's row was already in place).
        _seed_edge(
            cycle_db,
            src_session_id="conv-A",
            dst_native_id="native-B",
        )

        # The resolver-by-child code path is what runs when A's own edge
        # is upserted and we then sweep for matching parent rows.
        _run_resolve_for_child(cycle_db, src_session_id="conv-A")

        status, evidence = _fetch_edge_status(cycle_db, "conv-A")
        assert status == TopologyEdgeStatus.QUARANTINED.value
        assert evidence is not None
        payload = json.loads(evidence)
        assert payload["reason"] == "cycle_rejected"
        # The cycle path must include both endpoints.
        assert _sid("conv-A") in payload["cycle_path"]
        assert _sid("conv-B") in payload["cycle_path"]

        # And critically, A.parent_session_id stays NULL so the
        # fast-path ancestry walk does not enter the cycle.
        assert _fetch_parent(cycle_db, "conv-A") is None

    def test_two_node_cycle_via_parent_first_path(self, cycle_db: sqlite3.Connection) -> None:
        # Same topology but exercised through the parent-first entry point
        # (parent X is being saved; we look for unresolved edges pointing at X).
        _insert_session(
            cycle_db,
            session_id="conv-A",
            source_name="codex",
            provider_session_id="native-A",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-B",
            source_name="codex",
            provider_session_id="native-B",
            parent_session_id="conv-A",
        )
        _seed_edge(
            cycle_db,
            src_session_id="conv-A",
            dst_native_id="native-B",
        )

        # B is now being "saved" (already present in sessions from
        # the fixture), and we sweep edges that point at native-B. Without
        # cycle detection this would resolve the edge and backfill
        # A.parent_session_id = B, closing the loop.
        flipped = _run_resolve_for_parent(
            cycle_db,
            session_id="conv-B",
            origin=Origin.CODEX_SESSION.value,
            native_id="native-B",
        )
        assert flipped == 0  # nothing was successfully resolved

        status, evidence = _fetch_edge_status(cycle_db, "conv-A")
        assert status == TopologyEdgeStatus.QUARANTINED.value
        assert evidence is not None
        assert _fetch_parent(cycle_db, "conv-A") is None


class TestThreeNodeCycle:
    """A → B → C → A. C is being saved; resolving A→C would close the loop."""

    def test_three_node_cycle_quarantined(self, cycle_db: sqlite3.Connection) -> None:
        _insert_session(
            cycle_db,
            session_id="conv-A",
            source_name="codex",
            provider_session_id="native-A",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-B",
            source_name="codex",
            provider_session_id="native-B",
            parent_session_id="conv-A",
        )
        _insert_session(
            cycle_db,
            session_id="conv-C",
            source_name="codex",
            provider_session_id="native-C",
            parent_session_id="conv-B",
        )
        # A asserts unresolved edge to C → resolving would mean
        # A.parent = C → B → A → cycle.
        _seed_edge(
            cycle_db,
            src_session_id="conv-A",
            dst_native_id="native-C",
        )

        _run_resolve_for_child(cycle_db, src_session_id="conv-A")

        status, evidence = _fetch_edge_status(cycle_db, "conv-A")
        assert status == TopologyEdgeStatus.QUARANTINED.value
        payload = json.loads(evidence or "{}")
        # The recorded path should walk A → C → B → A.
        assert payload["cycle_path"][0] == _sid("conv-A")
        assert payload["cycle_path"][-1] == _sid("conv-A")
        assert _sid("conv-B") in payload["cycle_path"]
        assert _sid("conv-C") in payload["cycle_path"]
        assert _fetch_parent(cycle_db, "conv-A") is None


class TestSelfCycle:
    """A → A. The most pathological cycle shape."""

    def test_self_cycle_quarantined(self, cycle_db: sqlite3.Connection) -> None:
        _insert_session(
            cycle_db,
            session_id="conv-A",
            source_name="codex",
            provider_session_id="native-A",
            parent_session_id=None,
        )
        # A's edge points at its own native id.
        _seed_edge(
            cycle_db,
            src_session_id="conv-A",
            dst_native_id="native-A",
        )

        flipped = _run_resolve_for_parent(
            cycle_db,
            session_id="conv-A",
            origin=Origin.CODEX_SESSION.value,
            native_id="native-A",
        )
        assert flipped == 0

        status, evidence = _fetch_edge_status(cycle_db, "conv-A")
        assert status == TopologyEdgeStatus.QUARANTINED.value
        payload = json.loads(evidence or "{}")
        assert payload["cycle_path"] == [_sid("conv-A"), _sid("conv-A")]
        assert _fetch_parent(cycle_db, "conv-A") is None


class TestDiamondDagNoFalsePositive:
    """B → D and C → D — D is a shared parent for two children. This is a
    legitimate diamond DAG, not a cycle, and the resolver must process it
    cleanly without quarantining either edge."""

    def test_diamond_dag_resolves_both_children(self, cycle_db: sqlite3.Connection) -> None:
        _insert_session(
            cycle_db,
            session_id="conv-D",
            source_name="codex",
            provider_session_id="native-D",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-B",
            source_name="codex",
            provider_session_id="native-B",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-C",
            source_name="codex",
            provider_session_id="native-C",
            parent_session_id=None,
        )
        _seed_edge(
            cycle_db,
            src_session_id="conv-B",
            dst_native_id="native-D",
        )
        _seed_edge(
            cycle_db,
            src_session_id="conv-C",
            dst_native_id="native-D",
        )

        flipped = _run_resolve_for_parent(
            cycle_db,
            session_id="conv-D",
            origin=Origin.CODEX_SESSION.value,
            native_id="native-D",
        )
        assert flipped == 2  # both edges resolved

        for child in ("conv-B", "conv-C"):
            status, _ = _fetch_edge_status(cycle_db, child)
            assert status == TopologyEdgeStatus.RESOLVED.value
            assert _fetch_parent(cycle_db, child) == _sid("conv-D")


class TestSiblingNotQuarantined:
    """A cycle on one child must not quarantine a non-cyclic sibling that
    happens to share the resolver batch (both pointing at the same parent
    native id)."""

    def test_cycle_on_one_child_leaves_sibling_resolved(self, cycle_db: sqlite3.Connection) -> None:
        # X is the parent native id. Two children share it:
        #   - cyclic: X is descendant of cycle-child, so resolving creates cycle.
        #   - clean: independent child with no ancestry conflict.
        _insert_session(
            cycle_db,
            session_id="conv-cycle",
            source_name="codex",
            provider_session_id="native-cycle",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-X",
            source_name="codex",
            provider_session_id="native-X",
            parent_session_id="conv-cycle",  # X already descends from cycle-child
        )
        _insert_session(
            cycle_db,
            session_id="conv-clean",
            source_name="codex",
            provider_session_id="native-clean",
            parent_session_id=None,
        )
        _seed_edge(
            cycle_db,
            src_session_id="conv-cycle",
            dst_native_id="native-X",
        )
        _seed_edge(
            cycle_db,
            src_session_id="conv-clean",
            dst_native_id="native-X",
        )

        flipped = _run_resolve_for_parent(
            cycle_db,
            session_id="conv-X",
            origin=Origin.CODEX_SESSION.value,
            native_id="native-X",
        )
        # Only the clean child resolved; the cyclic one was quarantined.
        assert flipped == 1

        cycle_status, _ = _fetch_edge_status(cycle_db, "conv-cycle")
        clean_status, _ = _fetch_edge_status(cycle_db, "conv-clean")
        assert cycle_status == TopologyEdgeStatus.QUARANTINED.value
        assert clean_status == TopologyEdgeStatus.RESOLVED.value
        assert _fetch_parent(cycle_db, "conv-cycle") is None
        assert _fetch_parent(cycle_db, "conv-clean") == _sid("conv-X")


class TestIdempotency:
    """Re-running the resolver after quarantine produces no further state
    changes — the edge stays quarantined, the sessions row stays
    untouched."""

    def test_resolver_idempotent_on_quarantined_edge(self, cycle_db: sqlite3.Connection) -> None:
        _insert_session(
            cycle_db,
            session_id="conv-A",
            source_name="codex",
            provider_session_id="native-A",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-B",
            source_name="codex",
            provider_session_id="native-B",
            parent_session_id="conv-A",
        )
        _seed_edge(
            cycle_db,
            src_session_id="conv-A",
            dst_native_id="native-B",
        )

        # First pass quarantines.
        _run_resolve_for_parent(
            cycle_db,
            session_id="conv-B",
            origin=Origin.CODEX_SESSION.value,
            native_id="native-B",
        )
        status1, evidence1 = _fetch_edge_status(cycle_db, "conv-A")
        assert status1 == TopologyEdgeStatus.QUARANTINED.value

        # Re-running the parent resolver: the edge is no longer
        # ``unresolved``, so the SELECT returns no candidates and nothing
        # changes. No spurious second quarantine event, no parent backfill.
        _run_resolve_for_parent(
            cycle_db,
            session_id="conv-B",
            origin=Origin.CODEX_SESSION.value,
            native_id="native-B",
        )
        status2, evidence2 = _fetch_edge_status(cycle_db, "conv-A")
        assert status2 == TopologyEdgeStatus.QUARANTINED.value
        assert evidence2 == evidence1  # same payload, untouched
        assert _fetch_parent(cycle_db, "conv-A") is None

    def test_resolver_for_child_idempotent_on_quarantined_edge(self, cycle_db: sqlite3.Connection) -> None:
        _insert_session(
            cycle_db,
            session_id="conv-A",
            source_name="codex",
            provider_session_id="native-A",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-B",
            source_name="codex",
            provider_session_id="native-B",
            parent_session_id="conv-A",
        )
        _seed_edge(
            cycle_db,
            src_session_id="conv-A",
            dst_native_id="native-B",
        )

        _run_resolve_for_child(cycle_db, src_session_id="conv-A")
        _run_resolve_for_child(cycle_db, src_session_id="conv-A")

        status, _ = _fetch_edge_status(cycle_db, "conv-A")
        assert status == TopologyEdgeStatus.QUARANTINED.value
        assert _fetch_parent(cycle_db, "conv-A") is None


class TestQuarantineCount:
    """The diagnostic counter surfaces quarantined edges so the daemon
    workload probe can warn the operator."""

    def test_count_quarantined_session_links(self, cycle_db: sqlite3.Connection) -> None:
        import asyncio

        # Start with zero.
        count0 = asyncio.run(count_quarantined_session_links(_AsyncSqliteAdapter(cycle_db)))  # type: ignore[arg-type]
        assert count0 == 0

        # Create a cycle.
        _insert_session(
            cycle_db,
            session_id="conv-A",
            source_name="codex",
            provider_session_id="native-A",
            parent_session_id=None,
        )
        _insert_session(
            cycle_db,
            session_id="conv-B",
            source_name="codex",
            provider_session_id="native-B",
            parent_session_id="conv-A",
        )
        _seed_edge(
            cycle_db,
            src_session_id="conv-A",
            dst_native_id="native-B",
        )
        _run_resolve_for_child(cycle_db, src_session_id="conv-A")

        count1 = asyncio.run(count_quarantined_session_links(_AsyncSqliteAdapter(cycle_db)))  # type: ignore[arg-type]
        assert count1 == 1
