"""Regression coverage for polylogue-a7xr.2: converger/repair staleness agreement.

Prior bug: ``daemon/convergence_stages.py`` and ``storage/repair.py`` encoded
*different* staleness predicates for ``session_profiles``/
``session_latency_profiles`` rows whose owning session has ``sort_key_ms IS
NULL`` (a "timeless" session with no derivable temporal sort key). The
converger compared the profile's cached ``source_updated_at`` against the
session's ``updated_at_ms``; repair instead COALESCEd the missing sort key to
``0.0`` and compared it against the profile's ``source_sort_key`` — which
permanently flagged any NULL-``sort_key_ms`` session with a nonzero cached
``source_sort_key`` as stale, regardless of whether the converger considered
it fresh. That produced either indefinite repair churn (repair repeatedly
"fixing" a row the converger already materialized correctly) or missed
rebuilds, depending on which path ran first.

Both paths now compose their staleness check from the single
``session_profile_stale_predicate`` builder in
``polylogue/storage/insights/session/runtime.py``. This test proves the fix
two ways: (1) a fixture with ``sort_key_ms IS NULL`` and a nonzero cached
``source_sort_key`` is classified *identically* (both "fresh") by the real
convergence path (``convergence_stages._stale_session_profile_ids``) and the
real ops-repair path (``repair._targeted_session_insight_rebuild_ids``) when
``source_updated_at`` agrees with the session; (2) the same fixture is
classified identically as "stale" by both paths when ``source_updated_at``
disagrees. It would fail against the pre-fix repair predicate, which ignored
``source_updated_at`` entirely and used ``source_sort_key`` unconditionally.
"""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import polylogue.daemon.convergence_stages as convergence_stages
import polylogue.storage.repair as repair
from polylogue.storage.insights.session.runtime import (
    SESSION_INSIGHT_MATERIALIZATION_TYPES,
    session_profile_stale_predicate,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION

_SESSION_ID = "ts-null-sort-key"
_UPDATED_AT_MS = 1_780_000_000_000


def _build_fixture_db(path: Path, *, source_updated_at: str, source_sort_key: float) -> None:
    """Minimal schema covering every table ``repair._targeted_session_insight_rebuild_ids`` joins."""

    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                sort_key_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE TABLE session_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER,
                source_sort_key REAL,
                source_updated_at TEXT,
                work_event_count INTEGER NOT NULL DEFAULT 0,
                phase_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE session_latency_profiles (
                session_id TEXT PRIMARY KEY,
                materializer_version INTEGER,
                source_sort_key REAL,
                source_updated_at TEXT
            );
            CREATE TABLE session_work_events (session_id TEXT);
            CREATE TABLE session_phases (session_id TEXT);
            CREATE TABLE insight_materialization (
                insight_type TEXT,
                session_id TEXT,
                materializer_version INTEGER,
                source_sort_key_ms INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions (session_id, sort_key_ms, updated_at_ms) VALUES (?, NULL, ?)",
            (_SESSION_ID, _UPDATED_AT_MS),
        )
        conn.execute(
            """
            INSERT INTO session_profiles
                (session_id, materializer_version, source_sort_key, source_updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (_SESSION_ID, SESSION_INSIGHT_MATERIALIZER_VERSION, source_sort_key, source_updated_at),
        )
        conn.execute(
            """
            INSERT INTO session_latency_profiles
                (session_id, materializer_version, source_sort_key, source_updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (_SESSION_ID, SESSION_INSIGHT_MATERIALIZER_VERSION, source_sort_key, source_updated_at),
        )
        # Every non-thread materialization type fully stamped and sort-key-agreeing
        # (source_sort_key_ms compares directly against sort_key_ms, which is
        # NULL here, so COALESCE both sides to 0 for an exact NULL/NULL match).
        for insight_type in SESSION_INSIGHT_MATERIALIZATION_TYPES:
            if insight_type == "thread":
                continue
            conn.execute(
                "INSERT INTO insight_materialization (insight_type, session_id, materializer_version, "
                "source_sort_key_ms) VALUES (?, ?, ?, NULL)",
                (insight_type, _SESSION_ID, SESSION_INSIGHT_MATERIALIZER_VERSION),
            )
        conn.commit()
    finally:
        conn.close()


def _run_convergence_pass(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return convergence_stages._stale_session_profile_ids(conn, [_SESSION_ID])
    finally:
        conn.close()


def _run_repair_pass(db_path: Path) -> tuple[str, ...]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        result = repair._targeted_session_insight_rebuild_ids(conn, SimpleNamespace())
        assert result is not None, "fixture triggered an unexpected archive-wide rebuild fallback"
        return result
    finally:
        conn.close()


def test_null_sort_key_fresh_profile_agrees_between_convergence_and_repair(tmp_path: Path) -> None:
    """source_updated_at matches the session: both paths must call it fresh.

    Pre-fix, repair's COALESCE-to-0.0 branch flagged this row stale purely
    because ``source_sort_key`` (999.0) was nonzero — regardless of
    ``source_updated_at`` agreement — while the converger correctly called it
    fresh. That divergence is the bug this test pins.
    """

    db_path = tmp_path / "fresh.db"
    agreeing_source_updated_at = str(_UPDATED_AT_MS // 1000)
    _build_fixture_db(db_path, source_updated_at=agreeing_source_updated_at, source_sort_key=999.0)

    convergence_stale = _run_convergence_pass(db_path)
    repair_stale = _run_repair_pass(db_path)

    assert convergence_stale == [], "converger incorrectly considers the fresh NULL-sort-key profile stale"
    assert _SESSION_ID not in repair_stale, "repair incorrectly considers the fresh NULL-sort-key profile stale"


def test_null_sort_key_stale_profile_agrees_between_convergence_and_repair(tmp_path: Path) -> None:
    """source_updated_at disagrees with the session: both paths must call it stale."""

    db_path = tmp_path / "stale.db"
    disagreeing_source_updated_at = str((_UPDATED_AT_MS // 1000) - 3600)
    _build_fixture_db(db_path, source_updated_at=disagreeing_source_updated_at, source_sort_key=0.0)

    convergence_stale = _run_convergence_pass(db_path)
    repair_stale = _run_repair_pass(db_path)

    assert convergence_stale == [_SESSION_ID]
    assert _SESSION_ID in repair_stale


def test_repair_selects_zero_rows_immediately_after_convergence_agrees(tmp_path: Path) -> None:
    """Idempotence: once the converger calls a NULL-sort-key row fresh, repair must not re-flag it.

    This is the concrete manifestation of the bug's "repeated repair churn"
    consequence — a converged archive must not oscillate between the two
    passes.
    """

    db_path = tmp_path / "idempotent.db"
    agreeing_source_updated_at = str(_UPDATED_AT_MS // 1000)
    _build_fixture_db(db_path, source_updated_at=agreeing_source_updated_at, source_sort_key=999.0)

    assert _run_convergence_pass(db_path) == []
    assert _run_repair_pass(db_path) == ()


def test_session_profile_stale_predicate_has_exactly_one_definition() -> None:
    """Static companion check: no module reimplements the ABS/source_sort_key comparison.

    Complements the behavioral agreement tests above by asserting the
    textual pattern that caused the divergence (``ABS(COALESCE(<alias>.
    source_sort_key`` for the session_profiles/session_latency_profiles
    aliases) exists nowhere except inside
    ``session_profile_stale_predicate`` itself — a careless reintroduction of
    an inline copy in either convergence_stages.py or repair.py is caught
    here even if its NULL-branch semantics happened to be correct.
    """

    predicate_source = inspect.getsource(session_profile_stale_predicate)
    assert "ABS(COALESCE(" in predicate_source

    # "source_sort_key," (comma immediately after, no "_ms") uniquely identifies
    # the session_profiles/session_latency_profiles staleness comparison this
    # predicate owns — distinct from the unrelated insight_materialization
    # "source_sort_key_ms" family that legitimately stays inline.
    for module in (convergence_stages, repair):
        module_source = inspect.getsource(module)
        assert "source_sort_key," not in module_source, (
            f"{module.__name__} appears to reimplement the session-profile staleness ABS/source_sort_key "
            "comparison inline instead of composing session_profile_stale_predicate() "
            "(polylogue-a7xr.2)."
        )
