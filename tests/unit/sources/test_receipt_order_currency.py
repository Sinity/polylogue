"""Currency among raws follows durable receipt order, never a wall clock.

A live source that goes A -> B -> A re-mints A's content-derived raw id, so
``raw_sessions.acquired_at_ms`` keeps A's first sighting, and a wall-clock
rollback can stamp A's second receipt earlier than B's. In both cases A's
second receipt is the newest by insertion order, and it must be current.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.core.enums import Origin
from polylogue.sources.codex_state_projection import latest_retained_state_exports
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    upsert_raw_artifact,
    write_source_raw_session,
)
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture


def _observe(conn: sqlite3.Connection, *, origin: str, source_path: str, payload: bytes, acquired_at_ms: int) -> str:
    return write_source_raw_session(
        conn,
        origin=origin,
        source_path=source_path,
        source_index=0,
        payload=payload,
        acquired_at_ms=acquired_at_ms,
    )


def test_retained_state_export_follows_receipt_order_across_a_clock_rollback(tmp_path: Path) -> None:
    """Anti-vacuity: order receipts by ``acquired_at_ms`` first and B (300)
    outranks A's returned receipt (250)."""
    source_db = tmp_path / "source.db"
    state_path = str(tmp_path / "install" / "state_5.sqlite")
    initialize_runtime_source_fixture(source_db)
    with closing(sqlite3.connect(source_db)) as conn:
        origin = Origin.CODEX_SESSION.value
        raw_a = _observe(conn, origin=origin, source_path=state_path, payload=b"state A", acquired_at_ms=200)
        _observe(conn, origin=origin, source_path=state_path, payload=b"state B", acquired_at_ms=300)
        assert _observe(conn, origin=origin, source_path=state_path, payload=b"state A", acquired_at_ms=250) == raw_a
        conn.commit()

        exports = latest_retained_state_exports(conn)

    assert [(export.raw_id, export.observed_at_ms) for export in exports] == [(raw_a, 250)]


def test_artifact_carrier_follows_receipt_order_across_a_clock_rollback(tmp_path: Path) -> None:
    """One coordinate has one authority carrier. Anti-vacuity: compare the
    carriers' receipt stamps first and B (300) keeps the coordinate although A
    was observed after it."""
    source_db = tmp_path / "source.db"
    source_path = str(tmp_path / "journal.jsonl")
    initialize_runtime_source_fixture(source_db)
    origin = "claude-code-session"

    def _artifact(reason: str, observed_at_ms: int) -> ArchiveSourceArtifact:
        return ArchiveSourceArtifact(
            artifact_id="artifact-coordinate",
            origin=origin,
            source_path=source_path,
            source_index=0,
            artifact_kind="workflow_journal",
            classification_reason=reason,
            first_observed_at_ms=observed_at_ms,
            last_observed_at_ms=observed_at_ms,
        )

    with closing(sqlite3.connect(source_db)) as conn:
        raw_a = _observe(conn, origin=origin, source_path=source_path, payload=b"journal A", acquired_at_ms=200)
        raw_b = _observe(conn, origin=origin, source_path=source_path, payload=b"journal B", acquired_at_ms=300)
        upsert_raw_artifact(conn, raw_b, _artifact("journal B", 300))
        assert _observe(conn, origin=origin, source_path=source_path, payload=b"journal A", acquired_at_ms=250) == raw_a

        upsert_raw_artifact(conn, raw_a, _artifact("journal A", 250))

        carrier = conn.execute(
            "SELECT raw_id FROM raw_artifacts WHERE source_path = ? AND source_index = 0",
            (source_path,),
        ).fetchall()

    assert carrier == [(raw_a,)]
