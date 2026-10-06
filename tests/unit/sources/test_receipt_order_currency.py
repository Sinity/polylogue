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

import pytest

from polylogue.core.enums import Provider
from polylogue.storage.sqlite.agent_thread_state import read_provenance, read_thread_titles
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveSourceArtifact,
    upsert_raw_artifact,
    write_source_raw_session,
)
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture
from tests.infra.retained_replay import replay_retained_components
from tests.infra.thread_state import codex_state_export


def _observe(conn: sqlite3.Connection, *, origin: str, source_path: str, payload: bytes, acquired_at_ms: int) -> str:
    return write_source_raw_session(
        conn,
        origin=origin,
        source_path=source_path,
        canonical_source_path=source_path,
        source_index=0,
        payload=payload,
        acquired_at_ms=acquired_at_ms,
    )


@pytest.mark.parametrize("scope_order", [("left", "right"), ("right", "left")])
def test_retained_state_scopes_follow_receipt_order_across_a_clock_rollback(
    tmp_path: Path,
    scope_order: tuple[str, str],
) -> None:
    """Canonical replay keeps two scopes and lets A's returned receipt outrank B."""
    bootstrap_archive_root(tmp_path)
    left = tmp_path / "left" / "state_5.sqlite"
    right = tmp_path / "right" / "state_5.sqlite"
    a = codex_state_export([("left-thread", "State A")])
    b = codex_state_export([("left-thread", "State B")])
    c = codex_state_export([("right-thread", "State C")])
    raw_a = ""
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for scope in scope_order:
            if scope == "left":
                raw_a = archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=a,
                    source_path=str(left),
                    canonical_source_path=str(left),
                    acquired_at_ms=200,
                )
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=b,
                    source_path=str(left),
                    canonical_source_path=str(left),
                    acquired_at_ms=300,
                )
                assert (
                    archive.write_raw_payload(
                        provider=Provider.CODEX,
                        payload=a,
                        source_path=str(left),
                        canonical_source_path=str(left),
                        acquired_at_ms=250,
                    )
                    == raw_a
                )
            else:
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=c,
                    source_path=str(right),
                    canonical_source_path=str(right),
                    acquired_at_ms=100,
                )
        archive.commit()
    replay_retained_components(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        assert read_thread_titles(index, source_scope=str(left.parent)) == {"left-thread": "State A"}
        assert read_thread_titles(index, source_scope=str(right.parent)) == {"right-thread": "State C"}
        provenance = read_provenance(index, source_scope=str(left.parent))
        assert provenance is not None and (provenance.raw_id, provenance.observed_at_ms) == (raw_a, 250)


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
