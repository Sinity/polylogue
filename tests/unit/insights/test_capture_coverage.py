"""Capture-completeness coverage materialization tests (polylogue-3uw).

Proves the production reference path: real ``raw_hook_events``/``sessions``
tables in a bootstrap-initialized source.db/index.db, correlated by
:func:`compute_capture_coverage`, not a test-only mock of the computation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.insights.capture_coverage import (
    apply_coverage_to_enumeration,
    compute_capture_coverage,
    coverage_citation,
    render_capture_coverage_report,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

NOW_MS = 2_000_000_000_000
GRACE_MS = 15 * 60 * 1000


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    initialize_archive_database(root / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(root / "index.db", ArchiveTier.INDEX)
    return root


def _insert_hook_event(
    archive: Path,
    native_id: str,
    *,
    origin: str = "claude-code-session",
    event: str = "SessionStart",
    observed_at_ms: int = NOW_MS - GRACE_MS - 1,
) -> None:
    payload = {"event_type": event, "session_id": native_id}
    with sqlite3.connect(archive / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id,
                source_path, event_type, payload_json, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"hook-{native_id}-{event}",
                origin,
                native_id,
                native_id,
                f"/hooks/{native_id}.jsonl",
                event,
                json.dumps(payload),
                observed_at_ms,
            ),
        )


def _insert_session(archive: Path, native_id: str, *, origin: str = "claude-code-session") -> None:
    with sqlite3.connect(archive / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, content_hash, updated_at_ms)
            VALUES (?, ?, ?, ?)
            """,
            (native_id, origin, bytes.fromhex("11" * 32), NOW_MS),
        )


def _open(archive: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    return sqlite3.connect(archive / "source.db"), sqlite3.connect(archive / "index.db")


def test_matched_session_reports_full_coverage_no_misses(archive: Path) -> None:
    _insert_hook_event(archive, "session-ok")
    _insert_session(archive, "session-ok")
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )

    hook_source = next(s for s in assessment.sources if s.source == "hook_session_start")
    assert hook_source.status == "computed"
    assert hook_source.observed_count == 1
    assert hook_source.matched_count == 1
    assert hook_source.missing_count == 0
    assert assessment.known_miss_count == 0
    assert assessment.coverage_ratio == 1.0
    assert assessment.is_frame_complete is True


def test_missing_session_is_a_drillable_known_miss(archive: Path) -> None:
    _insert_hook_event(archive, "session-lost")
    # Deliberately no matching session row -- a seeded missed-session scenario.
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )

    assert assessment.known_miss_count == 1
    assert assessment.known_missing_native_ids == ("session-lost",)
    assert assessment.coverage_ratio == 0.0
    assert assessment.is_frame_complete is False


def test_recent_hook_event_inside_grace_window_is_not_yet_a_miss(archive: Path) -> None:
    _insert_hook_event(archive, "session-fresh", observed_at_ms=NOW_MS - 60_000)
    source_conn, index_conn = _open(archive)

    # A window entirely inside the grace period (the last minute, with a
    # 15-minute grace window) cannot yet distinguish "in-flight ingest"
    # from "missed" -- the source must report unknown, not a false miss.
    assessment = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=NOW_MS - 60_000,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        grace_window_ms=GRACE_MS,
        now_ms=NOW_MS,
    )

    hook_source = next(s for s in assessment.sources if s.source == "hook_session_start")
    assert hook_source.status == "unknown"


def test_unwired_origin_reports_hook_source_as_unknown_not_zero(archive: Path) -> None:
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="chatgpt-export",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )

    hook_source = next(s for s in assessment.sources if s.source == "hook_session_start")
    assert hook_source.status == "unknown"
    assert hook_source.reason is not None


def test_declared_uncomputed_sources_are_always_unknown_with_a_reason(archive: Path) -> None:
    _insert_hook_event(archive, "session-ok")
    _insert_session(archive, "session-ok")
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )

    unknown_kinds = {source.source for source in assessment.unknown_sources}
    assert "watcher_file_inventory" in unknown_kinds
    assert "browser_extension_observation" in unknown_kinds
    for source in assessment.unknown_sources:
        assert source.reason


def test_zero_computed_sources_makes_coverage_ratio_and_frame_unknown(archive: Path) -> None:
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="chatgpt-export",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )

    assert assessment.computed_sources == ()
    assert assessment.coverage_ratio is None
    assert assessment.is_frame_complete is False


def test_ref_is_stable_across_independently_constructed_equal_assessments(archive: Path) -> None:
    _insert_hook_event(archive, "session-a")
    _insert_session(archive, "session-a")
    source_conn_1, index_conn_1 = _open(archive)
    source_conn_2, index_conn_2 = _open(archive)

    first = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn_1,
        index_conn=index_conn_1,
        now_ms=NOW_MS,
        generations={"index": "gen-1"},
    )
    second = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn_2,
        index_conn=index_conn_2,
        now_ms=NOW_MS,
        generations={"index": "gen-1"},
    )

    assert first.ref == second.ref


def test_ref_changes_when_evidence_source_status_flips(archive: Path) -> None:
    _insert_hook_event(archive, "session-a")
    source_conn_1, index_conn_1 = _open(archive)
    without_session = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn_1,
        index_conn=index_conn_1,
        now_ms=NOW_MS,
    )

    _insert_session(archive, "session-a")
    source_conn_2, index_conn_2 = _open(archive)
    with_session = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn_2,
        index_conn=index_conn_2,
        now_ms=NOW_MS,
    )

    # Same identity question (both "computed" for hook_session_start): the
    # ref must be stable even though the miss count differs, since the ref
    # names the measurement question, not today's answer.
    assert without_session.ref == with_session.ref
    assert without_session.known_miss_count == 1
    assert with_session.known_miss_count == 0


def test_coverage_citation_carries_ref_and_headline_numbers(archive: Path) -> None:
    _insert_hook_event(archive, "session-lost")
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )
    citation = coverage_citation(assessment)

    assert citation["coverage_ref"] == assessment.ref
    assert citation["known_miss_count"] == 1
    assert citation["frame_complete"] is False


def test_seeded_missing_signal_flips_exact_enumeration_to_frame_incomplete(archive: Path) -> None:
    _insert_hook_event(archive, "session-lost")
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )

    assert apply_coverage_to_enumeration("exact", assessment) == "frame_incomplete"


def test_exact_enumeration_survives_full_coverage(archive: Path) -> None:
    _insert_hook_event(archive, "session-ok")
    _insert_session(archive, "session-ok")
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )

    assert apply_coverage_to_enumeration("exact", assessment) == "exact"


def test_render_report_is_drillable_per_origin(archive: Path) -> None:
    _insert_hook_event(archive, "session-lost")
    source_conn, index_conn = _open(archive)

    assessment = compute_capture_coverage(
        origin="claude-code-session",
        since_ms=0,
        until_ms=NOW_MS,
        source_conn=source_conn,
        index_conn=index_conn,
        now_ms=NOW_MS,
    )
    report = render_capture_coverage_report([assessment])

    assert "claude-code-session" in report
    assert assessment.ref in report
    assert "session-lost" in report
