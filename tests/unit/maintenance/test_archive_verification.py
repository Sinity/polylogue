"""Tests for the read-only archive verification gate (``verify-archive``).

Each deliberately-broken fixture proves a specific check trips on the exact
incoherence it claims to detect -- not merely that *some* check fails.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import (
    ARCHIVE_VERIFICATION_CHECK_NAMES,
    ArchiveVerificationCheck,
    ArchiveVerificationReport,
    verify_archive,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _seed_coherent_archive(root: Path) -> None:
    """Build a minimal but fully coherent 5-tier archive: one raw, one session."""
    initialize_active_archive_root(root)

    source_conn = _connect(root / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-1', 'codex-session', 'session', '/x', ?, 10, 100)
            """,
            (b"a" * 32,),
        )
        source_conn.execute(
            """
            INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, censused_at_ms)
            VALUES ('raw-1', 'fp', 'complete', 1, 100)
            """
        )
        source_conn.commit()
    finally:
        source_conn.close()

    index_conn = _connect(root / "index.db")
    try:
        index_conn.execute(
            """
            INSERT INTO sessions(native_id, origin, raw_id, content_hash, message_count)
            VALUES ('session', 'codex-session', 'raw-1', ?, 1)
            """,
            (b"s" * 32,),
        )
        index_conn.execute(
            """
            INSERT INTO messages(session_id, position, role, material_origin, content_hash)
            VALUES ('codex-session:session', 0, 'user', 'human_authored', ?)
            """,
            (b"m" * 32,),
        )
        index_conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, text)
            VALUES ('codex-session:session:0.0', 'codex-session:session', 0, 'text', 'hello world')
            """
        )
        index_conn.commit()
        index_conn.execute("ANALYZE blocks")
        index_conn.execute("ANALYZE messages")
        index_conn.execute("ANALYZE action_pairs")
        index_conn.commit()
    finally:
        index_conn.close()


def _check(report: ArchiveVerificationReport, name: str) -> ArchiveVerificationCheck:
    matches = [c for c in report.checks if c.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} check, found {len(matches)}"
    match = matches[0]
    assert isinstance(match, ArchiveVerificationCheck)
    return match


def test_coherent_archive_is_all_ok(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path)

    assert not report.blocking
    assert report.warning_count == 0
    assert {check.name for check in report.checks} == set(ARCHIVE_VERIFICATION_CHECK_NAMES)
    for check in report.checks:
        assert check.status is OutcomeStatus.OK, f"{check.name}: {check.summary}"


def test_missing_tier_trips_tier_schema_check(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    (tmp_path / "embeddings.db").unlink()

    report = verify_archive(tmp_path, checks=("tier-schema",))

    check = _check(report, "tier-schema")
    assert check.status is OutcomeStatus.ERROR
    assert report.blocking
    assert "embeddings" in check.summary
    assert check.evidence["tiers"]["embeddings"]["exists"] is False


def test_stale_schema_version_trips_tier_schema_check(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "user.db")
    try:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("tier-schema",))

    check = _check(report, "tier-schema")
    assert check.status is OutcomeStatus.ERROR
    assert "user" in check.summary
    assert check.evidence["tiers"]["user"]["actual_version"] == 1
    assert check.evidence["tiers"]["user"]["expected_version"] == ARCHIVE_TIER_SPECS[ArchiveTier.USER].version


def test_stale_pointer_trips_pointer_coherence_check(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    # Simulate an interrupted blue-green rebuild (polylogue-k8kj): a promoted
    # generation elsewhere, referenced by the pointer, while the conventional
    # index.db path is left behind as a stale regular file.
    generation_dir = tmp_path / ".index-generations" / "gen-2"
    generation_dir.mkdir(parents=True)
    (tmp_path / "index.db").rename(generation_dir / "index.db")
    initialize_active_archive_root(tmp_path)  # recreate a fresh, near-empty stale index.db
    (tmp_path / ".index-active-pointer").write_text(str(generation_dir / "index.db"), encoding="utf-8")

    report = verify_archive(tmp_path, checks=("pointer-coherence",))

    check = _check(report, "pointer-coherence")
    assert check.status is OutcomeStatus.ERROR
    assert report.blocking
    assert "k8kj" in check.summary
    assert check.evidence["active_index_resolved_path"] == str((generation_dir / "index.db").resolve())


def test_invalid_pointer_file_is_reported_as_error(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    (tmp_path / ".index-active-pointer").write_text("not-an-absolute-path", encoding="utf-8")

    report = verify_archive(tmp_path, checks=("pointer-coherence",))

    check = _check(report, "pointer-coherence")
    assert check.status is OutcomeStatus.ERROR
    assert "invalid active index pointer" in check.summary


def test_raw_with_complete_census_and_no_session_is_missing_work(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-orphaned-work', 'codex-session', 'never-materialized', '/y', ?, 10, 100)
            """,
            (b"b" * 32,),
        )
        source_conn.execute(
            """
            INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, censused_at_ms)
            VALUES ('raw-orphaned-work', 'fp', 'complete', 1, 100)
            """
        )
        source_conn.commit()
    finally:
        source_conn.close()

    report = verify_archive(tmp_path, checks=("source-index-coverage",))

    check = _check(report, "source-index-coverage")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["missing_work_count"] == 1
    assert "raw-orphaned-work" in check.evidence["missing_work_sample"]
    assert check.evidence["orphan_count"] == 0


def test_index_session_with_no_backing_raw_is_orphan(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    index_conn = _connect(tmp_path / "index.db")
    try:
        index_conn.execute(
            """
            INSERT INTO sessions(native_id, origin, raw_id, content_hash, message_count)
            VALUES ('orphan-session', 'codex-session', 'raw-does-not-exist', ?, 0)
            """,
            (b"o" * 32,),
        )
        index_conn.commit()
    finally:
        index_conn.close()

    report = verify_archive(tmp_path, checks=("source-index-coverage",))

    check = _check(report, "source-index-coverage")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["orphan_count"] == 1
    assert "raw-does-not-exist" in check.evidence["orphan_sample"]
    assert check.evidence["missing_work_count"] == 0


def test_deleted_fts_row_trips_message_fts_parity(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("DELETE FROM messages_fts")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("fts-parity",))

    check = _check(report, "fts-parity")
    assert check.status is OutcomeStatus.ERROR
    assert "messages_fts gap" in check.summary
    assert check.evidence["messages_fts"]["gap"] == 1
    assert check.evidence["messages_fts"]["worst_sessions"][0]["session_id"] == "codex-session:session"


def test_missing_trigram_row_trips_trigram_parity(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, tool_name, tool_input, tool_id)
            VALUES ('codex-session:session:0.1', 'codex-session:session', 1, 'tool_use', 'Bash',
                    '{"command": "ls -la"}', 'tool-1')
            """
        )
        conn.commit()
        # Simulate drift: the trigram shadow row is gone (as if the trigger
        # never fired, e.g. a schema regression removing it) while the source
        # block remains -- delete via the fts5 'delete' command form so the
        # shadow tables stay internally consistent, then never re-add it.
        row = conn.execute("SELECT rowid, tool_detail_text FROM blocks WHERE block_type = 'tool_use'").fetchone()
        conn.execute(
            "INSERT INTO blocks_command_trigram(blocks_command_trigram, rowid, tool_detail_text) VALUES ('delete', ?, ?)",
            (row[0], row[1]),
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("fts-parity",))

    check = _check(report, "fts-parity")
    assert check.status is OutcomeStatus.ERROR
    assert "blocks_command_trigram gap" in check.summary
    assert check.evidence["blocks_command_trigram"]["gap"] == 1


def test_dangling_resolved_dst_trips_lineage_sanity(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO session_links(
                src_session_id, dst_origin, dst_native_id, link_type,
                resolved_dst_session_id, observed_at_ms
            ) VALUES ('codex-session:session', 'codex-session', 'ghost', 'resume',
                      'codex-session:ghost-session-that-does-not-exist', 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("lineage-sanity",))

    check = _check(report, "lineage-sanity")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["dangling_resolved_dst_count"] == 1
    assert check.evidence["dangling_resolved_dst_sample"] == ["codex-session:ghost-session-that-does-not-exist"]


def test_dangling_branch_point_message_trips_lineage_sanity(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute(
            """
            INSERT INTO session_links(
                src_session_id, dst_origin, dst_native_id, link_type,
                branch_point_message_id, inheritance, observed_at_ms
            ) VALUES ('codex-session:session', 'codex-session', 'child', 'fork',
                      'codex-session:session:no-such-message', 'prefix-sharing', 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("lineage-sanity",))

    check = _check(report, "lineage-sanity")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["dangling_branch_point_count"] == 1
    assert check.evidence["dangling_branch_point_sample"][0]["branch_point_message_id"] == (
        "codex-session:session:no-such-message"
    )


def test_missing_sqlite_stat1_is_warning_not_error(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("DELETE FROM sqlite_stat1")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("planner-stats",))

    check = _check(report, "planner-stats")
    assert check.status is OutcomeStatus.WARNING
    assert not report.blocking  # warnings alone must not gate by default
    assert report.warning_count == 1
    assert "l3tk" in check.summary


def test_partial_analyze_coverage_is_reported_by_table(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("DELETE FROM sqlite_stat1 WHERE tbl = 'action_pairs'")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("planner-stats",))

    check = _check(report, "planner-stats")
    assert check.status is OutcomeStatus.WARNING
    assert check.evidence["missing_tables"] == ["action_pairs"]


def test_counts_summary_reports_origin_breakdown(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("counts-summary",))

    check = _check(report, "counts-summary")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["session_count"] == 1
    assert check.evidence["message_count"] == 1
    assert check.evidence["block_count"] == 1
    assert check.breakdown == {"codex-session": 1}


def test_missing_archive_root_reports_skips_not_crashes(tmp_path: Path) -> None:
    empty_root = tmp_path / "does-not-exist"

    report = verify_archive(empty_root)

    # tier-schema legitimately errors (every tier missing); the point of this
    # test is that the *other* checks degrade to an honest skip instead of
    # raising, not that the whole report reads as clean.
    assert report.blocking
    names_by_status = {check.name: check.status for check in report.checks}
    assert names_by_status["tier-schema"] is OutcomeStatus.ERROR  # every tier missing
    assert names_by_status["source-index-coverage"] is OutcomeStatus.SKIP
    assert names_by_status["fts-parity"] is OutcomeStatus.SKIP
    assert names_by_status["lineage-sanity"] is OutcomeStatus.SKIP
    assert names_by_status["planner-stats"] is OutcomeStatus.SKIP
    assert names_by_status["counts-summary"] is OutcomeStatus.SKIP


def test_one_check_raising_does_not_abort_the_others(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_coherent_archive(tmp_path)
    from polylogue.maintenance import archive_verification as module

    def _boom(_archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
        raise RuntimeError("synthetic failure")

    # Rebuild the registry entry pointing at the broken function, mirroring
    # how a real regression in one check function would surface: the crash
    # is contained to that check's own result, not the whole report.
    broken_specs = tuple(
        module.ArchiveVerificationCheckSpec(spec.name, spec.description, _boom) if spec.name == "fts-parity" else spec
        for spec in module.ARCHIVE_VERIFICATION_CHECKS
    )
    monkeypatch.setattr(module, "ARCHIVE_VERIFICATION_CHECKS", broken_specs)

    report = module.verify_archive(tmp_path)

    by_name = {check.name: check for check in report.checks}
    assert by_name["fts-parity"].status is OutcomeStatus.ERROR
    assert "synthetic failure" in by_name["fts-parity"].summary
    assert by_name["counts-summary"].status is OutcomeStatus.OK
    assert by_name["tier-schema"].status is OutcomeStatus.OK


def test_unknown_check_name_raises_value_error(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    with pytest.raises(ValueError, match="unknown archive verification check"):
        verify_archive(tmp_path, checks=("not-a-real-check",))


def test_report_to_json_is_json_document(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path)
    payload = report.to_json()

    checks_payload = payload["checks"]
    assert isinstance(checks_payload, list)

    assert payload["archive_root"] == str(tmp_path)
    assert payload["blocking"] is False
    assert len(checks_payload) == len(ARCHIVE_VERIFICATION_CHECK_NAMES)
    names = set()
    for entry in checks_payload:
        assert isinstance(entry, dict)
        names.add(entry["name"])
    assert names == set(ARCHIVE_VERIFICATION_CHECK_NAMES)
