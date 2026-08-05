"""Tests for the read-only archive verification gate (``verify-archive``).

Each deliberately-broken fixture proves a specific check trips on the exact
incoherence it claims to detect -- not merely that *some* check fails.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest

from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import (
    ARCHIVE_VERIFICATION_CHECK_NAMES,
    ARCHIVE_VERIFICATION_CHECKS,
    ARCHIVE_VERIFICATION_WAIVERS,
    REINDEX_ACCEPTANCE_CHECKS,
    REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS,
    ArchiveVerificationCheck,
    ArchiveVerificationCheckClass,
    ArchiveVerificationReport,
    ArchiveVerificationWaiver,
    verify_archive,
)
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.pathology_zoo import build_pathology_zoo, make_pathology_zoo_member_red
from tests.infra.workload_artifacts import SeededArchiveArtifact


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
        source_conn.execute(
            """
            INSERT INTO blob_refs(blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-1', 'raw_payload', '/x', 10, 100)
            """,
            (b"a" * 32,),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    index_conn = _connect(root / "index.db")
    try:
        index_conn.execute(
            """
            INSERT INTO sessions(
                native_id, origin, raw_id, parser_fingerprint, lowering_fingerprint, content_hash, message_count
            ) VALUES ('session', 'codex-session', 'raw-1', ?, ?, ?, 1)
            """,
            (parser_fingerprint_for_origin("codex-session"), lowering_fingerprint(), b"s" * 32),
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


def test_raw_with_no_typed_refusal_and_no_session_is_untyped_gap(tmp_path: Path) -> None:
    """A head with a real authority grant (not quarantined) and no parse
    error or declared-non-session census verdict that never materialized is
    the exact bug class I1 exists to catch: a silent, unexplained
    materialization failure. Ground truth is ``raw_sessions`` itself -- the
    census row here is deliberately 'complete' (would have satisfied the old,
    buggy universe) to prove the fix is about typing, not about the census.
    """
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, revision_authority
            )
            VALUES ('raw-orphaned-work', 'codex-session', 'never-materialized', '/y', ?, 10, 100, 'byte_proven')
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
    assert check.evidence["untyped_count"] == 1
    assert "raw-orphaned-work" in check.evidence["untyped_sample"]
    assert check.evidence["orphan_count"] == 0


def test_quarantined_head_with_no_session_is_warning_not_error(tmp_path: Path) -> None:
    """The polylogue-in24n bug class in reverse: a quarantined raw (the
    default ``revision_authority``) that never materialized is a *typed*
    gap -- reconciliation hasn't granted it write authority yet -- and must
    surface as WARN-level evidence, not silently invisible (the old bug) and
    not a hard ERROR (quarantine is an ordinary, common, self-explaining
    state; see t0m73/in24n live counts: 7,191 of 7,200 unindexed heads).
    """
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-quarantined', 'codex-session', 'not-yet-reconciled', '/z', ?, 10, 100)
            """,
            (b"c" * 32,),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    report = verify_archive(tmp_path, checks=("source-index-coverage",))

    check = _check(report, "source-index-coverage")
    assert check.status is OutcomeStatus.WARNING
    assert not report.blocking
    assert check.evidence["quarantined_count"] == 1
    assert "raw-quarantined" in check.evidence["quarantined_sample"]
    assert check.evidence["untyped_count"] == 0


def test_parse_error_head_with_no_session_is_ok(tmp_path: Path) -> None:
    """A head that failed to parse is explained by ``raw_sessions.parse_error``
    itself -- not a coverage problem this check should flag at all."""
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms,
                revision_authority, parse_error
            )
            VALUES ('raw-parse-error', 'codex-session', 'unparseable', '/w', ?, 10, 100, 'byte_proven', 'boom')
            """,
            (b"d" * 32,),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    report = verify_archive(tmp_path, checks=("source-index-coverage",))

    check = _check(report, "source-index-coverage")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["parse_error_count"] == 1
    assert check.evidence["untyped_count"] == 0
    assert check.evidence["quarantined_count"] == 0


def test_non_session_census_head_with_no_session_is_ok(tmp_path: Path) -> None:
    """A head the census declared not-a-session (e.g. a settings/config
    artifact) is a declared refusal, not a materialization gap."""
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, revision_authority
            )
            VALUES ('raw-non-session', 'codex-session', 'not-a-session', '/v', ?, 10, 100, 'byte_proven')
            """,
            (b"e" * 32,),
        )
        source_conn.execute(
            """
            INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, censused_at_ms)
            VALUES ('raw-non-session', 'fp', 'non_session', 0, 100)
            """
        )
        source_conn.commit()
    finally:
        source_conn.close()

    report = verify_archive(tmp_path, checks=("source-index-coverage",))

    check = _check(report, "source-index-coverage")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["non_session_count"] == 1
    assert check.evidence["untyped_count"] == 0


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
    assert check.evidence["untyped_count"] == 0


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


def test_missing_fts_trigger_trips_message_fts_parity(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("DROP TRIGGER messages_fts_ai")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("fts-parity",))

    check = _check(report, "fts-parity")
    assert check.status is OutcomeStatus.ERROR
    assert "messages_fts triggers missing" in check.summary
    assert check.evidence["messages_fts"]["triggers_present"] is False


def test_excess_fts_row_trips_message_fts_parity(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("DROP TRIGGER messages_fts_ad")
        conn.execute("DELETE FROM blocks")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("fts-parity",))

    check = _check(report, "fts-parity")
    assert check.status is OutcomeStatus.ERROR
    assert "messages_fts excess_rows=1" in check.summary
    assert check.evidence["messages_fts"]["excess_rows"] == 1


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
    assert names_by_status["enum-superset-check"] is OutcomeStatus.SKIP
    assert names_by_status["blob-refs-liveness"] is OutcomeStatus.SKIP
    assert names_by_status["embeddings-refs-liveness"] is OutcomeStatus.SKIP
    assert names_by_status["session-lineage-acyclic"] is OutcomeStatus.SKIP
    assert names_by_status["message-count-projection"] is OutcomeStatus.SKIP
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
        module.ArchiveVerificationCheckSpec(spec.name, spec.description, _boom, spec.check_class)
        if spec.name == "fts-parity"
        else spec
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


def test_enum_superset_check_passes_on_current_schema(tmp_path: Path) -> None:
    """Freshly bootstrapped DDL is always generated from the current enum
    (I2's failure mode requires a DDL frozen at an *older* enum version, which
    a fresh fixture can't reproduce) -- this proves the check reads clean
    against the schema this branch actually ships, not just that it never
    trips."""
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("enum-superset-check",))

    check = _check(report, "enum-superset-check")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["current_origin_vocabulary"]


def test_blob_ref_with_no_referent_trips_blob_refs_liveness(tmp_path: Path) -> None:
    """RED TWIN (I3): a blob_refs row surviving its referent's deletion is
    exactly the GC-liveness bug class this check exists to catch -- proves
    it fails on the specific incoherence it claims to detect, not merely
    that some check fails."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO blob_refs(blob_hash, ref_id, ref_type, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-does-not-exist', 'raw_payload', 10, 100)
            """,
            (b"f" * 32,),
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("blob-refs-liveness",))

    check = _check(report, "blob-refs-liveness")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["orphans_by_ref_type"]["raw_payload"] == 1
    assert "raw-does-not-exist" in check.evidence["orphan_samples_by_ref_type"]["raw_payload"]


def test_blob_refs_liveness_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("blob-refs-liveness",))

    check = _check(report, "blob-refs-liveness")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["orphans_by_ref_type"] == {
        "attachment": 0,
        "hook_payload": 0,
        "raw_payload": 0,
        "sidecar": 0,
    }


def test_blob_reference_closure_rejects_acquired_attachment_without_ref(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute(
            "INSERT INTO attachments (attachment_id, byte_count, blob_hash, acquisition_status, ref_count) "
            "VALUES ('orphan-acquired', 1, ?, 'acquired', 0)",
            (b"a" * 32,),
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("blob-reference-closure",))
    check = _check(report, "blob-reference-closure")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["acquired_attachment_missing_ref_count"] == 1


def test_attachment_blob_ref_joins_its_parent_raw_session(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO blob_refs(blob_hash, ref_id, ref_type, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-1', 'attachment', 10, 100)
            """,
            (b"a" * 32,),
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("blob-refs-liveness",))

    check = _check(report, "blob-refs-liveness")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["orphans_by_ref_type"]["attachment"] == 0


def test_orphaned_embedding_ref_trips_embeddings_refs_liveness(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "embeddings.db")
    try:
        conn.execute(
            """
            INSERT INTO message_embedding_refs(message_id, session_id, origin, embedding_input_hash)
            VALUES ('codex-session:session:no-such-message', 'codex-session:session', 'codex-session', ?)
            """,
            (b"g" * 32,),
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("embeddings-refs-liveness",))

    check = _check(report, "embeddings-refs-liveness")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["orphan_count"] == 1
    assert "codex-session:session:no-such-message" in check.evidence["orphan_sample"]


def test_embeddings_refs_liveness_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("embeddings-refs-liveness",))

    check = _check(report, "embeddings-refs-liveness")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["orphan_count"] == 0


def _insert_bare_session(conn: sqlite3.Connection, native_id: str, content_byte: bytes) -> None:
    conn.execute(
        """
        INSERT INTO sessions(native_id, origin, content_hash, message_count)
        VALUES (?, 'codex-session', ?, 0)
        """,
        (native_id, content_byte * 32),
    )


def test_parent_session_id_cycle_trips_session_lineage_acyclic(tmp_path: Path) -> None:
    """RED TWIN (I5): two sessions whose parent_session_id point at each
    other is the exact cycle the lineage-prefix-recomposition walker must
    never be handed -- proves this check detects it, not merely that some
    check fails."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        _insert_bare_session(conn, "cycle-a", b"\xa1")
        _insert_bare_session(conn, "cycle-b", b"\xb2")
        conn.commit()
        conn.execute("UPDATE sessions SET parent_session_id = 'codex-session:cycle-b' WHERE native_id = 'cycle-a'")
        conn.execute("UPDATE sessions SET parent_session_id = 'codex-session:cycle-a' WHERE native_id = 'cycle-b'")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("session-lineage-acyclic",))

    check = _check(report, "session-lineage-acyclic")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["cycle_member_sample"]
    assert {"codex-session:cycle-a", "codex-session:cycle-b"} <= set(check.evidence["cycle_member_sample"])


def test_session_lineage_acyclic_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        _insert_bare_session(conn, "child", b"\xc3")
        conn.commit()
        conn.execute("UPDATE sessions SET parent_session_id = 'codex-session:session' WHERE native_id = 'child'")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("session-lineage-acyclic",))

    check = _check(report, "session-lineage-acyclic")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["linked_session_count"] == 1


def test_drifted_message_count_trips_message_count_projection(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("UPDATE sessions SET message_count = 99 WHERE session_id = 'codex-session:session'")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("message-count-projection",))

    check = _check(report, "message-count-projection")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["drifted_session_count"] == 1
    assert "codex-session:session" in check.evidence["drifted_session_sample"]


def test_message_count_projection_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("message-count-projection",))

    check = _check(report, "message-count-projection")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["drifted_session_count"] == 0


def test_reindex_acceptance_rejects_missing_semantic_stamp_coverage(tmp_path: Path) -> None:
    """A candidate cannot promote when a session has no parser stamp."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET parser_fingerprint = NULL")

    report = verify_archive(tmp_path, checks=REINDEX_ACCEPTANCE_CHECKS)

    check = _check(report, "session-fingerprint-stamps")
    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["invalid_parser_stamp_count"] == 1
    assert check.evidence["fully_current_session_count"] == 0


def test_reindex_acceptance_rejects_missing_semantic_stamp_column(tmp_path: Path) -> None:
    """A pre-bootstrap candidate cannot bypass coverage by lacking the column."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute("ALTER TABLE sessions DROP COLUMN parser_fingerprint")

    report = verify_archive(tmp_path, checks=REINDEX_ACCEPTANCE_CHECKS)

    check = _check(report, "session-fingerprint-stamps")
    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert "missing fingerprint column(s): parser_fingerprint" in check.summary


def test_reindex_acceptance_rejects_stale_semantic_stamps(tmp_path: Path) -> None:
    """A well-formed historic stamp must fail the current-source comparison."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET parser_fingerprint = ?", ("0" * 64,))

    report = verify_archive(tmp_path, checks=REINDEX_ACCEPTANCE_CHECKS)

    check = _check(report, "session-fingerprint-stamps")
    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["parser_stale_by_origin"] == {"codex-session": 1}


def test_reindex_acceptance_rejects_mixed_semantic_stamps(tmp_path: Path) -> None:
    """A partly replayed candidate cannot hide a second parser/lowering vintage."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions(native_id, origin, parser_fingerprint, lowering_fingerprint, content_hash, message_count)
            VALUES ('other', 'codex-session', ?, ?, ?, 0)
            """,
            ("1" * 64, "2" * 64, b"o" * 32),
        )

    report = verify_archive(tmp_path, checks=REINDEX_ACCEPTANCE_CHECKS)

    check = _check(report, "session-fingerprint-stamps")
    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["parser_mixed_by_origin"] == {"codex-session": 2}
    assert check.evidence["lowering_distinct_count"] == 2


def test_missing_enum_value_trips_enum_superset_check(tmp_path: Path) -> None:
    """RED TWIN (I2): a live CHECK list frozen at an older, narrower Origin
    vocabulary is exactly the drift class this check exists to catch -- an
    additive enum change that didn't rebuild every table's DDL. A fresh
    fixture can't reproduce a real table falling behind (its DDL is always
    generated from the current enum), so this creates a synthetic table
    whose CHECK list the check's own regex recognizes (``origin ... IN
    (...)``), pinned to a single, deliberately-stale origin value."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute("CREATE TABLE frozen_origin_probe (origin TEXT NOT NULL CHECK(origin IN ('codex-session')))")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("enum-superset-check",))

    check = _check(report, "enum-superset-check")
    assert check.status is OutcomeStatus.ERROR
    assert "source.db:frozen_origin_probe.origin" in check.evidence["missing_by_column"]


def test_byte_dup_of_indexed_head_is_reported_as_evidence_not_hidden(tmp_path: Path) -> None:
    """I1 correction (2026-08-03 operator challenge): an unindexed head whose
    bytes duplicate an already-indexed raw is real evidence the gap isn't
    all-novel. This must show up as report evidence (byte_dup_of_indexed_count,
    novel_unindexed_count) without silently suppressing the underlying
    quarantined/untyped classification the head still gets."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-dup', 'codex-session', 'session-dup', '/y', ?, 10, 200)
            """,
            (b"a" * 32,),  # same blob_hash as raw-1, which IS indexed; defaults to quarantined
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("source-index-coverage",))

    check = _check(report, "source-index-coverage")
    assert check.status is OutcomeStatus.WARNING  # quarantined (default authority), not untyped -- doesn't block
    assert check.evidence["quarantined_count"] == 1
    assert check.evidence["byte_dup_of_indexed_count"] == 1
    assert check.evidence["novel_unindexed_count"] == 0


def test_stalled_backlog_trips_convergence_freshness(tmp_path: Path) -> None:
    """RED TWIN (I6): an unindexed backlog (a second, uncovered logical
    source) with no daemon/convergence activity recorded in ops.db is
    exactly the "stalled, not async lag" condition this check exists to
    catch -- the corrected criterion (gap>0 AND no recent activity) that
    replaced the prototype's too-weak "some activity ever happened"."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-gap', 'codex-session', 'session-gap', '/z', ?, 10, 300)
            """,
            (b"b" * 32,),
        )
        conn.commit()
    finally:
        conn.close()
    # ops.db exists (bootstrapped by _seed_coherent_archive) but carries no
    # daemon_events/daemon_stage_events/convergence_debt rows -- no activity.

    report = verify_archive(tmp_path, checks=("convergence-freshness",))

    check = _check(report, "convergence-freshness")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unindexed_backlog_gap"] == 1


@pytest.mark.frozen_clock_modules("polylogue.maintenance.archive_verification")
def test_recent_activity_downgrades_convergence_freshness_to_warning(tmp_path: Path, frozen_clock: Any) -> None:
    """A gap with recent daemon activity is WARNING (still converging), not
    ERROR (stalled) -- the criterion's second half, proven independently of
    the red twin above."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-gap', 'codex-session', 'session-gap', '/z', ?, 10, 300)
            """,
            (b"b" * 32,),
        )
        conn.commit()
    finally:
        conn.close()
    ops_conn = _connect(tmp_path / "ops.db")
    try:
        now_ms = int(frozen_clock.now().timestamp() * 1000)
        ops_conn.execute(
            "INSERT INTO daemon_events(ts_ms, kind, operation_id, payload_json) VALUES (?, 'ingest', 'op-1', '{}')",
            (now_ms,),
        )
        ops_conn.commit()
    finally:
        ops_conn.close()

    report = verify_archive(tmp_path, checks=("convergence-freshness",))

    check = _check(report, "convergence-freshness")
    assert check.status is OutcomeStatus.WARNING
    assert check.evidence["unindexed_backlog_gap"] == 1


def test_convergence_freshness_passes_with_no_gap(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("convergence-freshness",))

    check = _check(report, "convergence-freshness")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["unindexed_backlog_gap"] == 0


def test_dangling_assertion_target_trips_user_tier_refs(tmp_path: Path) -> None:
    """RED TWIN (I10): a user-tier assertion whose target session/message no
    longer resolves in index.db is a dangling reference -- silently
    unreachable from any normal target-scoped query, exactly the liveness
    gap this check exists to catch."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "user.db")
    try:
        conn.execute(
            """
            INSERT INTO assertions(assertion_id, target_ref, kind, body_text, created_at_ms, updated_at_ms)
            VALUES ('a-dangling', 'session:codex-session:no-such-session', 'note', 'orphaned note', 100, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("user-tier-refs",))

    check = _check(report, "user-tier-refs")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["dangling_session_ref_count"] == 1
    assert "a-dangling" in check.evidence["dangling_sample"]


def test_user_tier_refs_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "user.db")
    try:
        conn.execute(
            """
            INSERT INTO assertions(assertion_id, target_ref, kind, body_text, created_at_ms, updated_at_ms)
            VALUES ('a-live', 'session:codex-session:session', 'note', 'a live note', 100, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("user-tier-refs",))

    check = _check(report, "user-tier-refs")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["total_scoped_assertion_count"] == 1
    assert check.evidence["dangling_session_ref_count"] == 0


def test_excluded_cursor_with_live_next_retry_at_trips_vocabulary_honesty(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "ops.db")
    try:
        conn.execute(
            """
            INSERT INTO ingest_cursor(source_path, excluded, failure_count, next_retry_at, updated_at_ms)
            VALUES ('/poison.jsonl', 1, 5, '2026-08-04T00:00:00+00:00', 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("excluded-cursor-vocabulary-honesty",))

    check = _check(report, "excluded-cursor-vocabulary-honesty")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["mislabeled_count"] == 1
    assert "/poison.jsonl" in check.details


def test_excluded_cursor_vocabulary_honesty_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "ops.db")
    try:
        conn.execute(
            """
            INSERT INTO ingest_cursor(source_path, excluded, failure_count, next_retry_at, updated_at_ms)
            VALUES ('/poison.jsonl', 1, 5, NULL, 100)
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("excluded-cursor-vocabulary-honesty",))

    check = _check(report, "excluded-cursor-vocabulary-honesty")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["excluded_count"] == 1
    assert check.evidence["mislabeled_count"] == 0


def test_stalled_append_cursor_trips_freshness_check(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    now_ms = 10_000_000_000  # far enough past epoch that "now - threshold" stays positive
    stale_updated_at_ms = now_ms - int(3 * 60 * 60 * 1000)  # 3h old, past the 1h threshold
    conn = _connect(tmp_path / "ops.db")
    try:
        conn.execute(
            """
            INSERT INTO ingest_cursor(source_path, excluded, stat_size, byte_offset, updated_at_ms)
            VALUES ('/rollout-stalled.jsonl', 0, 100000000, 5000000, ?)
            """,
            (stale_updated_at_ms,),
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("stalled-append-cursor-freshness",))

    check = _check(report, "stalled-append-cursor-freshness")
    assert check.status is OutcomeStatus.WARNING
    assert check.evidence["stalled_count"] == 1
    assert check.evidence["total_lag_bytes"] == 95_000_000
    assert "/rollout-stalled.jsonl" in check.details


def test_recently_stalled_append_cursor_does_not_trip_freshness_check(tmp_path: Path) -> None:
    """A cursor briefly behind its file's size (writer mid-append) is not a
    finding -- only one stalled longer than the escalation threshold is."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "ops.db")
    try:
        conn.execute(
            """
            INSERT INTO ingest_cursor(source_path, excluded, stat_size, byte_offset, updated_at_ms)
            VALUES (
                '/rollout-active.jsonl', 0, 100000000, 5000000,
                CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("stalled-append-cursor-freshness",))

    check = _check(report, "stalled-append-cursor-freshness")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["stalled_count"] == 0


def test_stalled_append_cursor_freshness_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("stalled-append-cursor-freshness",))

    check = _check(report, "stalled-append-cursor-freshness")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["stalled_count"] == 0


def test_fully_quarantined_duplicate_group_trips_raw_quarantine_group_dedup(tmp_path: Path) -> None:
    """polylogue-zm4w8: two quarantined raws sharing (source_path, blob_hash),
    with no indexed twin anywhere for that blob_hash, is exactly the residual
    gap raw-byte-duplicate-supersession-apply cannot see (it requires an
    already-indexed twin). This must trip ERROR, not the WARN
    source-index-coverage already gives quarantined-but-unindexed heads.
    """
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        for raw_id in ("raw-dup-a", "raw-dup-b"):
            source_conn.execute(
                """
                INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
                VALUES (?, 'codex-session', ?, '/rollout-repeated.jsonl', ?, 10, 100)
                """,
                (raw_id, f"native-{raw_id}", b"d" * 32),
            )
        source_conn.commit()
    finally:
        source_conn.close()

    report = verify_archive(tmp_path, checks=("raw-quarantine-group-dedup",))

    check = _check(report, "raw-quarantine-group-dedup")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["group_count"] == 1
    assert check.evidence["duplicate_count"] == 1
    group_sample = check.evidence["group_sample"]
    assert len(group_sample) == 1
    assert group_sample[0]["source_path"] == "/rollout-repeated.jsonl"
    assert group_sample[0]["representative_raw_id"] == "raw-dup-a"
    assert group_sample[0]["duplicate_raw_ids"] == ["raw-dup-b"]


def test_quarantined_duplicate_with_indexed_twin_elsewhere_does_not_trip(tmp_path: Path) -> None:
    """A (source_path, blob_hash) group of >1 quarantined rows whose
    blob_hash ALSO appears on an already-indexed raw elsewhere is
    raw-byte-duplicate-supersession-apply's territory, not this check's --
    it must not double-flag content that actuator can already resolve.
    """
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        for raw_id in ("raw-dup-c", "raw-dup-d"):
            source_conn.execute(
                """
                INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
                VALUES (?, 'codex-session', ?, '/rollout-also-indexed.jsonl', ?, 10, 100)
                """,
                (raw_id, f"native-{raw_id}", b"s" * 32),
            )
        # raw-1's blob_hash (b"s" * 32) is the coherent-archive fixture's
        # already-indexed raw -- share it here to simulate an indexed twin.
        source_conn.execute("UPDATE raw_sessions SET blob_hash = ? WHERE raw_id = 'raw-1'", (b"s" * 32,))
        source_conn.commit()
    finally:
        source_conn.close()

    report = verify_archive(tmp_path, checks=("raw-quarantine-group-dedup",))

    check = _check(report, "raw-quarantine-group-dedup")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["group_count"] == 0
    assert check.evidence["already_resolved_group_count"] == 1


def test_raw_quarantine_group_dedup_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("raw-quarantine-group-dedup",))

    check = _check(report, "raw-quarantine-group-dedup")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["group_count"] == 0


@pytest.mark.parametrize("check_name", ARCHIVE_VERIFICATION_CHECK_NAMES)
def test_every_registry_check_does_not_error_on_the_real_pipeline_corpus(
    check_name: str, seeded_archive: SeededArchiveArtifact
) -> None:
    """Every check in the registry, run individually against a real-pipeline
    (acquire->parse->materialize->index), multi-provider synthetic corpus,
    must not report ``error``. This is the anti-vacuity backstop for the
    whole registry: a check that only ever runs against a single
    hand-inserted row (the other tests in this file) could pass by never
    actually exercising realistic multi-session, multi-provider shape.
    ``verify_archive`` is read-only, so the session-scoped fixture root is
    read directly with no per-test clone.
    """
    report = verify_archive(seeded_archive.root, checks=(check_name,))

    check = _check(report, check_name)
    assert check.status is not OutcomeStatus.ERROR, f"{check_name}: {check.summary}\n{check.evidence}"


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
        assert entry["check_class"] in {cls.value for cls in ArchiveVerificationCheckClass}
    assert names == set(ARCHIVE_VERIFICATION_CHECK_NAMES)


# ---------------------------------------------------------------------------
# Registry mechanism: class tagging, waivers, reindex acceptance subset
# (polylogue-t0m73 productization)
# ---------------------------------------------------------------------------


def test_every_registered_check_has_a_class_tag() -> None:
    """Construction-time contract: :class:`ArchiveVerificationCheckSpec` has
    no default for ``check_class``, so a check without a tag is a TypeError
    at import time -- this test just makes that guarantee explicit and
    readable rather than relying on incidental import success."""
    for spec in ARCHIVE_VERIFICATION_CHECKS:
        assert isinstance(spec.check_class, ArchiveVerificationCheckClass), spec.name


def test_pathology_zoo_contract_is_production_owned_and_registered() -> None:
    """The archive verifier, rather than tests, owns the zoo's enforcement boundary."""
    from polylogue.maintenance.pathology_zoo import PATHOLOGY_ZOO_MANIFEST

    assert len(PATHOLOGY_ZOO_MANIFEST) == 17
    assert "pathology-zoo-invariants" in ARCHIVE_VERIFICATION_CHECK_NAMES


def test_check_class_is_stamped_onto_every_report_check(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    by_name = {spec.name: spec for spec in ARCHIVE_VERIFICATION_CHECKS}

    report = verify_archive(tmp_path)

    for check in report.checks:
        assert isinstance(check, ArchiveVerificationCheck)
        assert check.check_class == by_name[check.name].check_class.value


def test_waived_check_still_reports_error_but_does_not_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED TWIN for the waiver mechanism itself: a waived check must keep
    reporting its true ``error`` status and evidence (the finding is never
    hidden) while :attr:`ArchiveVerificationReport.blocking` excludes it --
    proving the waiver changes the *gate*, not the *check*."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "embeddings.db")
    try:
        conn.execute(
            """
            INSERT INTO message_embedding_refs(message_id, session_id, origin, embedding_input_hash)
            VALUES ('codex-session:session:no-such-message', 'codex-session:session', 'codex-session', ?)
            """,
            (b"h" * 32,),
        )
        conn.commit()
    finally:
        conn.close()
    from polylogue.maintenance import archive_verification as module

    monkeypatch.setattr(
        module,
        "ARCHIVE_VERIFICATION_WAIVERS",
        {"embeddings-refs-liveness": ArchiveVerificationWaiver(bead_id="polylogue-test", reason="synthetic")},
    )

    report = module.verify_archive(tmp_path, checks=("embeddings-refs-liveness",))

    check = _check(report, "embeddings-refs-liveness")
    assert check.status is OutcomeStatus.ERROR  # the finding is never hidden
    assert check.waived_bead_id == "polylogue-test"
    assert check.evidence["waiver"]["bead_id"] == "polylogue-test"
    assert not report.blocking  # but the waived finding does not gate


def test_real_waiver_table_also_waives_embeddings_refs_liveness(tmp_path: Path) -> None:
    """Sanity twin for the waiver test above: with the real (unmodified)
    ``ARCHIVE_VERIFICATION_WAIVERS`` table, this same violation is ALSO
    non-blocking (waived), confirming the prior monkeypatched-table test's
    green result matches production waiver config rather than an artifact
    of the test's own patched table."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "embeddings.db")
    try:
        conn.execute(
            """
            INSERT INTO message_embedding_refs(message_id, session_id, origin, embedding_input_hash)
            VALUES ('codex-session:session:no-such-message', 'codex-session:session', 'codex-session', ?)
            """,
            (b"h" * 32,),
        )
        conn.commit()
    finally:
        conn.close()
    assert "embeddings-refs-liveness" in ARCHIVE_VERIFICATION_WAIVERS  # documents why this bug is currently waived

    report = verify_archive(tmp_path, checks=("embeddings-refs-liveness",))

    check = _check(report, "embeddings-refs-liveness")
    assert check.status is OutcomeStatus.ERROR
    assert check.waived_bead_id == "polylogue-feu0"
    assert not report.blocking  # waived by the real table too -- consistent with the mechanism test above


def test_reindex_acceptance_checks_are_all_registered_and_ground_truth_eligible() -> None:
    """Every name in :data:`REINDEX_ACCEPTANCE_CHECKS` must be a real
    registry check, and running it against an index-only root (mirroring a
    real generation directory, which has no source.db/user.db/embeddings.db)
    must never report ``error`` from a missing-tier false positive."""
    assert set(REINDEX_ACCEPTANCE_CHECKS) <= set(ARCHIVE_VERIFICATION_CHECK_NAMES)


def test_reindex_acceptance_subset_is_satisfiable_from_index_only_root(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier

    conn = _connect(tmp_path / "index.db")
    try:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
        conn.execute(
            """
            INSERT INTO sessions(native_id, origin, parser_fingerprint, lowering_fingerprint, content_hash, message_count)
            VALUES ('session', 'codex-session', ?, ?, ?, 0)
            """,
            (parser_fingerprint_for_origin("codex-session"), lowering_fingerprint(), b"s" * 32),
        )
        conn.commit()
        conn.execute("ANALYZE blocks")
        conn.execute("ANALYZE messages")
        conn.execute("ANALYZE action_pairs")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=list(REINDEX_ACCEPTANCE_CHECKS))

    assert not report.blocking, [check.summary for check in report.checks if check.status is OutcomeStatus.ERROR]
    for check in report.checks:
        assert check.status is not OutcomeStatus.ERROR, f"{check.name}: {check.summary}"


# ---------------------------------------------------------------------------
# RED-TWIN contract rule (polylogue-t0m73): structural enforcement
# ---------------------------------------------------------------------------

#: Maps every registry check name to the name of the test function in *this*
#: module that proves a fixture mutation flips it to a non-OK status. This
#: is the RED-TWIN contract rule's structural carrier: a check without an
#: entry here (or whose named function doesn't exist) fails
#: ``test_every_non_complexity_check_has_a_red_twin_test`` below, rather
#: than silently shipping a check nobody has proven can ever go red.
#: ``COMPLEXITY``-class checks (report-only, never pass/fail by design --
#: see :class:`ArchiveVerificationCheckClass`) are exempt.
RED_TWIN_TESTS: dict[str, str] = {
    "tier-schema": "test_missing_tier_trips_tier_schema_check",
    "pointer-coherence": "test_stale_pointer_trips_pointer_coherence_check",
    "source-index-coverage": "test_raw_with_no_typed_refusal_and_no_session_is_untyped_gap",
    "fts-parity": "test_deleted_fts_row_trips_message_fts_parity",
    "lineage-sanity": "test_dangling_resolved_dst_trips_lineage_sanity",
    "enum-superset-check": "test_missing_enum_value_trips_enum_superset_check",
    "blob-refs-liveness": "test_blob_ref_with_no_referent_trips_blob_refs_liveness",
    "blob-reference-closure": "test_blob_reference_closure_rejects_acquired_attachment_without_ref",
    "pathology-zoo-invariants": "test_pathology_zoo_invariants_red_twin",
    "embeddings-refs-liveness": "test_orphaned_embedding_ref_trips_embeddings_refs_liveness",
    "session-lineage-acyclic": "test_parent_session_id_cycle_trips_session_lineage_acyclic",
    "message-count-projection": "test_drifted_message_count_trips_message_count_projection",
    "session-fingerprint-stamps": "test_reindex_acceptance_rejects_missing_semantic_stamp_coverage",
    "planner-stats": "test_missing_sqlite_stat1_is_warning_not_error",
    "convergence-freshness": "test_stalled_backlog_trips_convergence_freshness",
    "user-tier-refs": "test_dangling_assertion_target_trips_user_tier_refs",
    "excluded-cursor-vocabulary-honesty": "test_excluded_cursor_with_live_next_retry_at_trips_vocabulary_honesty",
    "stalled-append-cursor-freshness": "test_stalled_append_cursor_trips_freshness_check",
    "raw-quarantine-group-dedup": "test_fully_quarantined_duplicate_group_trips_raw_quarantine_group_dedup",
    "corpus-absences": "test_corpus_absences_red_twin",
    "corpus-attachment-fidelity": "test_corpus_attachment_fidelity_red_twin",
    "corpus-revision-fidelity": "test_corpus_revision_fidelity_red_twin",
}


def test_corpus_absences_red_twin(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-corpus-absence', 'codex-session', 'missing', '/missing', ?, 1, 1)
            """,
            (b"c" * 32,),
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count
            ) VALUES ('raw-corpus-absence', 'codex:missing', 'missing', 'r1', ?, 1)
            """,
            (b"d" * 32,),
        )
    report = verify_archive(tmp_path, checks=("corpus-absences",))
    assert _check(report, "corpus-absences").status is OutcomeStatus.ERROR


def test_corpus_attachment_fidelity_red_twin(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute("INSERT INTO attachments(attachment_id) VALUES ('raw-attachment')")
        conn.execute(
            """
            INSERT INTO attachment_refs(attachment_id, session_id, message_id, position, upload_origin)
            VALUES ('raw-attachment', 'codex-session:session', 'codex-session:session:0.0', 0, 'drive')
            """
        )
    report = verify_archive(tmp_path, checks=("corpus-attachment-fidelity",))
    assert _check(report, "corpus-attachment-fidelity").status is OutcomeStatus.ERROR


def test_corpus_revision_fidelity_red_twin(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-corpus-revision', 'codex-session', 'session', '/revision', ?, 1, 1)
            """,
            (b"e" * 32,),
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count
            ) VALUES ('raw-corpus-revision', 'codex:session', 'session', 'r2', ?, 100)
            """,
            (b"f" * 32,),
        )
    report = verify_archive(tmp_path, checks=("corpus-revision-fidelity",))
    assert _check(report, "corpus-revision-fidelity").status is OutcomeStatus.ERROR


def test_pathology_zoo_invariants_red_twin(tmp_path: Path) -> None:
    """Every production manifest member makes its registered verifier red when mutated."""
    from polylogue.maintenance.pathology_zoo import PATHOLOGY_ZOO_MANIFEST

    zoo = build_pathology_zoo(tmp_path / "zoo")
    green = verify_archive(zoo.archive_root, checks=("pathology-zoo-invariants",))
    assert _check(green, "pathology-zoo-invariants").status is OutcomeStatus.OK

    for member in PATHOLOGY_ZOO_MANIFEST:
        mutated_root = tmp_path / member.member_id
        copytree(zoo.archive_root, mutated_root)
        make_pathology_zoo_member_red(mutated_root, member.member_id)

        red = verify_archive(mutated_root, checks=("pathology-zoo-invariants",))
        check = _check(red, "pathology-zoo-invariants")
        assert check.status is OutcomeStatus.ERROR, member.invariant.condition
        assert member.member_id in check.evidence["failed_member_ids"]


def test_pathology_zoo_candidate_check_uses_candidate_index_and_durable_source(tmp_path: Path) -> None:
    zoo = build_pathology_zoo(tmp_path / "zoo")
    candidate = tmp_path / "candidate-index.db"
    shutil.copy2(zoo.archive_root / "index.db", candidate)

    green = verify_archive(
        zoo.archive_root,
        checks=("pathology-zoo-invariants",),
        index_path_override=candidate,
    )
    assert _check(green, "pathology-zoo-invariants").status is OutcomeStatus.OK

    with _connect(candidate) as conn:
        conn.execute("UPDATE sessions SET message_count = 47 WHERE session_id = ?", ("codex-session:zoo-whale",))

    red = verify_archive(
        zoo.archive_root,
        checks=REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS,
        index_path_override=candidate,
    )
    check = _check(red, "pathology-zoo-invariants")
    assert check.status is OutcomeStatus.ERROR
    assert "whale-component" in check.evidence["failed_member_ids"]


def test_every_non_complexity_check_has_a_red_twin_test() -> None:
    """Structural enforcement of the RED-TWIN contract rule (polylogue-t0m73):
    every registry check whose class is not COMPLEXITY (a pass/fail
    invariant, not a report-only summary -- see
    :class:`ArchiveVerificationCheckClass`'s docstring) must have a
    registered red-twin test in :data:`RED_TWIN_TESTS`, and that test
    function must actually exist in this module. A new check added to
    :data:`ARCHIVE_VERIFICATION_CHECKS` without a red-twin entry fails this
    test immediately -- the meta-test the AC calls for, not a manual review
    convention that can be forgotten."""
    module_globals = globals()
    missing_entry = []
    missing_function = []
    for spec in ARCHIVE_VERIFICATION_CHECKS:
        if spec.check_class is ArchiveVerificationCheckClass.COMPLEXITY:
            continue
        test_name = RED_TWIN_TESTS.get(spec.name)
        if test_name is None:
            missing_entry.append(spec.name)
            continue
        if test_name not in module_globals or not callable(module_globals[test_name]):
            missing_function.append((spec.name, test_name))
    assert not missing_entry, f"registry check(s) with no RED_TWIN_TESTS entry: {missing_entry}"
    assert not missing_function, f"RED_TWIN_TESTS entries pointing at a missing test function: {missing_function}"


def test_red_twin_tests_only_reference_real_registry_checks() -> None:
    """Inverse of the above: a stale ``RED_TWIN_TESTS`` entry for a check
    name that no longer exists in the registry (e.g. after a rename) is
    itself a drift signal worth catching, not silently ignored."""
    assert set(RED_TWIN_TESTS) <= set(ARCHIVE_VERIFICATION_CHECK_NAMES)
