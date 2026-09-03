"""Tests for the read-only archive verification gate (``verify-archive``).

Each deliberately-broken fixture proves a specific check trips on the exact
incoherence it claims to detect -- not merely that *some* check fails.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.topology.edge import (
    HOOK_AUTHORITATIVE_LINK_METHOD,
    HOOK_CONTRADICTED_LINK_METHOD,
)
from polylogue.core.enums import ArtifactSupportStatus, Origin
from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import (
    ArchiveVerificationCheck,
    ArchiveVerificationReport,
    archive_verification_coverage,
    archive_verification_names_for_route,
    archive_verification_owner_adapters,
    passes_strict_acceptance,
    verify_archive,
)
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceArtifact, upsert_raw_artifact
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.pathology_zoo import (
    CLAUDE_VINTAGE_LIVE_PROOF_LOGICAL_SOURCE_KEY,
    CLAUDE_VINTAGE_LIVE_PROOF_ORIGIN,
    CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID,
)
from tests.infra.workload_artifacts import SeededArchiveArtifact

pytest_plugins = ("tests.infra.corpus_fixtures",)

ARCHIVE_VERIFICATION_CHECK_NAMES = archive_verification_names_for_route("live-archive")


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _insert_claude_identity_collision_rows(source_db: Path) -> tuple[str, ...]:
    """Add decoys that independently collide on origin and logical source key."""
    collision_rows = (
        (
            "same-origin-different-logical-key",
            CLAUDE_VINTAGE_LIVE_PROOF_ORIGIN,
            f"{CLAUDE_VINTAGE_LIVE_PROOF_ORIGIN}:collision:{CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID}",
            "foreign-origin/same-origin-different-key.json",
            "claude-ai",
        ),
        (
            "different-origin-same-logical-key",
            "chatgpt-export",
            CLAUDE_VINTAGE_LIVE_PROOF_LOGICAL_SOURCE_KEY,
            "foreign-origin/different-origin-same-key.json",
            "chatgpt",
        ),
    )
    with _connect(source_db) as conn:
        raw = conn.execute(
            """
            SELECT r.native_id, r.blob_hash, r.blob_size, r.acquired_at_ms,
                   m.normalized_content_hash, m.message_count
            FROM raw_sessions AS r
            JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
            WHERE r.origin = ? AND m.logical_source_key = ?
            ORDER BY r.source_path
            LIMIT 1
            """,
            (CLAUDE_VINTAGE_LIVE_PROOF_ORIGIN, CLAUDE_VINTAGE_LIVE_PROOF_LOGICAL_SOURCE_KEY),
        ).fetchone()
        assert raw is not None
        native_id, blob_hash, blob_size, acquired_at_ms, content_hash, message_count = raw
        for raw_id, origin, logical_source_key, source_path, capture_mode in collision_rows:
            conn.execute(
                """
                INSERT INTO raw_sessions(
                    raw_id, origin, native_id, source_path, blob_hash, blob_size,
                    acquired_at_ms, logical_source_key, revision_kind, source_revision,
                    revision_authority, capture_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'full', ?, 'byte_proven', ?)
                """,
                (
                    raw_id,
                    origin,
                    native_id,
                    source_path,
                    blob_hash,
                    blob_size,
                    acquired_at_ms,
                    logical_source_key,
                    f"{raw_id}-revision",
                    capture_mode,
                ),
            )
            conn.execute(
                """
                INSERT INTO raw_session_memberships(
                    raw_id, logical_source_key, provider_session_id, source_revision,
                    normalized_content_hash, message_count, revision_authority,
                    decision, decided_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, 'byte_proven', 'applied', ?)
                """,
                (
                    raw_id,
                    logical_source_key,
                    native_id,
                    f"{raw_id}-revision",
                    content_hash,
                    message_count,
                    acquired_at_ms,
                ),
            )
        conn.commit()
    return tuple(row[0] for row in collision_rows)


def _insert_claude_vintage_extra_revision(source_db: Path) -> str:
    """Add a third in-scope revision whose typed decision is otherwise ignored."""
    extra_raw_id = "claude-vintage-extra-revision"
    with _connect(source_db) as conn:
        raw = conn.execute(
            """
            SELECT r.native_id, r.blob_hash, r.blob_size, r.acquired_at_ms,
                   m.normalized_content_hash, m.message_count
            FROM raw_sessions AS r
            JOIN raw_session_memberships AS m ON m.raw_id = r.raw_id
            WHERE r.origin = ? AND m.logical_source_key = ?
            ORDER BY r.source_path
            LIMIT 1
            """,
            (CLAUDE_VINTAGE_LIVE_PROOF_ORIGIN, CLAUDE_VINTAGE_LIVE_PROOF_LOGICAL_SOURCE_KEY),
        ).fetchone()
        assert raw is not None
        native_id, blob_hash, blob_size, acquired_at_ms, content_hash, message_count = raw
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, logical_source_key, revision_kind, source_revision,
                revision_authority, capture_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'full', ?, 'byte_proven', 'claude-ai')
            """,
            (
                extra_raw_id,
                CLAUDE_VINTAGE_LIVE_PROOF_ORIGIN,
                native_id,
                "manual/claude-live-proof-extra.json",
                blob_hash,
                blob_size,
                acquired_at_ms,
                CLAUDE_VINTAGE_LIVE_PROOF_LOGICAL_SOURCE_KEY,
                "claude-vintage-extra-revision",
            ),
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, revision_authority,
                decision, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, 'byte_proven', 'superseded_prefix', ?)
            """,
            (
                extra_raw_id,
                CLAUDE_VINTAGE_LIVE_PROOF_LOGICAL_SOURCE_KEY,
                native_id,
                "claude-vintage-extra-revision",
                content_hash,
                message_count,
                acquired_at_ms,
            ),
        )
        conn.commit()
    return extra_raw_id


def _seed_coherent_archive(root: Path) -> None:
    """Build a minimal but fully coherent 5-tier archive: one raw, one session."""
    initialize_active_archive_root(root)

    blob_hash = BlobStore(root / "blob").write_from_bytes(b"coherent raw payload")[0]
    source_conn = _connect(root / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-1', 'codex-session', 'session', '/x', ?, 10, 100)
            """,
            (bytes.fromhex(blob_hash),),
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
            (bytes.fromhex(blob_hash),),
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
        index_conn.execute("ANALYZE session_links")
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
        # A check whose population is genuinely absent from the fixture
        # reports SKIP (not-applicable), which is coherent; anything
        # warning-or-worse is not.
        assert check.status in {OutcomeStatus.OK, OutcomeStatus.SKIP}, f"{check.name}: {check.summary}"


def test_raw_failure_lifecycle_accepts_only_typed_deferred_or_terminal_evidence(tmp_path: Path) -> None:
    """A real source-tier failure without its closed outcome blocks reindex."""
    _seed_coherent_archive(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parse_error = 'parser failed' WHERE raw_id = 'raw-1'")
        upsert_raw_artifact(
            conn,
            "raw-1",
            ArchiveSourceArtifact(
                artifact_id="failure-evidence",
                origin=Origin.CODEX_SESSION,
                source_path="/x",
                source_index=0,
                artifact_kind="terminal_corrupt_input",
                classification_reason="terminal_corrupt_input",
                support_status=ArtifactSupportStatus.DECODE_FAILED,
                first_observed_at_ms=100,
                last_observed_at_ms=100,
            ),
        )
        conn.commit()

    typed = verify_archive(tmp_path, checks=("raw-failure-lifecycle",))
    typed_check = _check(typed, "raw-failure-lifecycle")
    assert typed_check.status is OutcomeStatus.OK
    assert not typed.blocking
    assert typed_check.evidence["terminal"] == 1

    with sqlite3.connect(tmp_path / "source.db") as conn:
        upsert_raw_artifact(
            conn,
            "raw-1",
            ArchiveSourceArtifact(
                artifact_id="failure-evidence-deferred",
                origin=Origin.CODEX_SESSION,
                source_path="/x",
                source_index=0,
                artifact_kind="deferred_hot_jsonl_capture",
                classification_reason="deferred_hot_jsonl_capture",
                support_status=ArtifactSupportStatus.PARTIAL_DECODE,
                first_observed_at_ms=200,
                last_observed_at_ms=200,
            ),
        )
        conn.commit()

    deferred = verify_archive(tmp_path, checks=("raw-failure-lifecycle",))
    deferred_check = _check(deferred, "raw-failure-lifecycle")
    assert deferred_check.status is OutcomeStatus.WARNING
    assert deferred_check.evidence["deferred"] == 1

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM raw_artifacts WHERE raw_id = 'raw-1'")
        conn.commit()

    untyped = verify_archive(tmp_path, checks=("raw-failure-lifecycle",))
    untyped_check = _check(untyped, "raw-failure-lifecycle")
    assert untyped_check.status is OutcomeStatus.ERROR
    assert untyped.blocking
    assert untyped_check.evidence["unexplained"] == 1


def test_raw_failure_lifecycle_mutation_red_twin_for_validation_failures(tmp_path: Path) -> None:
    """Validation failures cannot be hidden by a parse-outcome artifact label."""
    _seed_coherent_archive(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET validation_status = 'failed' WHERE raw_id = 'raw-1'")
        upsert_raw_artifact(
            conn,
            "raw-1",
            ArchiveSourceArtifact(
                artifact_id="validation-evidence",
                origin=Origin.CODEX_SESSION,
                source_path="/x",
                source_index=0,
                artifact_kind="terminal_unsupported_shape",
                classification_reason="terminal_unsupported_shape",
                support_status=ArtifactSupportStatus.UNSUPPORTED_PARSEABLE,
                first_observed_at_ms=100,
                last_observed_at_ms=100,
            ),
        )
        conn.commit()

    report = verify_archive(tmp_path, checks=("raw-failure-lifecycle",))
    check = _check(report, "raw-failure-lifecycle")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["validation_failures"] == 1
    assert check.evidence["unexplained"] == 1


def test_cross_tier_reindex_profile_includes_raw_failure_lifecycle(tmp_path: Path) -> None:
    """Candidate acceptance cannot bypass the source failure lifecycle gate."""
    _seed_coherent_archive(tmp_path)
    candidate = tmp_path / "candidate-index.db"
    shutil.copy2(tmp_path / "index.db", candidate)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parse_error = 'parser failed' WHERE raw_id = 'raw-1'")
        conn.commit()

    report = verify_archive(
        tmp_path,
        checks=archive_verification_names_for_route("reindex-cross-tier-candidate"),
        index_path_override=candidate,
    )

    assert _check(report, "raw-failure-lifecycle").status is OutcomeStatus.ERROR


def test_domain_owner_adapters_share_result_seam(tmp_path: Path) -> None:
    """Anti-vacuity: either adapter disappearing would remove a real domain result."""
    _seed_coherent_archive(tmp_path)

    owners = archive_verification_owner_adapters(tmp_path)

    assert [owner.name for owner in owners] == ["fts-parity", "source-index-coverage"]
    results = [owner.check() for owner in owners if owner.check is not None]
    assert [result.name for result in results] == ["fts-parity", "source-index-coverage"]
    assert all(result.status is OutcomeStatus.OK for result in results)


def test_domain_owner_adapters_bind_cross_tier_candidate_need(tmp_path: Path) -> None:
    """The source owner remains callable when the index owner is not applicable."""
    _seed_coherent_archive(tmp_path)
    candidate = tmp_path / "candidate-index.db"
    shutil.copy2(tmp_path / "index.db", candidate)

    owners = archive_verification_owner_adapters(tmp_path, index_path_override=candidate)

    assert owners[0].check is None
    assert owners[1].check is not None
    result = owners[1].check()
    assert result.name == "source-index-coverage"
    assert result.status is OutcomeStatus.OK


def test_domain_declarations_compile_routes_without_pathology_catalogue(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    coverage = archive_verification_coverage(archive_root=tmp_path)
    assert coverage.candidate_id is None
    assert coverage.missing_production_routes == ()
    assert coverage.ownerless_checks == ()
    assert "pathology-zoo-invariants" in coverage.retirement_candidates
    assert {owner.semantic_owner for owner in coverage.declarations}

    canary = archive_verification_names_for_route("reindex-canary-candidate")
    assert canary == ("active-leaf-title-convergence",)
    assert "pathology-zoo-invariants" not in canary


def test_candidate_coverage_reports_runner_and_candidate_identity(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    candidate = tmp_path / "candidate-index.db"
    shutil.copy2(tmp_path / "index.db", candidate)

    coverage = archive_verification_coverage(
        archive_root=tmp_path,
        route="reindex-cross-tier-candidate",
        index_path_override=candidate,
    )

    assert coverage.candidate_id == str(candidate)
    assert coverage.declarations
    assert all(owner.candidate_check is not None for owner in coverage.declarations)


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


def test_source_index_coverage_census_deletion_does_not_hide_raw_head(tmp_path: Path) -> None:
    """RED TWIN (polylogue-r4jiu): census removal cannot shrink I1's universe.

    The raw source of truth keeps an unindexed, byte-proven logical head while
    the derived census ledger first records it and then loses it.  The actual
    ``verify_archive`` domain route must remain red after that deletion;
    the predecessor's census-derived query instead returned green because it
    selected only rows the ledger still described.
    """
    _seed_coherent_archive(tmp_path)
    source_path = tmp_path / "source.db"
    source_conn = _connect(source_path)
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, revision_authority
            )
            VALUES ('raw-census-deleted', 'codex-session', 'never-materialized', '/census-deleted', ?, 10, 100,
                    'byte_proven')
            """,
            (b"d" * 32,),
        )
        source_conn.execute(
            """
            INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, censused_at_ms)
            VALUES ('raw-census-deleted', 'fp', 'complete', 1, 100)
            """
        )
        source_conn.commit()

        assert _check(verify_archive(tmp_path, checks=("source-index-coverage",)), "source-index-coverage").status is (
            OutcomeStatus.ERROR
        )

        source_conn.execute("DELETE FROM raw_membership_census WHERE raw_id = 'raw-census-deleted'")
        source_conn.commit()
        raw = source_conn.execute(
            "SELECT raw_id, parse_error, revision_authority FROM raw_sessions WHERE raw_id = 'raw-census-deleted'"
        ).fetchone()
    finally:
        source_conn.close()

    assert raw == ("raw-census-deleted", None, "byte_proven")
    check = _check(verify_archive(tmp_path, checks=("source-index-coverage",)), "source-index-coverage")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["untyped_count"] == 1
    assert check.evidence["untyped_sample"] == ["raw-census-deleted"]

    owner = next(
        owner
        for owner in archive_verification_coverage(archive_root=tmp_path).declarations
        if owner.name == "source-index-coverage"
    )
    assert owner.semantic_owner == "source-materialization"
    assert owner.owned_reference == "test_source_index_coverage_census_deletion_does_not_hide_raw_head"


def test_source_index_coverage_groups_reacquisitions_by_logical_source_key(tmp_path: Path) -> None:
    """Different capture paths for one typed source are revisions, not gaps."""

    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            "UPDATE raw_sessions SET logical_source_key = 'codex-session:session' WHERE raw_id = 'raw-1'"
        )
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, logical_source_key, revision_authority
            )
            SELECT 'raw-reacquired', origin, NULL, '/new-capture-path',
                   blob_hash, blob_size, 200, 'codex:session', 'byte_proven'
            FROM raw_sessions WHERE raw_id = 'raw-1'
            """
        )
        source_conn.commit()
    finally:
        source_conn.close()

    check = _check(verify_archive(tmp_path, checks=("source-index-coverage",)), "source-index-coverage")

    assert check.status is OutcomeStatus.OK
    assert check.evidence["logical_head_count"] == 1
    assert check.evidence["unindexed_head_count"] == 0


def test_coverage_groups_retired_membership_identity_with_bound_revision(tmp_path: Path) -> None:
    """A retired raw keeps its sole membership identity after its raw key is nulled."""
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            "UPDATE raw_sessions SET logical_source_key = 'codex-session:session' WHERE raw_id = 'raw-1'"
        )
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, revision_authority
            )
            SELECT 'raw-retired-membership', origin, 'session', '/retired-membership',
                   blob_hash, blob_size, 200, 'quarantined'
            FROM raw_sessions WHERE raw_id = 'raw-1'
            """
        )
        source_conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count
            ) VALUES ('raw-retired-membership', 'codex:session', 'session', 'retired', ?, 1)
            """,
            (b"r" * 32,),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    report = verify_archive(tmp_path, checks=("source-index-coverage", "convergence-freshness"))

    coverage = _check(report, "source-index-coverage")
    freshness = _check(report, "convergence-freshness")
    assert coverage.status is OutcomeStatus.OK
    assert coverage.evidence["logical_head_count"] == 1
    assert freshness.status is OutcomeStatus.OK
    assert freshness.evidence["unindexed_backlog_gap"] == 0


def test_coverage_does_not_assign_a_shared_raw_to_one_membership_key(tmp_path: Path) -> None:
    """A raw with several retained identities must not inherit an arbitrary cohort."""
    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute("UPDATE raw_sessions SET logical_source_key = 'codex:session' WHERE raw_id = 'raw-1'")
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, revision_authority
            )
            SELECT 'raw-shared-membership', origin, NULL, '/shared-membership',
                   blob_hash, blob_size, 200, 'quarantined'
            FROM raw_sessions WHERE raw_id = 'raw-1'
            """
        )
        source_conn.executemany(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count
            ) VALUES ('raw-shared-membership', ?, ?, 'shared', ?, 1)
            """,
            [
                ("codex:session", "session", b"s" * 32),
                ("codex:other", "other", b"o" * 32),
            ],
        )
        source_conn.commit()
    finally:
        source_conn.close()

    check = _check(verify_archive(tmp_path, checks=("source-index-coverage",)), "source-index-coverage")

    assert check.status is OutcomeStatus.WARNING
    assert check.evidence["logical_head_count"] == 2
    assert check.evidence["quarantined_count"] == 1


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


def test_valid_byte_supersession_receipt_covers_unindexed_head(tmp_path: Path) -> None:
    """A receipt is authority only when its bytes and indexed twin revalidate."""

    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, revision_authority
            )
            SELECT 'raw-superseded', origin, 'duplicate', source_path, source_index,
                   blob_hash, blob_size, 200, 'byte_proven'
            FROM raw_sessions WHERE raw_id = 'raw-1'
            """
        )
        source_conn.execute(
            """
            INSERT INTO raw_byte_duplicate_supersession_receipts(
                raw_id, blob_hash, blob_size, duplicate_of_raw_id,
                duplicate_of_session_id, previous_revision_authority,
                promoted_at_ms, tool_version, backup_manifest_path, detail
            )
            SELECT 'raw-superseded', blob_hash, blob_size, 'raw-1',
                   'codex-session:session', 'quarantined', 200,
                   'test', '/verified/manifest.json', ''
            FROM raw_sessions WHERE raw_id = 'raw-superseded'
            """
        )
        source_conn.commit()
    finally:
        source_conn.close()

    check = _check(verify_archive(tmp_path, checks=("source-index-coverage",)), "source-index-coverage")

    assert check.status is OutcomeStatus.OK
    assert check.evidence["superseded_byte_duplicate_count"] == 1
    assert check.evidence["untyped_count"] == 0


def test_byte_supersession_receipt_requires_matching_source_semantics(tmp_path: Path) -> None:
    """Byte equality cannot collapse a raw whose path-specific replay semantics differ."""

    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, revision_authority
            )
            SELECT 'raw-cross-path-supersession', origin, 'duplicate', '/different-path', source_index,
                   blob_hash, blob_size, 200, 'byte_proven'
            FROM raw_sessions WHERE raw_id = 'raw-1'
            """
        )
        conn.execute(
            """
            INSERT INTO raw_byte_duplicate_supersession_receipts(
                raw_id, blob_hash, blob_size, duplicate_of_raw_id, duplicate_of_session_id,
                previous_revision_authority, promoted_at_ms, tool_version, backup_manifest_path, detail
            )
            SELECT 'raw-cross-path-supersession', blob_hash, blob_size, 'raw-1',
                   'codex-session:session', 'quarantined', 200, 'test', '/verified/manifest.json', ''
            FROM raw_sessions WHERE raw_id = 'raw-1'
            """
        )

    check = _check(verify_archive(tmp_path, checks=("source-index-coverage",)), "source-index-coverage")

    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["superseded_byte_duplicate_count"] == 0
    assert check.evidence["untyped_count"] == 1


def test_invalid_byte_supersession_receipt_does_not_cover_unindexed_head(tmp_path: Path) -> None:
    """A stale/mismatched receipt cannot authorize its own coverage result."""

    _seed_coherent_archive(tmp_path)
    source_conn = _connect(tmp_path / "source.db")
    try:
        source_conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, revision_authority
            ) VALUES (
                'raw-bad-receipt', 'codex-session', 'bad-receipt', '/bad-receipt',
                ?, 10, 200, 'byte_proven'
            )
            """,
            (b"z" * 32,),
        )
        source_conn.execute(
            """
            INSERT INTO raw_byte_duplicate_supersession_receipts(
                raw_id, blob_hash, blob_size, duplicate_of_raw_id,
                duplicate_of_session_id, previous_revision_authority,
                promoted_at_ms, tool_version, backup_manifest_path, detail
            ) VALUES (
                'raw-bad-receipt', ?, 10, 'raw-1', 'codex-session:session',
                'quarantined', 200, 'test', '/verified/manifest.json', ''
            )
            """,
            (b"z" * 32,),
        )
        source_conn.commit()
    finally:
        source_conn.close()

    check = _check(verify_archive(tmp_path, checks=("source-index-coverage",)), "source-index-coverage")

    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["superseded_byte_duplicate_count"] == 0
    assert check.evidence["untyped_count"] == 1


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


def test_contradiction_without_authoritative_winner_trips_the_check(tmp_path: Path) -> None:
    """A contradicted edge with nothing overruling it is a defect, not a resolution.

    The presence of contradictions is normal and reported as observability --
    it is what the write path exists to record. What must never happen is an
    edge marked "overruled by hook evidence" with no authoritative sibling for
    the same child, because that means inference was demoted by nothing.
    """
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO session_links(
                src_session_id, dst_origin, dst_native_id, link_type,
                status, method, observed_at_ms
            ) VALUES ('codex-session:session', 'codex-session', 'inferred-parent', 'subagent',
                      'authority-contradicted', ?, 100)
            """,
            (HOOK_CONTRADICTED_LINK_METHOD,),
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("hook-authority-topology-conflict",))
    check = _check(report, "hook-authority-topology-conflict")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unresolved_contradiction_count"] == 1
    assert check.evidence["unresolved_contradiction_sample"] == ["codex-session:session"]


def test_two_authoritative_parents_for_one_child_trips_the_check(tmp_path: Path) -> None:
    """ "Exactly one authoritative parent", not "some winner exists".

    A revised hook claim lands at a DIFFERENT primary key, so a child can
    accumulate two authoritative edges; composition would then choose between
    them by arrival order while a some-winner-exists check reported OK. The
    write path now supersedes the older edge, and this is the gate that
    catches any archive where that did not happen.
    """
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executemany(
            """
            INSERT INTO session_links(
                src_session_id, dst_origin, dst_native_id, link_type,
                status, method, observed_at_ms
            ) VALUES ('codex-session:session', 'codex-session', ?, 'subagent', NULL, ?, 100)
            """,
            [("parent-a", HOOK_AUTHORITATIVE_LINK_METHOD), ("parent-b", HOOK_AUTHORITATIVE_LINK_METHOD)],
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("hook-authority-topology-conflict",))
    check = _check(report, "hook-authority-topology-conflict")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["multi_authoritative_count"] == 1
    assert check.evidence["multi_authoritative_sample"] == ["codex-session:session:subagent=2"]


def test_resolved_contradiction_is_reported_without_erroring(tmp_path: Path) -> None:
    """The resolved shape -- loser plus authoritative winner -- is OK."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executemany(
            """
            INSERT INTO session_links(
                src_session_id, dst_origin, dst_native_id, link_type,
                status, method, observed_at_ms
            ) VALUES ('codex-session:session', 'codex-session', ?, 'subagent', ?, ?, 100)
            """,
            [
                ("inferred-parent", "authority-contradicted", HOOK_CONTRADICTED_LINK_METHOD),
                ("hook-parent", None, HOOK_AUTHORITATIVE_LINK_METHOD),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("hook-authority-topology-conflict",))
    check = _check(report, "hook-authority-topology-conflict")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["contradicted_count"] == 1
    assert check.evidence["authoritative_count"] == 1
    assert check.evidence["unresolved_contradiction_count"] == 0


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
        conn.execute("DELETE FROM sqlite_stat1 WHERE tbl = 'session_links'")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=("planner-stats",))

    check = _check(report, "planner-stats")
    assert check.status is OutcomeStatus.WARNING
    assert check.evidence["missing_tables"] == ["session_links"]


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


def test_one_check_raising_does_not_abort_the_others(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_coherent_archive(tmp_path)
    from polylogue.maintenance import archive_verification as module

    def _boom(_archive_root: Path, _sample_limit: int) -> ArchiveVerificationCheck:
        raise RuntimeError("synthetic failure")

    owners = module.archive_verification_domain_adapters(tmp_path)
    broken_owners = tuple(
        replace(owner, check=lambda: _boom(tmp_path, 10)) if owner.name == "fts-parity" else owner for owner in owners
    )
    monkeypatch.setattr(module, "archive_verification_domain_adapters", lambda *args, **kwargs: broken_owners)

    report = module.verify_archive(tmp_path)

    by_name = {check.name: check for check in report.checks}
    assert by_name["fts-parity"].status is OutcomeStatus.ERROR
    assert "synthetic failure" in by_name["fts-parity"].summary
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


@pytest.mark.parametrize("mutation", ("missing", "corrupt", "orphan"))
def test_full_blob_integrity_red_twin(tmp_path: Path, mutation: str) -> None:
    """Full archive verification rejects every physical blob failure class."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "source.db") as conn:
        raw_hash = bytes(conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = 'raw-1'").fetchone()[0]).hex()
    store = BlobStore(tmp_path / "blob")
    if mutation == "missing":
        store.blob_path(raw_hash).unlink()
    elif mutation == "corrupt":
        store.blob_path(raw_hash).write_bytes(b"corrupt bytes")
    else:
        store.write_from_bytes(b"orphan bytes")

    report = verify_archive(tmp_path, checks=("blob-integrity",))

    check = _check(report, "blob-integrity")
    assert check.status is OutcomeStatus.ERROR
    assert report.blocking is True
    finding_kinds = set(cast(dict[str, object], check.evidence["finding_counts"]))
    expected_kind = {
        "missing": "missing_referenced_blobs",
        "corrupt": "hash_mismatch",
        "orphan": "orphan_blobs",
    }[mutation]
    assert expected_kind in finding_kinds
    assert cast(dict[str, object], check.evidence["scan"])["full_scan"] is True


def test_acquired_unreachable_attachment_debt_is_blocking(tmp_path: Path) -> None:
    """Acquired bytes without an attachment_refs edge are not queryable."""
    _seed_coherent_archive(tmp_path)
    blob_hash, size = BlobStore(tmp_path / "blob").write_from_bytes(b"unreachable attachment")
    with _connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO attachments(attachment_id, blob_hash, byte_count, acquisition_status, ref_count)
            VALUES ('unreachable-attachment', ?, ?, 'acquired', 0)
            """,
            (bytes.fromhex(blob_hash), size),
        )
        conn.commit()

    report = verify_archive(tmp_path, checks=("attachment-coverage",))

    check = _check(report, "attachment-coverage")
    assert check.status is OutcomeStatus.ERROR
    assert report.blocking
    assert check.evidence["unreachable_count"] == 1
    assert "unreachable-attachment" in check.details[0]


def test_orphaned_embedding_ref_trips_embeddings_refs_liveness(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "embeddings.db")
    try:
        conn.execute(
            """
            INSERT INTO message_embedding_refs(message_id, session_id, origin, vector_derivation_hash)
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

    report = verify_archive(tmp_path, checks=archive_verification_names_for_route("reindex-index-candidate"))

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

    report = verify_archive(tmp_path, checks=archive_verification_names_for_route("reindex-index-candidate"))

    check = _check(report, "session-fingerprint-stamps")
    assert report.blocking
    assert check.status is OutcomeStatus.ERROR
    assert "missing fingerprint column(s): parser_fingerprint" in check.summary


def test_reindex_acceptance_rejects_stale_semantic_stamps(tmp_path: Path) -> None:
    """A well-formed historic stamp must fail the current-source comparison."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET parser_fingerprint = ?", ("0" * 64,))

    report = verify_archive(tmp_path, checks=archive_verification_names_for_route("reindex-index-candidate"))

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

    report = verify_archive(tmp_path, checks=archive_verification_names_for_route("reindex-index-candidate"))

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
        indexed_blob_hash = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = 'raw-1'").fetchone()[0]
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-dup', 'codex-session', 'session-dup', '/y', ?, 10, 200)
            """,
            (indexed_blob_hash,),  # same blob_hash as raw-1, which IS indexed; defaults to quarantined
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


def test_convergence_freshness_excludes_a_receipt_backed_duplicate(tmp_path: Path) -> None:
    """The production convergence-freshness route excludes a byte-identical
    unindexed duplicate only when its supersession receipt names an indexed
    twin. Removing the receipt must turn this route back into a backlog."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, revision_authority
            )
            SELECT 'raw-superseded-backlog', origin, 'duplicate-backlog', source_path, source_index,
                   blob_hash, blob_size, 200, 'byte_proven'
            FROM raw_sessions WHERE raw_id = 'raw-1'
            """
        )
        conn.execute(
            """
            INSERT INTO raw_byte_duplicate_supersession_receipts(
                raw_id, blob_hash, blob_size, duplicate_of_raw_id,
                duplicate_of_session_id, previous_revision_authority,
                promoted_at_ms, tool_version, backup_manifest_path, detail
            )
            SELECT 'raw-superseded-backlog', blob_hash, blob_size, 'raw-1',
                   'codex-session:session', 'quarantined', 200,
                   'test', '/verified/manifest.json', ''
            FROM raw_sessions WHERE raw_id = 'raw-superseded-backlog'
            """
        )
        conn.commit()
    finally:
        conn.close()

    check = _check(verify_archive(tmp_path, checks=("convergence-freshness",)), "convergence-freshness")

    assert check.status is OutcomeStatus.OK
    assert check.evidence["unindexed_backlog_gap"] == 0

    with _connect(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM raw_byte_duplicate_supersession_receipts WHERE raw_id = 'raw-superseded-backlog'")
        conn.commit()

    check = _check(verify_archive(tmp_path, checks=("convergence-freshness",)), "convergence-freshness")

    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unindexed_backlog_gap"] == 1


def test_convergence_freshness_counts_a_receipt_with_the_wrong_twin_bytes(tmp_path: Path) -> None:
    """The production convergence-freshness route keeps a duplicate in the
    backlog when mutating the receipt's indexed-twin bytes invalidates its
    supersession evidence."""
    _seed_coherent_archive(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, revision_authority
            ) VALUES (
                'raw-invalid-supersession-backlog', 'codex-session', 'invalid-duplicate',
                '/invalid-duplicate', ?, 10, 200, 'byte_proven'
            )
            """,
            (b"z" * 32,),
        )
        conn.execute(
            """
            INSERT INTO raw_byte_duplicate_supersession_receipts(
                raw_id, blob_hash, blob_size, duplicate_of_raw_id,
                duplicate_of_session_id, previous_revision_authority,
                promoted_at_ms, tool_version, backup_manifest_path, detail
            ) VALUES (
                'raw-invalid-supersession-backlog', ?, 10, 'raw-1',
                'codex-session:session', 'quarantined', 200,
                'test', '/verified/manifest.json', ''
            )
            """,
            (b"z" * 32,),
        )
        conn.commit()
    finally:
        conn.close()

    check = _check(verify_archive(tmp_path, checks=("convergence-freshness",)), "convergence-freshness")

    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unindexed_backlog_gap"] == 1


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


# ---------------------------------------------------------------------------
# active-leaf / title convergence (polylogue-2hwl)
# ---------------------------------------------------------------------------


def _insert_leaf_message(conn: sqlite3.Connection, native_id: str, position: int, *, active_leaf: int) -> None:
    conn.execute(
        """
        INSERT INTO messages(session_id, native_id, position, role, material_origin, content_hash, is_active_leaf)
        VALUES ('codex-session:session', ?, ?, 'assistant', 'assistant_authored', ?, ?)
        """,
        (native_id, position, bytes([position + 1]) * 32, active_leaf),
    )


def test_active_leaf_title_convergence_passes_on_coherent_archive(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)

    report = verify_archive(tmp_path, checks=("active-leaf-title-convergence",))

    check = _check(report, "active-leaf-title-convergence")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["multi_active_leaf_session_count"] == 0
    assert check.evidence["unresolvable_active_leaf_pointer_count"] == 0
    assert check.evidence["origin_titled_placeholder_count"] == 0


def test_second_active_leaf_trips_active_leaf_title_convergence(tmp_path: Path) -> None:
    """polylogue-2hwl defect (2): duplicate provider ids across merged chunks
    flagged every matching position, so one session carried several active
    leaves. 103 live sessions were in this state when the detector was first
    run (2026-08-03)."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        _insert_leaf_message(conn, "retry-a", 1, active_leaf=1)
        _insert_leaf_message(conn, "retry-b", 2, active_leaf=1)

    report = verify_archive(tmp_path, checks=("active-leaf-title-convergence",))

    check = _check(report, "active-leaf-title-convergence")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["multi_active_leaf_session_count"] == 1
    assert check.evidence["worst_active_leaf_count"] == 2
    assert check.evidence["multi_active_leaf_sample"][0]["session_id"] == "codex-session:session"


def test_single_active_leaf_does_not_trip_active_leaf_title_convergence(tmp_path: Path) -> None:
    """Anti-vacuity for the red twin above: one flagged leaf is the normal,
    converged state and must stay green, so the check is not merely counting
    ``is_active_leaf`` rows."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        _insert_leaf_message(conn, "only-leaf", 1, active_leaf=1)
        (leaf_message_id,) = conn.execute("SELECT message_id FROM messages WHERE native_id = 'only-leaf'").fetchone()
        conn.execute(
            "UPDATE sessions SET active_leaf_message_id = ? WHERE session_id = 'codex-session:session'",
            (leaf_message_id,),
        )

    report = verify_archive(tmp_path, checks=("active-leaf-title-convergence",))

    assert _check(report, "active-leaf-title-convergence").status is OutcomeStatus.OK


def test_unresolvable_active_leaf_pointer_trips_active_leaf_title_convergence(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute(
            "UPDATE sessions SET active_leaf_message_id = ? WHERE session_id = 'codex-session:session'",
            ("codex-session:session:does-not-exist",),
        )

    report = verify_archive(tmp_path, checks=("active-leaf-title-convergence",))

    check = _check(report, "active-leaf-title-convergence")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unresolvable_active_leaf_pointer_count"] == 1


def test_origin_titled_placeholder_trips_active_leaf_title_convergence(tmp_path: Path) -> None:
    """polylogue-2hwl defect (1): the merge kept a placeholder/None title even
    once a real one arrived. A bare native id stored *as* an ORIGIN-provenance
    title is that bug's stored residue."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET title = native_id, title_source = 'origin'")

    report = verify_archive(tmp_path, checks=("active-leaf-title-convergence",))

    check = _check(report, "active-leaf-title-convergence")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["origin_titled_placeholder_count"] == 1
    assert check.evidence["origin_titled_placeholder_sample"][0]["title"] == "session"


def test_untitled_session_without_origin_provenance_stays_green(tmp_path: Path) -> None:
    """The bare-id title fallback is legitimate when it does not claim ORIGIN
    provenance -- ``sources/parsers/chatgpt.py`` stores exactly that shape on
    purpose. Only the provenance lie is the defect."""
    _seed_coherent_archive(tmp_path)
    with _connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET title = native_id, title_source = NULL")

    report = verify_archive(tmp_path, checks=("active-leaf-title-convergence",))

    assert _check(report, "active-leaf-title-convergence").status is OutcomeStatus.OK


# ---------------------------------------------------------------------------
# ChatGPT parse-boundary content conservation (polylogue-xofj)
# ---------------------------------------------------------------------------


def _chatgpt_node(node_id: str, message_id: str, content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node_id,
        "message": {"id": message_id, "author": {"role": "assistant"}, "content": content},
    }


def _seed_chatgpt_conversation(
    root: Path,
    *,
    nodes: dict[str, Any],
    materialized_native_ids: tuple[str, ...],
    conversation_id: str = "conv-1",
    raw_id: str = "raw-chatgpt",
    acquired_at_ms: int = 1_000,
) -> None:
    """Add one acquired chatgpt-export conversation plus its indexed session."""
    payload = {"id": conversation_id, "title": "Conversation", "mapping": nodes}
    blob_hash, size = BlobStore(root / "blob").write_from_bytes(json.dumps(payload).encode())
    with _connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms)
            VALUES (?, 'chatgpt-export', ?, '/chatgpt.json', ?, ?, ?)
            """,
            (raw_id, conversation_id, bytes.fromhex(blob_hash), size, acquired_at_ms),
        )
    with _connect(root / "index.db") as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions(
                native_id, origin, raw_id, parser_fingerprint, lowering_fingerprint, content_hash, message_count
            ) VALUES (?, 'chatgpt-export', ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                raw_id,
                parser_fingerprint_for_origin("chatgpt-export"),
                lowering_fingerprint(),
                b"c" * 32,
                len(materialized_native_ids),
            ),
        )
        for position, native_id in enumerate(materialized_native_ids):
            conn.execute(
                """
                INSERT INTO messages(session_id, native_id, position, role, material_origin, content_hash)
                VALUES (?, ?, ?, 'assistant', 'assistant_authored', ?)
                """,
                (
                    f"chatgpt-export:{conversation_id}",
                    native_id,
                    position,
                    bytes([position + 1]) * 32,
                ),
            )


_CONSERVED_NODES: dict[str, Any] = {
    "n1": _chatgpt_node("n1", "m1", {"content_type": "text", "parts": ["hello"]}),
    "n2": _chatgpt_node("n2", "m2", {"content_type": "sonic_webpage", "result": "browsed page body"}),
}


def test_chatgpt_content_conservation_passes_when_every_node_materializes(tmp_path: Path) -> None:
    _seed_coherent_archive(tmp_path)
    _seed_chatgpt_conversation(tmp_path, nodes=_CONSERVED_NODES, materialized_native_ids=("m1", "m2"))

    report = verify_archive(tmp_path, checks=("chatgpt-content-conservation",))

    check = _check(report, "chatgpt-content-conservation")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["content_units_conserved"] == 2
    assert check.evidence["conserved_by_content_type"] == {"sonic_webpage": 1, "text": 1}


def test_dropped_raw_node_trips_chatgpt_content_conservation(tmp_path: Path) -> None:
    """polylogue-xofj's class: a content_type the parser has no branch for hits
    ``extract_messages_from_mapping``'s ``if not text and not content_blocks:
    continue`` and leaves no row, no event, and no typed refusal. Only
    re-reading the acquired bytes can see it, which is why this check's
    universe is the blob rather than any indexed relation."""
    _seed_coherent_archive(tmp_path)
    _seed_chatgpt_conversation(tmp_path, nodes=_CONSERVED_NODES, materialized_native_ids=("m1",))

    report = verify_archive(tmp_path, checks=("chatgpt-content-conservation",))

    check = _check(report, "chatgpt-content-conservation")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["content_units_dropped"] == 1
    assert check.evidence["dropped_by_content_type"] == {"sonic_webpage": 1}
    assert check.evidence["dropped_sample"][0]["provider_message_id"] == "m2"
    assert check.evidence["documents_with_dropped_content"] == 1


def test_chatgpt_conservation_ignores_nodes_with_no_content_payload(tmp_path: Path) -> None:
    """A node whose ``content`` carries only descriptors conserves nothing, so
    the parser dropping it is not a finding. Without this the census would be
    red on every export's empty code stubs."""
    _seed_coherent_archive(tmp_path)
    nodes = {
        "n1": _chatgpt_node("n1", "m1", {"content_type": "text", "parts": ["hello"]}),
        "n2": _chatgpt_node("n2", "m2", {"content_type": "code", "language": "python", "text": ""}),
    }
    _seed_chatgpt_conversation(tmp_path, nodes=nodes, materialized_native_ids=("m1",))

    report = verify_archive(tmp_path, checks=("chatgpt-content-conservation",))

    check = _check(report, "chatgpt-content-conservation")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["content_units_conserved"] == 1


def test_chatgpt_conservation_measures_only_the_newest_acquired_revision(tmp_path: Path) -> None:
    """A branch the user deleted in ChatGPT is legitimately absent from the
    current index; measuring superseded revisions would report supersession as
    a parser drop."""
    _seed_coherent_archive(tmp_path)
    _seed_chatgpt_conversation(
        tmp_path,
        nodes=_CONSERVED_NODES,
        materialized_native_ids=("m1",),
        raw_id="raw-chatgpt-old",
        acquired_at_ms=1_000,
    )
    _seed_chatgpt_conversation(
        tmp_path,
        nodes={"n1": _chatgpt_node("n1", "m1", {"content_type": "text", "parts": ["hello"]})},
        materialized_native_ids=(),
        raw_id="raw-chatgpt-new",
        acquired_at_ms=2_000,
    )

    report = verify_archive(tmp_path, checks=("chatgpt-content-conservation",))

    check = _check(report, "chatgpt-content-conservation")
    assert check.status is OutcomeStatus.OK
    assert check.evidence["content_units_conserved"] == 1
    assert check.evidence["content_units_dropped"] == 0


def test_chatgpt_conservation_excludes_documents_absent_from_the_index(tmp_path: Path) -> None:
    """A whole conversation missing from the index is ``corpus-absences``'
    finding; attributing its every node to the parser here would double-count
    one defect as two."""
    _seed_coherent_archive(tmp_path)
    _seed_chatgpt_conversation(tmp_path, nodes=_CONSERVED_NODES, materialized_native_ids=("m1", "m2"))
    with _connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM messages WHERE session_id = 'chatgpt-export:conv-1'")
        conn.execute("DELETE FROM sessions WHERE session_id = 'chatgpt-export:conv-1'")

    report = verify_archive(tmp_path, checks=("chatgpt-content-conservation",))

    check = _check(report, "chatgpt-content-conservation")
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["documents_absent_from_index"] == 1
    assert check.evidence["documents_measured"] == 0
    assert check.evidence["outcome_reason"] == "zero_candidate_overlap"


@pytest.mark.parametrize("check_name", ARCHIVE_VERIFICATION_CHECK_NAMES)
def test_every_live_declaration_does_not_error_on_the_real_pipeline_corpus(
    check_name: str, seeded_archive: SeededArchiveArtifact
) -> None:
    """Every live declaration, run individually against a real-pipeline
    (acquire->parse->materialize->index), multi-provider synthetic corpus,
    must not report ``error``. This is the anti-vacuity backstop for the
    live declaration set: a check that only ever runs against a single
    hand-inserted row (the other tests in this file) could pass by never
    actually exercising realistic multi-session, multi-provider shape.
    ``verify_archive`` is read-only, so the session-scoped fixture root is
    read directly with no per-test clone.
    """
    report = verify_archive(seeded_archive.root, checks=(check_name,))

    check = _check(report, check_name)
    assert check.status is not OutcomeStatus.ERROR, f"{check_name}: {check.summary}\n{check.evidence}"


def test_new_convergence_checks_measure_a_real_population_on_the_pipeline_corpus(
    seeded_archive: SeededArchiveArtifact,
) -> None:
    """Anti-vacuity for the two polylogue-t0m73 graduations: the parametrized
    corpus test above only asserts "not error", which a check that measured
    nothing would also satisfy. These two are the checks whose universes are
    origin-scoped (chatgpt raws) or population-scoped (sessions carrying an
    active leaf), so their green result is only meaningful if the population
    was non-empty."""
    report = verify_archive(
        seeded_archive.root,
        checks=("active-leaf-title-convergence", "chatgpt-content-conservation"),
    )

    active_leaf = _check(report, "active-leaf-title-convergence")
    assert active_leaf.status is OutcomeStatus.OK
    assert active_leaf.evidence["sessions_measured"] > 0
    assert active_leaf.evidence["sessions_with_active_leaf"] > 0

    conservation = _check(report, "chatgpt-content-conservation")
    assert conservation.status is OutcomeStatus.OK
    assert conservation.evidence["documents_measured"] > 0
    assert conservation.evidence["content_units_conserved"] > 0
    assert conservation.evidence["raws_scanned"] > 0


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
        assert isinstance(entry["evidence"], dict)
    assert names == set(ARCHIVE_VERIFICATION_CHECK_NAMES)
    coverage_payload = cast(dict[str, object], payload["coverage"])
    assert coverage_payload["denominator"] == len(ARCHIVE_VERIFICATION_CHECK_NAMES)


def test_strict_acceptance_requires_every_named_check() -> None:
    report = ArchiveVerificationReport(checks=[ArchiveVerificationCheck(name="present", status=OutcomeStatus.OK)])

    assert not passes_strict_acceptance(report, required_checks=("present", "missing"))


def test_reindex_acceptance_checks_are_declared_and_index_eligible() -> None:
    """Every index-candidate name must be a declared
    declared check, and running it against an index-only root (mirroring a
    real generation directory, which has no source.db/user.db/embeddings.db)
    must never report ``error`` from a missing-tier false positive."""
    assert set(archive_verification_names_for_route("reindex-index-candidate")) <= set(ARCHIVE_VERIFICATION_CHECK_NAMES)


def test_full_rebuild_candidate_profile_covers_cross_tier_acceptance_and_canary_stays_partial() -> None:
    expected = {
        "source-index-coverage",
        "fts-parity",
        "lineage-sanity",
        "session-lineage-acyclic",
        "blob-refs-liveness",
        "blob-integrity",
        "attachment-coverage",
        "raw-failure-lifecycle",
        "embeddings-refs-liveness",
        "user-tier-refs",
        "session-fingerprint-stamps",
        "message-count-projection",
        "excluded-cursor-vocabulary-honesty",
        "stalled-append-cursor-freshness",
        "corpus-absences",
        "corpus-attachment-fidelity",
        "corpus-revision-fidelity",
    }

    assert expected <= set(archive_verification_names_for_route("reindex-index-candidate")) | set(
        archive_verification_names_for_route("reindex-cross-tier-candidate")
    )
    assert archive_verification_names_for_route("reindex-canary-candidate") == ("active-leaf-title-convergence",)
    assert set(archive_verification_names_for_route("reindex-source-preflight")) <= set(
        ARCHIVE_VERIFICATION_CHECK_NAMES
    )


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
        conn.execute("ANALYZE session_links")
        conn.commit()
    finally:
        conn.close()

    report = verify_archive(tmp_path, checks=list(archive_verification_names_for_route("reindex-index-candidate")))

    assert not report.blocking, [check.summary for check in report.checks if check.status is OutcomeStatus.ERROR]
    for check in report.checks:
        assert check.status is not OutcomeStatus.ERROR, f"{check.name}: {check.summary}"


# ---------------------------------------------------------------------------
# RED-TWIN contract rule (polylogue-t0m73): structural enforcement
# ---------------------------------------------------------------------------

#: Domain declarations, not this test module, own red-twin identity. This view keeps
