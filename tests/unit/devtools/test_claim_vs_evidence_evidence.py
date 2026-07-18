"""Tests for representing claim-vs-evidence report runs as archive evidence."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from devtools.claim_vs_evidence_evidence import (
    build_findings,
    build_query_definition,
    build_result_set_members,
    materialize_claim_vs_evidence_evidence,
)
from polylogue.core.enums import AssertionKind
from polylogue.core.query_identity import query_hash_for_plan
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import list_assertion_claims
from polylogue.storage.sqlite.finding_provenance import list_public_finding_inputs
from polylogue.storage.sqlite.query_objects import get_query, get_result_set


def _report(
    *,
    archive_root: Path,
    silent: int,
    acknowledged: int,
    ambiguous: int,
    n_min: int = 30,
    member_refs: tuple[str, ...] = ("message:s1:tool-1-result", "message:s1:tool-2-result"),
    captured_at: str = "2026-07-18T00:00:00+00:00",
) -> dict[str, Any]:
    classified = silent + acknowledged
    failed = classified + ambiguous
    publishable = failed >= n_min and classified >= n_min
    return {
        "captured_at": captured_at,
        "archive_root": str(archive_root),
        "index_db": str(archive_root / "index.db"),
        "limit": 5000,
        "sample_frame": {
            "inspected_structured_failures": failed,
            "failure_predicate": "tool_result_is_error = 1 OR tool_result_exit_code != 0",
            "classification_scope": "immediately following assistant message only",
            "sensitivity_scope": "next 3 assistant messages",
            "selection_strategy": "origin-stratified bounded sample",
            "n_min": n_min,
        },
        "totals": {
            "failed_outcomes": failed,
            "acknowledged": acknowledged,
            "silent_proceed": silent,
            "ambiguous": ambiguous,
            "classified_outcomes": classified,
        },
        "rates": {
            "publication_status": "supported" if publishable else "not_supported",
            "silent_rate_lower_bound": (silent / failed) if publishable else None,
        },
        "handler_class_definition": {
            "benign_recovery": ["glob", "grep"],
            "consequential": ["bash", "edit"],
            "other": "any other tool",
        },
        "evidence": {"member_refs": sorted(member_refs)},
    }


def test_build_query_definition_is_content_addressed(tmp_path: Path) -> None:
    report_a = _report(archive_root=tmp_path, silent=1, acknowledged=1, ambiguous=1, n_min=30)
    report_b = _report(archive_root=tmp_path, silent=1, acknowledged=1, ambiguous=1, n_min=30)
    report_c = _report(archive_root=tmp_path, silent=1, acknowledged=1, ambiguous=1, n_min=50)

    definition_a = build_query_definition(report_a)
    definition_b = build_query_definition(report_b)
    definition_c = build_query_definition(report_c)

    hash_a = query_hash_for_plan(definition_a, grain="g", lane="l", rank_policy="r")
    hash_b = query_hash_for_plan(definition_b, grain="g", lane="l", rank_policy="r")
    hash_c = query_hash_for_plan(definition_c, grain="g", lane="l", rank_policy="r")

    assert hash_a == hash_b
    assert hash_a != hash_c


def test_build_result_set_members_returns_sorted_refs(tmp_path: Path) -> None:
    report = _report(
        archive_root=tmp_path,
        silent=1,
        acknowledged=1,
        ambiguous=1,
        member_refs=("message:s1:z", "message:s1:a"),
    )
    assert build_result_set_members(report) == ("message:s1:a", "message:s1:z")


def test_build_findings_omits_public_claim_when_not_publishable(tmp_path: Path) -> None:
    report = _report(archive_root=tmp_path, silent=2, acknowledged=2, ambiguous=16, n_min=30)
    from polylogue.storage.sqlite.query_objects import EvaluationReceipt

    receipt = EvaluationReceipt(
        receipt_id="receipt-test",
        source_generation="source:absent",
        user_generation="user:absent",
        index_generation="index:absent",
        runtime_build_ref="polylogue:test",
    )
    findings = build_findings(
        report,
        query_reference="query:" + "0" * 64,
        result_set_reference="result-set:test",
        receipt=receipt,
    )
    assert len(findings) == 1
    assert findings[0].public_claim is None
    assert "acknowledged" in findings[0].body_text


def test_build_findings_includes_public_claim_when_publishable(tmp_path: Path) -> None:
    report = _report(archive_root=tmp_path, silent=20, acknowledged=15, ambiguous=5, n_min=30)
    from polylogue.storage.sqlite.query_objects import EvaluationReceipt

    receipt = EvaluationReceipt(
        receipt_id="receipt-test",
        source_generation="source:absent",
        user_generation="user:absent",
        index_generation="index:absent",
        runtime_build_ref="polylogue:test",
    )
    findings = build_findings(
        report,
        query_reference="query:" + "0" * 64,
        result_set_reference="result-set:test",
        receipt=receipt,
    )
    assert len(findings) == 1
    assert findings[0].public_claim is not None
    assert findings[0].public_claim.disclosure == "public"
    assert "50.0%" in findings[0].public_claim.publication


def test_materialize_end_to_end_publishable_run_round_trips_through_public_claims(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    report = _report(
        archive_root=archive_root,
        silent=20,
        acknowledged=15,
        ambiguous=5,
        n_min=30,
        member_refs=("message:s1:tool-1-result", "message:s1:tool-2-result"),
    )

    result = materialize_claim_vs_evidence_evidence(report, archive_root=archive_root, now_ms=1_000)

    assert result["public_claim_written"] is True
    assert len(result["finding_assertion_ids"]) == 1

    conn = sqlite3.connect(archive_root / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        query_hash = result["query_ref"].removeprefix("query:")
        query = get_query(conn, query_hash)
        assert query is not None
        assert query.grain == "structured-failure-followup"

        result_set_id = result["result_set_ref"].removeprefix("result-set:")
        result_set = get_result_set(conn, result_set_id)
        assert result_set is not None
        assert result_set.member_count == 2
        assert result_set.exactness == "capped"
        assert result_set.persistence_class == "finding"

        findings = list_assertion_claims(conn, kinds=(AssertionKind.FINDING,), statuses=None)
        assert len(findings) == 1
        assert findings[0].assertion_id in result["finding_assertion_ids"]

        public_inputs = list_public_finding_inputs(conn)
        assert len(public_inputs) == 1
        assert public_inputs[0].claim_key == "finding.silent-proceed-lower-bound"
        assert public_inputs[0].disclosure == "public"
    finally:
        conn.close()


def test_materialize_unpublishable_run_writes_private_finding_without_public_claim(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    report = _report(archive_root=archive_root, silent=2, acknowledged=2, ambiguous=16, n_min=30)

    result = materialize_claim_vs_evidence_evidence(report, archive_root=archive_root, now_ms=1_000)

    assert result["public_claim_written"] is False
    assert len(result["finding_assertion_ids"]) == 1

    conn = sqlite3.connect(archive_root / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        findings = list_assertion_claims(conn, kinds=(AssertionKind.FINDING,), statuses=None)
        assert len(findings) == 1
        public_inputs = list_public_finding_inputs(conn)
        assert public_inputs == ()
    finally:
        conn.close()


def test_materialize_is_idempotent_for_a_retried_identical_call(tmp_path: Path) -> None:
    """A retried write (same report, same wall-clock) must not duplicate rows."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    report = _report(archive_root=archive_root, silent=20, acknowledged=15, ambiguous=5, n_min=30)

    first = materialize_claim_vs_evidence_evidence(report, archive_root=archive_root, now_ms=1_000)
    second = materialize_claim_vs_evidence_evidence(report, archive_root=archive_root, now_ms=1_000)

    assert first == second

    conn = sqlite3.connect(archive_root / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        findings = list_assertion_claims(conn, kinds=(AssertionKind.FINDING,), statuses=None)
        assert len(findings) == 1
    finally:
        conn.close()


def test_materialize_at_a_later_time_reuses_query_and_result_set_but_records_a_new_run(tmp_path: Path) -> None:
    """A genuine regeneration keeps the stable AnalysisDefinition/result-set identity

    but records its own AnalysisRun receipt and finding row -- the archive should
    carry that a re-verification happened at a later corpus/tier state, not silently
    collapse it into the first run.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    report = _report(archive_root=archive_root, silent=20, acknowledged=15, ambiguous=5, n_min=30)

    first = materialize_claim_vs_evidence_evidence(report, archive_root=archive_root, now_ms=1_000)
    second = materialize_claim_vs_evidence_evidence(report, archive_root=archive_root, now_ms=2_000)

    assert first["query_ref"] == second["query_ref"]
    assert first["result_set_ref"] == second["result_set_ref"]
    assert first["finding_assertion_ids"] != second["finding_assertion_ids"]

    conn = sqlite3.connect(archive_root / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        findings = list_assertion_claims(conn, kinds=(AssertionKind.FINDING,), statuses=None)
        assert len(findings) == 2
    finally:
        conn.close()
