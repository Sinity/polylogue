from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from polylogue.core.enums import AssertionStatus
from polylogue.insights.measurement.public_claims import (
    EvidenceIntegrityStatus,
    EvidenceIntegrityVerdict,
    MappingEvidenceIntegrityProvider,
    PublicClaimStatus,
    project_public_claims,
)
from polylogue.scenarios.corpus import claim_vs_evidence_findings
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    PublicClaimDeclaration,
    judge_assertion_candidate,
    upsert_findings_as_assertions,
)
from polylogue.storage.sqlite.finding_provenance import list_public_finding_inputs


def _connect_user_db(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "user.db"
    initialize_archive_database(path, ArchiveTier.USER)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _supported_verdict(finding_ref: str) -> EvidenceIntegrityVerdict:
    return EvidenceIntegrityVerdict(
        finding_ref=finding_ref,
        status=EvidenceIntegrityStatus.SUPPORTED,
        public_evidence_refs=("file:docs/findings/claim-vs-evidence.md#verdict",),
        as_of_epoch="2026-07-04T08:55:53.667311+00:00",
        frame_ref="file:.agent/demos/claim-vs-evidence/claim-vs-evidence.report.json#sample_frame",
        definition_ref="file:.agent/demos/claim-vs-evidence/claim-vs-evidence.report.json#definition",
    )


def test_seed_population_uses_snapshot_values_and_sanitized_demo_refs() -> None:
    findings = {finding.claim_key: finding for finding in claim_vs_evidence_findings()}

    assert set(findings) == {
        "finding.silent-proceed-lower-bound",
        "finding.handler-class-split",
        "finding.per-origin-inspection-counts",
    }
    silent = findings["finding.silent-proceed-lower-bound"]
    assert silent.statistic == {
        "op": "lower_bound",
        "value": 0.241,
        "unit": "ratio",
        "numerator": 1_205,
        "denominator": 5_000,
        "ambiguous": 3_375,
    }
    handlers = findings["finding.handler-class-split"].statistic["value"]
    assert handlers == {
        "consequential": {
            "failed_outcomes": 4_175,
            "silent_proceed": 930,
            "ambiguous": 2_842,
            "silent_rate_lower_bound": 0.22275449101796407,
        },
        "benign_recovery": {
            "failed_outcomes": 634,
            "silent_proceed": 172,
            "ambiguous": 455,
            "silent_rate_lower_bound": 0.27129337539432175,
        },
        "other": {
            "failed_outcomes": 191,
            "silent_proceed": 103,
            "ambiguous": 78,
            "silent_rate_lower_bound": 0.5392670157068062,
        },
    }
    origins = findings["finding.per-origin-inspection-counts"].statistic["value"]
    assert isinstance(origins, dict)
    assert origins["claude-code-session"] == {"inspected": 3_752, "requested": 3_752, "frame_total": 31_555}
    assert origins["codex-session"] == {"inspected": 1_241, "requested": 1_241, "frame_total": 10_429}
    assert origins["claude-ai-export"] == {"inspected": 7, "requested": 7, "frame_total": 49}
    assert all(
        ref.startswith("file:.agent/demos/claim-vs-evidence/") or ref == "file:docs/findings/claim-vs-evidence.md"
        for finding in findings.values()
        for ref in finding.evidence_refs
    )


def test_real_storage_lifecycle_controls_public_support(tmp_path: Path) -> None:
    conn = _connect_user_db(tmp_path)
    try:
        candidates = upsert_findings_as_assertions(conn, claim_vs_evidence_findings(), now_ms=1_000)
        conn.commit()
        candidate = next(item for item in candidates if item.key == "finding.silent-proceed-lower-bound")

        before = project_public_claims(
            list_public_finding_inputs(conn),
            (),
            integrity=MappingEvidenceIntegrityProvider({}),
        )
        before_claim = next(item for item in before if item.claim_key == candidate.key)
        assert before_claim.status is PublicClaimStatus.UNKNOWN
        assert before_claim.assertion_status is AssertionStatus.CANDIDATE
        assert before_claim.publication_review == "pending"

        judged = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision="accept",
            reason="Reviewed publication, sample frame, and caveat.",
            now_ms=2_000,
        )
        conn.commit()
        assert judged.resulting_assertion is not None
        active_ref = f"assertion:{judged.resulting_assertion.assertion_id}"
        rows_after_accept = int(conn.execute("SELECT count(*) FROM assertions").fetchone()[0])

        accepted = project_public_claims(
            list_public_finding_inputs(conn),
            (),
            integrity=MappingEvidenceIntegrityProvider({active_ref: _supported_verdict(active_ref)}),
        )
        accepted_claim = next(item for item in accepted if item.claim_key == candidate.key)

        assert accepted_claim.source_ref == active_ref
        assert accepted_claim.assertion_status is AssertionStatus.ACTIVE
        assert accepted_claim.judgment_ref == f"assertion:{judged.judgment.assertion_id}"
        assert accepted_claim.publication_review == "approved"
        assert accepted_claim.privacy_review == "approved"
        assert accepted_claim.status is PublicClaimStatus.SUPPORTED

        upsert_findings_as_assertions(conn, claim_vs_evidence_findings(), now_ms=3_000)
        conn.commit()
        assert int(conn.execute("SELECT count(*) FROM assertions").fetchone()[0]) == rows_after_accept

        stale = project_public_claims(
            list_public_finding_inputs(conn),
            (),
            integrity=MappingEvidenceIntegrityProvider(
                {
                    active_ref: EvidenceIntegrityVerdict(
                        finding_ref=active_ref,
                        status=EvidenceIntegrityStatus.STALE,
                        reason_codes=("source-epoch-advanced",),
                        as_of_epoch="2026-07-17T00:00:00+00:00",
                        frame_ref="file:.agent/demos/claim-vs-evidence/claim-vs-evidence.report.json#sample_frame-v2",
                        definition_ref="file:.agent/demos/claim-vs-evidence/claim-vs-evidence.report.json#definition",
                        public_remediation_refs=("run:claim-vs-evidence-rerun",),
                    )
                }
            ),
        )
        stale_claim = next(item for item in stale if item.claim_key == candidate.key)
        assert stale_claim.source_ref == active_ref
        assert stale_claim.status is PublicClaimStatus.STALE_NEEDS_RERUN
        assert int(conn.execute("SELECT count(*) FROM assertions").fetchone()[0]) == rows_after_accept
    finally:
        conn.close()


def test_rejected_candidate_cannot_render_supported(tmp_path: Path) -> None:
    conn = _connect_user_db(tmp_path)
    try:
        candidate = upsert_findings_as_assertions(conn, claim_vs_evidence_findings()[:1], now_ms=1_000)[0]
        judged = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision="reject",
            reason="Publication claim not accepted.",
            now_ms=2_000,
        )
        conn.commit()
        candidate_ref = f"assertion:{candidate.assertion_id}"

        claim = project_public_claims(
            list_public_finding_inputs(conn),
            (),
            integrity=MappingEvidenceIntegrityProvider({candidate_ref: _supported_verdict(candidate_ref)}),
        )[0]

        assert judged.resulting_assertion is None
        assert claim.assertion_status is AssertionStatus.REJECTED
        assert claim.status is PublicClaimStatus.NOT_SUPPORTED
        assert claim.blocker_codes == ("finding-rejected",)
    finally:
        conn.close()


def test_public_claim_writer_rejects_absolute_or_parent_traversal_refs(tmp_path: Path) -> None:
    conn = _connect_user_db(tmp_path)
    finding = claim_vs_evidence_findings()[0]
    assert finding.public_claim is not None
    try:
        for unsafe_ref in ("file:/home/operator/private.json", "file:../private.json", "file:C:\\private.json"):
            unsafe = replace(
                finding,
                public_claim=replace(finding.public_claim, public_evidence_refs=(unsafe_ref,)),
            )
            try:
                upsert_findings_as_assertions(conn, (unsafe,), now_ms=1_000)
            except ValueError as exc:
                assert "repository-relative" in str(exc)
            else:
                raise AssertionError(f"unsafe ref was accepted: {unsafe_ref}")
    finally:
        conn.close()


def test_held_private_declaration_survives_storage_and_blocks_publication(tmp_path: Path) -> None:
    conn = _connect_user_db(tmp_path)
    finding = claim_vs_evidence_findings()[0]
    assert finding.public_claim is not None
    held = replace(
        finding,
        public_claim=PublicClaimDeclaration(
            publication=finding.public_claim.publication,
            scope=finding.public_claim.scope,
            caveat=finding.public_claim.caveat,
            public_evidence_refs=finding.public_claim.public_evidence_refs,
            presets=finding.public_claim.presets,
            disclosure="held_private",
        ),
    )
    try:
        candidate = upsert_findings_as_assertions(conn, (held,), now_ms=1_000)[0]
        judged = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision="accept",
            now_ms=2_000,
        )
        conn.commit()
        assert judged.resulting_assertion is not None
        active_ref = f"assertion:{judged.resulting_assertion.assertion_id}"

        claim = project_public_claims(
            list_public_finding_inputs(conn),
            (),
            integrity=MappingEvidenceIntegrityProvider({active_ref: _supported_verdict(active_ref)}),
        )[0]

        assert claim.publication_review == "approved"
        assert claim.privacy_review == "held_private"
        assert claim.status is PublicClaimStatus.HELD_PRIVATE
    finally:
        conn.close()
