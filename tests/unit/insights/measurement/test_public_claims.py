from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from polylogue.core.enums import AssertionStatus, AssertionVisibility
from polylogue.insights.measurement.public_claims import (
    CapabilityClaimInput,
    EvidenceIntegrityStatus,
    EvidenceIntegrityVerdict,
    MappingEvidenceIntegrityProvider,
    PublicClaimPresetName,
    PublicClaimStatus,
    PublicFindingInput,
    build_public_claims_payload,
    project_public_claims,
    render_public_claims_markdown,
)


def _finding(**overrides: object) -> PublicFindingInput:
    values: dict[str, Any] = {
        "assertion_ref": "assertion:finding-active",
        "claim_key": "finding.demo",
        "publication": "The bounded demo observed 3 of 10 rows.",
        "scope": "One deterministic ten-row fixture.",
        "caveat": "This is not a population estimate.",
        "public_evidence_refs": ("file:docs/demo.md",),
        "presets": tuple(PublicClaimPresetName),
        "disclosure": "public",
        "statistic": {"op": "ratio", "value": 0.3, "unit": "ratio"},
        "finding_epoch": "epoch-1",
        "evaluation_ref": "file:docs/demo.md#evaluation",
        "frame_ref": "file:docs/demo.md#frame",
        "assertion_status": AssertionStatus.ACTIVE,
        "assertion_visibility": AssertionVisibility.PRIVATE,
        "author_kind": "user",
        "judgment_ref": None,
        "judgment_decision": None,
        "supersedes": (),
        "updated_at_ms": 10,
    }
    values.update(overrides)
    return PublicFindingInput(**values)


def _verdict(
    status: EvidenceIntegrityStatus,
    *,
    finding_ref: str = "assertion:finding-active",
    reason_codes: tuple[str, ...] = (),
) -> EvidenceIntegrityVerdict:
    qualified = status in {
        EvidenceIntegrityStatus.SUPPORTED,
        EvidenceIntegrityStatus.PARTIALLY_SUPPORTED,
    }
    return EvidenceIntegrityVerdict(
        finding_ref=finding_ref,
        status=status,
        public_evidence_refs=("file:docs/demo.md#receipt",),
        reason_codes=reason_codes,
        as_of_epoch="epoch-1" if qualified else None,
        frame_ref="file:docs/demo.md#frame" if qualified else None,
        definition_ref="file:docs/demo.md#definition" if qualified else None,
    )


@pytest.mark.parametrize(
    ("integrity_status", "public_status", "badge"),
    [
        (EvidenceIntegrityStatus.SUPPORTED, PublicClaimStatus.SUPPORTED, "[SUPPORTED]"),
        (
            EvidenceIntegrityStatus.PARTIALLY_SUPPORTED,
            PublicClaimStatus.PARTIALLY_SUPPORTED,
            "[PARTIALLY SUPPORTED]",
        ),
        (EvidenceIntegrityStatus.NOT_SUPPORTED, PublicClaimStatus.NOT_SUPPORTED, "[NOT SUPPORTED]"),
        (EvidenceIntegrityStatus.STALE, PublicClaimStatus.STALE_NEEDS_RERUN, "[STALE / NEEDS RERUN]"),
        (EvidenceIntegrityStatus.HELD_PRIVATE, PublicClaimStatus.HELD_PRIVATE, "[HELD PRIVATE]"),
        (EvidenceIntegrityStatus.CLOSED_LOOP, PublicClaimStatus.UNKNOWN, "[UNKNOWN · CLOSED LOOP]"),
        (EvidenceIntegrityStatus.CYCLE, PublicClaimStatus.UNKNOWN, "[UNKNOWN · CYCLE]"),
        (EvidenceIntegrityStatus.UNRESOLVED, PublicClaimStatus.UNKNOWN, "[UNKNOWN · UNRESOLVED]"),
        (
            EvidenceIntegrityStatus.FRAME_INCOMPLETE,
            PublicClaimStatus.UNKNOWN,
            "[UNKNOWN · FRAME INCOMPLETE]",
        ),
    ],
)
def test_integrity_verdicts_project_to_distinct_public_renderings(
    integrity_status: EvidenceIntegrityStatus,
    public_status: PublicClaimStatus,
    badge: str,
) -> None:
    finding = _finding()
    verdict = _verdict(integrity_status, reason_codes=(f"reason-{integrity_status.value}",))

    projected = project_public_claims(
        (finding,),
        (),
        integrity=MappingEvidenceIntegrityProvider({finding.assertion_ref: verdict}),
    )[0]

    assert projected.status is public_status
    assert projected.integrity_status is integrity_status
    assert projected.badge == badge
    if public_status is not PublicClaimStatus.SUPPORTED:
        assert f"integrity-{integrity_status.value}" in projected.blocker_codes


def test_broken_reference_is_distinct_from_explicitly_unsupported_evidence() -> None:
    finding = _finding()
    broken = _verdict(
        EvidenceIntegrityStatus.UNRESOLVED,
        reason_codes=("broken-reference",),
    )
    unsupported = _verdict(
        EvidenceIntegrityStatus.NOT_SUPPORTED,
        reason_codes=("incompatible-grounding",),
    )

    broken_claim = project_public_claims(
        (finding,),
        (),
        integrity=MappingEvidenceIntegrityProvider({finding.assertion_ref: broken}),
    )[0]
    unsupported_claim = project_public_claims(
        (finding,),
        (),
        integrity=MappingEvidenceIntegrityProvider({finding.assertion_ref: unsupported}),
    )[0]

    assert (broken_claim.status, broken_claim.integrity_status, broken_claim.badge) == (
        PublicClaimStatus.UNKNOWN,
        EvidenceIntegrityStatus.UNRESOLVED,
        "[UNKNOWN · UNRESOLVED]",
    )
    assert (unsupported_claim.status, unsupported_claim.integrity_status, unsupported_claim.badge) == (
        PublicClaimStatus.NOT_SUPPORTED,
        EvidenceIntegrityStatus.NOT_SUPPORTED,
        "[NOT SUPPORTED]",
    )


def test_missing_integrity_verdict_fails_closed_as_unresolved() -> None:
    finding = _finding()

    projected = project_public_claims(
        (finding,),
        (),
        integrity=MappingEvidenceIntegrityProvider({}),
    )[0]

    assert projected.status is PublicClaimStatus.UNKNOWN
    assert projected.integrity_status is EvidenceIntegrityStatus.UNRESOLVED
    assert projected.integrity_verdict_present is False
    assert projected.reason_codes == ("integrity-verdict-not-computed",)
    assert projected.public_remediation_refs == ("bead:polylogue-37t.14",)


def test_review_and_privacy_block_supported_verdicts() -> None:
    candidate = _finding(assertion_status=AssertionStatus.CANDIDATE, author_kind="detector")
    private = _finding(disclosure="held_private")
    rejected = _finding(assertion_status=AssertionStatus.REJECTED, author_kind="detector")
    provider = MappingEvidenceIntegrityProvider(
        {
            candidate.assertion_ref: _verdict(EvidenceIntegrityStatus.SUPPORTED),
            private.assertion_ref: _verdict(EvidenceIntegrityStatus.SUPPORTED),
            rejected.assertion_ref: _verdict(EvidenceIntegrityStatus.SUPPORTED),
        }
    )

    candidate_claim = project_public_claims((candidate,), (), integrity=provider)[0]
    private_claim = project_public_claims((private,), (), integrity=provider)[0]
    rejected_claim = project_public_claims((rejected,), (), integrity=provider)[0]

    assert candidate_claim.status is PublicClaimStatus.UNKNOWN
    assert candidate_claim.blocker_codes == ("finding-candidate",)
    assert private_claim.status is PublicClaimStatus.HELD_PRIVATE
    assert private_claim.blocker_codes == ("publication-held-private",)
    assert private_claim.publication == "Claim text withheld pending public privacy review."
    assert private_claim.public_evidence_refs == ()
    assert private_claim.statistic is None
    assert rejected_claim.status is PublicClaimStatus.NOT_SUPPORTED
    assert rejected_claim.blocker_codes == ("finding-rejected",)


def test_integrity_or_declaration_privacy_hold_dominates_lifecycle_and_redacts() -> None:
    finding = _finding(
        assertion_status=AssertionStatus.CANDIDATE,
        author_kind="detector",
    )
    integrity_held = project_public_claims(
        (finding,),
        (),
        integrity=MappingEvidenceIntegrityProvider(
            {finding.assertion_ref: _verdict(EvidenceIntegrityStatus.HELD_PRIVATE)}
        ),
    )[0]
    declaration_held = project_public_claims(
        (replace(finding, disclosure="held_private"),),
        (),
        integrity=MappingEvidenceIntegrityProvider(
            {finding.assertion_ref: _verdict(EvidenceIntegrityStatus.SUPPORTED)}
        ),
    )[0]

    for claim in (integrity_held, declaration_held):
        assert claim.status is PublicClaimStatus.HELD_PRIVATE
        assert claim.privacy_review == "held_private"
        assert claim.publication == "Claim text withheld pending public privacy review."
        assert claim.public_evidence_refs == ()
        assert claim.statistic is None
        assert claim.finding_epoch is None
        assert claim.evaluation_ref is None
    assert integrity_held.blocker_codes == ("integrity-held_private",)
    assert declaration_held.blocker_codes == ("publication-held-private",)


def test_supersession_selects_one_live_finding_and_conflicting_actives_fail_closed() -> None:
    old = _finding(assertion_ref="assertion:old", updated_at_ms=10)
    current = _finding(
        assertion_ref="assertion:current",
        supersedes=("assertion:old",),
        judgment_ref="assertion:judgment",
        judgment_decision="accept",
        updated_at_ms=20,
    )
    provider = MappingEvidenceIntegrityProvider(
        {current.assertion_ref: _verdict(EvidenceIntegrityStatus.SUPPORTED, finding_ref=current.assertion_ref)}
    )

    selected = project_public_claims((old, current), (), integrity=provider)[0]

    assert selected.source_ref == current.assertion_ref
    assert selected.status is PublicClaimStatus.SUPPORTED

    conflicting = replace(current, assertion_ref="assertion:other", supersedes=(), updated_at_ms=30)
    conflict_provider = MappingEvidenceIntegrityProvider(
        {
            current.assertion_ref: _verdict(EvidenceIntegrityStatus.SUPPORTED, finding_ref=current.assertion_ref),
            conflicting.assertion_ref: _verdict(
                EvidenceIntegrityStatus.SUPPORTED, finding_ref=conflicting.assertion_ref
            ),
        }
    )
    failed_closed = project_public_claims((current, conflicting), (), integrity=conflict_provider)[0]

    assert failed_closed.source_ref == conflicting.assertion_ref
    assert failed_closed.status is PublicClaimStatus.UNKNOWN
    assert failed_closed.blocker_codes == ("multiple-live-findings",)


def test_evidence_epoch_advance_degrades_same_finding_without_rewriting_it() -> None:
    finding = _finding()
    supported_provider = MappingEvidenceIntegrityProvider(
        {finding.assertion_ref: _verdict(EvidenceIntegrityStatus.SUPPORTED)}
    )
    stale_provider = MappingEvidenceIntegrityProvider(
        {
            finding.assertion_ref: EvidenceIntegrityVerdict(
                finding_ref=finding.assertion_ref,
                status=EvidenceIntegrityStatus.STALE,
                reason_codes=("source-epoch-advanced",),
                as_of_epoch="epoch-2",
                frame_ref="file:docs/demo.md#frame-v2",
                definition_ref="file:docs/demo.md#definition",
                public_remediation_refs=("run:demo-rerun",),
            )
        }
    )

    supported = project_public_claims((finding,), (), integrity=supported_provider)[0]
    stale = project_public_claims((finding,), (), integrity=stale_provider)[0]

    assert supported.source_ref == stale.source_ref == finding.assertion_ref
    assert supported.finding_epoch == stale.finding_epoch == "epoch-1"
    assert supported.status is PublicClaimStatus.SUPPORTED
    assert stale.status is PublicClaimStatus.STALE_NEEDS_RERUN
    assert stale.verdict_as_of_epoch == "epoch-2"
    assert stale.reason_codes == ("source-epoch-advanced",)


def test_all_presets_reuse_one_projection_status_and_include_qualifiers() -> None:
    finding = _finding()
    capability = CapabilityClaimInput(
        claim_key="category.demo",
        publication="The demo has a local evidence surface.",
        scope="The deterministic fixture.",
        caveat="This is a capability statement, not a measured outcome.",
        public_evidence_refs=("file:README.md",),
        presets=tuple(PublicClaimPresetName),
    )
    claims = project_public_claims(
        (finding,),
        (capability,),
        integrity=MappingEvidenceIntegrityProvider(
            {finding.assertion_ref: _verdict(EvidenceIntegrityStatus.PARTIALLY_SUPPORTED)}
        ),
    )

    statuses: dict[PublicClaimPresetName, dict[str, object]] = {}
    for preset in PublicClaimPresetName:
        payload = build_public_claims_payload(claims, preset)
        raw_claims = cast(list[dict[str, object]], payload["claims"])
        statuses[preset] = {cast(str, item["claim_key"]): item["status"] for item in raw_claims}

    assert len({tuple(sorted(status.items())) for status in statuses.values()}) == 1

    readme_claim = cast(
        list[dict[str, object]],
        build_public_claims_payload(claims, PublicClaimPresetName.README)["claims"],
    )[0]
    launch_claim = cast(
        list[dict[str, object]],
        build_public_claims_payload(claims, PublicClaimPresetName.LAUNCH)["claims"],
    )[0]
    findings_claim = cast(
        list[dict[str, object]],
        build_public_claims_payload(claims, PublicClaimPresetName.FINDINGS_PAGE)["claims"],
    )[0]
    assert "review" not in readme_claim and "reason_codes" not in readme_claim
    assert "review" not in launch_claim and "reason_codes" in launch_claim
    assert "review" in findings_claim and "reason_codes" in findings_claim

    markdown = render_public_claims_markdown(claims, PublicClaimPresetName.FINDINGS_PAGE)
    assert "`finding.demo` [PARTIALLY SUPPORTED]" in markdown
    assert "verdict as-of=epoch-1" in markdown
    assert "definition=file:docs/demo.md#definition" in markdown
    assert "`category.demo` [CAPABILITY ONLY]" in markdown


def test_public_boundary_rejects_private_paths_in_copy_and_receipts() -> None:
    with pytest.raises(ValueError, match="private or absolute path"):
        CapabilityClaimInput(
            claim_key="category.private",
            publication="Read /home/operator/private/archive.db.",
            scope="One local archive.",
            caveat="Not public.",
            public_evidence_refs=("file:README.md",),
            presets=(PublicClaimPresetName.README,),
        )

    with pytest.raises(ValueError, match="private or absolute path"):
        EvidenceIntegrityVerdict(
            finding_ref="assertion:finding-active",
            status=EvidenceIntegrityStatus.STALE,
            public_remediation_refs=("run:/home/operator/rerun",),
        )

    with pytest.raises(ValueError, match="private or absolute path"):
        _finding(statistic={"breakdown": {"receipt": "/home/operator/private/result.json"}})


def test_supported_verdict_requires_epoch_frame_and_definition() -> None:
    with pytest.raises(ValueError, match="require as_of_epoch, frame_ref, and definition_ref"):
        EvidenceIntegrityVerdict(
            finding_ref="assertion:finding-active",
            status=EvidenceIntegrityStatus.SUPPORTED,
        )


def test_mismatched_verdict_ref_is_rejected() -> None:
    finding = _finding()
    provider = MappingEvidenceIntegrityProvider(
        {finding.assertion_ref: _verdict(EvidenceIntegrityStatus.SUPPORTED, finding_ref="assertion:wrong")}
    )

    with pytest.raises(ValueError, match="verdict ref mismatch"):
        project_public_claims((finding,), (), integrity=provider)
