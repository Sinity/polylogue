"""Every declared subject carries exactly one recorded outcome.

Anti-vacuity: each test names the receipt shape that must not be accepted --
an absent subject, an unconserved zero diff, a narrowed package, a zero-material
claim the frontier contradicts -- and asserts the blocking outcome. Report the
outcome from the receipt alone, or iterate the receipts instead of the
denominator, and these go red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument, json_document
from polylogue.schemas.provider_denominator import DenominatorSubject, ProviderDenominator
from polylogue.schemas.provider_reconciliation import (
    ProviderMatrix,
    load_receipts,
    reconcile_provider_matrix,
)
from polylogue.schemas.source_frontier import (
    FrontierCheck,
    FrontierFinding,
    FrontierRoot,
    FrontierSubject,
    RootBaseline,
    SchemaFrontier,
)

CONFIGURATION: JSONDocument = {"privacy_level": "standard", "source_selection": "declared_frontier"}


def _denominator(*subjects: DenominatorSubject) -> ProviderDenominator:
    return ProviderDenominator(subjects=tuple(sorted(subjects)), findings=())


def _required(token: str) -> DenominatorSubject:
    return DenominatorSubject(
        subject=token,
        disposition="generation_required",
        origins=(f"{token}-session",),
        executable_origins=(f"{token}-session",),
        requires_package=True,
        reason=None,
    )


def _frontier(
    token: str,
    *,
    members: int,
    zero_material_reason: str | None = None,
    mutability: str = "frozen",
) -> SchemaFrontier:
    root = FrontierRoot(
        path=Path("/declared/root"),
        scope="synthetic declared root",
        mutability=mutability,
        zero_material_reason=zero_material_reason,
    )
    baseline = RootBaseline(
        subject=token,
        root=str(root.path),
        member_count=members,
        byte_count=members * 10,
        members=(),
    )
    return SchemaFrontier(
        subjects=(FrontierSubject(token, (root,)),),
        baselines=(baseline,),
    )


def _check(frontier: SchemaFrontier, *, added: int = 0) -> FrontierCheck:
    findings = tuple(
        FrontierFinding(
            "member_added",
            baseline.subject,
            baseline.root,
            "the declared root admits a member the baseline does not record",
            member=f"grown-{index}",
            severity="notice",
        )
        for baseline in frontier.baselines
        for index in range(added)
    )
    return FrontierCheck(
        baseline_digest=frontier.baseline_digest,
        declaration_digest=frontier.declaration_digest,
        findings=findings,
        checked_roots=1,
        checked_members=sum(item.member_count for item in frontier.baselines) + added,
        content_verified=False,
    )


def _receipt(
    token: str,
    *,
    candidates: int,
    included: int,
    samples: int,
    statuses: tuple[str, ...],
    narrowed: tuple[str, ...] = (),
    exit_code: int = 0,
) -> dict[str, object]:
    return {
        "subject": token,
        "exit_code": exit_code,
        "argv": ["devtools", "schema", "commit", "--provider", token, "--full-corpus", "--frontier"],
        "result": {
            "provider": token,
            "success": exit_code == 0,
            "sample_count": samples,
            "narrowed": bool(narrowed),
            "versions": [
                {
                    "version": f"v{index + 1}",
                    "status": status,
                    "sample_count": samples,
                    "added_paths": [],
                    "narrowed_paths": list(narrowed) if index == 0 else [],
                }
                for index, status in enumerate(statuses)
            ],
            "phase_receipt": {
                "source": {
                    "source_candidate_count": candidates,
                    "source_included_candidate_count": included,
                    "source_candidate_terminal_outcomes": {
                        "included": included,
                        "intentionally_excluded": candidates - included,
                    },
                    "source_input_manifest_digest": "0" * 64,
                    "source_recipe": {
                        "implementation_fingerprint": "f" * 64,
                        "structure_fingerprint": "e" * 64,
                        "statistics_fingerprint": "d" * 64,
                    },
                }
            },
        },
    }


def _reconcile(
    denominator: ProviderDenominator,
    frontier: SchemaFrontier,
    receipts: list[dict[str, object]],
    *,
    added: int = 0,
) -> ProviderMatrix:
    return reconcile_provider_matrix(
        frontier=frontier,
        check=_check(frontier, added=added),
        receipts=load_receipts(receipts),
        code_revision="0123456789abcdef0123456789abcdef01234567",
        inference_configuration=CONFIGURATION,
        denominator=denominator,
    )


def test_a_subject_the_pass_never_reached_is_recorded_as_not_run() -> None:
    """The outcome that a receipt-driven report cannot express at all."""

    denominator = _denominator(_required("codex"), _required("hermes"))
    frontier = _frontier("codex", members=4)
    matrix = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=4, included=4, samples=40, statuses=("changed",))],
    )

    outcomes = {item.subject: item.outcome for item in matrix.subjects}
    assert outcomes == {"codex": "generated", "hermes": "not_run"}
    assert not matrix.ok
    assert any("hermes" in blocker for blocker in matrix.blockers)


def test_a_generated_subject_records_its_route_evidence() -> None:
    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)
    matrix = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=4, included=3, samples=40, statuses=("changed", "unchanged"))],
    )

    assert matrix.ok
    entry = matrix.subjects[0]
    assert entry.outcome == "generated"
    assert entry.counts is not None
    assert entry.counts.conserves
    assert "3 of 4 inventoried candidate(s)" in entry.reason


def test_a_zero_diff_is_accepted_only_with_a_reconciled_denominator() -> None:
    """A zero diff whose candidate count does not reconcile is a failure.

    Anti-vacuity: the only difference between the two passes below is whether
    the route inventoried the admitted member set. Accept a zero diff on route
    evidence alone and the second assertion goes red.
    """

    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)

    conserved = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=4, included=4, samples=40, statuses=("unchanged",))],
    )
    assert conserved.subjects[0].outcome == "zero_diff"
    assert conserved.ok

    unconserved = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=2, included=2, samples=40, statuses=("unchanged",))],
    )
    assert unconserved.subjects[0].outcome == "failed"
    assert not unconserved.ok


def test_narrowing_without_adjudication_is_a_failure() -> None:
    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)
    matrix = _reconcile(
        denominator,
        frontier,
        [
            _receipt(
                "codex",
                candidates=4,
                included=4,
                samples=40,
                statuses=("changed",),
                narrowed=("session_document$.messages[*].role",),
            )
        ],
    )

    assert matrix.subjects[0].outcome == "failed"
    assert "narrowed" in matrix.subjects[0].reason


def test_a_zero_sample_count_over_admitted_material_is_a_failure() -> None:
    """Zero samples is only explainable by an empty denominator."""

    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)
    matrix = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=4, included=0, samples=0, statuses=())],
    )

    assert matrix.subjects[0].outcome == "failed"
    assert "unexplained zero sample count" in matrix.subjects[0].reason


def test_zero_eligible_material_needs_the_frontier_to_agree() -> None:
    """A declared zero denominator is checked against the frontier, not trusted.

    Anti-vacuity: the same terminal receipt is accepted for an empty declared
    root and refused for a root that admits members.
    """

    denominator = _denominator(_required("grok"))
    terminal = {
        "subject": "grok",
        "exit_code": 0,
        "argv": ["devtools", "schema", "commit", "--provider", "grok", "--full-corpus", "--frontier"],
        "result": {
            "provider": "grok",
            "success": False,
            "terminal": "zero_eligible_material",
            "reason": "no Grok export has ever been acquired",
            "sample_count": 0,
            "versions": [],
        },
    }

    empty = _frontier("grok", members=0, zero_material_reason="no Grok export has ever been acquired")
    accepted = _reconcile(denominator, empty, [dict(terminal)])
    assert accepted.subjects[0].outcome == "proven_non_applicable"
    assert accepted.ok

    populated = _frontier("grok", members=7, zero_material_reason="no Grok export has ever been acquired")
    refused = _reconcile(denominator, populated, [dict(terminal)])
    assert refused.subjects[0].outcome == "failed"
    assert not refused.ok


def test_an_incomplete_invocation_blocks_the_matrix() -> None:
    """A pass that did not declare the frontier is not bound to the baseline."""

    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)
    receipt = _receipt("codex", candidates=4, included=4, samples=40, statuses=("changed",))
    receipt["argv"] = ["devtools", "schema", "commit", "--provider", "codex"]

    matrix = _reconcile(denominator, frontier, [receipt])
    assert not matrix.ok
    assert any("--full-corpus" in blocker and "--frontier" in blocker for blocker in matrix.blockers)


def test_a_mixed_generator_revision_blocks_the_matrix() -> None:
    denominator = _denominator(_required("codex"), _required("hermes"))
    frontier = _frontier("codex", members=4)
    first = _receipt("codex", candidates=4, included=4, samples=40, statuses=("changed",))
    second = _receipt("hermes", candidates=0, included=0, samples=0, statuses=())
    recipe = json_document(
        json_document(json_document(json_document(second["result"])["phase_receipt"])["source"])["source_recipe"]
    )
    recipe["implementation_fingerprint"] = "a" * 64

    matrix = _reconcile(denominator, frontier, [first, second])
    assert any("mixed 2 generator revisions" in blocker for blocker in matrix.blockers)


def test_a_duplicate_receipt_is_refused() -> None:
    receipt = _receipt("codex", candidates=1, included=1, samples=1, statuses=("changed",))
    with pytest.raises(ValueError, match="two run receipts claim subject codex"):
        load_receipts([receipt, dict(receipt)])


def test_the_matrix_binds_what_decided_it() -> None:
    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)
    matrix = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=4, included=4, samples=40, statuses=("changed",))],
    )
    payload = matrix.to_payload()

    for field in (
        "baseline_digest",
        "frontier_declaration_digest",
        "provider_declaration_digest",
        "code_revision",
        "generator_semantics",
        "inference_configuration",
        "matrix_digest",
    ):
        assert payload[field], field
    assert json_document(payload["generator_semantics"])["implementation_fingerprint"] == ["f" * 64]


def test_append_root_growth_is_explained_drift_not_a_conservation_failure() -> None:
    """A live root observed twice cannot be required to hold still.

    An ``append`` root legitimately gains members while a session runs, and the
    frontier check observes it at a different instant than the generation did.
    Conservation therefore requires that no recorded baseline member was
    dropped, not that the two observations agree exactly.

    Anti-vacuity: dropping *below* the retained baseline is still a failure,
    and the same shortfall on a frozen root is a failure too.
    """

    denominator = _denominator(_required("codex"))
    live = _frontier("codex", members=100, mutability="append")

    # The baseline recorded 100 members, the generation saw 103, and the later
    # frontier check saw 105.
    grew = _reconcile(
        denominator,
        live,
        [_receipt("codex", candidates=103, included=103, samples=980, statuses=("unchanged",))],
        added=5,
    )
    entry = grew.subjects[0]
    assert entry.counts is not None
    assert entry.counts.conserves
    assert "append-root growth" in entry.counts.conservation_detail
    assert entry.outcome == "zero_diff"

    lost = _reconcile(
        denominator,
        live,
        [_receipt("codex", candidates=60, included=60, samples=600, statuses=("unchanged",))],
        added=5,
    )
    assert lost.subjects[0].outcome == "failed"

    frozen = _reconcile(
        denominator,
        _frontier("codex", members=100),
        [_receipt("codex", candidates=103, included=103, samples=980, statuses=("unchanged",))],
        added=5,
    )
    assert frozen.subjects[0].outcome == "failed"
    assert frozen.subjects[0].counts is not None
    assert "frozen roots" in frozen.subjects[0].counts.conservation_detail


def test_an_unaccounted_terminal_outcome_breaks_conservation() -> None:
    """Every inventoried candidate must carry exactly one terminal outcome."""

    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)
    receipt = _receipt("codex", candidates=4, included=4, samples=40, statuses=("unchanged",))
    source = json_document(json_document(json_document(receipt["result"])["phase_receipt"])["source"])
    source["source_candidate_terminal_outcomes"] = {"included": 3}

    matrix = _reconcile(denominator, frontier, [receipt])
    assert matrix.subjects[0].outcome == "failed"
    assert matrix.subjects[0].counts is not None
    assert "do not account for" in matrix.subjects[0].counts.conservation_detail


def test_a_generated_subject_also_needs_a_reconciled_denominator() -> None:
    """An incomplete pass is a failure whether or not a version changed.

    Conservation was consulted only on the zero-diff branch, so a run that
    inventoried half the declared corpus and reported one changed version
    became ``generated``, produced no blocker, and the matrix exited
    successfully -- omitting half the frontier is exactly what the
    conservation term exists to catch, and a changed version does not excuse
    it.

    Anti-vacuity: the two passes below differ only in whether the route
    inventoried the admitted member set; both report a changed version. Check
    conservation after the changed-version branch and the first assertion goes
    red.
    """

    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)

    unconserved = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=2, included=2, samples=40, statuses=("changed",))],
    )
    assert unconserved.subjects[0].outcome == "failed"
    assert "did not consume the declared denominator" in unconserved.subjects[0].reason
    assert not unconserved.ok

    conserved = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=4, included=4, samples=40, statuses=("changed",))],
    )
    assert conserved.subjects[0].outcome == "generated"
    assert conserved.ok
