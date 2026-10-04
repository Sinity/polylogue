"""Every declared subject carries exactly one recorded outcome.

Anti-vacuity: each test names the receipt shape that must not be accepted --
an absent subject, an unconserved zero diff, a narrowed package, a zero-material
claim the frontier contradicts -- and asserts the blocking outcome. Report the
outcome from the receipt alone, or iterate the receipts instead of the
denominator, and these go red.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from polylogue.core.json import JSONValue

from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument, json_document
from polylogue.schemas.privacy_config import PrivacyConfig
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

CONFIGURATION: JSONDocument = PrivacyConfig().to_payload()


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
            "inference_configuration": CONFIGURATION,
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
            "inference_configuration": CONFIGURATION,
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


def test_matrix_publication_rejects_a_partial_subject_set() -> None:
    """A partial regeneration cannot publish aggregate provenance.

    Anti-vacuity: removing the publication guard lets this malformed matrix
    serialize successfully, which would mis-bind the run revision to the
    omitted denominator subject.
    """

    denominator = _denominator(_required("codex"), _required("hermes"))
    frontier = _frontier("codex", members=1)
    matrix = _reconcile(
        denominator,
        frontier,
        [_receipt("codex", candidates=1, included=1, samples=1, statuses=("changed",))],
    )

    with pytest.raises(ValueError, match="complete denominator"):
        replace(matrix, subjects=(matrix.subjects[0],))


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


@pytest.mark.parametrize(
    ("candidates", "included", "terminal"),
    [
        (4, 5, {"included": 4}),
        (4, 3, {"included": 5, "intentionally_excluded": -1}),
    ],
)
def test_negative_or_out_of_range_counts_break_conservation(
    candidates: int,
    included: int,
    terminal: dict[str, int],
) -> None:
    """Malformed arithmetic must not manufacture a conserved denominator.

    Anti-vacuity: accepting either an included count above the inventory or a
    negative terminal bucket lets a malformed receipt sum to the inventory and
    incorrectly produce a successful provider outcome.
    """

    denominator = _denominator(_required("codex"))
    frontier = _frontier("codex", members=4)
    receipt = _receipt("codex", candidates=candidates, included=included, samples=40, statuses=("unchanged",))
    source = json_document(json_document(json_document(receipt["result"])["phase_receipt"])["source"])
    source["source_candidate_terminal_outcomes"] = cast("JSONValue", terminal)

    matrix = _reconcile(denominator, frontier, [receipt])
    entry = matrix.subjects[0]
    assert entry.outcome == "failed"
    assert entry.counts is not None
    assert not entry.counts.conserves


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


@pytest.mark.parametrize("recorded", [None, "malformed", {"level": "permissive"}])
def test_reconciliation_refuses_unbound_or_mismatched_policy(recorded: object) -> None:
    receipt = _receipt("codex", candidates=1, included=1, samples=1, statuses=("changed",))
    result = receipt["result"]
    assert isinstance(result, dict)
    result["inference_configuration"] = recorded
    with pytest.raises(ValueError, match="inference configuration"):
        _reconcile(_denominator(_required("codex")), _frontier("codex", members=1), [receipt])


@pytest.mark.parametrize("privacy", [None, "standard", "permissive"])
def test_reconcile_cli_checks_receipt_policy_before_writing(
    privacy: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from devtools import schema_reconcile
    from polylogue.schemas import provider_reconciliation

    frontier = _frontier("codex", members=1)
    monkeypatch.setattr(schema_reconcile, "load_frontier", lambda _path: frontier)
    monkeypatch.setattr(schema_reconcile, "check_frontier", _check)
    monkeypatch.setattr(
        provider_reconciliation, "derive_provider_denominator", lambda: _denominator(_required("codex"))
    )
    receipt = _receipt("codex", candidates=1, included=1, samples=1, statuses=("changed",))
    result = receipt["result"]
    assert isinstance(result, dict)
    policy = PrivacyConfig(level="permissive").to_payload()
    result["inference_configuration"] = policy
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "codex.json").write_text(json.dumps(receipt))
    destination = tmp_path / "matrix.json"
    destination.write_text("existing matrix")
    args = ["--receipts", str(receipts), "--write", str(destination), "--json", "--code-revision", "synthetic"]
    if privacy is not None:
        args.extend(["--privacy", privacy])
    exit_code = schema_reconcile.main(args)
    output = capsys.readouterr()
    if privacy == "permissive":
        assert exit_code == 0
        assert json.loads(output.out)["inference_configuration"] == policy
        assert json.loads(destination.read_text())["inference_configuration"] == policy
    else:
        assert exit_code == 1
        assert output.out == ""
        assert destination.read_text() == "existing matrix"


def test_policy_payload_preserves_all_custom_fields_and_rule_order() -> None:
    from polylogue.schemas.operator.inference import privacy_config_from_payload

    policy = PrivacyConfig(
        level="strict",
        safe_enum_max_length=19,
        high_entropy_min_length=7,
        cross_conv_min_count=11,
        cross_conv_proportional=False,
        field_overrides={"specific.*": "deny", "*": "allow"},
        allow_value_patterns=["z*", "a*"],
        deny_value_patterns=["first*", "second*"],
    )
    import json

    rebuilt = privacy_config_from_payload(json.loads(json.dumps(policy.to_payload(), sort_keys=True)))
    assert rebuilt is not None
    assert rebuilt.to_payload() == policy.to_payload()
    assert list(rebuilt.field_overrides) == ["specific.*", "*"]
    assert rebuilt.allow_value_patterns == ["z*", "a*"]


@pytest.mark.parametrize("change", ["rule_order", "bool_as_int"])
def test_reconciliation_preserves_wire_rule_order_and_json_types(change: str) -> None:
    policy = PrivacyConfig(field_overrides={"specific.*": "deny", "*": "allow"}).to_payload()
    receipt = _receipt("codex", candidates=1, included=1, samples=1, statuses=("changed",))
    result = receipt["result"]
    assert isinstance(result, dict)
    recorded = dict(policy)
    if change == "rule_order":
        rules = recorded["field_overrides"]
        assert isinstance(rules, list)
        recorded["field_overrides"] = list(reversed(rules))
    else:
        recorded["cross_conv_proportional"] = 0
    result["inference_configuration"] = recorded
    with pytest.raises(ValueError, match="inference configuration"):
        reconcile_provider_matrix(
            frontier=_frontier("codex", members=1),
            check=_check(_frontier("codex", members=1)),
            receipts=load_receipts([receipt]),
            code_revision="synthetic",
            inference_configuration=policy,
            denominator=_denominator(_required("codex")),
        )
