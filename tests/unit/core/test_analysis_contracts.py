"""Non-vacuous laws for the shared evidence/analysis contract kernel."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from polylogue.core.analysis_contracts import (
    AnalysisContractError,
    BasketPointer,
    DefinitionIdentity,
    EvaluationWorld,
    ExperimentReceipt,
    FindingRecord,
    ImprovementLoopContract,
    IncompatibleRelationError,
    RelationManifest,
    ResultEnvelope,
    TypedReceiptEnvelope,
    UnsupportedProtocolError,
    claim_class_for,
    claims_view,
    require_shared_loop_contract,
)
from polylogue.core.evidence_integrity import EvidenceIntegrityStatus
from polylogue.core.evidence_value import FrameCoverage
from polylogue.core.refs import ObjectRef


def _ref(kind: str, name: str) -> ObjectRef:
    return ObjectRef(kind=kind, object_id=name)  # type: ignore[arg-type]


def _world(*, runtime: str = "build:v1") -> EvaluationWorld:
    return EvaluationWorld(
        source_generation="source:v1",
        user_generation="user:v1",
        index_generation="index:v1",
        runtime_build_ref=runtime,
        frame_ref=_ref("context-snapshot", "frame:v1"),
        resolved_bounds={"since": "2026-01-01T00:00:00+00:00"},
    )


def _definition(*, protocol: str = "metric.v1") -> DefinitionIdentity:
    return DefinitionIdentity(
        kind="metric",
        protocol_version=protocol,
        content={"construct": "cost", "unit": "usd"},
    )


def _relation(*, grain: str = "session", enumeration: str = "exact") -> RelationManifest:
    definition = _definition()
    coverage = FrameCoverage(
        intended_frame="all captured sessions",
        grain=grain,
        denominator="captured sessions",
        intended_count=2,
        observed_count=1,
        supported_count=1,
        complete=False,
        intended_refs=(_ref("session", "s1"), _ref("session", "s2")),
        observed_refs=(_ref("session", "s1"),),
    )
    return RelationManifest(
        relation_ref=_ref("result-set", "result:v1"),
        definition_ref=definition.ref,
        evaluation_world=_world(),
        grain=grain,
        coverage=coverage,
        enumeration=enumeration,  # type: ignore[arg-type]
        measurement_authority=("model-derived",),
        member_refs=(_ref("session", "s1"),),
    )


def test_definition_protocol_version_is_part_of_identity() -> None:
    assert _definition(protocol="metric.v1").ref != _definition(protocol="metric.v2").ref

    with pytest.raises(UnsupportedProtocolError):
        DefinitionIdentity("metric", "metric", {"construct": "cost"})


def test_evaluation_world_changes_for_runtime_or_bound_mutations() -> None:
    assert _world().world_id != _world(runtime="build:v2").world_id
    assert _world() != replace(_world(), resolved_bounds={"since": "2026-02-01T00:00:00+00:00"})


def test_result_preserves_exact_enumeration_frame_gap_and_model_authority() -> None:
    relation = _relation()
    result = ResultEnvelope(
        result_ref=_ref("result-set", "result:v1"),
        definition_ref=relation.definition_ref,
        evaluation_world=relation.evaluation_world,
        relation=relation,
        value_state="known",
        value=3.5,
    )

    wire = result.to_dict()
    assert wire["enumeration"] == "exact"
    assert wire["coverage"]["complete"] is False  # type: ignore[index]
    assert wire["measurement_authority"] == ["model-derived"]
    assert result.enumeration == "exact"


def test_exact_enumeration_rejects_sampling_interval_but_allows_frame_incompleteness() -> None:
    relation = _relation()
    with pytest.raises(AnalysisContractError, match="sampling intervals"):
        ResultEnvelope(
            result_ref=_ref("result-set", "result:v1"),
            definition_ref=relation.definition_ref,
            evaluation_world=relation.evaluation_world,
            relation=relation,
            value_state="known",
            value=1,
            sampling_interval={"method": "bootstrap"},
        )


def test_sensitive_ad_hoc_definition_is_not_durable_without_promotion() -> None:
    with pytest.raises(AnalysisContractError, match="promotion or citation"):
        DefinitionIdentity(
            "pattern",
            "pattern.v1",
            {"literal": "secret"},
            privacy_class="secret",
            durability="user",
        )

    promoted = DefinitionIdentity(
        "pattern",
        "pattern.v1",
        {"literal": "secret"},
        privacy_class="secret",
        durability="user",
        promoted=True,
        retention_policy={"ttl": "30d"},
        excision_link="excision:pattern-1",
    )
    assert promoted.excisable


def test_relation_composition_fails_closed_on_grain_mismatch() -> None:
    left = _relation(grain="session")
    right = _relation(grain="message")
    assert not left.compatible_with(right)
    with pytest.raises(IncompatibleRelationError):
        left.require_compatible_with(right)


def test_causal_claim_requires_experiment_receipt_and_complete_receipts() -> None:
    with pytest.raises(AnalysisContractError, match="ExperimentDefinition"):
        claim_class_for(causal_requested=True, experiment=None)

    incomplete = ExperimentReceipt(
        definition_ref=_ref("experiment", "experiment:v1"),
        assignment_ref=None,
        exposure_ref=_ref("observed-event", "exposure:v1"),
        frame_ref=_ref("cohort", "frame:v1"),
        exclusion_ref=None,
        stopping_ref=_ref("observed-event", "stop:v1"),
        outcome_refs=(_ref("observed-event", "outcome:v1"),),
        evaluation_world=_world(),
    )
    with pytest.raises(AnalysisContractError, match="assignment"):
        claim_class_for(causal_requested=True, experiment=incomplete)


def test_finding_support_is_a_view_over_assertion_lifecycle_and_integrity() -> None:
    finding = FindingRecord(
        finding_ref=_ref("finding", "f1"),
        status="candidate",  # type: ignore[arg-type]
        evidence_refs=(_ref("result-set", "r1"),),
        integrity_status=EvidenceIntegrityStatus.SUPPORTED,
        definition_ref=_definition().ref,
        frame_ref=_ref("context-snapshot", "frame:v1"),
    )
    assert not finding.current_supported
    accepted = finding.with_status("accepted")  # type: ignore[arg-type]
    assert accepted.current_supported
    assert accepted.context_injectable
    assert not replace(accepted, integrity_status=EvidenceIntegrityStatus.CYCLE).current_supported
    assert claims_view((finding, accepted)) == (accepted,)


def test_basket_is_a_versioned_workspace_pointer_not_a_duplicate_member_store() -> None:
    first = BasketPointer(_ref("workspace", "w"), _ref("result-set", "r"), "merkle:v1")
    second = replace(first, relation_version="merkle:v2")
    assert first.ref != second.ref
    assert first.to_dict()["relation_ref"] == "result-set:r"


def test_two_loop_pilots_share_scheduler_and_state_protocol() -> None:
    scheduler = _ref("run", "scheduler:v1")
    state = _ref("workspace", "loop-state")
    first = ImprovementLoopContract(_ref("improvement-loop", "a"), "loop.v1", scheduler, state, "curriculum")
    second = ImprovementLoopContract(_ref("improvement-loop", "b"), "loop.v1", scheduler, state, "recovery")
    require_shared_loop_contract(first, second)

    forked = replace(second, scheduler_ref=_ref("run", "scheduler:v2"))
    with pytest.raises(AnalysisContractError, match="scheduler/state contract"):
        require_shared_loop_contract(first, forked)


def test_canonical_payload_cannot_be_mutated_after_identity() -> None:
    """A content-addressed identity must not change under the reader's feet.

    ``_canonical`` returned ordinary dicts and lists, so a consumer that saved
    ``definition.ref_text`` could mutate ``definition.content`` and have the
    same object produce a different ref -- and receipts persisted before the
    mutation disagreed with every later serialization. The same applied to
    ``EvaluationWorld.resolved_bounds`` and its ``world_id``.

    Anti-vacuity: making ``_canonical`` return plain ``dict``/``list`` again
    turns every ``pytest.raises`` here green-to-red, and the ref/world_id
    equalities below then fail because the mutation lands.
    """
    definition = DefinitionIdentity(
        kind="metric",
        protocol_version="metric.v1",
        content={"construct": "cost", "tags": ["a", "b"], "bounds": {"since": "2026-01-01"}},
    )
    before = definition.ref_text
    with pytest.raises(TypeError, match="immutable"):
        definition.content["construct"] = "latency"  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        definition.content["bounds"]["since"] = "1999-01-01"  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        cast(dict[str, object], definition.content).setdefault("extra", 1)
    with pytest.raises(AttributeError):
        cast(list[object], definition.content["tags"]).append("c")
    assert definition.ref_text == before

    world = _world()
    world_before = world.world_id
    with pytest.raises(TypeError, match="immutable"):
        world.resolved_bounds["since"] = "1999-01-01"  # type: ignore[index]
    assert world.world_id == world_before
    # Still an ordinary JSON payload for every reader and hasher.
    assert isinstance(definition.content, dict)
    assert json.loads(json.dumps(definition.canonical_payload))["content"]["tags"] == ["a", "b"]


def test_result_value_requires_a_known_state() -> None:
    """A value behind a non-``known`` state is two answers to one question.

    Anti-vacuity: dropping the ``value is not None`` guard makes the three
    ``pytest.raises`` cases construct successfully, and dropping the
    vocabulary check admits ``value_state="maybe"``. The ``known`` and
    valueless cases below pin that the rule is not a blanket refusal.
    """
    relation = _relation()

    def envelope(state: str, value: object | None = None) -> ResultEnvelope:
        return ResultEnvelope(
            result_ref=relation.relation_ref,
            definition_ref=relation.definition_ref,
            evaluation_world=relation.evaluation_world,
            relation=relation,
            value_state=cast(Any, state),
            value=value,
        )

    for state in ("unknown", "unavailable", "redacted"):
        with pytest.raises(AnalysisContractError, match="cannot carry a value"):
            envelope(state, 42)
        assert envelope(state).value is None
    with pytest.raises(AnalysisContractError, match="unsupported result value state"):
        envelope("maybe")
    assert envelope("known", 42).value == 42


def test_receipt_privacy_inherits_the_embedded_definition() -> None:
    """A durable receipt cannot declassify the definition it serializes.

    ``TypedReceiptEnvelope.to_dict`` emits ``definition.to_dict()`` verbatim,
    so a ``secret`` unpromoted ad-hoc definition inside a ``private``/``audit``
    receipt persisted its content to the audit tier with no retention or
    excision metadata. Classification follows the content carried.

    Anti-vacuity: reading ``self.privacy_class`` instead of
    ``effective_privacy_class`` makes the first construction succeed. The
    promoted case below pins that a properly retained secret definition is
    still allowed, so the rule is not a blanket ban.
    """
    secret = DefinitionIdentity(
        kind="metric",
        protocol_version="metric.v1",
        content={"construct": "secret-cost"},
        privacy_class="secret",
    )
    with pytest.raises(AnalysisContractError, match="promotion or citation"):
        TypedReceiptEnvelope(
            object_ref=_ref("analysis-run", "run:v1"),
            definition=secret,
            evaluation_world=_world(),
            durability="audit",
            privacy_class="private",
        )

    retained = DefinitionIdentity(
        kind="metric",
        protocol_version="metric.v1",
        content={"construct": "secret-cost"},
        privacy_class="secret",
        durability="audit",
        promoted=True,
        retention_policy={"keep": "90d"},
        excision_link="excision:1",
    )
    receipt = TypedReceiptEnvelope(
        object_ref=_ref("analysis-run", "run:v1"),
        definition=retained,
        evaluation_world=_world(),
        durability="audit",
        privacy_class="private",
        promoted=True,
        retention_policy={"keep": "90d"},
        excision_link="excision:1",
    )
    assert receipt.effective_privacy_class == "secret"
    # An ephemeral receipt is not durable persistence, so it stays allowed.
    assert (
        TypedReceiptEnvelope(
            object_ref=_ref("analysis-run", "run:v2"),
            definition=secret,
            evaluation_world=_world(),
            durability="ephemeral",
        ).effective_privacy_class
        == "secret"
    )


def test_shared_loop_contract_compares_the_whole_state_ref() -> None:
    """Two pilots forking their state object do not share a loop contract.

    ``compatible_with`` compared only ``state_ref.kind``, so
    ``workspace:curriculum-state`` and ``workspace:recovery-state`` -- the
    exact per-loop fork this function exists to reject -- were accepted.

    Anti-vacuity: restoring the ``.kind`` comparison makes the
    ``pytest.raises`` fail; the shared-state pair below keeps it from becoming
    a refusal of every second pilot.
    """
    scheduler = _ref("run", "scheduler:v1")
    shared = _ref("workspace", "loop-state")
    first = ImprovementLoopContract(_ref("improvement-loop", "a"), "loop.v1", scheduler, shared, "curriculum")
    second = ImprovementLoopContract(_ref("improvement-loop", "b"), "loop.v1", scheduler, shared, "recovery")
    require_shared_loop_contract(first, second)

    forked = replace(second, state_ref=_ref("workspace", "recovery-state"))
    assert forked.state_ref.kind == first.state_ref.kind
    with pytest.raises(AnalysisContractError, match="scheduler/state contract"):
        require_shared_loop_contract(first, forked)
