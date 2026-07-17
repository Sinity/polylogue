"""Tests for comparative-judgment build/serialize round-tripping (rxdo.9.11)."""

from __future__ import annotations

from polylogue.core.enums import ComparativeVerdict
from polylogue.insights.judgment.comparative import (
    build_comparative_judgment,
    comparative_judgment_from_value,
    comparative_judgment_to_value,
)
from polylogue.insights.judgment.types import JudgeIdentity

_JUDGE = JudgeIdentity(actor_ref="agent:claude-sonnet-5", execution_context_id="ctx-hash-abc")


def test_build_is_deterministic_for_identical_inputs() -> None:
    kwargs: dict[str, object] = {
        "items": ["finding:a", "finding:b"],
        "dimension": "correctness",
        "verdict": ComparativeVerdict.PREFER_LEFT,
        "judge": _JUDGE,
        "blinded": True,
        "rubric_id": "rubric-1",
        "rubric_version": 1,
        "decided_at_ms": 1000,
    }
    first = build_comparative_judgment(**kwargs)  # type: ignore[arg-type]
    second = build_comparative_judgment(**kwargs)  # type: ignore[arg-type]
    assert first.judgment_id == second.judgment_id


def test_build_id_changes_with_verdict() -> None:
    base: dict[str, object] = {
        "items": ["finding:a", "finding:b"],
        "dimension": "correctness",
        "judge": _JUDGE,
        "blinded": True,
        "rubric_id": "rubric-1",
        "rubric_version": 1,
        "decided_at_ms": 1000,
    }
    left = build_comparative_judgment(verdict=ComparativeVerdict.PREFER_LEFT, **base)  # type: ignore[arg-type]
    right = build_comparative_judgment(verdict=ComparativeVerdict.PREFER_RIGHT, **base)  # type: ignore[arg-type]
    assert left.judgment_id != right.judgment_id


def test_pairwise_round_trips_through_value_json() -> None:
    judgment = build_comparative_judgment(
        items=["finding:a", "finding:b"],
        dimension="correctness",
        verdict=ComparativeVerdict.PREFER_LEFT,
        judge=_JUDGE,
        blinded=True,
        rubric_id="rubric-1",
        rubric_version=2,
        decided_at_ms=5000,
        evidence_refs=["session:s1"],
        rationale="A is more correct.",
        rationale_visible=True,
    )
    value = comparative_judgment_to_value(judgment)
    restored = comparative_judgment_from_value(judgment.judgment_id, value, evidence_refs=("session:s1",))

    assert restored.items == judgment.items
    assert restored.dimension == judgment.dimension
    assert restored.verdict == judgment.verdict
    assert restored.judge == judgment.judge
    assert restored.blinded == judgment.blinded
    assert restored.rubric_id == judgment.rubric_id
    assert restored.rubric_version == judgment.rubric_version
    assert restored.evidence_refs == ("session:s1",)
    assert restored.rationale == "A is more correct."
    assert restored.decided_at_ms == 5000


def test_nwise_ordering_round_trips_as_a_list_not_a_scalar() -> None:
    judgment = build_comparative_judgment(
        items=["finding:a", "finding:b", "finding:c"],
        dimension="usefulness",
        verdict=["finding:c", "finding:a", "finding:b"],
        judge=_JUDGE,
        blinded=False,
        rubric_id="rubric-2",
        rubric_version=1,
        decided_at_ms=42,
    )
    value = comparative_judgment_to_value(judgment)
    assert value["verdict"] == ["finding:c", "finding:a", "finding:b"]
    restored = comparative_judgment_from_value(judgment.judgment_id, value)
    assert restored.verdict == ("finding:c", "finding:a", "finding:b")
    assert restored.is_ordering


def test_hidden_rationale_is_not_serialized() -> None:
    judgment = build_comparative_judgment(
        items=["finding:a", "finding:b"],
        dimension="correctness",
        verdict=ComparativeVerdict.TIE,
        judge=_JUDGE,
        blinded=True,
        rubric_id="rubric-1",
        rubric_version=1,
        decided_at_ms=1,
        rationale="secret reasoning",
        rationale_visible=False,
    )
    value = comparative_judgment_to_value(judgment)
    assert value["rationale"] is None
