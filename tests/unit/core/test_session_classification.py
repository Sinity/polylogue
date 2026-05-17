"""Tests for the heuristic session classifier (issue #1130).

Pins the typed taxonomy, classifier determinism, evidence-citation
contract, and per-category behavior on representative fixture profiles.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from polylogue.archive.conversation.extraction import WorkEvent, WorkEventKind
from polylogue.archive.session.session_profile import SessionProfile
from polylogue.insights.classification import (
    CLASSIFIER_FAMILY,
    CLASSIFIER_VERSION,
    EvidenceCite,
    SessionCategory,
    SessionClassification,
    classify_session,
)

# ---------------------------------------------------------------------------
# Profile + work-event factories
# ---------------------------------------------------------------------------


def _profile(
    *,
    conversation_id: str = "conv-1",
    provider: str = "claude-code",
    work_events: tuple[WorkEvent, ...] = (),
    tool_categories: dict[str, int] | None = None,
    repo_names: tuple[str, ...] = (),
    file_paths_touched: tuple[str, ...] = (),
    message_count: int = 4,
    tool_use_count: int = 0,
) -> SessionProfile:
    return SessionProfile(
        conversation_id=conversation_id,
        provider=provider,
        title=None,
        created_at=datetime(2026, 5, 17, 9, 0),
        updated_at=datetime(2026, 5, 17, 10, 0),
        message_count=message_count,
        substantive_count=message_count,
        tool_use_count=tool_use_count,
        thinking_count=0,
        attachment_count=0,
        word_count=120,
        total_cost_usd=0.0,
        total_duration_ms=0,
        tool_categories=tool_categories or {},
        repo_paths=(),
        cwd_paths=(),
        branch_names=(),
        file_paths_touched=file_paths_touched,
        languages_detected=(),
        repo_names=repo_names,
        work_events=work_events,
        phases=(),
        first_message_at=datetime(2026, 5, 17, 9, 0),
        last_message_at=datetime(2026, 5, 17, 10, 0),
        canonical_session_date=date(2026, 5, 17),
    )


def _event(kind: WorkEventKind, index: int = 0, *, confidence: float = 0.9) -> WorkEvent:
    return WorkEvent(
        kind=kind,
        start_index=index,
        end_index=index,
        confidence=confidence,
        evidence=(kind.value,),
        file_paths=(),
        tools_used=(kind.value,),
        summary=f"{kind.value} event",
        start_time=datetime(2026, 5, 17, 9, index),
        end_time=datetime(2026, 5, 17, 9, index, 30),
    )


# ---------------------------------------------------------------------------
# Taxonomy invariants
# ---------------------------------------------------------------------------


def test_session_category_is_closed_enum() -> None:
    """SessionCategory is a closed taxonomy — string values must round-trip."""

    values = {member.value for member in SessionCategory}
    # Every value is a non-empty snake_case identifier.
    assert all(value and value.replace("_", "").isalnum() for value in values)
    # Membership is round-trippable: SessionCategory(value) reconstructs.
    for member in SessionCategory:
        assert SessionCategory(member.value) is member


def test_taxonomy_covers_every_work_event_kind() -> None:
    """Every WorkEventKind must map to a SessionCategory (no silent drops)."""

    from polylogue.insights.classification import _WORK_EVENT_TO_CATEGORY

    missing = [kind for kind in WorkEventKind if kind not in _WORK_EVENT_TO_CATEGORY]
    assert not missing, f"WorkEventKind variants without category mapping: {missing}"


def test_unclassified_variant_exists() -> None:
    """A no-signal session must have a typed UNCLASSIFIED fallback."""

    assert SessionCategory.UNCLASSIFIED.value == "unclassified"


def test_classification_rejects_unknown_label_at_type_level() -> None:
    """SessionClassification.category cannot be an unknown string."""

    with pytest.raises(ValidationError):
        SessionClassification(
            category="not_a_real_category",  # type: ignore[arg-type]
            confidence=0.5,
            support_level="weak",
            evidence=(),
        )


def test_confidence_is_bounded() -> None:
    """confidence is constrained to [0, 1] by the model."""

    with pytest.raises(ValidationError):
        SessionClassification(
            category=SessionCategory.DEBUGGING,
            confidence=1.5,
            support_level="strong",
            evidence=(),
        )
    with pytest.raises(ValidationError):
        SessionClassification(
            category=SessionCategory.DEBUGGING,
            confidence=-0.1,
            support_level="strong",
            evidence=(),
        )


def test_evidence_weight_is_bounded() -> None:
    with pytest.raises(ValidationError):
        EvidenceCite(field="x", value="y", weight=1.5)


# ---------------------------------------------------------------------------
# Classifier behavior per category
# ---------------------------------------------------------------------------


def test_classifies_debugging_session() -> None:
    profile = _profile(
        work_events=(_event(WorkEventKind.DEBUGGING, 0), _event(WorkEventKind.DEBUGGING, 1)),
        tool_categories={"debug": 4},
        repo_names=("polylogue",),
        file_paths_touched=("polylogue/storage/sqlite/connection.py",),
        tool_use_count=8,
    )
    result = classify_session(profile)
    assert result.category is SessionCategory.DEBUGGING
    assert result.confidence > 0.5
    assert result.evidence  # non-empty


def test_classifies_refactoring_session() -> None:
    profile = _profile(
        work_events=(_event(WorkEventKind.REFACTORING, 0), _event(WorkEventKind.REFACTORING, 1)),
        repo_names=("polylogue",),
    )
    result = classify_session(profile)
    assert result.category is SessionCategory.REFACTORING


def test_classifies_feature_session_from_implementation_events() -> None:
    profile = _profile(
        work_events=(_event(WorkEventKind.IMPLEMENTATION, 0), _event(WorkEventKind.IMPLEMENTATION, 1)),
        tool_categories={"write": 3, "edit": 5},
    )
    result = classify_session(profile)
    assert result.category is SessionCategory.FEATURE


def test_classifies_testing_session() -> None:
    profile = _profile(
        work_events=(_event(WorkEventKind.TESTING, 0),),
        tool_categories={"test": 4},
    )
    result = classify_session(profile)
    assert result.category is SessionCategory.TESTING


def test_classifies_exploration_session_from_research() -> None:
    profile = _profile(
        work_events=(_event(WorkEventKind.RESEARCH, 0), _event(WorkEventKind.RESEARCH, 1)),
        tool_categories={"search": 6},
    )
    result = classify_session(profile)
    assert result.category is SessionCategory.EXPLORATION


def test_classifies_conversation_when_no_tools_and_no_files() -> None:
    profile = _profile(message_count=8, tool_use_count=0)
    result = classify_session(profile)
    assert result.category is SessionCategory.CONVERSATION
    # Conversation fallback cites the deciding fields explicitly.
    cite_fields = {cite.field for cite in result.evidence}
    assert "tool_use_count" in cite_fields
    assert "message_count" in cite_fields


def test_classifies_unclassified_when_empty() -> None:
    profile = _profile(message_count=0, tool_use_count=0)
    result = classify_session(profile)
    assert result.category is SessionCategory.UNCLASSIFIED
    assert result.confidence == 0.0
    assert result.evidence == ()
    assert result.support_level == "weak"


# ---------------------------------------------------------------------------
# Evidence-cite invariant
# ---------------------------------------------------------------------------


def test_non_unclassified_classifications_carry_evidence() -> None:
    """AC: every non-unclassified label must cite at least one evidence field."""

    profiles = [
        _profile(work_events=(_event(WorkEventKind.DEBUGGING),)),
        _profile(work_events=(_event(WorkEventKind.TESTING),), tool_categories={"test": 2}),
        _profile(work_events=(_event(WorkEventKind.IMPLEMENTATION),)),
        _profile(message_count=5, tool_use_count=0),  # conversation
    ]
    for profile in profiles:
        result = classify_session(profile)
        if result.category is SessionCategory.UNCLASSIFIED:
            continue
        assert result.evidence, f"{result.category} should carry evidence"
        for cite in result.evidence:
            assert cite.field  # named profile field
            assert cite.value  # non-empty observation
            assert 0.0 <= cite.weight <= 1.0


def test_evidence_is_ordered_by_weight_desc() -> None:
    profile = _profile(
        work_events=(_event(WorkEventKind.DEBUGGING, 0), _event(WorkEventKind.DEBUGGING, 1)),
        tool_categories={"debug": 3},
        repo_names=("polylogue",),
        file_paths_touched=("a.py", "b.py"),
    )
    result = classify_session(profile)
    weights = [cite.weight for cite in result.evidence]
    assert weights == sorted(weights, reverse=True)


# ---------------------------------------------------------------------------
# Determinism + versioning
# ---------------------------------------------------------------------------


def test_classifier_is_deterministic() -> None:
    """AC: rebuild deterministic for fixed input."""

    profile = _profile(
        work_events=(_event(WorkEventKind.DEBUGGING, 0), _event(WorkEventKind.TESTING, 1)),
        tool_categories={"test": 2, "debug": 3},
        repo_names=("polylogue",),
    )
    results = [classify_session(profile) for _ in range(5)]
    first = results[0]
    for other in results[1:]:
        assert other == first


def test_classifier_version_is_pinned_on_results() -> None:
    profile = _profile(work_events=(_event(WorkEventKind.DEBUGGING),))
    result = classify_session(profile)
    assert result.classifier_version == CLASSIFIER_VERSION
    assert result.classifier_family == CLASSIFIER_FAMILY


def test_tie_resolution_is_taxonomy_order() -> None:
    """Equal-weight ties resolve in SessionCategory declaration order."""

    # DEBUGGING is declared before REFACTORING, so a tie picks DEBUGGING.
    profile = _profile(
        work_events=(
            _event(WorkEventKind.DEBUGGING, 0, confidence=0.5),
            _event(WorkEventKind.REFACTORING, 1, confidence=0.5),
        ),
    )
    result = classify_session(profile)
    assert result.category is SessionCategory.DEBUGGING


# ---------------------------------------------------------------------------
# Auto-tag contract for the M2M tag system (suggestion-grade)
# ---------------------------------------------------------------------------


def test_auto_tag_has_distinguishing_prefix() -> None:
    """Auto-tags must be distinguishable from user-authored tags (#817)."""

    profile = _profile(work_events=(_event(WorkEventKind.DEBUGGING),))
    result = classify_session(profile)
    assert result.auto_tag.startswith("auto:")
    assert result.category.value in result.auto_tag


# ---------------------------------------------------------------------------
# Frozen-model invariant (immutability)
# ---------------------------------------------------------------------------


def test_classification_model_is_frozen() -> None:
    profile = _profile(work_events=(_event(WorkEventKind.DEBUGGING),))
    result = classify_session(profile)
    with pytest.raises(ValidationError):
        result.confidence = 0.0
