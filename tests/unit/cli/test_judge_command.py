from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.judge import JudgeCandidateRow, _edit_and_accept, _render_queue_health, judge_command
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.surfaces.payloads import (
    AssertionBulkJudgmentItemPayload,
    AssertionBulkJudgmentPayload,
    AssertionCandidateQueueHealthPayload,
    AssertionClaimPayload,
    AssertionJudgmentPayload,
    AssertionJudgmentResultPayload,
)


def _judgment_recorder(issued: list[tuple[str, dict[str, object]]], payload: AssertionBulkJudgmentPayload) -> object:
    """Stand in for the daemon's ``mutation.judgment.record`` handler.

    `judge` no longer calls the Python facade: a review batch lowers to a
    declared operation, and the command rebuilds its payload from the recorded
    result. The refs, decision and inject flag that used to be read off the
    facade call are now read off the recorded operation payload's ``reviews``.
    """

    def _served(_config: object, name: str, sent: dict[str, object]) -> dict[str, object]:
        issued.append((name, dict(sent)))
        return {
            "status": "ok",
            "affected_count": payload.applied_count,
            "result": payload.model_dump(mode="json"),
        }

    return _served


def _reviews(issued: list[tuple[str, dict[str, object]]]) -> list[dict[str, object]]:
    assert [name for name, _sent in issued] == ["mutation.judgment.record"]
    sent = issued[0][1]
    assert sent["judgment_kind"] == "assertion-review"
    reviews = sent["reviews"]
    assert isinstance(reviews, list)
    return reviews


def _env() -> AppEnv:
    return cast(AppEnv, SimpleNamespace(polylogue=SimpleNamespace(), config=MagicMock()))


def _candidate() -> AssertionClaimPayload:
    return AssertionClaimPayload(
        assertion_id="candidate-judge-1",
        target_ref="session:judge",
        kind=AssertionKind.FINDING,
        body_text="A finding should await a reviewer.",
        evidence_refs=("session:judge",),
        status=AssertionStatus.CANDIDATE,
        visibility=AssertionVisibility.PRIVATE,
        context_policy={"inject": False, "promotion_required": True},
        created_at_ms=1,
        updated_at_ms=1,
    )


def test_judge_mutation_requires_candidate_ref() -> None:
    """A decision option cannot be parsed without its explicit candidate ref."""

    invocation = CliRunner().invoke(judge_command, ["--accept"])

    assert invocation.exit_code == 2
    assert "requires an argument" in invocation.output


def test_queue_health_text_does_not_invent_legacy_receipt_counters(capsys: pytest.CaptureFixture[str]) -> None:
    payload = AssertionCandidateQueueHealthPayload(
        state="pending",
        observed_at_ms=1,
        pending_count=1,
        judgment_scheduler_receipt_status="completed",
        judgment_scheduler_receipt_reason="legacy_receipt",
    )

    _render_queue_health(payload, "text")

    output = capsys.readouterr().out
    assert "reason=legacy_receipt" not in output
    assert "receipt counts:" not in output


def test_queue_health_text_renders_typed_receipt_persistence_flags(capsys: pytest.CaptureFixture[str]) -> None:
    payload = AssertionCandidateQueueHealthPayload(
        state="scheduler-stalled",
        observed_at_ms=1,
        pending_count=1,
        judgment_scheduler_receipt_status="failed",
        judgment_scheduler_receipt_retryable=True,
        judgment_scheduler_receipt_retry_route="next tick",
        judgment_scheduler_receipt_batch_limit=10,
        judgment_scheduler_receipt_considered=2,
        judgment_scheduler_receipt_failed=2,
        judgment_scheduler_receipt_persistence_degraded=True,
        judgment_scheduler_receipt_persistence_recovered=False,
    )

    _render_queue_health(payload, "text")

    output = capsys.readouterr().out
    assert "degraded=True; recovered=False" in output
    assert "considered=2" in output


def test_judge_injection_requires_explicit_flag() -> None:
    """Acceptance defaults to non-injecting unless the operator opts in."""

    payload = AssertionBulkJudgmentPayload(items=(), applied_count=0, idempotent_count=0, failed_count=0)
    issued: list[tuple[str, dict[str, object]]] = []

    with patch(
        "polylogue.cli.archive_query._submit_mutation_operation",
        side_effect=_judgment_recorder(issued, payload),
    ):
        invocation = CliRunner().invoke(
            judge_command,
            ["--accept", "assertion:candidate-judge-1", "--format", "json"],
            obj=_env(),
            catch_exceptions=False,
        )

    assert invocation.exit_code == 0
    assert _reviews(issued)[0]["inject"] is False


def test_judge_noninteractive_accept_uses_bulk_lifecycle_payload() -> None:
    candidate = _candidate()
    result = AssertionJudgmentResultPayload(
        candidate=candidate.model_copy(update={"status": AssertionStatus.ACCEPTED}),
        judgment=AssertionJudgmentPayload(
            judgment_id="judgment-1",
            candidate_ref="assertion:candidate-judge-1",
            decision="accept",
            decided_at_ms=2,
            resulting_assertion_ref="assertion:active-judge-1",
        ),
        resulting_assertion=candidate.model_copy(
            update={"assertion_id": "active-judge-1", "status": AssertionStatus.ACTIVE}
        ),
    )
    payload = AssertionBulkJudgmentPayload(
        items=(
            AssertionBulkJudgmentItemPayload(
                candidate_ref="assertion:candidate-judge-1", outcome="applied", result=result
            ),
        ),
        applied_count=1,
        idempotent_count=0,
        failed_count=0,
    )
    issued: list[tuple[str, dict[str, object]]] = []

    with patch(
        "polylogue.cli.archive_query._submit_mutation_operation",
        side_effect=_judgment_recorder(issued, payload),
    ):
        invocation = CliRunner().invoke(
            judge_command,
            ["--accept", "assertion:candidate-judge-1", "--inject", "--format", "json"],
            obj=_env(),
            catch_exceptions=False,
        )

    assert invocation.exit_code == 0
    assert json.loads(invocation.output)["applied_count"] == 1
    review = _reviews(issued)[0]
    assert review["candidate_ref"] == "assertion:candidate-judge-1"
    assert review["inject"] is True


def test_judge_edit_preserves_the_candidate_lifecycle_kind() -> None:
    payload = AssertionBulkJudgmentPayload(items=(), applied_count=0, idempotent_count=0, failed_count=0)
    issued: list[tuple[str, dict[str, object]]] = []
    env = _env()
    selected = JudgeCandidateRow(
        assertion_id="candidate-transform-1",
        kind=AssertionKind.TRANSFORM_CANDIDATE.value,
        target_ref="session:judge",
        body="Operator wording for a decision candidate.",
        evidence_refs=(),
    )

    with patch(
        "polylogue.cli.archive_query._submit_mutation_operation",
        side_effect=_judgment_recorder(issued, payload),
    ):
        _edit_and_accept(env, selected=selected, edited_body="Edited decision wording.", inject=True)

    review = _reviews(issued)[0]
    assert review["decision"] == "supersede"
    assert review["replacement_body_text"] == "Edited decision wording."
    assert review["replacement_kind"] is None


def test_judge_accept_all_of_kind_applies_the_real_queue_filters() -> None:
    finding = _candidate()
    payload = AssertionBulkJudgmentPayload(items=(), applied_count=0, idempotent_count=0, failed_count=0)
    from polylogue.surfaces.action_affordances import assertion_candidate_review_affordances
    from polylogue.surfaces.payloads import AssertionCandidateReviewItemPayload, AssertionCandidateReviewListPayload

    review_item = AssertionCandidateReviewItemPayload(
        candidate_ref="assertion:candidate-judge-1",
        review_status="pending",
        candidate=finding,
        action_affordances=assertion_candidate_review_affordances(candidate_ref="assertion:candidate-judge-1"),
    )
    review_payload = AssertionCandidateReviewListPayload(
        items=(review_item,),
        total=1,
        limit=1,
        candidate_statuses=(AssertionStatus.CANDIDATE,),
    )
    polylogue = SimpleNamespace(list_assertion_candidate_reviews=AsyncMock(return_value=review_payload))
    env = SimpleNamespace(polylogue=polylogue, config=MagicMock())
    issued: list[tuple[str, dict[str, object]]] = []

    with patch(
        "polylogue.cli.archive_query._submit_mutation_operation",
        side_effect=_judgment_recorder(issued, payload),
    ):
        invocation = CliRunner().invoke(
            judge_command,
            ["--accept-all-of-kind", "--kind", "finding", "--since", "1970-01-01", "--format", "json"],
            obj=env,
            catch_exceptions=False,
        )

    assert invocation.exit_code == 0
    assert json.loads(invocation.output)["applied_count"] == 0
    assert [review["candidate_ref"] for review in _reviews(issued)] == ["assertion:candidate-judge-1"]
    polylogue.list_assertion_candidate_reviews.assert_awaited_once_with(
        target_ref=None,
        kinds=(AssertionKind.FINDING,),
        statuses=(AssertionStatus.CANDIDATE,),
        limit=None,
    )
