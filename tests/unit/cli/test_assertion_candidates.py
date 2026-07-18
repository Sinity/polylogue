from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from click.testing import CliRunner

from polylogue.cli.commands.judge import judge_command
from polylogue.cli.query_verbs import mark_verb
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.surfaces.action_affordances import CandidateReviewDecision, assertion_candidate_review_affordances
from polylogue.surfaces.payloads import (
    AssertionBulkJudgmentItemPayload,
    AssertionBulkJudgmentPayload,
    AssertionCandidateReviewItemPayload,
    AssertionCandidateReviewListPayload,
    AssertionCandidateReviewStatus,
    AssertionClaimPayload,
    AssertionEvidencePreviewPayload,
    AssertionJudgmentPayload,
    AssertionJudgmentResultPayload,
)


def _claim(status: AssertionStatus = AssertionStatus.CANDIDATE) -> AssertionClaimPayload:
    return AssertionClaimPayload(
        assertion_id="candidate-cli-1",
        target_ref="session:cli",
        kind=AssertionKind.TRANSFORM_CANDIDATE,
        value={"candidate_kind": "decision"},
        body_text="Keep candidate review explicit.",
        evidence_refs=("session:cli",),
        status=status,
        visibility=AssertionVisibility.PRIVATE,
        context_policy={"inject": False, "promotion_required": True},
        created_at_ms=1,
        updated_at_ms=1,
    )


def _review_item(status: AssertionStatus = AssertionStatus.CANDIDATE) -> AssertionCandidateReviewItemPayload:
    review_status_by_assertion_status: dict[AssertionStatus, AssertionCandidateReviewStatus] = {
        AssertionStatus.CANDIDATE: "pending",
        AssertionStatus.ACCEPTED: "accepted",
        AssertionStatus.REJECTED: "rejected",
        AssertionStatus.DEFERRED: "deferred",
        AssertionStatus.SUPERSEDED: "superseded",
    }
    review_status = review_status_by_assertion_status[status]
    disabled: dict[CandidateReviewDecision, str] | None = (
        None
        if status is AssertionStatus.CANDIDATE
        else {
            "accept": f"candidate_{review_status}",
            "reject": f"candidate_{review_status}",
            "defer": f"candidate_{review_status}",
            "supersede": f"candidate_{review_status}",
        }
    )
    return AssertionCandidateReviewItemPayload(
        candidate_ref="assertion:candidate-cli-1",
        review_status=review_status,
        candidate=_claim(status=status),
        claim_summary="Keep candidate review explicit.",
        source_ref="agent:producer",
        source_kind="agent",
        age_ms=2_000,
        evidence_previews=(
            AssertionEvidencePreviewPayload(
                ref="session:cli",
                state="resolved",
                kind="session",
                title="CLI evidence session",
                excerpt="Evidence supporting the candidate.",
                open_commands=("polylogue read session:cli",),
            ),
        ),
        evidence_total_count=1,
        evidence_resolution="resolved",
        action_affordances=assertion_candidate_review_affordances(
            candidate_ref="assertion:candidate-cli-1",
            disabled_reasons=disabled,
        ),
    )


def test_candidates_list_emits_shared_json_payload() -> None:
    payload = AssertionCandidateReviewListPayload(
        items=(_review_item(),),
        total=1,
        limit=50,
        candidate_statuses=(AssertionStatus.CANDIDATE,),
    )
    env = SimpleNamespace(polylogue=SimpleNamespace(list_assertion_candidate_reviews=AsyncMock(return_value=payload)))

    result = CliRunner().invoke(judge_command, ["--list", "--format", "json"], obj=env, catch_exceptions=False)

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["mode"] == "assertion-candidate-review-list"
    assert rendered["items"][0]["claim_summary"] == "Keep candidate review explicit."
    assert rendered["items"][0]["evidence_previews"][0]["state"] == "resolved"
    assert rendered["items"][0]["evidence_previews"][0]["open_commands"] == ["polylogue read session:cli"]
    env.polylogue.list_assertion_candidate_reviews.assert_awaited_once_with(
        target_ref=None,
        kinds=None,
        statuses=(AssertionStatus.CANDIDATE,),
        limit=50,
    )


def test_candidates_review_emits_status_and_disabled_action_reasons() -> None:
    item = _review_item(AssertionStatus.DEFERRED)
    payload = AssertionCandidateReviewListPayload(
        items=(item,),
        total=1,
        limit=50,
        target_ref=None,
        candidate_statuses=(AssertionStatus.CANDIDATE, AssertionStatus.DEFERRED),
    )
    env = SimpleNamespace(polylogue=SimpleNamespace(list_assertion_candidate_reviews=AsyncMock(return_value=payload)))

    result = CliRunner().invoke(judge_command, ["--review", "--format", "json"], obj=env, catch_exceptions=False)

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["durable_assertions_excluded"] is True
    assert rendered["items"][0]["review_status"] == "deferred"
    disabled = {
        action["id"]: action["availability"]["disabled_reason"] for action in rendered["items"][0]["action_affordances"]
    }
    assert disabled == {
        "assertion_candidate.accept": "candidate_deferred",
        "assertion_candidate.reject": "candidate_deferred",
        "assertion_candidate.defer": "candidate_deferred",
        "assertion_candidate.supersede": "candidate_deferred",
    }


def test_candidates_accept_emits_bulk_judgment_payload() -> None:
    judgment = AssertionJudgmentPayload(
        judgment_id="judgment-cli-1",
        candidate_ref="assertion:candidate-cli-1",
        decision="accept",
        reason="confirmed",
        actor_ref="user:local",
        decided_at_ms=2,
        resulting_assertion_ref="assertion:active-cli-1",
        evidence_refs=("session:cli", "assertion:candidate-cli-1"),
    )
    result_payload = AssertionJudgmentResultPayload(
        candidate=_claim(status=AssertionStatus.ACCEPTED),
        judgment=judgment,
        resulting_assertion=_claim(status=AssertionStatus.ACTIVE).model_copy(
            update={
                "assertion_id": "active-cli-1",
                "kind": AssertionKind.DECISION,
                "supersedes": ("assertion:candidate-cli-1",),
            }
        ),
    )
    payload = AssertionBulkJudgmentPayload(
        items=(
            AssertionBulkJudgmentItemPayload(
                candidate_ref="assertion:candidate-cli-1", outcome="applied", result=result_payload
            ),
        ),
        applied_count=1,
        idempotent_count=0,
        failed_count=0,
    )
    env = SimpleNamespace(polylogue=SimpleNamespace(judge_assertion_candidates=AsyncMock(return_value=payload)))

    result = CliRunner().invoke(
        judge_command,
        ["--accept", "assertion:candidate-cli-1", "--reason", "confirmed", "--format", "json"],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["items"][0]["result"]["judgment"]["decision"] == "accept"
    call = env.polylogue.judge_assertion_candidates.await_args.kwargs
    assert call["items"][0].candidate_ref == "assertion:candidate-cli-1"
    assert call["items"][0].inject is False


def test_mark_candidates_public_group_is_retired() -> None:
    """The query-first mark workflow no longer registers a second lifecycle."""

    assert "candidates" not in mark_verb.commands
