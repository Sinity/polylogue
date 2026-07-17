from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from click.testing import CliRunner

from polylogue.cli.commands.judge import JudgeCandidateRow, _edit_and_accept, judge_command
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.surfaces.payloads import (
    AssertionBulkJudgmentItemPayload,
    AssertionBulkJudgmentPayload,
    AssertionClaimPayload,
    AssertionJudgmentPayload,
    AssertionJudgmentResultPayload,
)


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
    polylogue = SimpleNamespace(judge_assertion_candidates=AsyncMock(return_value=payload))
    env = cast(AppEnv, SimpleNamespace(polylogue=polylogue))

    invocation = CliRunner().invoke(
        judge_command,
        ["--accept", "assertion:candidate-judge-1", "--inject", "--format", "json"],
        obj=env,
        catch_exceptions=False,
    )

    assert invocation.exit_code == 0
    assert json.loads(invocation.output)["applied_count"] == 1
    item = polylogue.judge_assertion_candidates.await_args.kwargs["items"][0]
    assert item.candidate_ref == "assertion:candidate-judge-1"
    assert item.inject is True


def test_judge_edit_preserves_the_candidate_lifecycle_kind() -> None:
    payload = AssertionBulkJudgmentPayload(items=(), applied_count=0, idempotent_count=0, failed_count=0)
    polylogue = SimpleNamespace(judge_assertion_candidates=AsyncMock(return_value=payload))
    env = cast(AppEnv, SimpleNamespace(polylogue=polylogue))
    selected = JudgeCandidateRow(
        assertion_id="candidate-transform-1",
        kind=AssertionKind.TRANSFORM_CANDIDATE.value,
        target_ref="session:judge",
        body="Operator wording for a decision candidate.",
        evidence_refs=(),
    )

    _edit_and_accept(env, selected=selected, edited_body="Edited decision wording.", inject=True)

    item = polylogue.judge_assertion_candidates.await_args.kwargs["items"][0]
    assert item.decision == "supersede"
    assert item.replacement_body_text == "Edited decision wording."
    assert item.replacement_kind is None


def test_judge_accept_all_of_kind_applies_the_real_queue_filters() -> None:
    finding = _candidate()
    decision = finding.model_copy(update={"assertion_id": "candidate-decision-1", "kind": AssertionKind.DECISION})
    payload = AssertionBulkJudgmentPayload(items=(), applied_count=0, idempotent_count=0, failed_count=0)
    polylogue = SimpleNamespace(
        list_assertion_candidates=AsyncMock(return_value=[finding, decision]),
        judge_assertion_candidates=AsyncMock(return_value=payload),
    )
    env = SimpleNamespace(polylogue=polylogue)

    invocation = CliRunner().invoke(
        judge_command,
        ["--accept-all-of-kind", "--kind", "finding", "--since", "1970-01-01", "--format", "json"],
        obj=env,
        catch_exceptions=False,
    )

    assert invocation.exit_code == 0
    assert json.loads(invocation.output)["applied_count"] == 0
    items = polylogue.judge_assertion_candidates.await_args.kwargs["items"]
    assert [item.candidate_ref for item in items] == ["assertion:candidate-judge-1"]
    polylogue.list_assertion_candidates.assert_awaited_once_with(kinds=(AssertionKind.FINDING,), limit=None)
