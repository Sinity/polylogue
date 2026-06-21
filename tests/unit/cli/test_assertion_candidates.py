from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from click.testing import CliRunner

from polylogue.cli.query_verbs import mark_candidates_group
from polylogue.core.enums import AssertionKind
from polylogue.surfaces.payloads import AssertionClaimPayload, AssertionJudgmentPayload, AssertionJudgmentResultPayload


def _claim(status: str = "candidate") -> AssertionClaimPayload:
    return AssertionClaimPayload(
        assertion_id="candidate-cli-1",
        target_ref="session:cli",
        kind=AssertionKind.TRANSFORM_CANDIDATE,
        value={"candidate_kind": "decision"},
        body_text="Keep candidate review explicit.",
        evidence_refs=("session:cli",),
        status=status,
        visibility="private",
        context_policy={"inject": False, "promotion_required": True},
        created_at_ms=1,
        updated_at_ms=1,
    )


def test_candidates_list_emits_shared_json_payload() -> None:
    env = SimpleNamespace(polylogue=SimpleNamespace(list_assertion_candidates=AsyncMock(return_value=[_claim()])))

    result = CliRunner().invoke(mark_candidates_group, ["list", "--format", "json"], obj=env, catch_exceptions=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["total"] == 1
    assert payload["statuses"] == ["candidate"]
    assert payload["items"][0]["assertion_id"] == "candidate-cli-1"
    env.polylogue.list_assertion_candidates.assert_awaited_once_with(target_ref=None, limit=50)


def test_candidates_accept_emits_judgment_result_payload() -> None:
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
    payload = AssertionJudgmentResultPayload(
        candidate=_claim(status="accepted"),
        judgment=judgment,
        resulting_assertion=_claim(status="active").model_copy(
            update={
                "assertion_id": "active-cli-1",
                "kind": AssertionKind.DECISION,
                "supersedes": ("assertion:candidate-cli-1",),
            }
        ),
    )
    env = SimpleNamespace(polylogue=SimpleNamespace(judge_assertion_candidate=AsyncMock(return_value=payload)))

    result = CliRunner().invoke(
        mark_candidates_group,
        ["accept", "assertion:candidate-cli-1", "--reason", "confirmed", "--format", "json"],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["judgment"]["decision"] == "accept"
    assert rendered["resulting_assertion"]["kind"] == "decision"
    env.polylogue.judge_assertion_candidate.assert_awaited_once_with(
        candidate_ref="assertion:candidate-cli-1",
        decision="accept",
        reason="confirmed",
        actor_ref="user:local",
        replacement_kind=None,
        replacement_body_text=None,
    )
