"""Unit coverage for the cold, wire-only continuity model lane.

Anti-vacuity conditions named per test.  The common bypass this module is
built to catch: a lane that accepts any model answer, or that quietly falls
back to an in-process registry when the wire misbehaves, would stay green
while certifying a route no cold client can use.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from devtools.continuity_cold_model import (
    ColdModelLaneError,
    ColdModelPlan,
    ColdModelPlanStep,
    ColdModelVariancePolicy,
    HTTPColdModelBackend,
    ScriptedColdModelBackend,
    WireDiscoveryCapture,
    build_cold_prompt,
    grade_execution,
    grade_formulation,
    parse_cold_model_plan,
    reconcile_registry_coverage,
)
from devtools.continuity_scenarios import CONTINUITY_SCENARIOS, continuity_scenario

_MODULE = Path(__file__).resolve().parents[3] / "devtools" / "continuity_cold_model.py"
_AUTHOR_PLANS = Path(__file__).resolve().parents[2] / "data" / "continuity" / "author-recorded-plans.json"


def _capture(tools: tuple[str, ...] = ("query", "status", "explain", "read", "get", "context")) -> WireDiscoveryCapture:
    return WireDiscoveryCapture(
        transport_name="mcp-stdio-json-rpc",
        protocol_version="2025-11-25",
        tools=tools,
        tool_schemas={name: {"input_schema": {}} for name in tools},
        examples=({"key": "example-a"},),
        hops=2,
        schema_digest="schema",
        catalog_digest="catalog",
    )


class TestPlanContract:
    """Parsing is strict; a junk answer is a named formulation failure."""

    def test_valid_plan_round_trips(self) -> None:
        plan = parse_cold_model_plan(
            json.dumps(
                {
                    "steps": [{"tool": "query", "arguments": {"expression": "messages where x", "limit": 2}}],
                    "stop_conditions": ["no continuation"],
                    "citation_fields": ["message_id"],
                    "uncertainty": "medium",
                }
            )
        )
        assert plan.plan_signature == ("query:messages",)
        assert plan.tools == ("query",)

    @pytest.mark.parametrize(
        ("answer", "kind"),
        [
            pytest.param("not json at all", "plan_unparsable", id="junk"),
            pytest.param("[]", "plan_not_an_object", id="list"),
            pytest.param('{"steps": [], "uncertainty": "low"}', "plan_without_steps", id="nosteps"),
            pytest.param('{"steps": [{"tool": "sql"}], "uncertainty": "low"}', "plan_unknown_tool", id="tool"),
            pytest.param(
                '{"steps": [{"tool": "query"}], "uncertainty": "certain"}',
                "plan_uncertainty_invalid",
                id="unc",
            ),
            pytest.param(
                '{"steps": [{"tool": "query"}], "uncertainty": "low", "stop_conditions": "x"}',
                "plan_field_not_a_list",
                id="stops",
            ),
        ],
    )
    def test_bad_answer_is_named(self, answer: str, kind: str) -> None:
        """Anti-vacuity: were parsing lenient, an empty plan would pass every axis."""
        with pytest.raises(ColdModelLaneError) as excinfo:
            parse_cold_model_plan(answer)
        assert excinfo.value.kind == kind
        assert excinfo.value.axis == "formulation"

    def test_fenced_json_is_accepted(self) -> None:
        plan = parse_cold_model_plan(
            '```json\n{"steps": [{"tool": "status", "arguments": {}}], "uncertainty": "low"}\n```'
        )
        assert plan.tools == ("status",)


def test_author_recorded_plans_remain_synthetic_parseable_fixtures() -> None:
    """Keep the recorded plans usable without presenting them as model output."""
    fixture = json.loads(_AUTHOR_PLANS.read_text(encoding="utf-8"))
    assert fixture["family"] == "synthetic"
    assert fixture["generation"]["kind"] == "author-written"
    assert fixture["generation"]["model_evidence"] is False
    assert set(fixture["plans"]) == {scenario.scenario_id for scenario in CONTINUITY_SCENARIOS}
    for answer in fixture["plans"].values():
        assert parse_cold_model_plan(json.dumps(answer)).steps


class TestColdness:
    """The prompt carries sparse wording and public discovery, nothing else."""

    def test_prompt_excludes_the_answer_key(self) -> None:
        """Anti-vacuity: a prompt leaking route steps makes the lane self-fulfilling."""
        scenario = continuity_scenario("resume")
        prompt = build_cold_prompt(scenario.sparse_prompt, _capture(), max_calls=scenario.budget.max_calls)
        assert scenario.sparse_prompt in prompt
        assert scenario.fixture_key not in prompt
        for step in scenario.route_steps:
            assert step.step_id not in prompt
            assert step.plan_atom not in prompt
        for signature in scenario.equivalent_plan_signatures:
            for atom in signature:
                assert atom not in prompt

    def test_no_registry_catalog_import(self) -> None:
        """Anti-vacuity: importing QUERY_DISCOVERY_EXAMPLES restores the deleted fallback."""
        tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert "QUERY_DISCOVERY_EXAMPLES" not in imported
        assert not any(name.startswith("polylogue.archive.query.discovery") for name in imported)


class TestGrading:
    """Axes are graded independently and only after the plan is committed."""

    def test_declared_plan_passes_every_axis(self) -> None:
        scenario = continuity_scenario("resume")
        plan = ColdModelPlan(
            steps=tuple(ColdModelPlanStep(step.tool, dict(step.argument_dict())) for step in scenario.route_steps),
            stop_conditions=("no continuation",),
            citation_fields=("message_id",),
            uncertainty="medium",
        )
        grades = {grade.axis: grade.status for grade in grade_formulation(scenario, plan, _capture())}
        assert grades == {
            "discovery": "pass",
            "formulation": "pass",
            "plan_equivalence": "pass",
            "citations": "pass",
            "uncertainty": "pass",
            "stop_conditions": "pass",
        }

    def test_off_route_plan_fails_equiv(self) -> None:
        """Anti-vacuity: a lane that passes any plan grades nothing."""
        scenario = continuity_scenario("resume")
        plan = ColdModelPlan(
            steps=(ColdModelPlanStep("query", {"expression": "sessions where y"}),),
            stop_conditions=("stop",),
            citation_fields=("message_id",),
            uncertainty="low",
        )
        grades = {grade.axis: grade.status for grade in grade_formulation(scenario, plan, _capture())}
        assert grades["plan_equivalence"] == "fail"
        assert grades["formulation"] == "pass"

    def test_disallowed_surface_fails_formulation(self) -> None:
        scenario = continuity_scenario("resume")
        plan = ColdModelPlan(
            steps=(ColdModelPlanStep("get", {"id": "x"}),),
            stop_conditions=("stop",),
            citation_fields=("message_id",),
            uncertainty="low",
        )
        grades = {grade.axis: grade.status for grade in grade_formulation(scenario, plan, _capture())}
        assert grades["formulation"] == "fail"

    def test_missing_wire_tool_fails_discovery(self) -> None:
        scenario = continuity_scenario("resume")
        plan = ColdModelPlan(
            steps=(ColdModelPlanStep("query", {"expression": "messages where x"}),),
            stop_conditions=("stop",),
            citation_fields=("message_id",),
            uncertainty="low",
        )
        grades = {grade.axis: grade.status for grade in grade_formulation(scenario, plan, _capture(("status",)))}
        assert grades["discovery"] == "fail"

    def test_execution_axes_from_route(self) -> None:
        passing = grade_execution({"status": "pass", "diagnostics": []})
        assert {grade.axis: grade.status for grade in passing} == {"execution": "pass", "projection": "pass"}
        failing = grade_execution(
            {"status": "fail", "diagnostics": [{"failure_class": "source_coverage", "kind": "fact_mismatch"}]}
        )
        assert {grade.axis: grade.status for grade in failing} == {"execution": "fail", "projection": "fail"}


class TestRegistryCoverage:
    """Missing, duplicate and stale scenarios are failures, not skips."""

    def test_full_registry_reconciles(self) -> None:
        ids = [scenario.scenario_id for scenario in CONTINUITY_SCENARIOS]
        assert reconcile_registry_coverage(ids) == ()

    def test_missing_scenario_reported(self) -> None:
        """Anti-vacuity: a silently-skipped scenario would otherwise read as green."""
        ids = [scenario.scenario_id for scenario in CONTINUITY_SCENARIOS][1:]
        errors = reconcile_registry_coverage(ids)
        assert len(errors) == 1
        assert "no result" in errors[0]

    def test_duplicate_and_stale(self) -> None:
        ids = [scenario.scenario_id for scenario in CONTINUITY_SCENARIOS]
        errors = reconcile_registry_coverage([*ids, ids[0], "invented-scenario"])
        assert any("duplicate" in error for error in errors)
        assert any("stale" in error for error in errors)


class TestVariancePolicy:
    """A failed attempt remains a failure; the lane never retries silently."""

    def test_product_failure_not_variance(self) -> None:
        policy = ColdModelVariancePolicy()
        assert policy.disposition(passes=1, product_failure=True) == "fail"

    def test_single_attempt_outcomes(self) -> None:
        policy = ColdModelVariancePolicy()
        assert policy.disposition(passes=1, product_failure=False) == "pass"
        assert policy.disposition(passes=0, product_failure=False) == "fail"

    def test_retry_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ColdModelVariancePolicy(attempts=2, required_passes=1)


class TestScriptedBackend:
    """The hermetic backend refuses to invent an answer it was not given."""

    def test_unknown_scenario_raises(self) -> None:
        backend = ScriptedColdModelBackend(answers={"known question": "{}"})
        assert backend.identity.family == "scripted"
        assert backend.identity.model == "synthetic-test-plan"
        prompt = build_cold_prompt("unknown question", _capture(), max_calls=4)
        with pytest.raises(ColdModelLaneError) as excinfo:
            backend.answer(prompt)
        assert excinfo.value.kind == "scripted_answer_missing"


class TestResponsesBackend:
    def test_no_tools_request_and_native_receipt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        requests: list[dict[str, object]] = []

        def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float) -> httpx.Response:
            requests.append({"url": url, "body": json, "headers": headers, "timeout": timeout})
            return httpx.Response(
                200,
                json={
                    "id": "resp-synthetic",
                    "model": "gpt-6-sol",
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
                    "usage": {"input_tokens": 12, "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 1}},
                },
            )

        monkeypatch.setattr(httpx, "post", fake_post)
        backend = HTTPColdModelBackend("openai-responses", "gpt-6-sol", "https://api.openai.com/v1", "test")
        answer = backend.answer("synthetic prompt")

        assert requests[0]["url"] == "https://api.openai.com/v1/responses"
        body = requests[0]["body"]
        assert isinstance(body, dict)
        assert body["model"] == "gpt-6-sol"
        assert "tools" not in body
        assert body["store"] is False
        assert answer.reported_model == "gpt-6-sol"
        assert answer.native_token_counts["output_tokens_details"] == {"reasoning_tokens": 1}

    def test_incomplete_response_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        monkeypatch.setattr(
            httpx,
            "post",
            lambda *args, **kwargs: httpx.Response(200, json={"status": "incomplete", "output": []}),
        )
        backend = HTTPColdModelBackend("openai-responses", "gpt-6-sol", "https://api.openai.com/v1", "test")
        with pytest.raises(ColdModelLaneError) as excinfo:
            backend.answer("synthetic prompt")
        assert excinfo.value.kind == "backend_answer_incomplete"
