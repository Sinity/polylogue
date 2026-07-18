from __future__ import annotations

import json
from pathlib import Path

from devtools.continuity_replay import evaluate_replay
from polylogue.product.continuity_scenarios import CONTINUITY_SCENARIOS, continuity_scenario

FIXTURE = json.loads((Path(__file__).parents[2] / "data" / "continuity" / "incident.json").read_text(encoding="utf-8"))


def test_continuity_catalog_contains_all_operator_jobs_and_incident_variant() -> None:
    assert {scenario.scenario_id for scenario in CONTINUITY_SCENARIOS} == {
        "resume",
        "forensic-debug",
        "prior-art",
        "decision",
        "postmortem",
        "cost",
        "self-inspection",
        "mcp-query-transaction",
        "parallel-claude-incident",
    }
    assert all(scenario.required_facts for scenario in CONTINUITY_SCENARIOS)
    assert all(scenario.canonical_plan_families for scenario in CONTINUITY_SCENARIOS)
    assert all(scenario.mutation_cases for scenario in CONTINUITY_SCENARIOS)


def test_oracle_is_independent_and_classifies_transport_failures() -> None:
    fixture = {"resume": ["session:resume-1"]}
    scenario = continuity_scenario("resume")

    assert scenario.classify(fixture, observed_refs=["session:resume-1"], calls=2) == "pass"
    assert scenario.classify(fixture, observed_refs=[], calls=2, observed_failure="execution") == "execution"
    assert scenario.classify(fixture, observed_refs=[], calls=11) == "discovery"


def test_replay_emits_expected_and_observed_refs() -> None:
    result = evaluate_replay(
        "parallel-claude-incident",
        {
            "parallel-claude-incident": [
                "session:coordinator",
                "run:wf-1",
                "commit:abc123",
                "bead:xyz",
            ]
        },
        {
            "refs": ["session:coordinator", "run:wf-1", "commit:abc123", "bead:xyz"],
            "calls": 7,
        },
    )
    assert result["classification"] == "pass"
    assert result["expected_refs"] == result["observed_refs"]


def test_incident_known_answer_preserves_independent_census() -> None:
    result = evaluate_replay(
        "parallel-claude-incident",
        FIXTURE,
        {
            "refs": ["session:coordinator", "run:wf-1", "commit:abc123", "bead:xyz"],
            "calls": 8,
            "answer": {
                "coordinator_session": "cf0c6474-da22-44be-af3e-666037aa5ea4",
                "run_ref": "wf_54d4fb2e-841",
                "workflow_invocations": 4,
                "call_keys": 50,
                "attempt_transcripts": 91,
                "result_records": 65,
                "completed_call_keys": 49,
                "unresolved_call_keys": 1,
                "other_child_sessions": 38,
                "source": "independent incident census",
            },
            "mutations": {
                "lost-continuation-state": "detected",
                "global-delegation-materialization": "detected",
                "wrong-workflow-membership": "detected",
            },
        },
    )

    assert result["classification"] == "pass"
    assert result["expected_answer"] == {
        "coordinator_session": "cf0c6474-da22-44be-af3e-666037aa5ea4",
        "run_ref": "wf_54d4fb2e-841",
        "workflow_invocations": 4,
        "call_keys": 50,
        "attempt_transcripts": 91,
        "result_records": 65,
        "completed_call_keys": 49,
        "unresolved_call_keys": 1,
        "other_child_sessions": 38,
        "source": "independent incident census",
    }


def test_replay_rejects_duplicate_or_nonprogressing_results() -> None:
    scenario = continuity_scenario("resume")
    fixture = {"resume": ["session:resume-1"]}

    assert scenario.classify(fixture, observed_refs=["session:resume-1", "session:resume-1"], calls=2) == "projection"
    assert (
        scenario.classify(
            fixture,
            observed_refs=["session:resume-1"],
            calls=2,
            non_progressing_continuation=True,
        )
        == "execution"
    )
