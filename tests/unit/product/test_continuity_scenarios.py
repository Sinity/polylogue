from __future__ import annotations

from devtools.continuity_replay import evaluate_replay
from polylogue.product.continuity_scenarios import CONTINUITY_SCENARIOS, continuity_scenario


def test_continuity_catalog_contains_all_operator_jobs_and_incident_variant() -> None:
    assert {scenario.scenario_id for scenario in CONTINUITY_SCENARIOS} == {
        "resume",
        "forensic-debug",
        "prior-art",
        "decision",
        "postmortem",
        "cost",
        "self-inspection",
        "parallel-claude-incident",
    }
    assert all(scenario.required_facts for scenario in CONTINUITY_SCENARIOS)
    assert all(scenario.canonical_plan_families for scenario in CONTINUITY_SCENARIOS)


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
