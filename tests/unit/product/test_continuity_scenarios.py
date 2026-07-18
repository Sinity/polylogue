from __future__ import annotations

from copy import deepcopy

from devtools.continuity_replay import (
    compare_observed_facts,
    grade_incident_attempt,
    materialize_route_arguments,
    project_fact,
)
from polylogue.core.json import require_json_document
from polylogue.product.continuity_scenarios import (
    CONTINUITY_SCENARIOS,
    ContinuityFactProjection,
    continuity_scenario,
)
from polylogue.product.workflows import QUERY_ACTION_WORKFLOW_BY_ID
from polylogue.scenarios import NamedScenarioSource, ScenarioProjectionSourceKind
from tests.infra.archive_scenarios import ScenarioContentBlock
from tests.infra.continuity import load_continuity_catalog

_EXPECTED_SCENARIOS = {
    "resume",
    "forensic-debug",
    "prior-art",
    "decision",
    "postmortem",
    "cost",
    "self-inspection",
    "parallel-claude-incident",
}


def test_continuity_catalog_extends_existing_scenario_source_seam() -> None:
    assert {scenario.scenario_id for scenario in CONTINUITY_SCENARIOS} == _EXPECTED_SCENARIOS
    for scenario in CONTINUITY_SCENARIOS:
        assert isinstance(scenario, NamedScenarioSource)
        assert scenario.projection_source_kind is ScenarioProjectionSourceKind.VALIDATION_LANE
        assert scenario.privacy_level == "synthetic"
        assert scenario.operator_question
        assert scenario.route_steps
        assert scenario.fact_projections
        assert scenario.required_facts == tuple(fact.name for fact in scenario.fact_projections)
        payload = scenario.scenario_payload()
        assert payload["fixture_key"] == scenario.scenario_id
        assert payload["operator_question"] == scenario.operator_question
        assert payload["route_steps"]
        assert payload["fact_projections"]


def test_declarations_use_only_allowed_public_tools_and_existing_workflows() -> None:
    for scenario in CONTINUITY_SCENARIOS:
        route_tools = {step.tool for step in scenario.route_steps}
        assert route_tools == set(scenario.allowed_query_surfaces)
        assert set(scenario.workflow_ids) <= set(QUERY_ACTION_WORKFLOW_BY_ID)
        assert len(scenario.required_facts) == len(set(scenario.required_facts))
        assert scenario.canonical_plan_families
        assert scenario.route_plan_signature in scenario.equivalent_plan_signatures
        assert scenario.discovery_requirements
        assert scenario.coverage_inventory
        assert scenario.result_semantics.target_population
        assert scenario.result_semantics.evidence_contract
        assert scenario.stop_conditions
        assert scenario.failure_taxonomy
        assert scenario.mutation_cases
        for step in scenario.route_steps:
            if step.paginate:
                assert step.item_identity_path
                expression = step.argument_dict().get("expression")
                assert isinstance(expression, str)
                unit = expression.split(maxsplit=1)[0]
                assert step.exact_count_probe is (unit in {"messages", "actions", "assertions", "files"})


def test_independent_manifest_matches_every_declared_fact_inventory() -> None:
    catalog = load_continuity_catalog()
    oracles = catalog["oracles"]
    assert isinstance(oracles, dict)
    assert set(oracles) == _EXPECTED_SCENARIOS
    for scenario in CONTINUITY_SCENARIOS:
        oracle = oracles[scenario.fixture_key]
        assert isinstance(oracle, dict)
        facts = oracle["facts"]
        assert isinstance(facts, dict)
        assert set(facts) == set(scenario.required_facts)
        assert oracle["source_refs"]


def test_parallel_incident_curriculum_keeps_features_and_expected_grades_independent() -> None:
    catalog = load_continuity_catalog()
    corpus = require_json_document(catalog["corpus"], context="continuity corpus")
    incident = require_json_document(corpus["parallel_incident"], context="incident corpus")
    curriculum = incident["attempt_curriculum"]
    assert isinstance(curriculum, list)
    assert len(curriculum) == 6
    for item in curriculum:
        record = require_json_document(item, context="incident curriculum record")
        assert "grade" not in record
        assert "expected" not in record

    oracles = require_json_document(catalog["oracles"], context="continuity oracles")
    oracle = require_json_document(oracles["parallel-claude-incident"], context="incident oracle")
    grades = require_json_document(oracle["attempt_grades"], context="incident attempt grades")
    assert set(grades) == {
        "candidate-list",
        "exact-operator-phrase",
        "sonnet-lexical-proxy",
        "sessions-only-query",
        "correct-topology",
        "correct-delegation",
    }


def test_incident_attempt_grader_matches_t8t_failure_curriculum() -> None:
    assert grade_incident_attempt({"query_shape": "candidate-list", "physical_size": "oversized"}) == (
        "reasonable_oversized"
    )
    assert grade_incident_attempt({"query_shape": "exact-operator-phrase", "corpus_match": "false"}) == (
        "wrong_corpus_assumption"
    )
    assert (
        grade_incident_attempt({"query_shape": "sonnet-lexical-proxy", "structure_discovery": "absent"})
        == "weak_lexical_proxy"
    )
    assert (
        grade_incident_attempt({"query_shape": "sessions-only-query", "shipped_instruction": "advertised"})
        == "product_induced_hidden_grammar"
    )
    assert grade_incident_attempt({"query_shape": "correct-topology", "outcome": "transport-failure"}) == (
        "execution_failure"
    )
    assert grade_incident_attempt({"query_shape": "correct-delegation", "outcome": "timeout"}) == ("execution_failure")
    assert grade_incident_attempt({"query_shape": "sessions-only-query", "shipped_instruction": "absent"}) == (
        "unreasonable_query"
    )


def test_route_templates_are_parameterized_by_the_caller_corpus() -> None:
    catalog = load_continuity_catalog()
    scenario = continuity_scenario("parallel-claude-incident")
    [member_step, *_] = scenario.route_steps
    default_arguments = materialize_route_arguments(member_step, catalog)
    assert default_arguments["expression"] == (
        'messages where text:parallel-child AND text:"workflow_run:wf_synthetic_841"'
    )

    live_catalog = deepcopy(catalog)
    live_corpus = require_json_document(live_catalog["corpus"], context="live corpus")
    live_incident = require_json_document(live_corpus["parallel_incident"], context="live incident")
    live_incident["run_ref"] = "wf_authorized_live_001"
    live_arguments = materialize_route_arguments(member_step, live_catalog)
    assert live_arguments["expression"] == (
        'messages where text:parallel-child AND text:"workflow_run:wf_authorized_live_001"'
    )


def test_fact_projectors_cover_single_unique_count_and_regex_oracles() -> None:
    observations = require_json_document(
        {
            "step": {
                "items": [
                    {"session_id": "s1", "text": "call_key:call-01 result_record:yes"},
                    {"session_id": "s2", "text": "call_key:call-02 result_record:yes"},
                    {"session_id": "s2", "text": "call_key:call-02"},
                ]
            }
        },
        context="projector observations",
    )
    assert (
        project_fact(
            ContinuityFactProjection("sessions", "step", ("items", "*", "session_id"), "unique_count"),
            observations,
        )
        == 2
    )
    assert project_fact(
        ContinuityFactProjection("ids", "step", ("items", "*", "session_id"), "unique_values"),
        observations,
    ) == ["s1", "s2"]
    assert (
        project_fact(
            ContinuityFactProjection(
                "call_keys",
                "step",
                ("items", "*", "text"),
                "regex_unique_count",
                r"call_key:(call-\d{2})",
            ),
            observations,
        )
        == 2
    )
    assert (
        project_fact(
            ContinuityFactProjection(
                "results",
                "step",
                ("items", "*", "text"),
                "regex_count",
                r"result_record:yes",
            ),
            observations,
        )
        == 2
    )


def test_mutated_planted_fact_reports_exact_expected_and_observed_values() -> None:
    diagnostics = compare_observed_facts(
        expected={"attempt_transcripts": 92, "call_keys": 50},
        observed={"attempt_transcripts": 91, "call_keys": 50},
        source_refs=("fixture:corpus.parallel_incident",),
    )
    assert diagnostics == [
        {
            "kind": "fact_mismatch",
            "failure_class": "projection",
            "fact": "attempt_transcripts",
            "expected": 92,
            "observed": 91,
            "source_refs": ["fixture:corpus.parallel_incident"],
        }
    ]


def test_existing_archive_scenario_block_seam_can_plant_structural_failures() -> None:
    payload = ScenarioContentBlock.tool_result(
        "failed",
        tool_name="Bash",
        tool_id="call-1",
        is_error=True,
        exit_code=2,
    ).to_payload()
    assert payload == {
        "type": "tool_result",
        "text": "failed",
        "tool_name": "Bash",
        "tool_id": "call-1",
        "tool_result_is_error": 1,
        "tool_result_exit_code": 2,
    }
