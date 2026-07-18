"""Known-answer continuity replay through real MCP and focused mutation routes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from devtools.continuity_replay import replay_archive
from polylogue.core.json import JSONDocument, json_document_list, require_json_document
from tests.infra.continuity import ContinuityFixtureSeed, load_continuity_catalog, seed_continuity_archive
from tests.infra.continuity_mutations import continuity_mutation, continuity_mutation_names


@pytest.fixture(scope="module")
def continuity_corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, JSONDocument, ContinuityFixtureSeed]:
    archive_root = tmp_path_factory.mktemp("continuity-replay") / "archive"
    catalog = load_continuity_catalog()
    seed = seed_continuity_archive(archive_root, catalog=catalog)
    return archive_root, catalog, seed


def test_fixture_compiler_plants_corrected_parallel_incident_population(
    continuity_corpus: tuple[Path, JSONDocument, ContinuityFixtureSeed],
) -> None:
    """Direct SQLite census guards the builder independently of query_units."""

    _, _, seed = continuity_corpus
    assert seed.direct_facts["coordinator_children"] == 129
    assert seed.direct_facts["incident_members"] == 91
    assert seed.direct_facts["other_children"] == 38
    assert seed.direct_facts["workflow_invocations"] == 4
    assert seed.direct_facts["incident_curriculum_cases"] == 6
    assert seed.direct_facts["usage_total_tokens"] == 1950


@pytest.mark.asyncio
async def test_all_scenarios_pass_through_official_mcp_stdio_json_rpc(
    continuity_corpus: tuple[Path, JSONDocument, ContinuityFixtureSeed],
) -> None:
    """Exercises SDK discovery/calls, stdio framing, FastMCP, DSL lowering, SQLite, and continuations."""

    archive_root, catalog, _ = continuity_corpus
    report = await replay_archive(archive_root, catalog)

    assert report["transport"] == "mcp-stdio-json-rpc"
    assert report["status"] == "pass"
    assert report["passed"] == 8
    assert report["failed"] == 0
    discovery = require_json_document(report["discovery_receipt"], context="stdio discovery")
    assert discovery["transport"] == "mcp-stdio-json-rpc"
    assert discovery["protocol_version"] != "in-process-registration"
    assert isinstance(discovery["tool_count"], int)
    assert discovery["tool_count"] >= 4

    result_documents = json_document_list(report["results"])
    results = {str(result["scenario"]): result for result in result_documents}
    incident = results["parallel-claude-incident"]
    observed_facts = require_json_document(incident["observed_facts"], context="incident observed facts")
    oracle_catalog = require_json_document(catalog["oracles"], context="continuity oracles")
    incident_oracle = require_json_document(
        oracle_catalog["parallel-claude-incident"],
        context="parallel incident oracle",
    )
    expected_facts = require_json_document(incident_oracle["facts"], context="parallel incident facts")
    assert observed_facts == expected_facts
    assert incident["observed_attempt_grades"] == incident_oracle["attempt_grades"]

    budget = require_json_document(incident["budget"], context="incident budget")
    assert isinstance(budget["observed_calls"], int)
    assert budget["observed_calls"] > 4
    assert budget["cancellation_exercised"] is False
    receipts = json_document_list(incident["route_receipts"])
    member_receipt = next(receipt for receipt in receipts if receipt["step_id"] == "incident-members")
    assert member_receipt["page_count"] == 6
    assert member_receipt["enumerated_item_count"] == 91
    assert member_receipt["unique_identity_count"] == 91
    assert member_receipt["page_total_sum"] == 91
    assert member_receipt["page_totals_match_items"] is True
    count_probe = require_json_document(member_receipt["count_probe"], context="member count probe")
    assert count_probe["selected_rows_exact"] == 91
    assert member_receipt["population_count_verified"] is True
    assert member_receipt["exact_enumeration_verified"] is True
    assert [page["page_total"] for page in json_document_list(member_receipt["pages"])] == [17, 17, 17, 17, 17, 6]
    assert isinstance(member_receipt["query_ref"], str)
    assert isinstance(member_receipt["result_ref"], str)

    cost = results["cost"]
    [cost_receipt] = json_document_list(cost["route_receipts"])
    [cost_page] = json_document_list(cost_receipt["pages"])
    assert cost_page["status"] == "response_budget_exceeded"
    assert cost_page["budget_exceeded"] is True
    assert isinstance(cost_page["response_sha256"], str)
    assert len(cost_page["response_sha256"]) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_name", continuity_mutation_names())
async def test_named_continuity_mutation_curriculum_fails_with_expected_diagnosis(
    continuity_corpus: tuple[Path, JSONDocument, ContinuityFixtureSeed],
    mutation_name: str,
) -> None:
    """Every transport/source/discovery/reasoning mutation named by t8t is executable."""

    archive_root, catalog, _ = continuity_corpus
    mutation = continuity_mutation(mutation_name)
    report = await replay_archive(
        archive_root,
        catalog,
        scenario_names=(mutation.scenario,),
        transport="registered",
        argument_mutator=mutation.argument_mutator,
        response_mutator=mutation.response_mutator,
        discovery_mutator=mutation.discovery_mutator,
    )

    assert report["status"] == "fail"
    [result] = json_document_list(report["results"])
    diagnostics = json_document_list(result["diagnostics"])
    assert any(
        diagnostic["kind"] == mutation.expected_kind and diagnostic["failure_class"] == mutation.expected_failure_class
        for diagnostic in diagnostics
    ), diagnostics


@pytest.mark.asyncio
async def test_dropping_incident_workflow_filter_fails_known_answer_oracle(
    continuity_corpus: tuple[Path, JSONDocument, ContinuityFixtureSeed],
) -> None:
    """Representative production mutation: query_units receives an over-broad member expression."""

    archive_root, catalog, _ = continuity_corpus

    def drop_workflow_filter(tool: str, arguments: dict[str, object], invocation: int) -> dict[str, object]:
        del invocation
        expression = arguments.get("expression")
        if tool == "query_units" and isinstance(expression, str):
            arguments["expression"] = expression.replace(' AND text:"workflow_run:wf_synthetic_841"', "")
        return arguments

    report = await replay_archive(
        archive_root,
        catalog,
        scenario_names=("parallel-claude-incident",),
        transport="registered",
        argument_mutator=drop_workflow_filter,
    )

    assert report["status"] == "fail"
    [result] = json_document_list(report["results"])
    assert result["classification"] == "source_coverage"
    observed_facts = require_json_document(result["observed_facts"], context="mutation observed facts")
    assert observed_facts["attempt_transcripts"] == 129
    diagnostics = json_document_list(result["diagnostics"])
    assert {
        "kind": "fact_mismatch",
        "failure_class": "source_coverage",
        "fact": "attempt_transcripts",
        "expected": 91,
        "observed": 129,
        "source_refs": [
            "fixture:corpus.parallel_incident",
            "bead:polylogue-z9gh.7",
            "bead:polylogue-t8t",
        ],
    } in diagnostics


@pytest.mark.asyncio
async def test_mutated_planted_fact_is_diagnosed_without_changing_route_output(
    continuity_corpus: tuple[Path, JSONDocument, ContinuityFixtureSeed],
) -> None:
    """Representative oracle mutation: a bad planted value identifies expected/observed skew."""

    archive_root, catalog, _ = continuity_corpus
    mutated = deepcopy(catalog)
    oracles = mutated["oracles"]
    assert isinstance(oracles, dict)
    incident = oracles["parallel-claude-incident"]
    assert isinstance(incident, dict)
    facts = incident["facts"]
    assert isinstance(facts, dict)
    facts["attempt_transcripts"] = 92

    report = await replay_archive(
        archive_root,
        mutated,
        scenario_names=("parallel-claude-incident",),
        transport="registered",
    )

    assert report["status"] == "fail"
    [result] = json_document_list(report["results"])
    observed_facts = require_json_document(result["observed_facts"], context="oracle-mutation observed facts")
    assert observed_facts["attempt_transcripts"] == 91
    diagnostics = json_document_list(result["diagnostics"])
    assert {
        "kind": "fact_mismatch",
        "failure_class": "source_coverage",
        "fact": "attempt_transcripts",
        "expected": 92,
        "observed": 91,
        "source_refs": [
            "fixture:corpus.parallel_incident",
            "bead:polylogue-z9gh.7",
            "bead:polylogue-t8t",
        ],
    } in diagnostics
