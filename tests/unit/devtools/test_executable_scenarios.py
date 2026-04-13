from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import (
    CorpusSpec,
    ExecutableScenario,
    ScenarioProjectionSourceKind,
    polylogue_execution,
    pytest_execution,
)


@dataclass(frozen=True, kw_only=True)
class _ExecutableFixture(ExecutableScenario):
    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_LANE


def test_executable_scenario_exposes_pytest_targets() -> None:
    scenario = _ExecutableFixture(
        name="machine-contract",
        description="machine contract lane",
        execution=pytest_execution("tests/unit/cli/test_machine_contract.py"),
        origin="authored.validation-lane",
        operation_targets=("cli.json-contract",),
        tags=("contract", "json"),
    )

    projection = scenario.to_projection_entry()

    assert scenario.tests == ("tests/unit/cli/test_machine_contract.py",)
    assert projection.source_kind is ScenarioProjectionSourceKind.VALIDATION_LANE
    assert projection.name == "machine-contract"
    assert projection.operation_targets == ("cli.json-contract",)
    assert projection.tags == ("contract", "json")
    assert projection.source_payload["execution"]["kind"] == "pytest"
    assert projection.source_payload["execution"]["argv"] == ["tests/unit/cli/test_machine_contract.py"]


def test_executable_scenario_projection_payload_preserves_corpus_specs() -> None:
    scenario = _ExecutableFixture(
        name="synthetic-lane",
        description="synthetic contract lane",
        execution=pytest_execution("tests/unit/cli/test_machine_contract.py"),
        corpus_specs=(CorpusSpec.for_provider("chatgpt", count=2),),
        origin="authored.validation-lane",
    )

    projection = scenario.to_projection_entry()

    assert projection.source_payload["corpus_specs"][0]["provider"] == "chatgpt"


def test_executable_scenario_infers_schema_query_metadata_from_polylogue_execution() -> None:
    scenario = _ExecutableFixture(
        name="gen-schema-list",
        description="schema list contract",
        execution=polylogue_execution("schema", "list", "--json"),
    )

    assert scenario.path_targets == ("schema-list-query-loop",)
    assert scenario.artifact_targets == (
        "schema_packages",
        "schema_cluster_manifests",
        "inferred_corpus_specs",
        "inferred_corpus_scenarios",
        "schema_list_results",
    )
    assert scenario.operation_targets == ("query-schema-catalog",)


def test_executable_scenario_infers_schema_explain_metadata_from_polylogue_execution() -> None:
    scenario = _ExecutableFixture(
        name="gen-schema-explain-chatgpt",
        description="schema explain contract",
        execution=polylogue_execution("schema", "explain", "--provider", "chatgpt", "--json"),
    )

    assert scenario.path_targets == ("schema-explain-query-loop",)
    assert scenario.artifact_targets == ("schema_packages", "schema_explanation_results")
    assert scenario.operation_targets == ("query-schema-explanations",)


def test_executable_scenario_infers_run_render_metadata_from_polylogue_execution() -> None:
    scenario = _ExecutableFixture(
        name="run-render-contract",
        description="render contract",
        execution=polylogue_execution("run", "render", "--format", "html"),
    )

    assert scenario.path_targets == ("conversation-render-loop",)
    assert scenario.artifact_targets == (
        "conversation_render_projection",
        "rendered_conversation_artifacts",
    )
    assert scenario.operation_targets == ("render-conversations",)


def test_executable_scenario_infers_run_site_metadata_from_polylogue_execution() -> None:
    scenario = _ExecutableFixture(
        name="run-site-contract",
        description="site contract",
        execution=polylogue_execution("run", "site"),
    )

    assert scenario.path_targets == ("site-publication-loop",)
    assert scenario.artifact_targets == (
        "conversation_render_projection",
        "site_conversation_pages",
        "site_publication_manifest",
        "publication_records",
    )
    assert scenario.operation_targets == ("publish-site",)
