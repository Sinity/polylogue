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


def _dict_payload(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _dict_list_payload(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    payloads: list[dict[str, object]] = []
    for item in value:
        payloads.append(_dict_payload(item))
    return payloads


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
    execution_payload = _dict_payload(projection.source_payload["execution"])

    assert scenario.tests == ("tests/unit/cli/test_machine_contract.py",)
    assert projection.source_kind is ScenarioProjectionSourceKind.VALIDATION_LANE
    assert projection.name == "machine-contract"
    assert projection.operation_targets == ("cli.json-contract",)
    assert projection.tags == ("contract", "json")
    assert execution_payload["kind"] == "pytest"
    assert execution_payload["argv"] == ["tests/unit/cli/test_machine_contract.py"]


def test_executable_scenario_projection_payload_preserves_corpus_specs() -> None:
    scenario = _ExecutableFixture(
        name="synthetic-lane",
        description="synthetic contract lane",
        execution=pytest_execution("tests/unit/cli/test_machine_contract.py"),
        corpus_specs=(CorpusSpec.for_provider("chatgpt", count=2),),
        origin="authored.validation-lane",
    )

    projection = scenario.to_projection_entry()
    corpus_specs = _dict_list_payload(projection.source_payload["corpus_specs"])

    assert corpus_specs[0]["provider"] == "chatgpt"


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


def test_executable_scenario_infers_run_acquire_metadata_from_polylogue_execution() -> None:
    scenario = _ExecutableFixture(
        name="run-acquire-contract",
        description="acquire contract",
        execution=polylogue_execution("run", "acquire"),
    )

    assert scenario.path_targets == ("source-acquisition-loop",)
    assert scenario.artifact_targets == (
        "configured_sources",
        "source_payload_stream",
        "raw_validation_state",
        "artifact_observation_rows",
    )
    assert scenario.operation_targets == ("acquire-raw-conversations",)


def test_executable_scenario_infers_run_parse_metadata_from_polylogue_execution() -> None:
    scenario = _ExecutableFixture(
        name="run-parse-contract",
        description="parse contract",
        execution=polylogue_execution("run", "parse"),
    )

    assert scenario.path_targets == (
        "source-acquisition-loop",
        "raw-reparse-loop",
        "raw-archive-ingest-loop",
    )
    assert scenario.artifact_targets == (
        "configured_sources",
        "source_payload_stream",
        "raw_validation_state",
        "artifact_observation_rows",
        "validation_backlog",
        "parse_backlog",
        "parse_quarantine",
        "archive_conversation_rows",
    )
    assert scenario.operation_targets == (
        "acquire-raw-conversations",
        "plan-validation-backlog",
        "plan-parse-backlog",
        "ingest-archive-runtime",
    )


def test_executable_scenario_infers_run_embed_metadata_from_polylogue_execution() -> None:
    scenario = _ExecutableFixture(
        name="run-embed-contract",
        description="embed contract",
        execution=polylogue_execution("run", "embed", "--limit", "5"),
    )

    assert scenario.path_targets == ("embedding-materialization-loop",)
    assert scenario.artifact_targets == (
        "archive_conversation_rows",
        "embedding_metadata_rows",
        "embedding_status_rows",
        "message_embedding_vectors",
    )
    assert scenario.operation_targets == ("materialize-transcript-embeddings",)


def test_executable_scenario_infers_embed_stats_metadata_from_polylogue_execution() -> None:
    scenario = _ExecutableFixture(
        name="embed-stats-contract",
        description="embed stats contract",
        execution=polylogue_execution("embed", "--stats", "--json"),
    )

    assert scenario.path_targets == ("retrieval-band-readiness-loop", "embedding-status-query-loop")
    assert scenario.artifact_targets == (
        "embedding_metadata_rows",
        "embedding_status_rows",
        "message_embedding_vectors",
        "action_event_readiness",
        "session_product_readiness",
        "retrieval_band_readiness",
        "embedding_status_results",
    )
    assert scenario.operation_targets == (
        "project-retrieval-band-readiness",
        "query-embedding-status",
        "cli.json-contract",
    )


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
