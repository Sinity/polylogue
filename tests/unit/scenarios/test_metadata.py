from __future__ import annotations

from unittest.mock import MagicMock

from polylogue.scenarios import (
    ScenarioMetadata,
    runtime_artifact_graph,
    runtime_artifact_target_names,
    runtime_operation_target_names,
)


def test_scenario_metadata_from_payload_normalizes_strings_and_targets() -> None:
    metadata = ScenarioMetadata.from_payload(
        {
            "origin": "generated.contract",
            "artifact_targets": ["doctor_runtime", "message_fts"],
            "operation_targets": ("cli.doctor",),
            "tags": ["generated", "json-contract"],
        }
    )

    assert metadata.origin == "generated.contract"
    assert metadata.artifact_targets == ("doctor_runtime", "message_fts")
    assert metadata.operation_targets == ("cli.doctor",)
    assert metadata.tags == ("generated", "json-contract")


def test_scenario_metadata_from_object_falls_back_for_mock_attributes() -> None:
    obj = MagicMock()

    metadata = ScenarioMetadata.from_object(obj)

    assert metadata.origin == "authored"
    assert metadata.artifact_targets == ()
    assert metadata.operation_targets == ()
    assert metadata.tags == ()


def test_scenario_metadata_payload_omits_empty_collections() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        artifact_targets=("doctor_runtime",),
        operation_targets=(),
        tags=("generated",),
    )

    assert metadata.to_payload() == {
        "origin": "generated.contract",
        "artifact_targets": ["doctor_runtime"],
        "tags": ["generated"],
    }


def test_runtime_target_names_include_declared_runtime_specs() -> None:
    assert "action_event_rows" in runtime_artifact_target_names()
    assert "project-action-event-health" in runtime_operation_target_names()


def test_scenario_metadata_resolves_only_runtime_declared_targets() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        artifact_targets=("action_event_rows", "message_fts"),
        operation_targets=("project-action-event-health", "benchmark.storage.crud"),
    )

    assert metadata.runtime_artifact_targets() == ("action_event_rows",)
    assert metadata.runtime_operation_targets() == ("project-action-event-health",)


def test_runtime_artifact_graph_exposes_resolved_specs() -> None:
    graph = runtime_artifact_graph()

    assert "action_event_rows" in graph.artifact_names()
    assert "materialize-session-products" in graph.operation_names()


def test_scenario_metadata_resolves_runtime_specs() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        artifact_targets=("action_event_rows", "message_fts"),
        operation_targets=("project-action-event-health", "benchmark.storage.crud"),
    )

    assert tuple(artifact.name for artifact in metadata.resolve_runtime_artifacts()) == ("action_event_rows",)
    assert tuple(operation.name for operation in metadata.resolve_runtime_operations()) == (
        "project-action-event-health",
    )
