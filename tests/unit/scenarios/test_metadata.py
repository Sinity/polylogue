from __future__ import annotations

from unittest.mock import MagicMock

from polylogue.scenarios import (
    ScenarioMetadata,
    declared_operation_target_names,
    runtime_artifact_graph,
    runtime_artifact_target_names,
    runtime_maintenance_target_names,
    runtime_operation_target_names,
    runtime_path_target_names,
)


def test_scenario_metadata_from_payload_normalizes_strings_and_targets() -> None:
    metadata = ScenarioMetadata.from_payload(
        {
            "origin": "generated.contract",
            "path_targets": ["action-event-repair-loop"],
            "artifact_targets": ["doctor_runtime", "message_fts"],
            "operation_targets": ("cli.doctor",),
            "maintenance_targets": ["dangling_fts"],
            "tags": ["generated", "json-contract"],
        }
    )

    assert metadata.origin == "generated.contract"
    assert metadata.path_targets == ("action-event-repair-loop",)
    assert metadata.artifact_targets == ("doctor_runtime", "message_fts")
    assert metadata.operation_targets == ("cli.doctor",)
    assert metadata.maintenance_targets == ("dangling_fts",)
    assert metadata.tags == ("generated", "json-contract")


def test_scenario_metadata_from_object_falls_back_for_mock_attributes() -> None:
    obj = MagicMock()

    metadata = ScenarioMetadata.from_object(obj)

    assert metadata.origin == "authored"
    assert metadata.artifact_targets == ()
    assert metadata.operation_targets == ()
    assert metadata.maintenance_targets == ()
    assert metadata.tags == ()


def test_scenario_metadata_payload_omits_empty_collections() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        path_targets=("raw-reparse-loop",),
        artifact_targets=("doctor_runtime",),
        operation_targets=(),
        maintenance_targets=("session_products",),
        tags=("generated",),
    )

    assert metadata.to_payload() == {
        "origin": "generated.contract",
        "path_targets": ["raw-reparse-loop"],
        "artifact_targets": ["doctor_runtime"],
        "maintenance_targets": ["session_products"],
        "tags": ["generated"],
    }


def test_runtime_target_names_include_declared_runtime_specs() -> None:
    assert "action_event_rows" in runtime_artifact_target_names()
    assert "project-action-event-health" in runtime_operation_target_names()
    assert "action-event-repair-loop" in runtime_path_target_names()
    assert "session_products" in runtime_maintenance_target_names()
    assert "cli.json-contract" in declared_operation_target_names()
    assert "benchmark.storage.crud" in declared_operation_target_names()


def test_scenario_metadata_resolves_only_runtime_declared_targets() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        path_targets=("action-event-repair-loop",),
        artifact_targets=("action_event_rows", "message_fts"),
        operation_targets=("project-action-event-health", "benchmark.storage.crud"),
        maintenance_targets=("action_event_read_model", "missing.target"),
    )

    assert metadata.runtime_path_targets() == ("action-event-repair-loop",)
    assert metadata.runtime_artifact_targets() == ("action_event_rows", "message_fts")
    assert metadata.runtime_operation_targets() == ("project-action-event-health",)
    assert metadata.runtime_maintenance_targets() == ("action_event_read_model",)


def test_runtime_artifact_graph_exposes_resolved_specs() -> None:
    graph = runtime_artifact_graph()

    assert "action_event_rows" in graph.artifact_names()
    assert "materialize-session-products" in graph.operation_names()
    assert "session_products" in graph.maintenance_target_names()


def test_scenario_metadata_resolves_runtime_specs() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        path_targets=("action-event-repair-loop",),
        artifact_targets=("action_event_rows", "message_fts"),
        operation_targets=("project-action-event-health", "benchmark.storage.crud"),
        maintenance_targets=("action_event_read_model",),
    )

    assert tuple(path.name for path in metadata.resolve_runtime_paths()) == ("action-event-repair-loop",)
    assert tuple(artifact.name for artifact in metadata.resolve_runtime_artifacts()) == (
        "action_event_rows",
        "message_fts",
    )
    assert tuple(operation.name for operation in metadata.resolve_runtime_operations()) == (
        "project-action-event-health",
    )
    assert tuple(target.name for target in metadata.resolve_runtime_maintenance_targets()) == (
        "action_event_read_model",
    )


def test_scenario_metadata_resolves_declared_operation_targets() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        operation_targets=("project-action-event-health", "benchmark.storage.crud", "missing.operation"),
    )

    assert metadata.declared_operation_targets() == (
        "project-action-event-health",
        "benchmark.storage.crud",
    )
    assert tuple(operation.name for operation in metadata.resolve_declared_operations()) == (
        "project-action-event-health",
        "benchmark.storage.crud",
    )
