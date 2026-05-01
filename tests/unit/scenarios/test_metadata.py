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
            "docs_role": "quickstart",
            "caption": "Doctor detects repairable action-event gaps.",
            "narrative_order": 10,
            "audience": ["operator"],
            "demonstrates": ["repair-preview", "json-envelope"],
            "privacy_level": "synthetic",
            "media": ["terminal"],
            "visual_style": "plain-terminal",
        }
    )

    assert metadata.origin == "generated.contract"
    assert metadata.path_targets == ("action-event-repair-loop",)
    assert metadata.artifact_targets == ("doctor_runtime", "message_fts")
    assert metadata.operation_targets == ("cli.doctor",)
    assert metadata.maintenance_targets == ("dangling_fts",)
    assert metadata.tags == ("generated", "json-contract")
    assert metadata.docs_role == "quickstart"
    assert metadata.caption == "Doctor detects repairable action-event gaps."
    assert metadata.narrative_order == 10
    assert metadata.audience == ("operator",)
    assert metadata.demonstrates == ("repair-preview", "json-envelope")
    assert metadata.privacy_level == "synthetic"
    assert metadata.media == ("terminal",)
    assert metadata.visual_style == "plain-terminal"


def test_scenario_metadata_from_object_falls_back_for_mock_attributes() -> None:
    obj = MagicMock()

    metadata = ScenarioMetadata.from_object(obj)

    assert metadata.origin == "authored"
    assert metadata.artifact_targets == ()
    assert metadata.operation_targets == ()
    assert metadata.maintenance_targets == ()
    assert metadata.tags == ()
    assert metadata.docs_role == ""
    assert metadata.narrative_order is None


def test_scenario_metadata_payload_omits_empty_collections() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        path_targets=("raw-reparse-loop",),
        artifact_targets=("doctor_runtime",),
        operation_targets=(),
        maintenance_targets=("session_products",),
        tags=("generated",),
        docs_role="reference",
        caption="Session insights are visible to docs projections.",
        narrative_order=20,
        audience=("maintainer",),
        demonstrates=("session-insights",),
        privacy_level="synthetic",
        media=("markdown",),
        visual_style="reference-table",
    )

    assert metadata.to_payload() == {
        "origin": "generated.contract",
        "path_targets": ["raw-reparse-loop"],
        "artifact_targets": ["doctor_runtime"],
        "maintenance_targets": ["session_products"],
        "tags": ["generated"],
        "docs_role": "reference",
        "caption": "Session insights are visible to docs projections.",
        "narrative_order": 20,
        "audience": ["maintainer"],
        "demonstrates": ["session-insights"],
        "privacy_level": "synthetic",
        "media": ["markdown"],
        "visual_style": "reference-table",
    }


def test_scenario_metadata_merges_presentation_fields_generally() -> None:
    default = ScenarioMetadata(
        docs_role="tour",
        caption="Default caption",
        narrative_order=30,
        audience=("operator",),
        demonstrates=("query",),
        privacy_level="synthetic",
        media=("terminal",),
        visual_style="plain",
        tags=("default",),
    )
    explicit = ScenarioMetadata(
        caption="Explicit caption",
        audience=("maintainer",),
        demonstrates=("repair",),
        media=("screenshot",),
        tags=("explicit",),
    )

    merged = explicit.with_default_targets(default)

    assert merged.docs_role == "tour"
    assert merged.caption == "Explicit caption"
    assert merged.narrative_order == 30
    assert merged.audience == ("maintainer",)
    assert merged.demonstrates == ("repair",)
    assert merged.privacy_level == "synthetic"
    assert merged.media == ("screenshot",)
    assert merged.visual_style == "plain"
    assert merged.tags == ("explicit", "default")


def test_runtime_target_names_include_declared_runtime_specs() -> None:
    assert "action_event_rows" in runtime_artifact_target_names()
    assert "project-action-event-readiness" in runtime_operation_target_names()
    assert "action-event-repair-loop" in runtime_path_target_names()
    assert "session_products" in runtime_maintenance_target_names()
    assert "cli.json-contract" in declared_operation_target_names()
    assert "benchmark.storage.crud" in declared_operation_target_names()


def test_scenario_metadata_resolves_only_runtime_declared_targets() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        path_targets=("action-event-repair-loop",),
        artifact_targets=("action_event_rows", "message_fts"),
        operation_targets=("project-action-event-readiness", "benchmark.storage.crud"),
        maintenance_targets=("action_event_read_model", "missing.target"),
    )

    assert metadata.runtime_path_targets() == ("action-event-repair-loop",)
    assert metadata.runtime_artifact_targets() == ("action_event_rows", "message_fts")
    assert metadata.runtime_operation_targets() == ("project-action-event-readiness",)
    assert metadata.runtime_maintenance_targets() == ("action_event_read_model",)


def test_runtime_artifact_graph_exposes_resolved_specs() -> None:
    graph = runtime_artifact_graph()

    assert "action_event_rows" in graph.artifact_names()
    assert "materialize-session-insights" in graph.operation_names()
    assert "session_products" in graph.maintenance_target_names()


def test_scenario_metadata_resolves_runtime_specs() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        path_targets=("action-event-repair-loop",),
        artifact_targets=("action_event_rows", "message_fts"),
        operation_targets=("project-action-event-readiness", "benchmark.storage.crud"),
        maintenance_targets=("action_event_read_model",),
    )

    assert tuple(path.name for path in metadata.resolve_runtime_paths()) == ("action-event-repair-loop",)
    assert tuple(artifact.name for artifact in metadata.resolve_runtime_artifacts()) == (
        "action_event_rows",
        "message_fts",
    )
    assert tuple(operation.name for operation in metadata.resolve_runtime_operations()) == (
        "project-action-event-readiness",
    )
    assert tuple(target.name for target in metadata.resolve_runtime_maintenance_targets()) == (
        "action_event_read_model",
    )


def test_scenario_metadata_resolves_declared_operation_targets() -> None:
    metadata = ScenarioMetadata(
        origin="generated.contract",
        operation_targets=("project-action-event-readiness", "benchmark.storage.crud", "missing.operation"),
    )

    assert metadata.declared_operation_targets() == (
        "project-action-event-readiness",
        "benchmark.storage.crud",
    )
    assert tuple(operation.name for operation in metadata.resolve_declared_operations()) == (
        "project-action-event-readiness",
        "benchmark.storage.crud",
    )
