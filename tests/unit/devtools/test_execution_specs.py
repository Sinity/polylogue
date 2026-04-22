from __future__ import annotations

from polylogue.scenarios import (
    CorpusRequest,
    ExecutionKind,
    PipelineProbeInputMode,
    PipelineProbeRequest,
    composite_execution,
    devtools_execution,
    memory_budget_execution,
    pipeline_probe_execution,
    polylogue_execution,
    pytest_execution,
    runner_execution,
)


def test_pytest_execution_exposes_pytest_targets() -> None:
    execution = pytest_execution("tests/unit/core/test_filters.py", "-q")

    assert execution.kind is ExecutionKind.PYTEST
    assert execution.command == ("pytest", "tests/unit/core/test_filters.py", "-q")
    assert execution.pytest_targets == ("tests/unit/core/test_filters.py", "-q")


def test_pytest_execution_normalizes_leading_pytest_binary() -> None:
    execution = pytest_execution("pytest", "-m", "machine_contract")

    assert execution.command == ("pytest", "-m", "machine_contract")
    assert execution.pytest_targets == ("-m", "machine_contract")


def test_polylogue_execution_renders_runtime_and_display_forms() -> None:
    execution = polylogue_execution("doctor", "--json")

    assert execution.kind is ExecutionKind.POLYLOGUE
    assert execution.command == ("polylogue", "--plain", "doctor", "--json")
    assert execution.display_command == ("polylogue", "doctor", "--json")
    assert execution.polylogue_invoke_args == ("--plain", "doctor", "--json")


def test_polylogue_doctor_targeted_execution_uses_maintenance_target_catalog_metadata() -> None:
    execution = polylogue_execution(
        "doctor",
        "--repair",
        "--target",
        "action_event_read_model",
        "--target",
        "session_products",
    )

    assert execution.metadata.operation_targets == (
        "materialize-action-events",
        "materialize-session-products",
        "project-action-event-readiness",
        "project-session-product-readiness",
    )
    assert execution.metadata.maintenance_targets == (
        "action_event_read_model",
        "session_products",
    )


def test_polylogue_doctor_target_aliases_resolve_through_catalog() -> None:
    execution = polylogue_execution("doctor", "--target", "action_events")

    assert execution.metadata.operation_targets == ("project-action-event-readiness",)
    assert execution.metadata.maintenance_targets == ("action_event_read_model",)


def test_pipeline_probe_execution_renders_control_plane_command() -> None:
    execution = pipeline_probe_execution(
        PipelineProbeRequest(
            stage="parse",
            corpus_request=CorpusRequest(providers=("chatgpt",), count=5, messages_min=4, messages_max=12, seed=42),
        )
    )

    assert execution.kind is ExecutionKind.PIPELINE_PROBE
    assert execution.command == (
        "devtools",
        "pipeline-probe",
        "--provider",
        "chatgpt",
        "--stage",
        "parse",
    )
    assert execution.display_command == execution.command


def test_memory_budget_execution_wraps_structured_execution() -> None:
    wrapped = polylogue_execution("doctor", "--json")
    execution = memory_budget_execution(1536, wrapped)

    assert execution.kind is ExecutionKind.MEMORY_BUDGET
    assert execution.command == (
        "devtools",
        "query-memory-budget",
        "--max-rss-mb",
        "1536",
        "--",
        "polylogue",
        "--plain",
        "doctor",
        "--json",
    )
    assert execution.display_command == (
        "devtools",
        "query-memory-budget",
        "--max-rss-mb",
        "1536",
        "--",
        "polylogue",
        "doctor",
        "--json",
    )


def test_execution_spec_round_trips_payload() -> None:
    execution = devtools_execution("lab-scenario", "run", "archive-smoke", "--tier", "0")

    restored = type(execution).from_payload(execution.to_payload())

    assert restored == execution


def test_nested_execution_spec_round_trips_payload() -> None:
    execution = memory_budget_execution(
        1536,
        pipeline_probe_execution(
            PipelineProbeRequest(
                stage="parse",
                corpus_request=CorpusRequest(
                    providers=("chatgpt",),
                    count=5,
                    messages_min=4,
                    messages_max=12,
                    seed=42,
                ),
            )
        ),
    )

    restored = type(execution).from_payload(execution.to_payload())

    assert restored == execution


def test_pipeline_probe_request_round_trips_archive_subset_payload() -> None:
    request = PipelineProbeRequest(
        input_mode=PipelineProbeInputMode.ARCHIVE_SUBSET,
        stage="parse",
        sample_per_provider=50,
        workdir="/tmp/probe",
        json_out="/tmp/probe.json",
        max_total_ms=10000,
        max_peak_rss_mb=512,
    )

    restored = PipelineProbeRequest.from_payload(request.to_payload())

    assert restored == request


def test_composite_execution_has_members_only() -> None:
    execution = composite_execution("lane-a", "lane-b")

    assert execution.kind is ExecutionKind.COMPOSITE
    assert execution.is_composite is True
    assert execution.command is None
    assert execution.members == ("lane-a", "lane-b")


def test_runner_execution_has_runner_only() -> None:
    execution = runner_execution("startup-readiness")

    assert execution.kind is ExecutionKind.RUNNER
    assert execution.is_runner is True
    assert execution.command is None
    assert execution.pytest_targets == ()
    assert execution.runner == "startup-readiness"
