from __future__ import annotations

from polylogue.scenarios import (
    ExecutionKind,
    command_execution,
    composite_execution,
    devtools_execution,
    memory_budget_execution,
    polylogue_execution,
    pytest_execution,
    runner_execution,
)


def test_command_execution_exposes_command_shape() -> None:
    execution = command_execution("devtools", "verify-showcase")

    assert execution.kind is ExecutionKind.COMMAND
    assert execution.command == ("devtools", "verify-showcase")
    assert execution.pytest_targets == ()
    assert execution.members == ()


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


def test_devtools_execution_renders_control_plane_command() -> None:
    execution = devtools_execution("pipeline-probe", "--provider", "chatgpt", "--stage", "parse")

    assert execution.kind is ExecutionKind.DEVTOOLS
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
    execution = polylogue_execution("audit", "--only", "exercises")

    restored = type(execution).from_payload(execution.to_payload())

    assert restored == execution


def test_nested_execution_spec_round_trips_payload() -> None:
    execution = memory_budget_execution(
        1536,
        devtools_execution("pipeline-probe", "--provider", "chatgpt", "--stage", "parse"),
    )

    restored = type(execution).from_payload(execution.to_payload())

    assert restored == execution


def test_composite_execution_has_members_only() -> None:
    execution = composite_execution("lane-a", "lane-b")

    assert execution.kind is ExecutionKind.COMPOSITE
    assert execution.is_composite is True
    assert execution.command is None
    assert execution.members == ("lane-a", "lane-b")


def test_runner_execution_has_runner_only() -> None:
    execution = runner_execution("startup-health")

    assert execution.kind is ExecutionKind.RUNNER
    assert execution.is_runner is True
    assert execution.command is None
    assert execution.pytest_targets == ()
    assert execution.runner == "startup-health"
