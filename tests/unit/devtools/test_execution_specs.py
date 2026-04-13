from __future__ import annotations

from devtools.execution_specs import (
    ExecutionKind,
    command_execution,
    composite_execution,
    pytest_execution,
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
    assert execution.command == ("tests/unit/core/test_filters.py", "-q")
    assert execution.pytest_targets == ("tests/unit/core/test_filters.py", "-q")


def test_composite_execution_has_members_only() -> None:
    execution = composite_execution("lane-a", "lane-b")

    assert execution.kind is ExecutionKind.COMPOSITE
    assert execution.is_composite is True
    assert execution.command is None
    assert execution.members == ("lane-a", "lane-b")
