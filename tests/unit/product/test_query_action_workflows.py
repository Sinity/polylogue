"""Executable query-action workflow registry and demo golden paths (#2305)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.core.json import loads
from polylogue.operations.action_contracts import ACTION_CONTRACT_BY_PATH, action_affordance_payloads
from polylogue.product.workflows import (
    EXECUTABLE_WORKFLOW_GOLDEN_PATHS,
    ExecutableWorkflowGoldenPath,
    JsonExpectation,
)
from polylogue.surfaces.action_affordances import ActionAffordancePayload


def _demo_env(root: Path) -> dict[str, str]:
    return {
        "POLYLOGUE_ARCHIVE_ROOT": str(root),
        "POLYLOGUE_FORCE_PLAIN": "1",
        "POLYLOGUE_NO_COLOR": "1",
        "NO_COLOR": "1",
        "COLUMNS": "120",
    }


def _seed_demo_archive(runner: CliRunner, tmp_path: Path) -> dict[str, str]:
    env = _demo_env(tmp_path / "archive")
    result = runner.invoke(
        cli,
        ["demo", "seed", "--with-overlays", "--format", "json"],
        env=env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return env


def _parse_json_output(output: str) -> Any:
    return loads(output)


def _resolve_json_path(payload: Any, path: tuple[str | int, ...]) -> Any:
    current = payload
    for segment in path:
        if isinstance(segment, int):
            assert isinstance(current, list), f"expected list before index {segment!r}, got {type(current).__name__}"
            current = current[segment]
        else:
            assert isinstance(current, Mapping), f"expected object before key {segment!r}, got {type(current).__name__}"
            assert segment in current, f"missing JSON key {segment!r} in {sorted(current)}"
            current = current[segment]
    return current


def _assert_json_expectation(payload: Any, expectation: JsonExpectation) -> None:
    value = _resolve_json_path(payload, expectation.path)
    match expectation.kind:
        case "any":
            return
        case "object":
            assert isinstance(value, Mapping)
        case "array":
            assert isinstance(value, list)
        case "string":
            assert isinstance(value, str)
        case "integer":
            assert isinstance(value, int) and not isinstance(value, bool)
        case "number":
            assert isinstance(value, int | float) and not isinstance(value, bool)
        case "boolean":
            assert isinstance(value, bool)
        case "null":
            assert value is None


def test_action_affordance_payload_has_no_flat_compatibility_aliases() -> None:
    stale_aliases = {
        "input_unit",
        "cardinality_state",
        "safety_level",
        "confirmation_command",
        "selection_command",
        "destination_support",
        "format_support",
        "default_format",
        "machine_envelope",
        "disabled_reason",
        "estimated_cost",
        "next_actions",
        "guards",
        "completion_context",
        "requires_daemon",
    }

    assert stale_aliases.isdisjoint(ActionAffordancePayload.model_fields)


def test_shared_action_affordance_payload_uses_grouped_contract_fields() -> None:
    payload_by_id = {payload.id: payload for payload in action_affordance_payloads()}
    read = payload_by_id["read"]

    assert read.target == "selection"
    assert read.input.unit == "query_result_set"
    assert read.execution.cardinality_state == "explicit_multi"
    assert read.execution.guards == ("single_match_unless_all_or_first", "file_destination_requires_out")
    assert "browser" in read.output.destination_support
    assert read.output.format_support == ("human", "json", "ndjson")
    assert read.safety.safety_level == "safe"
    assert read.safety.selection_command == "polylogue find QUERY then select"
    assert "continue" in read.availability.next_actions

    delete = payload_by_id["delete"]
    assert delete.safety.safety_level == "destructive"
    assert delete.safety.confirmation_command == "polylogue find QUERY then delete --dry-run"
    assert delete.availability.next_actions == ("find",)


@pytest.mark.parametrize("golden", EXECUTABLE_WORKFLOW_GOLDEN_PATHS, ids=lambda entry: entry.id)
def test_demo_archive_golden_path_executes_registry_command(
    golden: ExecutableWorkflowGoldenPath,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    env = _seed_demo_archive(runner, tmp_path)

    result = runner.invoke(cli, list(golden.command), env=env, catch_exceptions=False)
    assert result.exit_code == 0, result.output

    for expected in golden.stdout_contains:
        assert expected in result.output

    known_affordances = {payload.id for payload in action_affordance_payloads()}
    assert set(golden.required_affordance_ids) <= known_affordances
    assert golden.action_path in ACTION_CONTRACT_BY_PATH

    if golden.output_kind == "human":
        return

    payload = _parse_json_output(result.output)
    if golden.output_kind == "json_object":
        assert isinstance(payload, dict)
    elif golden.output_kind == "json_array":
        assert isinstance(payload, list)

    for expectation in golden.json_expectations:
        _assert_json_expectation(payload, expectation)
