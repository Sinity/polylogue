"""Subprocess proofs for the root CLI machine-error contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.lib.json import JSONDocument, json_document
from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

pytestmark = [pytest.mark.integration, pytest.mark.machine_contract]


def _parse_json(stdout: str) -> JSONDocument:
    return json_document(json.loads(stdout))


def test_script_entrypoint_invalid_flag_emits_json_error(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(["doctor", "--json", "--bad-flag"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code != 0
    payload = _parse_json(result.stdout)
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_arguments"
    assert payload["details"] == {"option": "--bad-flag"}
    assert "No such option" in str(payload["message"])
    assert "No such option" not in result.stderr


def test_module_entrypoint_invalid_flag_emits_json_error(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(
        ["doctor", "--json", "--bad-flag"],
        env=workspace["env"],
        cwd=tmp_path,
        entrypoint="module",
    )

    assert result.exit_code != 0
    payload = _parse_json(result.stdout)
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_arguments"
    assert payload["details"] == {"option": "--bad-flag"}
    assert "No such option" in str(payload["message"])
    assert "No such option" not in result.stderr


def test_module_entrypoint_command_validation_emits_json_error(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(
        ["doctor", "--json", "--proof", "--artifact-limit", "0"],
        env=workspace["env"],
        cwd=tmp_path,
        entrypoint="module",
    )

    assert result.exit_code != 0
    payload = _parse_json(result.stdout)
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_arguments"
    assert "--artifact-limit must be a positive integer" in str(payload["message"])
    assert "Traceback" not in result.stderr


def test_script_entrypoint_success_still_uses_success_envelope(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(["doctor", "--json"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    payload = _parse_json(result.stdout)
    assert payload["status"] == "ok"
    assert "result" in payload
