"""Integration tests for the tags command surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

pytestmark = [pytest.mark.integration]


def test_cli_tags_accepts_format_json(tmp_path: Path) -> None:
    workspace = setup_isolated_workspace(tmp_path)

    result = run_cli(["--plain", "tags", "--format", "json"], env=workspace["env"], cwd=tmp_path)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["result"] == {"tags": {}}
