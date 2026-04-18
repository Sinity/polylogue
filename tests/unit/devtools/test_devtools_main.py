from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import pytest

import devtools.__main__ as devtools_main
from devtools.command_catalog import COMMANDS


def test_list_commands_json_includes_generated_surface(capsys: pytest.CaptureFixture[str]) -> None:
    assert devtools_main.main(["--list-commands", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    commands = {entry["name"] for entry in payload["commands"]}
    assert "artifact-graph" in commands
    assert "scenario-projections" in commands
    assert "render-devtools-reference" in commands
    assert "status" in commands


def test_list_commands_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert devtools_main.main(["--list-commands"]) == 0
    captured = capsys.readouterr()
    assert "generated surfaces:" in captured.out
    assert "artifact-graph" in captured.out
    assert "scenario-projections" in captured.out
    assert "render-devtools-reference" in captured.out


def test_global_json_flag_is_forwarded_to_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str] | None] = []

    def fake_main(argv: list[str] | None) -> int:
        captured.append(argv)
        return 0

    class FakeSpec:
        @staticmethod
        def resolve_main() -> Callable[[list[str] | None], int]:
            return fake_main

    monkeypatch.setitem(COMMANDS, "status", cast(Any, FakeSpec()))

    assert devtools_main.main(["--json", "status"]) == 0
    assert captured == [["--json"]]


def test_help_uses_devtools_prog_name(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        devtools_main.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: devtools ")
