from __future__ import annotations

import json

import pytest

import devtools.__main__ as devtools_main


def test_list_commands_json_includes_generated_surface(capsys) -> None:
    assert devtools_main.main(["--list-commands", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    commands = {entry["name"] for entry in payload["commands"]}
    assert "render-devtools-reference" in commands
    assert "status" in commands


def test_list_commands_human_output(capsys) -> None:
    assert devtools_main.main(["--list-commands"]) == 0
    captured = capsys.readouterr()
    assert "generated surfaces:" in captured.out
    assert "render-devtools-reference" in captured.out


def test_global_json_flag_is_forwarded_to_command(monkeypatch) -> None:
    captured: list[list[str] | None] = []

    def fake_main(argv):
        captured.append(argv)
        return 0

    class FakeSpec:
        @staticmethod
        def resolve_main():
            return fake_main

    monkeypatch.setitem(devtools_main.COMMANDS, "status", FakeSpec())

    assert devtools_main.main(["--json", "status"]) == 0
    assert captured == [["--json"]]


def test_help_uses_devtools_prog_name(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        devtools_main.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: devtools ")
