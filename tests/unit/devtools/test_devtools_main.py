from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import devtools.__main__ as devtools_main
from devtools import project_motd, system_exit
from devtools.command_catalog import COMMAND_SPECS, COMMANDS, CommandSpec
from devtools.generated_surfaces import GeneratedSurface


class _UnprintableCode:
    def __str__(self) -> str:
        raise RuntimeError("str failed")

    def __repr__(self) -> str:
        raise RuntimeError("repr failed")


def test_list_commands_json_includes_generated_surface(capsys: pytest.CaptureFixture[str]) -> None:
    assert devtools_main.main(["--list-commands", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    commands = {entry["name"] for entry in payload["commands"]}
    assert commands == {spec.name for spec in COMMAND_SPECS}


def test_list_commands_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert devtools_main.main(["--list-commands"]) == 0
    captured = capsys.readouterr()
    assert "generated surfaces:" in captured.out
    for spec in COMMAND_SPECS:
        assert spec.name in captured.out


def test_global_json_flag_is_forwarded_to_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str] | None] = []

    def fake_main(argv: list[str] | None) -> int:
        captured.append(argv)
        return 0

    fake_module = ModuleType("_polylogue_devtools_test_fake")
    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(__import__("sys").modules, fake_module.__name__, fake_module)

    monkeypatch.setitem(
        COMMANDS,
        "status",
        CommandSpec("status", "core", "fake status", fake_module.__name__),
    )

    assert devtools_main.main(["--json", "status"]) == 0
    assert captured == [["--json"]]


def test_nested_render_command_dispatches_to_catalog_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str] | None] = []

    def fake_main(argv: list[str] | None) -> int:
        captured.append(argv)
        return 0

    fake_module = ModuleType("_polylogue_devtools_test_nested_fake")
    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(__import__("sys").modules, fake_module.__name__, fake_module)

    monkeypatch.setitem(
        COMMANDS,
        "render",
        CommandSpec("render", "generated surfaces", "fake render", fake_module.__name__),
    )

    assert devtools_main.main(["render", "all", "--check"]) == 0
    assert captured == [["all", "--check"]]


def test_bare_verify_dispatches_without_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str] | None] = []

    def fake_main(argv: list[str] | None) -> int:
        captured.append(argv)
        return 0

    fake_module = ModuleType("_polylogue_devtools_test_verify_fake")
    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(__import__("sys").modules, fake_module.__name__, fake_module)

    monkeypatch.setitem(
        COMMANDS,
        "verify",
        CommandSpec("verify", "verification", "fake verify", fake_module.__name__),
    )

    assert devtools_main.main(["verify"]) == 0
    assert captured == [[]]


def test_verify_flags_reach_the_wrapped_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str] | None] = []

    def fake_main(argv: list[str] | None) -> int:
        captured.append(argv)
        return 0

    fake_module = ModuleType("_polylogue_devtools_test_verify_flag_fake")
    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(__import__("sys").modules, fake_module.__name__, fake_module)

    monkeypatch.setitem(
        COMMANDS,
        "verify",
        CommandSpec("verify", "verification", "fake verify", fake_module.__name__),
    )

    assert devtools_main.main(["verify", "--quick"]) == 0
    assert captured == [["--quick"]]


@pytest.mark.parametrize(
    "code, expected",
    [(None, 1), (False, 1), (True, 1), (0, 0), (-7, -7), (7, 7)],
)
def test_system_exit_codes_keep_python_cli_semantics(
    monkeypatch: pytest.MonkeyPatch, code: object, expected: int
) -> None:
    fake_module = ModuleType("_polylogue_devtools_test_system_exit_fake")

    def fake_main(_argv: list[str] | None) -> int:
        raise SystemExit(code)

    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setitem(
        COMMANDS,
        "status",
        CommandSpec("status", "core", "fake status", fake_module.__name__),
    )

    assert devtools_main.main(["status"]) == expected


@pytest.mark.parametrize("code", ["", "0", 0.0, "command failed", object(), _UnprintableCode()])
def test_non_integer_system_exit_fails_closed_and_preserves_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], code: object
) -> None:
    """Anti-vacuity: removing the dispatch translation makes this return 0 and lose the message."""
    fake_module = ModuleType("_polylogue_devtools_test_system_exit_message_fake")

    def fake_main(_argv: list[str] | None) -> int:
        raise SystemExit(code)

    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setitem(
        COMMANDS,
        "status",
        CommandSpec("status", "core", "fake status", fake_module.__name__),
    )

    assert devtools_main.main(["status"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    if isinstance(code, _UnprintableCode):
        assert captured.err == "<unprintable SystemExit code>\n"
    else:
        assert captured.err == f"{code}\n"


def test_non_integer_system_exit_message_is_bounded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = "x" * 5000
    fake_module = ModuleType("_polylogue_devtools_test_system_exit_bounded_fake")

    def fake_main(_argv: list[str] | None) -> int:
        raise SystemExit(payload)

    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setitem(
        COMMANDS,
        "status",
        CommandSpec("status", "core", "fake status", fake_module.__name__),
    )

    assert devtools_main.main(["status"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("x" * 253) + "...\n"


def test_click_dispatch_and_project_motd_delegate_to_shared_system_exit_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[SystemExit] = []

    def fake_translate(exc: SystemExit) -> system_exit.SystemExitTranslation:
        calls.append(exc)
        return system_exit.SystemExitTranslation(code=1)

    monkeypatch.setattr(system_exit, "translate_system_exit", fake_translate)

    fake_module = ModuleType("_polylogue_devtools_test_shared_system_exit_fake")

    def fake_main(_argv: list[str] | None) -> int:
        raise SystemExit("click route")

    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setitem(
        COMMANDS,
        "status",
        CommandSpec("status", "core", "fake status", fake_module.__name__),
    )

    def surface_main(_argv: list[str] | None) -> int:
        raise SystemExit("motd route")

    surface = GeneratedSurface("fake", "Fake", "test", (), surface_main)
    assert devtools_main.main(["status"]) == 1
    assert project_motd.run_check(tmp_path, surface) == "stale"
    assert [exc.code for exc in calls] == ["click route", "motd route"]


def test_why_unreadable_receipt_fails_through_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from devtools import why

    run_json = tmp_path / "run" / "run.json"
    run_json.parent.mkdir()
    run_json.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(why, "VERIFY_RUNS_DIR", tmp_path)

    assert devtools_main.main(["why"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"why: cannot read {run_json}" in captured.err


def test_nested_archive_command_dispatches_to_catalog_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str] | None] = []

    def fake_main(argv: list[str] | None) -> int:
        captured.append(argv)
        return 0

    fake_module = ModuleType("_polylogue_devtools_test_archive_fake")
    fake_module.__dict__["main"] = fake_main
    monkeypatch.setitem(__import__("sys").modules, fake_module.__name__, fake_module)

    monkeypatch.setitem(
        COMMANDS,
        "archive lineage-validation",
        CommandSpec("archive lineage-validation", "archive", "fake archive command", fake_module.__name__),
    )

    assert devtools_main.main(["archive", "lineage-validation", "node-id", "--json"]) == 0
    assert captured == [["node-id", "--json"]]


def test_help_output_includes_devtools_prog_name(capsys: pytest.CaptureFixture[str]) -> None:
    # main() catches the SystemExit that Click raises with standalone_mode=True
    # and returns the exit code as an int.
    assert devtools_main.main(["--help"]) == 0
    captured = capsys.readouterr()
    # Click outputs "Usage: devtools ..." or "Usage: python -m devtools ..."
    assert "devtools" in captured.out
    assert "Options" in captured.out
