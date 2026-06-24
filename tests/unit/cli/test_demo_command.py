from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.scenarios import DEMO_CLAUDE_CODE_SESSION_ID


def test_demo_seed_and_verify_json_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()

    seed = runner.invoke(cli, ["demo", "seed", "--with-overlays", "--format", "json"])
    assert seed.exit_code == 0, seed.output
    seed_payload = json.loads(seed.output)
    assert seed_payload["session_count"] == 3
    assert seed_payload["message_count"] == 19
    assert seed_payload["overlays_seeded"] is True

    verify = runner.invoke(cli, ["demo", "verify", "--require-overlays", "--format", "json"])
    assert verify.exit_code == 0, verify.output
    verify_payload = json.loads(verify.output)
    assert verify_payload["ok"] is True
    assert DEMO_CLAUDE_CODE_SESSION_ID in verify_payload["query_hits"]
    assert verify_payload["absolute_path_leaks"] == []


def test_demo_script_prints_copy_pastable_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "script", "--root", str(tmp_path / "archive")])

    assert result.exit_code == 0, result.output
    assert "POLYLOGUE_ARCHIVE_ROOT" in result.output
    assert "polylogue demo seed" in result.output
    assert "polylogue demo verify" in result.output
    assert "--with-overlays --format json" in result.output
    assert "--require-overlays --format json" in result.output
    assert str(tmp_path / "archive") in result.output


def test_demo_script_seed_and_verify_commands_are_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    runner = CliRunner()
    script = runner.invoke(cli, ["demo", "script", "--root", str(archive_root)])
    assert script.exit_code == 0, script.output

    exports: dict[str, str] = {}
    demo_commands: list[list[str]] = []
    for line in script.output.splitlines():
        if line.startswith("export "):
            name, value = line.removeprefix("export ").split("=", maxsplit=1)
            exports[name] = shlex.split(value)[0]
            monkeypatch.setenv(name, exports[name])
        elif line.startswith("polylogue demo "):
            expanded = os.path.expandvars(line)
            demo_commands.append(shlex.split(expanded)[1:])

    assert [command[:2] for command in demo_commands] == [["demo", "seed"], ["demo", "verify"]]
    assert exports["POLYLOGUE_ARCHIVE_ROOT"] == str(archive_root)

    seed = runner.invoke(cli, demo_commands[0])
    assert seed.exit_code == 0, seed.output
    seed_payload = json.loads(seed.output)
    assert seed_payload["session_count"] == 3
    assert seed_payload["message_count"] == 19
    assert seed_payload["overlays_seeded"] is True

    verify = runner.invoke(cli, demo_commands[1])
    assert verify.exit_code == 0, verify.output
    verify_payload = json.loads(verify.output)
    assert verify_payload["ok"] is True
    assert verify_payload["overlays_present"] is True
    assert DEMO_CLAUDE_CODE_SESSION_ID in verify_payload["query_hits"]
    assert verify_payload["absolute_path_leaks"] == []
