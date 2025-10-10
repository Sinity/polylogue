from __future__ import annotations

import json

from pathlib import Path
from types import SimpleNamespace

from polylogue.automation import SCRIPT_PATH, cron_snippet, describe_targets, systemd_snippet
from polylogue.cli import run_automation_cli
from polylogue.commands import CommandEnv


class DummyConsole:
    def print(self, *args, **kwargs):  # pragma: no cover - not needed
        pass


class DummyUI:
    plain = True
    console = DummyConsole()


def test_systemd_snippet_includes_paths(tmp_path):
    snippet = systemd_snippet(
        target_key="codex",
        interval="15m",
        working_dir=tmp_path,
        extra_args=["--out", str(tmp_path / "out")],
        boot_delay="1m",
    )
    assert "polylogue-sync-codex.service" in snippet
    assert str(SCRIPT_PATH) in snippet
    assert f"WorkingDirectory={tmp_path}" in snippet
    assert "OnUnitActiveSec=15m" in snippet
    assert "--collapse-threshold 24" in snippet
    assert "--out" in snippet and str(tmp_path / "out") in snippet


def test_cron_snippet_default_schedule(tmp_path):
    log_path = "/tmp/polylogue-test.log"
    snippet = cron_snippet(
        target_key="claude-code",
        schedule="*/20 * * * *",
        working_dir=tmp_path,
        log_path=log_path,
        extra_args=["--html"],
        state_env="$HOME/.local/state",
    )
    assert snippet.startswith("*/20 * * * * ")
    assert f"cd {tmp_path}" in snippet
    assert log_path in snippet
    assert "--html" in snippet
    assert "--collapse-threshold 20" in snippet


def test_describe_targets_roundtrip():
    data = describe_targets()
    assert "codex" in data and "claude-code" in data and "drive-sync" in data and "gemini-render" in data
    codex = describe_targets("codex")
    assert list(codex.keys()) == ["codex"]
    assert codex["codex"]["command"][0] == "sync-codex"
    drive = describe_targets("drive-sync")
    assert drive["drive-sync"]["command"][0] == "sync"


def test_run_automation_cli_systemd(capsys, tmp_path):
    args = SimpleNamespace(
        automation_format="systemd",
        target="drive-sync",
        interval="10m",
        boot_delay="1m",
        working_dir=str(tmp_path),
        out=str(tmp_path / "out"),
        extra_arg=["--plain"],
        collapse_threshold=20,
        html=True,
    )
    run_automation_cli(args, CommandEnv(ui=DummyUI()))
    output = capsys.readouterr().out
    assert "polylogue-sync-drive.service" in output
    assert "--collapse-threshold 20" in output
    assert "--html" in output
    assert "--folder-name AI Studio" in output
    assert "--plain" in output


def test_run_automation_cli_describe(capsys):
    args = SimpleNamespace(
        automation_format="describe",
        target="codex",
        working_dir=None,
        out=None,
        extra_arg=[],
        collapse_threshold=None,
        html=False,
        interval=5,
        boot_delay=2,
    )
    run_automation_cli(args, CommandEnv(ui=DummyUI()))
    payload = json.loads(capsys.readouterr().out)
    assert payload["codex"]["command"][0] == "sync-codex"
    assert "defaults" in payload["codex"]
