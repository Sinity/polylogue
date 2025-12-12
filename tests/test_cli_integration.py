from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.commands import CommandEnv
from polylogue.cli.click_app import main
from polylogue.cli.doctor import run_doctor_cli
from polylogue.doctor import DoctorIssue, DoctorReport
from polylogue import ui as ui_module
from polylogue.ui import UI
from polylogue import util as util_module
from polylogue import paths as paths_module
from tests.conftest import _configure_state


def _configure_isolated_state(monkeypatch, root: Path) -> None:
    state_home = _configure_state(monkeypatch, root)
    data_home = root / "data"
    config_home = root / "config"
    cache_home = root / "cache"
    for path in (data_home, config_home, cache_home):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    monkeypatch.setattr(util_module, "DATA_HOME", data_home, raising=False)
    monkeypatch.setattr(util_module, "CONFIG_HOME", config_home, raising=False)
    monkeypatch.setattr(util_module, "CACHE_HOME", cache_home, raising=False)
    monkeypatch.setattr(paths_module, "DATA_HOME", data_home, raising=False)
    monkeypatch.setattr(paths_module, "CONFIG_HOME", config_home, raising=False)
    monkeypatch.setattr(paths_module, "CACHE_HOME", cache_home, raising=False)


def test_cli_search_runs_without_name_error(monkeypatch, tmp_path, capsys):
    """Test search command handles missing results gracefully (updated from inspect search)."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "search", "missing"])

    main()

    captured = capsys.readouterr().out
    assert "No results found." in captured


def test_doctor_cli_handles_bracket_paths(monkeypatch, tmp_path, capsys):
    _configure_isolated_state(monkeypatch, tmp_path)

    monkeypatch.setattr(ui_module.shutil, "which", lambda _cmd: "/usr/bin/fake")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["gum", "format"]:
            return SimpleNamespace(stdout="## Doctor\nPaths: [/tmp/foo[bar].json]", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(ui_module.subprocess, "run", fake_run)

    report = DoctorReport(
        checked={"codex": 1},
        issues=[DoctorIssue("codex", Path("/tmp/foo[bar].json"), "Codex sessions directory missing", "warning")],
    )
    monkeypatch.setattr("polylogue.cli.doctor.doctor_run", lambda **_: report)

    ui = UI(plain=False)
    env = CommandEnv(ui=ui)
    args = SimpleNamespace(codex_dir=None, claude_code_dir=None, limit=5, json=False)

    run_doctor_cli(args, env)

    captured = capsys.readouterr().out
    assert "Paths: [/tmp/foo[bar].json]" in captured
    assert any(cmd[:2] == ["gum", "format"] for cmd in calls)
