"""Tests for consolidated command structure (browse, maintain, config)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from polylogue.cli.app import main
from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.browse import run_browse_cli
from polylogue.cli.maintain import run_maintain_cli
from polylogue.commands import CommandEnv
from tests.conftest import _configure_state


class DummyConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        self.lines.append(text)


class DummyUI:
    plain = True

    def __init__(self):
        self.console = DummyConsole()


def _configure_isolated_state(monkeypatch, root: Path) -> None:
    """Configure isolated state for testing."""
    from polylogue import util as util_module
    from polylogue import paths as paths_module

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


# ===== Parser Structure Tests =====

def test_click_has_browse_command():
    """Verify browse command exists in Click CLI."""
    assert "browse" in click_cli.commands


def test_click_has_maintain_command():
    """Verify maintain command exists in Click CLI."""
    assert "maintain" in click_cli.commands


def test_click_has_config_subcommands():
    """Verify config command has init/set/show/edit subcommands."""
    config_group = click_cli.commands.get("config")
    assert config_group is not None
    assert hasattr(config_group, "commands")
    assert {"init", "set", "show", "edit"}.issubset(set(config_group.commands))


def test_click_browse_has_all_subcommands():
    """Verify browse has all expected subcommands."""
    browse_group = click_cli.commands.get("browse")
    assert browse_group is not None
    assert {"branches", "stats", "status", "runs", "inbox"}.issubset(set(browse_group.commands))


def test_click_maintain_has_all_subcommands():
    """Verify maintain has all expected subcommands."""
    maintain_group = click_cli.commands.get("maintain")
    assert maintain_group is not None
    assert {"prune", "doctor", "index", "restore"}.issubset(set(maintain_group.commands))


def test_click_rejects_old_inspect_command():
    """Verify old inspect command is not recognized."""
    assert "inspect" not in click_cli.commands


def test_click_rejects_old_prune_command():
    """Verify old prune command is not recognized."""
    assert "prune" not in click_cli.commands


def test_click_accepts_status_command():
    """Verify status remains available as a top-level command."""
    assert "status" in click_cli.commands


def test_click_rejects_old_settings_command():
    """Verify old settings command is not recognized."""
    assert "settings" not in click_cli.commands


def test_click_accepts_env_command():
    """Verify env command is available as a top-level command."""
    assert "env" in click_cli.commands


# ===== Dispatcher Tests =====

def test_browse_dispatcher_handles_branches():
    """Verify browse dispatcher correctly routes to branches."""
    from polylogue.cli.app import run_inspect_branches

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(
        browse_cmd="branches",
        provider=None,
        slug=None,
        conversation_id=None,
        min_branches=1,
        branch=None,
        diff=False,
        html_mode="auto",
        out=None,
        theme=None,
        no_picker=True,
        open=False,
    )

    # Should not raise
    run_browse_cli(args, env)


def test_browse_dispatcher_handles_stats():
    """Verify browse dispatcher correctly routes to stats."""
    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(
        browse_cmd="stats",
        dir=None,
        provider=None,
        slug=None,
        conversation_id=None,
        since=None,
        until=None,
        json=False,
        out=None,
    )

    # Should not raise
    run_browse_cli(args, env)


def test_browse_dispatcher_handles_status(monkeypatch, tmp_path):
    """Verify browse dispatcher correctly routes to status."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(
        browse_cmd="status",
        json=False,
        json_lines=False,
        watch=False,
        dump_only=False,
        interval=5.0,
        dump=None,
        dump_limit=100,
        runs_limit=200,
        providers=None,
        summary=None,
        summary_only=False,
    )

    # Should not raise
    run_browse_cli(args, env)


def test_browse_dispatcher_handles_runs(monkeypatch, tmp_path):
    """Verify browse dispatcher correctly routes to runs."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(
        browse_cmd="runs",
        limit=50,
        providers=None,
        commands=None,
        since=None,
        until=None,
        json=False,
    )

    # Should not raise
    run_browse_cli(args, env)


def test_browse_dispatcher_rejects_invalid_subcommand():
    """Verify browse dispatcher rejects invalid subcommands."""
    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(browse_cmd="invalid")

    with pytest.raises(SystemExit):
        run_browse_cli(args, env)


def test_maintain_dispatcher_handles_prune(monkeypatch, tmp_path):
    """Verify maintain dispatcher correctly routes to prune."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(
        maintain_cmd="prune",
        dirs=None,
        dry_run=True,
    )

    # Should not raise
    run_maintain_cli(args, env)


def test_maintain_dispatcher_handles_doctor(monkeypatch, tmp_path):
    """Verify maintain dispatcher correctly routes to doctor."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(
        maintain_cmd="doctor",
        codex_dir=None,
        claude_code_dir=None,
        limit=5,
        json=False,
    )

    # Should not raise
    run_maintain_cli(args, env)


def test_maintain_dispatcher_handles_index(monkeypatch, tmp_path):
    """Verify maintain dispatcher correctly routes to index."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(
        maintain_cmd="index",
        subcmd="check",
        repair=False,
        skip_qdrant=True,
        json=False,
    )

    # Should not raise
    run_maintain_cli(args, env)


def test_maintain_dispatcher_rejects_invalid_subcommand():
    """Verify maintain dispatcher rejects invalid subcommands."""
    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(maintain_cmd="invalid")

    with pytest.raises(SystemExit):
        run_maintain_cli(args, env)


def test_config_dispatcher_handles_show(monkeypatch, tmp_path, capsys):
    """Verify config dispatcher correctly routes to show."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = argparse.Namespace(
        config_cmd="show",
        json=True,
    )

    from polylogue.cli.app import _dispatch_config
    _dispatch_config(args, env)

    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert "settingsPath" in parsed
    assert "outputs" in parsed


# ===== Integration Tests =====

def test_browse_branches_integration(monkeypatch, tmp_path, capsys):
    """Test browse branches command end-to-end."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "browse", "branches", "--no-picker"])

    main()

    # Should not crash
    output = capsys.readouterr()
    assert output or True  # Just verify it runs


def test_browse_stats_integration(monkeypatch, tmp_path, capsys):
    """Test browse stats command end-to-end."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "browse", "stats", "--json"])

    main()

    output = capsys.readouterr().out
    # Should produce JSON output
    if output.strip():
        parsed = json.loads(output)
        assert isinstance(parsed, (dict, list))


def test_maintain_prune_dry_run_integration(monkeypatch, tmp_path, capsys):
    """Test maintain prune --dry-run command end-to-end."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "maintain", "prune", "--dry-run"])

    main()

    # Should not crash
    assert True


def test_config_show_json_integration(monkeypatch, tmp_path, capsys):
    """Test config show --json command end-to-end."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "config", "show", "--json"])

    main()

    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert "settingsPath" in parsed
    assert "outputs" in parsed


# ===== Backwards Compatibility Rejection Tests =====

def test_old_inspect_command_fails(monkeypatch, tmp_path):
    """Verify old 'inspect' command is rejected."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "inspect", "branches"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2  # argparse error


def test_old_prune_command_fails(monkeypatch, tmp_path):
    """Verify old 'prune' command is rejected."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "prune", "--dry-run"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2  # argparse error


def test_status_command_runs(monkeypatch, tmp_path):
    """Verify 'status' command is accepted and runs."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "status", "--summary", "-", "--summary-only"])

    main()


def test_env_command_runs(monkeypatch, tmp_path):
    """Verify 'env' command is accepted and runs."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "env"])

    main()


def test_old_settings_command_fails(monkeypatch, tmp_path):
    """Verify old 'settings' command is rejected."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "settings", "--html", "on"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2  # argparse error
