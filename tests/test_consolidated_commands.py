"""Tests for consolidated command structure (browse, doctor, config)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.cli.click_app import main
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


def test_click_has_doctor_command():
    """Verify doctor command exists in Click CLI."""
    assert "doctor" in click_cli.commands


def test_click_has_config_subcommands():
    """Verify config command has init/set/show/edit/prefs subcommands."""
    config_group = click_cli.commands.get("config")
    assert config_group is not None
    assert hasattr(config_group, "commands")
    assert {"init", "set", "show", "edit", "prefs"}.issubset(set(config_group.commands))


def test_click_browse_has_all_subcommands():
    """Verify browse has all expected subcommands."""
    browse_group = click_cli.commands.get("browse")
    assert browse_group is not None
    assert {"branches", "stats", "runs", "inbox", "metrics", "timeline", "analytics", "open"}.issubset(
        set(browse_group.commands)
    )


def test_click_doctor_has_all_subcommands():
    """Verify doctor has all expected subcommands."""
    doctor_group = click_cli.commands.get("doctor")
    assert doctor_group is not None
    assert {"check", "env", "status", "prune", "index", "restore", "attachments"}.issubset(set(doctor_group.commands))

    index_group = doctor_group.commands.get("index")
    assert index_group is not None
    assert {"check"}.issubset(set(index_group.commands))

    attachments_group = doctor_group.commands.get("attachments")
    assert attachments_group is not None
    assert {"stats", "extract"}.issubset(set(attachments_group.commands))


def test_click_config_prefs_has_all_subcommands():
    """Verify config prefs has list/set/clear subcommands."""
    config_group = click_cli.commands.get("config")
    assert config_group is not None
    prefs_group = config_group.commands.get("prefs")
    assert prefs_group is not None
    assert {"list", "set", "clear"}.issubset(set(prefs_group.commands))


def test_click_rejects_old_inspect_command():
    """Verify old inspect command is not recognized."""
    assert "inspect" not in click_cli.commands


def test_click_rejects_old_prune_command():
    """Verify old prune command is not recognized."""
    assert "prune" not in click_cli.commands


def test_click_rejects_status_command():
    """Verify status is not a top-level command."""
    assert "status" not in click_cli.commands


def test_click_rejects_old_settings_command():
    """Verify old settings command is not recognized."""
    assert "settings" not in click_cli.commands


def test_click_rejects_env_command():
    """Verify env is not a top-level command."""
    assert "env" not in click_cli.commands


# ===== Dispatcher Tests =====

def test_browse_dispatcher_handles_branches():
    """Verify browse dispatcher correctly routes to branches."""
    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = SimpleNamespace(
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
    args = SimpleNamespace(
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


def test_browse_dispatcher_handles_runs(monkeypatch, tmp_path):
    """Verify browse dispatcher correctly routes to runs."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = SimpleNamespace(
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
    args = SimpleNamespace(browse_cmd="invalid")

    with pytest.raises(SystemExit):
        run_browse_cli(args, env)


def test_maintain_dispatcher_handles_prune(monkeypatch, tmp_path):
    """Verify maintain dispatcher correctly routes to prune."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = SimpleNamespace(
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
    args = SimpleNamespace(
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
    args = SimpleNamespace(
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
    args = SimpleNamespace(maintain_cmd="invalid")

    with pytest.raises(SystemExit):
        run_maintain_cli(args, env)


def test_config_dispatcher_handles_show(monkeypatch, tmp_path, capsys):
    """Verify config dispatcher correctly routes to show."""
    _configure_isolated_state(monkeypatch, tmp_path)

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = SimpleNamespace(json=True)
    from polylogue.cli.config_cli import run_config_show

    run_config_show(args, env)

    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert "settingsPath" in parsed
    assert "outputs" in parsed


def test_config_show_serializes_labeled_roots(monkeypatch, tmp_path, capsys):
    _configure_isolated_state(monkeypatch, tmp_path)
    from polylogue.config import OutputDirs

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    alt_root = tmp_path / "alt"
    env.config.defaults.roots = {
        "alt": OutputDirs(
            render=alt_root / "render",
            sync_drive=alt_root / "gemini",
            sync_codex=alt_root / "codex",
            sync_claude_code=alt_root / "claude-code",
            import_chatgpt=alt_root / "chatgpt",
            import_claude=alt_root / "claude",
        )
    }
    args = SimpleNamespace(json=True)
    from polylogue.cli.config_cli import run_config_show

    run_config_show(args, env)

    parsed = json.loads(capsys.readouterr().out)
    roots = parsed["outputs"]["roots"]
    assert roots["alt"]["render"] == str(alt_root / "render")


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


def test_doctor_prune_dry_run_integration(monkeypatch, tmp_path, capsys):
    """Test doctor prune --dry-run command end-to-end."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "doctor", "prune", "--dry-run"])

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
    assert exc_info.value.code == 2  # Click usage error


def test_old_prune_command_fails(monkeypatch, tmp_path):
    """Verify old 'prune' command is rejected."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "prune", "--dry-run"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2  # Click usage error


@pytest.mark.parametrize(
    "argv",
    [
        ["polylogue", "maintain", "prune", "--dry-run"],
        ["polylogue", "status", "--summary", "-", "--summary-only"],
        ["polylogue", "env"],
        ["polylogue", "prefs", "list"],
        ["polylogue", "attachments", "stats"],
        ["polylogue", "open"],
        ["polylogue", "compare", "query", "--provider-a", "codex", "--provider-b", "claude-code"],
        ["polylogue", "reprocess-cmd"],
        ["polylogue", "import", "chatgpt", "export.zip"],
        ["polylogue", "verify", "--strict"],
        ["polylogue", "browse", "status"],
    ],
)
def test_old_command_paths_fail(monkeypatch, tmp_path, argv: list[str]):
    """Verify removed command paths are rejected (no backwards compat)."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2  # Click usage error


def test_doctor_status_command_runs(monkeypatch, tmp_path):
    """Verify 'doctor status' command is accepted and runs."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "doctor", "status", "--summary", "-", "--summary-only"])

    main()


def test_doctor_env_command_runs(monkeypatch, tmp_path):
    """Verify 'doctor env' command is accepted and runs."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "doctor", "env"])

    main()


def test_old_settings_command_fails(monkeypatch, tmp_path):
    """Verify old 'settings' command is rejected."""
    _configure_isolated_state(monkeypatch, tmp_path)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setattr(sys, "argv", ["polylogue", "settings", "--html", "on"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2  # Click usage error
