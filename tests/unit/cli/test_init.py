"""Tests for the first-run ``polylogue init`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.cli.commands.init import (
    detect_chat_sources,
    render_starter_toml,
    starter_config_path,
)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME and XDG roots at a clean tmp directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    return home


def test_detect_chat_sources_marks_missing_as_absent(isolated_home: Path) -> None:
    detected = detect_chat_sources()
    families = {d.family for d in detected}
    assert {"claude-code", "codex", "gemini-cli", "hermes", "antigravity", "hooks"}.issubset(families)
    # Fresh home: none of these directories exist yet.
    assert all(d.present is False for d in detected)


def test_detect_chat_sources_marks_present(isolated_home: Path) -> None:
    (isolated_home / ".claude" / "projects").mkdir(parents=True)
    (isolated_home / ".codex" / "sessions").mkdir(parents=True)

    detected = detect_chat_sources()
    by_family = {d.family: d for d in detected}
    assert by_family["claude-code"].present is True
    assert by_family["codex"].present is True
    assert by_family["gemini-cli"].present is False


def test_render_starter_toml_lists_present_and_comments_absent(isolated_home: Path) -> None:
    """The starter config loads, and only describes sources in comments.

    Anti-vacuity: emit ``[sources] roots`` or the unread ``[daemon] host``
    again and loading the rendered file raises ``ConfigError``.
    """
    from polylogue.config import load_polylogue_config

    (isolated_home / ".claude" / "projects").mkdir(parents=True)
    detected = detect_chat_sources()
    body = render_starter_toml(detected)

    assert "[archive]" in body
    assert "[daemon.api]" in body
    for line in body.splitlines():
        if ".claude/projects" in line or "codex/sessions" in line:
            assert line.lstrip().startswith("#"), f"sources are described, not configured: {line!r}"
    assert any("claude/projects" in line and "(present)" in line for line in body.splitlines())

    config = isolated_home / "polylogue.toml"
    config.write_text(body, encoding="utf-8")
    load_polylogue_config(config_path=config)


def test_render_starter_toml_persists_a_hermes_root_override(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detected ``POLYLOGUE_HERMES_ROOT`` override survives past init.

    ``detect_chat_sources`` reports the override's path as present, but the
    renderer only listed it in a comment: once the one-shot environment
    variable that produced the starter file is gone, ``polylogued run`` fell
    back to ``~/.hermes`` and never watched the source ``init`` claimed to
    detect and record.

    Anti-vacuity: rendering only the comment line (the old behavior) makes
    the ``sources.hermes.root`` assertions below fail.
    """
    from polylogue.config import load_polylogue_config

    hermes_root = isolated_home / "srv-hermes"
    hermes_root.mkdir()
    monkeypatch.setenv("POLYLOGUE_HERMES_ROOT", str(hermes_root))

    detected = detect_chat_sources()
    body = render_starter_toml(detected)
    assert "[sources.hermes]" in body
    assert f'root = "{hermes_root}"' in body

    config = isolated_home / "polylogue.toml"
    config.write_text(body, encoding="utf-8")
    monkeypatch.delenv("POLYLOGUE_HERMES_ROOT", raising=False)
    settings = load_polylogue_config(config_path=config)
    assert settings.hermes_root == str(hermes_root)


def test_init_command_writes_starter_config(isolated_home: Path) -> None:
    (isolated_home / ".claude" / "projects").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(cli, ["--plain", "init"], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    target = starter_config_path()
    assert target.exists()
    body = target.read_text(encoding="utf-8")
    assert "[archive]" in body
    assert ".claude/projects" in body
    # A first, successful write must say so — not falsely claim the config
    # "already exists" (the message used to re-check target.exists() *after*
    # writing it, which is always True; polylogue-jnj.8).
    assert "Wrote starter config" in result.output
    assert "already exists" not in result.output.lower()
    assert "polylogue demo seed" in result.output


def test_init_command_dry_run_does_not_write(isolated_home: Path) -> None:
    runner = CliRunner()
    target = starter_config_path()
    assert not target.exists()
    result = runner.invoke(cli, ["--plain", "init", "--dry-run"], catch_exceptions=False)
    assert result.exit_code == 0
    assert not target.exists()
    assert "[archive]" in result.output


def test_init_command_refuses_overwrite_without_force(isolated_home: Path) -> None:
    target = starter_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# preserved\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["--plain", "init"], catch_exceptions=False)
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "# preserved\n"
    assert "already exists" in result.output.lower()


def test_init_command_force_overwrites(isolated_home: Path) -> None:
    target = starter_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# stale\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["--plain", "init", "--force"], catch_exceptions=False)
    assert result.exit_code == 0
    body = target.read_text(encoding="utf-8")
    assert "[archive]" in body
    assert body != "# stale\n"


def test_init_command_json_format_is_machine_readable(isolated_home: Path) -> None:
    (isolated_home / ".claude" / "projects").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(cli, ["--plain", "init", "--dry-run", "--format", "json"], catch_exceptions=False)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["written"] is False
    assert "claude-code" in payload["present_families"]
    assert any(d["family"] == "codex" and d["present"] is False for d in payload["detected"])


def test_status_reports_absent_daemon_on_a_fresh_install(isolated_home: Path) -> None:
    """Operational status reports the daemon boundary when no snapshot exists.

    A fresh install has no archive, so the bounded first-run diagnostic is the
    honest answer. It used to be preempted by the ``--daemon-url`` refusal,
    which rendered as "Daemon: running" (polylogue-2d8oq). Anti-vacuity: a
    status route that claims liveness it never observed turns this red.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--plain", "--no-daemon", "ops", "status"], catch_exceptions=False, env={"POLYLOGUE_DAEMON": "off"}
    )
    assert "Daemon: running" not in result.output
    assert "Daemon: not running" in result.output
    assert "No archive found." in result.output


def test_status_reports_absent_daemon_after_init(isolated_home: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(cli, ["--plain", "init"], catch_exceptions=False)
    assert init_result.exit_code == 0

    result = runner.invoke(
        cli, ["--plain", "--no-daemon", "ops", "status"], catch_exceptions=False, env={"POLYLOGUE_DAEMON": "off"}
    )
    assert "Daemon: running" not in result.output
    assert "Daemon: not running" in result.output
