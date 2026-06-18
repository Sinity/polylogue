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
    (isolated_home / ".claude" / "projects").mkdir(parents=True)
    detected = detect_chat_sources()
    body = render_starter_toml(detected)

    assert "[archive]" in body
    assert "[sources]" in body
    assert "[daemon]" in body
    assert "roots = [" in body
    assert ".claude/projects" in body
    # Absent ones must appear only as commented hints.
    for line in body.splitlines():
        if "codex/sessions" in line:
            assert line.lstrip().startswith("#"), f"absent source must be commented: {line!r}"


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


def test_status_first_run_hint_suggests_init(isolated_home: Path) -> None:
    """Bare status on a fresh install must point at ``polylogue init``."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--plain", "ops", "status"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "polylogue init" in result.output


def test_status_first_run_hint_drops_init_after_init(isolated_home: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(cli, ["--plain", "init"], catch_exceptions=False)
    assert init_result.exit_code == 0

    result = runner.invoke(cli, ["--plain", "ops", "status"], catch_exceptions=False)
    assert result.exit_code == 0
    # Once the starter config exists, the hint shifts to the daemon.
    assert "polylogue init" not in result.output
    assert "polylogued run" in result.output
