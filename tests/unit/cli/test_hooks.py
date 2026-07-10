"""Behavioral tests for one-command harness hook wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.hooks import CLAUDE_CODE_EVENTS, CODEX_EVENTS, hook_main, resolve_events, settings_path


@pytest.fixture
def isolated_hook_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    return home


def _polylogue_commands(document: dict[str, object]) -> list[str]:
    commands: list[str] = []
    hooks = document.get("hooks")
    if not isinstance(hooks, dict):
        return commands
    for groups in hooks.values():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            for handler in group["hooks"]:
                if (
                    isinstance(handler, dict)
                    and isinstance(handler.get("command"), str)
                    and "polylogue-hook" in handler["command"]
                ):
                    commands.append(handler["command"])
    return commands


def test_install_recommended_is_idempotent_and_preserves_unrelated_hooks(isolated_hook_home: Path) -> None:
    target = settings_path("claude-code")
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "theme": "dark",
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "existing-policy"}],
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    first = runner.invoke(
        cli,
        ["--plain", "hooks", "install", "--harness", "claude-code", "--events", "recommended", "--json"],
        catch_exceptions=False,
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    assert first_payload["changed"] is True
    assert first_payload["written"] is True

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["theme"] == "dark"
    assert document["hooks"]["PreToolUse"][0]["hooks"] == [{"type": "command", "command": "existing-policy"}]
    commands = _polylogue_commands(document)
    assert len(commands) == 5
    assert "polylogue-hook SessionStart --provider claude-code" in commands

    before_second = target.read_bytes()
    second = runner.invoke(
        cli,
        ["--plain", "hooks", "install", "--harness", "claude-code", "--events", "recommended", "--json"],
        catch_exceptions=False,
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    assert second_payload["changed"] is False
    assert second_payload["written"] is False
    assert target.read_bytes() == before_second


def test_install_dry_run_does_not_create_settings(isolated_hook_home: Path) -> None:
    target = settings_path("claude-code")
    result = CliRunner().invoke(
        cli,
        ["--plain", "hooks", "install", "--harness", "claude-code", "--dry-run", "--json"],
        catch_exceptions=False,
    )
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["changed"] is True
    assert payload["written"] is False
    assert "SessionStart" in payload["diff"]
    assert not target.exists()


def test_uninstall_removes_only_polylogue_handlers(isolated_hook_home: Path) -> None:
    runner = CliRunner()
    install = runner.invoke(
        cli,
        ["--plain", "hooks", "install", "--harness", "claude-code"],
        catch_exceptions=False,
    )
    assert install.exit_code == 0, install.output
    target = settings_path("claude-code")
    document = json.loads(target.read_text(encoding="utf-8"))
    document["hooks"]["Stop"].append({"hooks": [{"type": "command", "command": "keep-me"}]})
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    uninstall = runner.invoke(
        cli,
        ["--plain", "hooks", "uninstall", "--harness", "claude-code", "--json"],
        catch_exceptions=False,
    )
    assert uninstall.exit_code == 0, uninstall.output
    remaining = json.loads(target.read_text(encoding="utf-8"))
    assert _polylogue_commands(remaining) == []
    assert remaining["hooks"]["Stop"] == [{"hooks": [{"type": "command", "command": "keep-me"}]}]


def test_codex_install_uses_hooks_json_and_preserves_config_toml(isolated_hook_home: Path) -> None:
    codex_home = isolated_hook_home / ".codex"
    codex_home.mkdir(parents=True)
    config = codex_home / "config.toml"
    config.write_text('[features]\nhooks = true\nmodel = "gpt-test"\n', encoding="utf-8")
    before = config.read_bytes()

    result = CliRunner().invoke(
        cli,
        ["--plain", "hooks", "install", "--harness", "codex", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert config.read_bytes() == before
    document = json.loads(settings_path("codex").read_text(encoding="utf-8"))
    commands = _polylogue_commands(document)
    assert len(commands) == 5
    assert all("--provider codex" in command for command in commands)


def test_hook_runtime_provider_override_records_codex_event(
    isolated_hook_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO('{"session_id":"session-1","model":"gpt-test"}'))
    assert hook_main(["SessionStart", "--provider", "codex"]) == 0
    sidecar = isolated_hook_home.parent / "data" / "polylogue" / "hooks" / "codex-session-1.jsonl"
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["provider"] == "codex"
    assert record["event_type"] == "SessionStart"


def test_all_event_catalogs_include_current_harness_lifecycle_edges() -> None:
    assert resolve_events("claude-code", "all") == CLAUDE_CODE_EVENTS
    assert {
        "InstructionsLoaded",
        "PostToolBatch",
        "SubagentStop",
        "WorktreeRemove",
        "PostCompact",
        "SessionEnd",
    } <= set(CLAUDE_CODE_EVENTS)
    assert resolve_events("codex", "all") == CODEX_EVENTS
    assert {"PreCompact", "PostCompact", "SubagentStart", "SubagentStop"} <= set(CODEX_EVENTS)


def test_status_reports_wired_vs_recommended(isolated_hook_home: Path) -> None:
    runner = CliRunner()
    install = runner.invoke(
        cli,
        ["--plain", "hooks", "install", "--harness", "claude-code", "--events", "SessionStart"],
        catch_exceptions=False,
    )
    assert install.exit_code == 0, install.output

    status = runner.invoke(
        cli,
        ["--plain", "hooks", "status", "--harness", "claude-code", "--json"],
        catch_exceptions=False,
    )
    payload = json.loads(status.output)["harnesses"][0]
    assert payload["wired_events"] == ["SessionStart"]
    assert payload["missing_recommended_events"] == [
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
    ]
