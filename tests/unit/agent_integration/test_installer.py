"""Ownership and native-client contract tests for the production installer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import pytest
import yaml

from polylogue.agent_integration.assets import agent_asset_digest, read_agent_asset
from polylogue.agent_integration.installer import (
    AgentIntegrationManager,
    InstallOptions,
    NativeConfigConflict,
    StateIntegrityError,
)


def _commands(root: Path) -> tuple[Path, Path]:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    commands = []
    for name in ("polylogue", "polylogue-mcp"):
        path = bin_dir / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        commands.append(path)
    return commands[0], commands[1]


def _manager(
    tmp_path: Path, *, extra_env: dict[str, str] | None = None
) -> tuple[AgentIntegrationManager, Path, Path, Path]:
    home = tmp_path / "home"
    home.mkdir()
    polylogue, server = _commands(tmp_path)
    environment = {"HOME": str(home), "PATH": str(polylogue.parent)}
    environment.update(extra_env or {})
    return AgentIntegrationManager(home=home, environment=environment), home, polylogue, server


def _options(polylogue: Path, server: Path, **changes: object) -> InstallOptions:
    values: dict[str, object] = {
        "clients": ("claude-code", "codex", "gemini", "hermes"),
        "role": "read",
        "archive_root": server.parent.parent / "archive",
        "config_path": server.parent.parent / "polylogue.toml",
        "server_command": str(server),
        "polylogue_command": str(polylogue),
    }
    values.update(changes)
    return InstallOptions(**values)  # type: ignore[arg-type]


def test_all_clients_install_native_mcp_and_guidance(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)

    receipt = manager.install(_options(polylogue, server))

    assert receipt["ok"] is True
    claude_mcp = json.loads((home / ".claude.json").read_text())
    assert claude_mcp["mcpServers"]["polylogue"]["args"] == ["--role", "read"]
    claude_settings = json.loads((home / ".claude" / "settings.json").read_text())
    command = claude_settings["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert command.endswith(" agent session-start --client claude-code")
    assert "[mcp_servers.polylogue]" in (home / ".codex" / "config.toml").read_text()
    assert "# >>> polylogue agent integration:codex-guidance >>>" in (home / ".codex" / "AGENTS.md").read_text()
    gemini = json.loads((home / ".gemini" / "settings.json").read_text())
    assert gemini["mcpServers"]["polylogue"]["env"]["POLYLOGUE_CONFIG"].endswith("polylogue.toml")
    hermes = yaml.safe_load((home / ".hermes" / "config.yaml").read_text())
    assert hermes["mcp_servers"]["polylogue"]["command"] == str(server)
    skill = (home / ".hermes" / "skills" / "productivity" / "polylogue" / "SKILL.md").read_text()
    assert skill.startswith("---\nname: polylogue\n")
    assert read_agent_asset("standing-manual.md") in skill
    assert manager.doctor()["ok"] is True


def test_install_is_idempotent_without_rewriting_files(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    options = _options(polylogue, server)
    manager.install(options)
    files = [path for path in home.rglob("*") if path.is_file()]
    mtimes = {path: path.stat().st_mtime_ns for path in files}
    time.sleep(0.01)

    manager.install(options)

    assert all(path.stat().st_mtime_ns == mtimes[path] for path in files)


def test_role_and_archive_upgrade_reconciles_owned_values(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    manager.install(_options(polylogue, server))

    manager.install(
        _options(
            polylogue,
            server,
            role="review",
            archive_root=tmp_path / "second-archive",
            config_path=tmp_path / "second.toml",
        )
    )

    claude = json.loads((home / ".claude.json").read_text())["mcpServers"]["polylogue"]
    assert claude["args"] == ["--role", "review"]
    assert claude["env"]["POLYLOGUE_ARCHIVE_ROOT"] == str((tmp_path / "second-archive").resolve())
    assert "review" in (home / ".codex" / "config.toml").read_text()


def test_operator_additions_survive_uninstall_and_clean_owned_dirs(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    manager.install(_options(polylogue, server))
    settings_path = home / ".gemini" / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["theme"] = "operator-choice"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    operator_file = home / ".codex" / "operator.txt"
    operator_file.write_text("keep\n")

    receipt = manager.uninstall()

    assert receipt["ok"] is True
    assert json.loads(settings_path.read_text()) == {"theme": "operator-choice"}
    assert operator_file.read_text() == "keep\n"
    assert not (home / ".claude").exists()
    assert not (home / ".hermes").exists()
    assert not (home / ".local").exists()


def test_clean_home_uninstall_removes_only_installer_created_tree(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    manager.install(_options(polylogue, server))

    manager.uninstall()

    assert list(home.iterdir()) == []


def test_preexisting_equal_native_value_is_not_claimed_or_removed(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    expected = {
        "command": str(server),
        "args": ["--role", "read"],
        "env": {
            "POLYLOGUE_ARCHIVE_ROOT": str((tmp_path / "archive").resolve()),
            "POLYLOGUE_CONFIG": str((tmp_path / "polylogue.toml").resolve()),
        },
    }
    path = home / ".gemini" / "settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"mcpServers": {"polylogue": expected}}, indent=2) + "\n")

    manager.install(_options(polylogue, server, clients=("gemini",), guidance="off", include_reference=False))
    status = manager.status()
    operation = status["clients"][0]["operations"][0]  # type: ignore[index]
    assert operation["state"] == "satisfied-unowned"
    manager.uninstall()

    assert json.loads(path.read_text())["mcpServers"]["polylogue"] == expected


def test_operator_conflict_fails_closed_and_rolls_back_other_clients(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    codex = home / ".codex" / "config.toml"
    codex.parent.mkdir()
    codex.write_text('[mcp_servers.polylogue]\ncommand = "operator-server"\n')

    with pytest.raises(NativeConfigConflict, match="operator-owned Codex MCP"):
        manager.install(_options(polylogue, server, clients=("claude-code", "codex")))

    assert not (home / ".claude.json").exists()
    assert codex.read_text() == '[mcp_servers.polylogue]\ncommand = "operator-server"\n'


def test_drifted_owned_content_is_retained_on_uninstall(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    manager.install(_options(polylogue, server, clients=("hermes",)))
    skill = home / ".hermes" / "skills" / "productivity" / "polylogue" / "SKILL.md"
    skill.write_text(skill.read_text() + "\noperator edit\n")

    receipt = manager.uninstall(("hermes",))

    assert receipt["ok"] is False
    assert skill.exists()
    assert manager.status()["blocking"] is True


def test_corrupt_state_blocks_status_doctor_and_uninstall(tmp_path: Path) -> None:
    manager, _, polylogue, server = _manager(tmp_path)
    manager.install(_options(polylogue, server, clients=("gemini",)))
    state = json.loads(manager.state_path.read_text())
    state["clients"]["gemini"]["role"] = "admin"
    manager.state_path.write_text(json.dumps(state))

    assert manager.status()["state_integrity"] == "failed"
    assert manager.doctor()["blocking"] is True
    with pytest.raises(StateIntegrityError):
        manager.uninstall()


def test_symlinked_native_config_is_refused(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    target = tmp_path / "operator-settings.json"
    target.write_text("{}\n")
    gemini_root = home / ".gemini"
    gemini_root.mkdir()
    (gemini_root / "settings.json").symlink_to(target)

    with pytest.raises(NativeConfigConflict, match="symlinked"):
        manager.install(_options(polylogue, server, clients=("gemini",)))

    assert target.read_text() == "{}\n"


def test_opt_down_removes_exact_guidance_but_keeps_mcp(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    manager.install(_options(polylogue, server, clients=("codex",)))

    manager.install(
        _options(
            polylogue,
            server,
            clients=("codex",),
            guidance="mcp-only",
            include_reference=False,
        )
    )

    assert not (home / ".codex" / "AGENTS.md").exists()
    assert not (home / ".codex" / "polylogue-reference.md").exists()
    assert "[mcp_servers.polylogue]" in (home / ".codex" / "config.toml").read_text()
    doctor = manager.doctor()
    assert doctor["ok"] is False
    assert any("opted down" in problem for problem in cast(list[str], doctor["problems"]))


def test_codex_guidance_relocates_when_override_becomes_active(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    options = _options(polylogue, server, clients=("codex",))
    manager.install(options)
    agents = home / ".codex" / "AGENTS.md"
    override = home / ".codex" / "AGENTS.override.md"
    assert agents.exists()
    override.write_text("Operator global override.\n")
    assert manager.doctor()["blocking"] is True

    manager.install(options)

    if agents.exists():
        assert "polylogue agent integration" not in agents.read_text()
    assert "Operator global override." in override.read_text()
    assert "polylogue agent integration:codex-guidance" in override.read_text()
    assert manager.doctor()["ok"] is True


def test_provider_specific_profile_roots_are_respected(tmp_path: Path) -> None:
    roots = {
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude-profile"),
        "CODEX_HOME": str(tmp_path / "codex-profile"),
        "GEMINI_CLI_HOME": str(tmp_path / "gemini-profile"),
        "HERMES_HOME": str(tmp_path / "hermes-profile"),
    }
    manager, _, polylogue, server = _manager(tmp_path, extra_env=roots)

    manager.install(_options(polylogue, server))

    assert (tmp_path / "claude-profile" / "settings.json").exists()
    assert (tmp_path / "claude-profile" / ".claude.json").exists()
    assert (tmp_path / "codex-profile" / "config.toml").exists()
    assert (tmp_path / "gemini-profile" / ".gemini" / "settings.json").exists()
    assert (tmp_path / "hermes-profile" / "config.yaml").exists()


def test_replace_clients_removes_unselected_exact_operations(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    manager.install(_options(polylogue, server, clients=("claude-code", "gemini")))

    receipt = manager.install(_options(polylogue, server, clients=("gemini",), replace_clients=True))

    assert receipt["removed_clients"] == ["claude-code"]
    assert not (home / ".claude.json").exists()
    assert (home / ".gemini" / "settings.json").exists()


def test_state_records_current_asset_digest(tmp_path: Path) -> None:
    manager, _, polylogue, server = _manager(tmp_path)
    manager.install(_options(polylogue, server, clients=("gemini",)))

    state = json.loads(manager.state_path.read_text())

    assert state["asset_digest"] == agent_asset_digest()
    assert state["integrity"]


def test_malformed_claude_session_start_shape_is_not_overwritten(tmp_path: Path) -> None:
    manager, home, polylogue, server = _manager(tmp_path)
    settings = home / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"hooks": {"SessionStart": {"operator": "value"}}}) + "\n")

    with pytest.raises(NativeConfigConflict, match="SessionStart must be a list"):
        manager.install(_options(polylogue, server, clients=("claude-code",)))

    assert json.loads(settings.read_text()) == {"hooks": {"SessionStart": {"operator": "value"}}}
    assert not (home / ".claude.json").exists()
