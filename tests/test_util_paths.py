from __future__ import annotations

from pathlib import Path

from polylogue import util


def test_resolve_claude_code_project_root_prefers_config_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("POLYLOGUE_CLAUDE_CODE_PROJECTS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    config_root = tmp_path / ".config" / "claude" / "projects"
    config_root.mkdir(parents=True, exist_ok=True)

    resolved = util.resolve_claude_code_project_root()

    assert resolved == config_root


def test_resolve_claude_code_project_root_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom-projects"
    custom.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("POLYLOGUE_CLAUDE_CODE_PROJECTS", str(custom))

    resolved = util.resolve_claude_code_project_root()

    assert resolved == custom
