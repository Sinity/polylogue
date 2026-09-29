from __future__ import annotations

from pathlib import Path

import pytest

from devtools.isolated_environment import isolated_home_environment
from polylogue.config import resolve_runtime_config


def test_isolated_environment_disables_the_cwd_config_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A project-local ``polylogue.toml`` in the launch cwd must not leak in.

    ``_user_config_path`` falls through to ``<cwd>/polylogue.toml`` when no
    ``POLYLOGUE_CONFIG`` override is set. The dev-loop proof deliberately
    runs the isolated daemon from the repository root, so simply dropping
    an inherited ``POLYLOGUE_CONFIG`` (rather than pointing it at the
    isolated home) leaves that fallback live.

    Anti-vacuity: popping ``POLYLOGUE_CONFIG`` instead of overriding it lets
    ``resolve_runtime_config`` pick up a real ``<cwd>/polylogue.toml`` (here
    standing in for an operator's untracked project-local file) even though
    HOME and every XDG root were replaced.
    """
    home = tmp_path / "isolated-home"
    home.mkdir()
    project_cwd = tmp_path / "checkout"
    project_cwd.mkdir()
    (project_cwd / "polylogue.toml").write_text(
        '[maintenance]\nbackup_verify_tmpdir = "/an/operators/real/scratch"\n', encoding="utf-8"
    )

    env = isolated_home_environment({}, home=home)
    assert env["POLYLOGUE_CONFIG"] == str(home / "unconfigured-polylogue.toml")

    monkeypatch.chdir(project_cwd)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = resolve_runtime_config()
    assert config.backup_verify_tmpdir is None
