from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_polylogue(args: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(cwd / "polylogue.py"), *args]
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, check=False)


def _common_env(tmp_path: Path, force_plain: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    if force_plain:
        env["POLYLOGUE_FORCE_PLAIN"] = "1"
    else:
        env.pop("POLYLOGUE_FORCE_PLAIN", None)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    return env


def test_plain_config_init_creates_files(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env = _common_env(tmp_path)

    result = _run_polylogue(["config", "init", "--force"], cwd=repo_root, env=env)
    assert result.returncode == 0, result.stderr

    config_home = Path(env["XDG_CONFIG_HOME"]) / "polylogue"
    assert (config_home / "settings.json").exists()
    assert (config_home / "config.json").exists()


def test_plain_sync_codex_processes_session(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env = _common_env(tmp_path)

    codex_base = tmp_path / "codex"
    codex_base.mkdir(parents=True, exist_ok=True)
    fixture = repo_root / "tests" / "fixtures" / "golden" / "codex" / "codex-golden.jsonl"
    (codex_base / "session.jsonl").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    result = _run_polylogue(
        ["sync", "codex", "--base-dir", str(codex_base), "--all"],
        cwd=repo_root,
        env=env,
    )
    assert result.returncode == 0, result.stderr

    output_dir = Path(env["XDG_DATA_HOME"]) / "polylogue" / "archive" / "codex"
    assert any(output_dir.rglob("conversation.md"))
