"""Subprocess-based CLI runner for tests that need isolated module state.

The CLI tests need subprocess isolation because:
1. Module-level caching in paths.py evaluates XDG paths at import time
2. monkeypatch.setenv() runs after import, so cached values aren't updated
3. CliRunner runs in-process, so module reloading is fragile and incomplete

This helper runs the CLI as a subprocess with custom environment variables,
ensuring complete isolation of module state.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CliResult:
    """Result from running CLI via subprocess."""

    exit_code: int
    stdout: str
    stderr: str
    output: str  # Combined stdout + stderr for compatibility

    @property
    def success(self) -> bool:
        return self.exit_code == 0


def run_cli(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 60.0,
) -> CliResult:
    """Run polylogue CLI as subprocess with isolated environment.

    Args:
        args: CLI arguments (e.g., ["sync", "--stage", "all"])
        env: Environment variables to set (merged with minimal clean env)
        cwd: Working directory for the command
        timeout: Maximum execution time in seconds

    Returns:
        CliResult with exit_code, stdout, stderr, and combined output
    """
    # Start with minimal environment to avoid inheriting user's config
    # Must include HOME for uv/Python to function, but use temp HOME to isolate config
    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        # Required for uv to work
        "UV_SYSTEM_PYTHON": "1",
        # Disable Qdrant in tests
        "QDRANT_URL": "",
        "QDRANT_API_KEY": "",
        # Use test-provided HOME if set via env, otherwise use system HOME
        "HOME": os.environ.get("HOME", "/tmp"),
    }

    # Add any custom environment variables (these override the defaults)
    if env:
        clean_env.update(env)

    # Run from the project root to ensure uv finds the project
    project_root = Path(__file__).parent.parent.parent

    result = subprocess.run(
        ["uv", "run", "polylogue"] + args,
        capture_output=True,
        text=True,
        env=clean_env,
        cwd=project_root,  # Always run from project root for uv
        timeout=timeout,
    )

    return CliResult(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        output=result.stdout + result.stderr,
    )


def setup_isolated_workspace(tmp_path: Path) -> dict[str, Any]:
    """Create an isolated workspace for subprocess CLI tests.

    The key to isolation is setting HOME to a temp directory. This ensures:
    - CLAUDE_CODE_PATH (Path.home() / ".claude/projects") won't find real data
    - CODEX_PATH (Path.home() / ".codex/sessions") won't find real data
    - XDG directories fall back to HOME-based defaults

    Returns a dict with:
        - env: Environment variables to pass to run_cli
        - paths: Dict of key paths (archive_root, render_root, inbox, db_path)
    """
    # Use a fake HOME to isolate from real user data
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)

    # XDG directories relative to fake home
    data_dir = fake_home / ".local/share"
    state_dir = fake_home / ".local/state"
    config_dir = fake_home / ".config"

    # Polylogue-specific directories
    polylogue_data = data_dir / "polylogue"
    polylogue_config = config_dir / "polylogue"
    polylogue_state = state_dir / "polylogue"

    # Create inbox (this is where test data goes)
    inbox = polylogue_data / "inbox"
    render_root = polylogue_data / "render"
    db_path = polylogue_data / "polylogue.db"

    for path in [polylogue_data, polylogue_config, polylogue_state, inbox, render_root]:
        path.mkdir(parents=True, exist_ok=True)

    # Environment variables - HOME is the key to isolation
    env = {
        "HOME": str(fake_home),
        "XDG_DATA_HOME": str(data_dir),
        "XDG_STATE_HOME": str(state_dir),
        "XDG_CONFIG_HOME": str(config_dir),
        "XDG_CACHE_HOME": str(fake_home / ".cache"),
        # Disable Qdrant for tests
        "QDRANT_URL": "",
        "QDRANT_API_KEY": "",
    }

    paths = {
        "home": fake_home,
        "archive_root": polylogue_data,
        "render_root": render_root,
        "inbox": inbox,
        "data_dir": data_dir,
        "state_dir": state_dir,
        "db_path": db_path,
    }

    return {"env": env, "paths": paths}
