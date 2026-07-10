"""Smoke tests for installed entry-point scripts.

Verifies that polylogue, polylogued, and polylogue-mcp are
accessible as entry points after ``pip install polylogue``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME_SCRIPTS = ("polylogue", "polylogued", "polylogue-mcp")


def _find_entry_points() -> dict[str, str]:
    """Find entry points installed for this checkout or active interpreter."""
    repo_bin = Path(__file__).resolve().parents[3] / ".venv" / "bin"
    if (repo_bin / "python").is_file():
        return {name: str(repo_bin / name) for name in RUNTIME_SCRIPTS if (repo_bin / name).is_file()}

    interpreter_bin = Path(sys.executable).resolve().parent
    scripts: dict[str, str] = {}
    for name in RUNTIME_SCRIPTS:
        interpreter_script = interpreter_bin / name
        found = str(interpreter_script) if interpreter_script.is_file() else shutil.which(name)
        if found is not None:
            scripts[name] = found
    return scripts


def _run_help(executable: str) -> tuple[int, str, str]:
    """Run ``<executable> --help`` and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "POLYLOGUE_FORCE_PLAIN": "1"},
    )
    return result.returncode, result.stdout, result.stderr


def test_polylogue_help() -> None:
    """``polylogue --help`` exits 0."""
    scripts = _find_entry_points()
    assert "polylogue" in scripts, "polylogue entry point not installed"
    exit_code, stdout, stderr = _run_help(scripts["polylogue"])
    assert exit_code == 0, f"polylogue --help failed with exit {exit_code}\nstdout: {stdout}\nstderr: {stderr}"
    assert "Usage:" in stdout or "usage:" in stdout.lower() or "polylogue" in stdout.lower()


def test_polylogued_help() -> None:
    """``polylogued --help`` exits 0."""
    scripts = _find_entry_points()
    assert "polylogued" in scripts, "polylogued entry point not installed"
    exit_code, stdout, stderr = _run_help(scripts["polylogued"])
    assert exit_code == 0, f"polylogued --help failed with exit {exit_code}\nstdout: {stdout}\nstderr: {stderr}"
    assert "Usage:" in stdout or "usage:" in stdout.lower() or "polylogued" in stdout.lower()


def test_polylogue_mcp_help() -> None:
    """``polylogue-mcp --help`` exits 0."""
    scripts = _find_entry_points()
    assert "polylogue-mcp" in scripts, "polylogue-mcp entry point not installed"
    exit_code, stdout, stderr = _run_help(scripts["polylogue-mcp"])
    assert exit_code == 0, f"polylogue-mcp --help failed with exit {exit_code}\nstdout: {stdout}\nstderr: {stderr}"
    assert "Usage:" in stdout or "usage:" in stdout.lower() or "polylogue-mcp" in stdout.lower()


def test_all_three_entry_points_found() -> None:
    """Verify all three entry-point scripts are found in PATH when installed."""
    scripts = _find_entry_points()
    missing = {"polylogue", "polylogued", "polylogue-mcp"} - set(scripts.keys())
    assert not missing, f"Installed entry points not found: {', '.join(sorted(missing))}"
    assert scripts, "Expected at least one entry point to be found"
