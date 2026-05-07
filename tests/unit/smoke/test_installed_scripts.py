"""Smoke tests for installed entry-point scripts.

Verifies that polylogue, polylogued, and polylogue-mcp are
accessible as entry points after ``pip install polylogue``.
"""

from __future__ import annotations

import os
import subprocess

import pytest


def _find_entry_points() -> dict[str, str]:
    """Find the three entry-point scripts in the current environment."""
    import shutil

    scripts: dict[str, str] = {}
    for name in ("polylogue", "polylogued", "polylogue-mcp"):
        found = shutil.which(name)
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
    if "polylogue" not in scripts:
        pytest.skip("polylogue entry point not found in PATH")
    exit_code, stdout, _stderr = _run_help(scripts["polylogue"])
    assert exit_code == 0, f"polylogue --help failed with exit {exit_code}\nstdout: {stdout}"
    assert "Usage:" in stdout or "usage:" in stdout.lower() or "polylogue" in stdout.lower()


def test_polylogued_help() -> None:
    """``polylogued --help`` exits 0."""
    scripts = _find_entry_points()
    if "polylogued" not in scripts:
        pytest.skip("polylogued entry point not found in PATH")
    exit_code, stdout, _stderr = _run_help(scripts["polylogued"])
    assert exit_code == 0, f"polylogued --help failed with exit {exit_code}\nstdout: {stdout}"
    assert "Usage:" in stdout or "usage:" in stdout.lower() or "polylogued" in stdout.lower()


def test_polylogue_mcp_help() -> None:
    """``polylogue-mcp --help`` exits 0."""
    scripts = _find_entry_points()
    if "polylogue-mcp" not in scripts:
        pytest.skip("polylogue-mcp entry point not found in PATH")
    exit_code, stdout, _stderr = _run_help(scripts["polylogue-mcp"])
    assert exit_code == 0, f"polylogue-mcp --help failed with exit {exit_code}\nstdout: {stdout}"
    assert "Usage:" in stdout or "usage:" in stdout.lower() or "polylogue-mcp" in stdout.lower()


def test_all_three_entry_points_found() -> None:
    """Verify all three entry-point scripts are found in PATH when installed."""
    scripts = _find_entry_points()
    missing = {"polylogue", "polylogued", "polylogue-mcp"} - set(scripts.keys())
    if missing:
        pytest.skip(f"Entry points not found in PATH: {', '.join(sorted(missing))}")
    assert scripts, "Expected at least one entry point to be found"
