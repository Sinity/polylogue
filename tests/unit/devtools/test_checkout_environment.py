"""Devtools rebinds checkout-naming variables inherited from another checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from devtools.checkout_guard import (
    ForeignInterpreterError,
    assert_interpreter_belongs_to,
    normalize_checkout_environment,
)


def _checkout(root: Path) -> Path:
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "polylogue"\n', encoding="utf-8")
    (root / ".venv" / "bin").mkdir(parents=True)
    return root


def test_variables_naming_another_checkout_are_rebound_to_this_one(tmp_path: Path) -> None:
    """A shell started in the primary checkout cannot steer a worktree's tools.

    Anti-vacuity: skip any branch of ``normalize_checkout_environment`` and the
    corresponding variable below still names the primary checkout.
    """
    primary = _checkout(tmp_path / "primary")
    worktree = _checkout(tmp_path / "worktree")
    environ = {
        "POLYLOGUE_REPO_ROOT": str(primary),
        "POLYLOGUE_ROOT": str(primary),
        "VIRTUAL_ENV": str(primary / ".venv"),
        "PATH": os.pathsep.join([str(primary / ".venv" / "bin"), "/usr/bin"]),
        "PYTHONPATH": os.pathsep.join([str(primary / "src"), "/nix/store/site-packages"]),
    }

    corrected = normalize_checkout_environment(worktree, environ)

    assert environ["POLYLOGUE_REPO_ROOT"] == str(worktree.resolve())
    assert environ["POLYLOGUE_ROOT"] == str(worktree.resolve())
    assert environ["VIRTUAL_ENV"] == str(worktree.resolve() / ".venv")
    assert environ["PATH"].split(os.pathsep) == [str(worktree.resolve() / ".venv" / "bin"), "/usr/bin"]
    assert environ["PYTHONPATH"] == "/nix/store/site-packages"
    assert len(corrected) == 5


def test_an_environment_already_bound_to_this_checkout_is_left_alone(tmp_path: Path) -> None:
    worktree = _checkout(tmp_path / "worktree").resolve()
    environ = {
        "POLYLOGUE_ROOT": str(worktree),
        "VIRTUAL_ENV": str(worktree / ".venv"),
        "PATH": os.pathsep.join([str(worktree / ".venv" / "bin"), "/usr/bin"]),
    }
    before = dict(environ)

    assert normalize_checkout_environment(worktree, environ) == []
    assert environ == before


def test_the_checkout_venv_is_moved_first_and_relative_entries_are_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: keep an existing venv entry in place, or skip relative
    resolution, and ``/opt/old/bin`` still wins or ``../primary`` survives."""
    primary = _checkout(tmp_path / "primary")
    worktree = _checkout(tmp_path / "worktree").resolve()
    (primary / ".direnv" / "bin").mkdir(parents=True)
    monkeypatch.chdir(worktree)
    environ = {"PATH": os.pathsep.join(["/opt/old/bin", "../primary/.direnv/bin", str(worktree / ".venv" / "bin")])}

    normalize_checkout_environment(worktree, environ)

    assert environ["PATH"].split(os.pathsep) == [str(worktree / ".venv" / "bin"), "/opt/old/bin"]


def test_an_interpreter_from_another_checkout_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: drop the prefix check and the foreign venv runs unrefused."""
    primary = _checkout(tmp_path / "primary")
    worktree = _checkout(tmp_path / "worktree")
    monkeypatch.setattr(sys, "prefix", str(primary / ".venv"))

    with pytest.raises(ForeignInterpreterError, match="another checkout's interpreter"):
        assert_interpreter_belongs_to(worktree, context="devtools")

    monkeypatch.setattr(sys, "prefix", str(worktree / ".venv"))
    assert_interpreter_belongs_to(worktree, context="devtools")
    monkeypatch.setattr(sys, "prefix", "/nix/store/python")
    assert_interpreter_belongs_to(worktree, context="devtools")
