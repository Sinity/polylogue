"""Devtools rebinds checkout-naming variables inherited from another checkout."""

from __future__ import annotations

import os
from pathlib import Path

from devtools.checkout_guard import normalize_checkout_environment


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
