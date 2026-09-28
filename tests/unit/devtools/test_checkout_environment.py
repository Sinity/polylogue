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


def test_ownership_ignores_git_ceilings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ceiling that hides the primary checkout from git discovery does not make its venv ours.

    Anti-vacuity: use ceiling-aware discovery for ownership and both the PATH
    entry and the interpreter below are accepted.
    """
    primary = _checkout(tmp_path / "primary")
    worktree = _checkout(tmp_path / "worktree")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(primary))
    environ = {"PATH": os.pathsep.join([str(primary / ".venv" / "bin"), "/usr/bin"])}

    normalize_checkout_environment(worktree, environ)

    assert str(primary / ".venv" / "bin") not in environ["PATH"].split(os.pathsep)
    monkeypatch.setattr(sys, "prefix", str(primary / ".venv"))
    with pytest.raises(ForeignInterpreterError):
        assert_interpreter_belongs_to(worktree, context="devtools")


def test_foreign_entries_leave_the_running_import_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: skip the ``sys.path`` filter and the primary site-packages stays importable."""
    primary = _checkout(tmp_path / "primary")
    worktree = _checkout(tmp_path / "worktree")
    foreign = str(primary / ".venv" / "lib")
    monkeypatch.setattr(sys, "path", [str(worktree), foreign, "/nix/store/site-packages"])
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})

    corrected = normalize_checkout_environment(worktree)

    assert sys.path == [str(worktree), "/nix/store/site-packages"]
    assert any("sys.path" in item for item in corrected)


def test_a_checkout_nested_inside_this_one_is_foreign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: treat every descendant of ``root`` as owned and the nested
    clone's venv is kept on PATH and its interpreter accepted."""
    outer = _checkout(tmp_path / "outer")
    nested = _checkout(outer / "vendor" / "clone")
    environ = {
        "PATH": os.pathsep.join([str(nested / ".venv" / "bin"), "/usr/bin"]),
        "VIRTUAL_ENV": str(nested / ".venv"),
    }

    normalize_checkout_environment(outer, environ)

    assert str(nested / ".venv" / "bin") not in environ["PATH"].split(os.pathsep)
    assert environ["VIRTUAL_ENV"] == str(outer.resolve() / ".venv")
    monkeypatch.setattr(sys, "prefix", str(nested / ".venv"))
    with pytest.raises(ForeignInterpreterError):
        assert_interpreter_belongs_to(outer, context="devtools")


def test_a_foreign_interpreter_is_refused_before_the_import_path_is_cleaned(tmp_path: Path) -> None:
    """``python -m devtools`` on another checkout's venv exits 125, not ModuleNotFoundError.

    Anti-vacuity: normalize before refusing in ``devtools/__main__.py`` and the
    foreign interpreter's click disappears first, so the exit is 1.
    """
    import subprocess

    primary = _checkout(tmp_path / "primary")
    fake_prefix = primary / ".venv"
    repo_root = Path(__file__).resolve().parents[3]
    script = (
        "import sys, runpy\n"
        f"sys.prefix = {str(fake_prefix)!r}\n"
        f"sys.argv = ['devtools', '--help']\n"
        f"runpy.run_path({str(repo_root / 'devtools' / '__main__.py')!r}, run_name='__main__')\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)

    assert completed.returncode == 125, completed.stderr
    assert "another checkout's interpreter" in completed.stderr
