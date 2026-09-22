"""An absent virtualenv is reported as such, not as a missing tool.

A ``.claude/worktrees/<agent>`` checkout that has never entered the devshell has
no ``.venv`` at all. Reporting the first gate's ``ruff`` as unavailable names
neither the cause nor the remedy, and ruff is in fact installed.

Anti-vacuity: delete the ``unprovisioned_environment`` branch in
``executable_gate_result`` and ``test_an_absent_venv_is_named_with_its_remedy``
goes red -- the diagnosis falls back to ``gate_missing_executable``.
"""

from __future__ import annotations

from pathlib import Path

from devtools.required_gate import (
    executable_gate_result,
    foreign_environment_binding,
    unprovisioned_environment,
)


def test_an_absent_venv_is_named_with_its_remedy(tmp_path: Path) -> None:
    result = executable_gate_result([str(tmp_path / ".venv" / "bin" / "ruff"), "format"], gate="gate format")
    assert result.ok is False
    assert result.diagnosis == "gate_unprovisioned_environment"
    assert str(tmp_path / ".venv") in result.details[0]
    assert "nix develop" in result.details[0]
    # The remedy must not suggest borrowing another checkout's environment.
    assert "Never share or symlink" in result.details[0]


def test_a_provisioned_venv_missing_one_tool_is_still_a_missing_executable(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    result = executable_gate_result([str(tmp_path / ".venv" / "bin" / "ruff"), "format"], gate="gate format")
    assert result.diagnosis == "gate_missing_executable"


def test_an_installed_tool_passes(tmp_path: Path) -> None:
    binary = tmp_path / ".venv" / "bin" / "ruff"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    result = executable_gate_result([str(binary), "format"], gate="gate format")
    assert result.ok is True
    assert result.diagnosis == "gate_passed"


def test_a_path_resolved_tool_is_not_classified_as_a_venv(tmp_path: Path) -> None:
    del tmp_path
    assert unprovisioned_environment("some-tool-not-on-path") is None
    assert unprovisioned_environment(None) is None
    assert unprovisioned_environment("/usr/bin/ruff") is None


def test_cross_venv_shebang_is_refused(tmp_path: Path) -> None:
    """A copied venv keeps the original tree's interpreter, and it still runs.

    This is the shape the environment contract already forbids by prose
    ("never share or symlink another checkout's .venv"): copying one in leaves
    every console script's shebang naming the ORIGINAL checkout's python. The
    script launches, so the gate's verdict looks ordinary while the kernel
    resolved the other tree's dependencies.

    Anti-vacuity: dropping the ``is_relative_to(venv_root)`` containment check
    in ``_resolved`` makes this test report ``gate_passed`` -- the foreign
    interpreter exists, which is exactly why mere existence is not enough.
    """
    other = tmp_path / "other-checkout" / ".venv" / "bin"
    other.mkdir(parents=True)
    foreign_python = other / "python3"
    foreign_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    foreign_python.chmod(0o755)

    here = tmp_path / "here" / ".venv" / "bin"
    here.mkdir(parents=True)
    copied = here / "ruff"
    copied.write_text(f"#!{foreign_python}\nprint('x')\n", encoding="utf-8")
    copied.chmod(0o755)

    result = executable_gate_result([str(copied), "format"], gate="gate format")

    assert result.ok is False
    assert result.diagnosis == "gate_foreign_environment"
    assert str(foreign_python) in result.details[0]
    assert "re-provision" in result.details[0]


def test_own_venv_shebang_passes(tmp_path: Path) -> None:
    """The opposite direction: a correctly provisioned venv must not be refused.

    ``uv`` writes every console script's shebang as ``<this venv>/bin/python3``,
    so containment is satisfied by every supported layout. Without this case a
    blanket shebang refusal would pass the test above and break every checkout.
    """
    here = tmp_path / ".venv" / "bin"
    here.mkdir(parents=True)
    own_python = here / "python3"
    own_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    own_python.chmod(0o755)
    script = here / "ruff"
    script.write_text(f"#!{own_python}\nprint('x')\n", encoding="utf-8")
    script.chmod(0o755)

    result = executable_gate_result([str(script), "format"], gate="gate format")

    assert result.ok is True
    assert result.diagnosis == "gate_passed"


def test_tool_outside_a_venv_is_unaffected(tmp_path: Path) -> None:
    """Nix store wrappers and system tools legitimately name a foreign python."""
    foreign = tmp_path / "store" / "bin" / "python3"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    foreign.chmod(0o755)
    tool = tmp_path / "usr" / "bin" / "some-tool"
    tool.parent.mkdir(parents=True)
    tool.write_text(f"#!{foreign}\nprint('x')\n", encoding="utf-8")
    tool.chmod(0o755)

    assert foreign_environment_binding(str(tool)) is None
    assert executable_gate_result([str(tool)], gate="gate format").diagnosis == "gate_passed"


def test_absent_interpreter_stays_missing(tmp_path: Path) -> None:
    here = tmp_path / ".venv" / "bin"
    here.mkdir(parents=True)
    script = here / "ruff"
    script.write_text(f"#!{tmp_path / 'deleted' / 'bin' / 'python3'}\nprint('x')\n", encoding="utf-8")
    script.chmod(0o755)

    result = executable_gate_result([str(script), "format"], gate="gate format")

    assert result.diagnosis == "gate_missing_executable"
