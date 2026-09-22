"""Shared fail-closed evidence contract for required non-pytest gates."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateResult:
    gate: str
    executable: str | None
    executable_available: bool | None
    required_count: int
    inspected_count: int
    unreadable_count: int = 0
    missing_count: int = 0
    stale_count: int = 0
    error_count: int = 0
    semantic_violation_count: int = 0
    diagnosis: str = "gate_passed"
    enforced: bool = True
    details: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.diagnosis in {"gate_passed", "not_enforced"}

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "polylogue.required-gate-result",
            "gate": self.gate,
            "status": "not_enforced" if self.diagnosis == "not_enforced" else "passed" if self.ok else "failed",
            "gate_passed": None if self.diagnosis == "not_enforced" else self.ok,
            "executable": self.executable,
            "executable_available": self.executable_available,
            "required_count": self.required_count,
            "inspected_count": self.inspected_count,
            "unreadable_count": self.unreadable_count,
            "missing_count": self.missing_count,
            "stale_count": self.stale_count,
            "error_count": self.error_count,
            "semantic_violation_count": self.semantic_violation_count,
            "diagnosis": self.diagnosis,
            "enforced": self.enforced,
            "details": list(self.details),
        }


def _venv_root(executable: str) -> Path | None:
    """The ``.venv`` a gate executable is addressed inside, if it is in one."""
    parts = Path(executable).parts
    if len(parts) < 3 or parts[-2] != "bin" or parts[-3] != ".venv":
        return None
    return Path(*parts[:-2])


def _shebang_interpreter(path: Path) -> str | None:
    """The absolute interpreter a script's ``#!`` names, if it names one."""
    try:
        with path.open("rb") as handle:
            first_line = handle.readline().decode("utf-8", errors="replace").strip()
    except OSError:
        return None
    if not first_line.startswith("#!"):
        return None
    interpreter = first_line[2:].split(maxsplit=1)[0]
    return interpreter if interpreter.startswith("/") else None


def _resolved(executable: str, env: Mapping[str, str] | None) -> bool:
    if os.path.dirname(executable):
        path = Path(executable)
        if not path.is_file() or not os.access(executable, os.X_OK):
            return False
        # A relocated or copied venv leaves console scripts behind whose
        # shebang still names the ORIGINAL checkout's interpreter. Two cases,
        # and only rejecting the first left the dangerous one open:
        #
        #   * the original tree is gone -- the script exists but cannot be
        #     launched at all, so it is an incomplete gate tool;
        #   * the original tree is still there -- the script launches happily
        #     and the kernel runs the OTHER checkout's environment, which is
        #     exactly the non-hermetic dependency resolution this gate's
        #     checkout-local addressing exists to prevent. Nothing downstream
        #     can see that it happened, because the verdict looks ordinary.
        #
        # Only a shebang naming a DIFFERENT ``.venv`` interpreter is refused.
        # That is the copied-venv shape exactly. An ambient or Nix-store
        # interpreter is a separate question this gate does not decide, and
        # refusing it here would reject ordinary shell wrappers that live
        # inside a venv's ``bin``.
        interpreter = _shebang_interpreter(path)
        if interpreter is not None and not Path(interpreter).is_file():
            return False
        return foreign_environment_binding(executable) is None
    return shutil.which(executable, path=(env or os.environ).get("PATH")) is not None


#: Told to whoever hits an unprovisioned checkout. The devshell's shellHook
#: builds and syncs the venv; sharing another checkout's venv is not a
#: substitute, because its editable install resolves the product to that tree.
UNPROVISIONED_ENVIRONMENT_REMEDY = (
    "this checkout has no .venv: run `nix develop --accept-flake-config --command true` here. "
    "Never share or symlink another checkout's .venv -- its editable install would run that tree's product."
)


#: Told to whoever copied a venv in instead of provisioning one. The script
#: launches, which is precisely why nothing else notices.
FOREIGN_ENVIRONMENT_REMEDY = (
    "re-provision this checkout's .venv with `nix develop --accept-flake-config --command true`; "
    "a copied or relocated venv keeps the original tree's interpreter and resolves that tree's dependencies."
)


def unprovisioned_environment(executable: str | None) -> str | None:
    """The absent virtualenv root a gate executable was resolved against, if any.

    ``venv_bin`` addresses tools inside the invoking checkout deliberately, so
    a missing tool is normally a real gate failure. An entirely absent ``.venv``
    is a different fact: nothing is installed, the tool is not missing, and
    reporting the first tool as unavailable describes neither.
    """
    if not executable:
        return None
    venv_root = _venv_root(executable)
    if venv_root is None:
        return None
    return None if venv_root.is_dir() else str(venv_root)


def foreign_environment_binding(executable: str | None) -> str | None:
    """The other checkout's interpreter this console script would launch, if any.

    A ``.venv`` that was copied or relocated keeps its console-script shebangs
    pointing at the interpreter of the tree it was built in. When that tree
    still exists the script runs, resolving the gate's dependencies out of the
    other checkout -- silently, because the verdict looks like any other.

    The binding is only reported when the shebang names an interpreter inside
    a different ``.venv``: that is the copied-venv shape, and nothing else
    produces it. A shell or store interpreter inside a venv's ``bin`` is an
    ordinary wrapper, not a foreign environment.
    """
    if not executable:
        return None
    venv_root = _venv_root(executable)
    if venv_root is None:
        return None
    interpreter = _shebang_interpreter(Path(executable))
    if interpreter is None:
        return None
    interpreter_venv = _venv_root(interpreter)
    if interpreter_venv is None or interpreter_venv == venv_root:
        return None
    return None if not Path(interpreter).is_file() else interpreter


def executable_gate_result(command: Sequence[str], *, gate: str, env: Mapping[str, str] | None = None) -> GateResult:
    """Preflight the executable owned by a required subprocess gate."""
    executable = str(command[0]) if command else None
    available = executable is not None and _resolved(executable, env)
    diagnosis: str
    details: tuple[str, ...]
    if available:
        diagnosis, details = "gate_passed", ()
    elif (venv_root := unprovisioned_environment(executable)) is not None:
        diagnosis = "gate_unprovisioned_environment"
        details = (f"{venv_root}: {UNPROVISIONED_ENVIRONMENT_REMEDY}",)
    elif (interpreter := foreign_environment_binding(executable)) is not None:
        diagnosis = "gate_foreign_environment"
        details = (
            f"{executable}: shebang names {interpreter}, outside this checkout's .venv. {FOREIGN_ENVIRONMENT_REMEDY}",
        )
    else:
        diagnosis, details = "gate_missing_executable", (str(executable),)
    return GateResult(
        gate=gate,
        executable=executable,
        executable_available=available,
        required_count=1,
        inspected_count=1 if available else 0,
        missing_count=0 if available else 1,
        diagnosis=diagnosis,
        details=details,
    )


#: The one sync command that provisions the repository-audit tooling every
#: analysis gate needs (grimp, ast-grep, import-linter, vulture).
#:
#: ``audit`` is a ``[dependency-groups]`` entry, not an optional-dependencies
#: extra, so ``uv sync --extra audit`` errors outright; and the obvious-looking
#: ``uv sync --extra dev --frozen`` *prunes* the group back out again. Both
#: mistakes leave a checkout whose analysis gates cannot run, which is why the
#: whole command lives here once rather than being paraphrased per gate.
AUDIT_GROUP_SYNC_COMMAND = "uv sync --extra dev --group audit --frozen"


def missing_analysis_dependency_gate_result(
    *, gate: str, dependency: str, reason: str, remedy: str, required_count: int = 1
) -> GateResult:
    """Classify "this checkout cannot run the check" separately from a finding.

    A gate whose analysis library is absent inspected nothing, so reporting it
    through ``semantic_violation_count`` would put an unprovisioned environment
    in the same channel as a real finding.  ``gate_missing_executable`` is the
    wrong name for it too: the executable is the interpreter and it resolved
    fine.  This diagnosis exists so a reader can separate the two at a glance.
    """

    return GateResult(
        gate=gate,
        executable=None,
        executable_available=None,
        required_count=required_count,
        inspected_count=0,
        missing_count=1,
        diagnosis="gate_missing_analysis_dependency",
        details=(f"{dependency} is not importable ({reason}); run `{remedy}`",),
    )


def evidence_gate_result(
    *,
    gate: str,
    required_count: int,
    inspected_count: int,
    unreadable_count: int = 0,
    missing_count: int = 0,
    stale_count: int = 0,
    error_count: int = 0,
    semantic_violation_count: int = 0,
    executable: str | None = None,
    executable_available: bool | None = None,
    enforced: bool = True,
    details: Sequence[str] = (),
) -> GateResult:
    """Build a result where empty or unavailable required evidence is failure."""
    if not enforced:
        diagnosis = "not_enforced"
    elif executable_available is False:
        diagnosis = "gate_missing_executable"
    elif missing_count:
        diagnosis = "gate_missing_input"
    elif stale_count:
        diagnosis = "gate_stale_evidence"
    elif unreadable_count:
        diagnosis = "gate_unreadable_input"
    elif error_count:
        diagnosis = "gate_input_error"
    elif semantic_violation_count:
        diagnosis = "gate_semantic_violation"
    elif required_count == 0:
        diagnosis = "gate_empty_required_population"
    elif inspected_count < required_count:
        diagnosis = "gate_incomplete_inspection"
    else:
        diagnosis = "gate_passed"
    return GateResult(
        gate=gate,
        executable=executable,
        executable_available=executable_available,
        required_count=required_count,
        inspected_count=inspected_count,
        unreadable_count=unreadable_count,
        missing_count=missing_count,
        stale_count=stale_count,
        error_count=error_count,
        semantic_violation_count=semantic_violation_count,
        diagnosis=diagnosis,
        enforced=enforced,
        details=tuple(str(detail) for detail in details),
    )


__all__ = [
    "AUDIT_GROUP_SYNC_COMMAND",
    "GateResult",
    "evidence_gate_result",
    "executable_gate_result",
    "missing_analysis_dependency_gate_result",
]
