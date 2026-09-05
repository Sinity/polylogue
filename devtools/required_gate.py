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


def _resolved(executable: str, env: Mapping[str, str] | None) -> bool:
    if os.path.dirname(executable):
        path = Path(executable)
        if not path.is_file() or not os.access(executable, os.X_OK):
            return False
        # A relocated venv can leave console scripts behind whose shebang
        # still names the old worktree interpreter.  Such a file exists but
        # cannot be launched, so classify it as an incomplete gate tool.
        try:
            first_line = path.open("rb").readline().decode("utf-8", errors="replace").strip()
        except OSError:
            return False
        if first_line.startswith("#!"):
            interpreter = first_line[2:].split(maxsplit=1)[0]
            if interpreter.startswith("/") and not Path(interpreter).is_file():
                return False
        return True
    return shutil.which(executable, path=(env or os.environ).get("PATH")) is not None


#: Told to whoever hits an unprovisioned checkout. The devshell's shellHook
#: builds and syncs the venv; sharing another checkout's venv is not a
#: substitute, because its editable install resolves the product to that tree.
UNPROVISIONED_ENVIRONMENT_REMEDY = (
    "this checkout has no .venv: run `nix develop --accept-flake-config --command true` here. "
    "Never share or symlink another checkout's .venv -- its editable install would run that tree's product."
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
    parts = Path(executable).parts
    if len(parts) < 3 or parts[-2] != "bin" or parts[-3] != ".venv":
        return None
    venv_root = Path(*parts[:-2])
    return None if venv_root.is_dir() else str(venv_root)


def executable_gate_result(command: Sequence[str], *, gate: str, env: Mapping[str, str] | None = None) -> GateResult:
    """Preflight the executable owned by a required subprocess gate."""
    executable = str(command[0]) if command else None
    available = executable is not None and _resolved(executable, env)
    if available:
        diagnosis, details = "gate_passed", ()
    elif (venv_root := unprovisioned_environment(executable)) is not None:
        diagnosis = "gate_unprovisioned_environment"
        details = (f"{venv_root}: {UNPROVISIONED_ENVIRONMENT_REMEDY}",)
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


__all__ = ["GateResult", "evidence_gate_result", "executable_gate_result"]
