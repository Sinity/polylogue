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
        return Path(executable).is_file() and os.access(executable, os.X_OK)
    return shutil.which(executable, path=(env or os.environ).get("PATH")) is not None


def executable_gate_result(command: Sequence[str], *, gate: str, env: Mapping[str, str] | None = None) -> GateResult:
    """Preflight the executable owned by a required subprocess gate."""
    executable = str(command[0]) if command else None
    available = executable is not None and _resolved(executable, env)
    return GateResult(
        gate=gate,
        executable=executable,
        executable_available=available,
        required_count=1,
        inspected_count=1 if available else 0,
        missing_count=0 if available else 1,
        diagnosis="gate_passed" if available else "gate_missing_executable",
        details=() if available else (str(executable),),
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
