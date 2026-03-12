"""Machine-consumable CLI error and success envelopes.

When ``--json`` is passed, all CLI output — including failures — must be
valid JSON on stdout.  This module provides the stable envelope shapes
and a top-level exception handler that Click's default error path cannot
satisfy (Click writes plain text to stderr before command logic runs).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

INVALID_ARGUMENTS = "invalid_arguments"
INVALID_PATH = "invalid_path"
RUNTIME_ERROR = "runtime_error"
DEPENDENCY_MISSING = "dependency_missing"
UNSUPPORTED_ENVIRONMENT = "unsupported_environment"


# ---------------------------------------------------------------------------
# Envelope dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MachineError:
    """Structured error envelope emitted when ``--json`` is active."""

    code: str
    message: str
    command: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": "error",
            "code": self.code,
            "message": self.message,
        }
        if self.command:
            out["command"] = self.command
        if self.details:
            out["details"] = self.details
        return out

    def emit(self, *, exit_code: int = 1) -> None:
        """Write JSON to stdout and exit."""
        sys.stdout.write(json.dumps(self.to_dict(), indent=2))
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise SystemExit(exit_code)


@dataclass(frozen=True, slots=True)
class MachineSuccess:
    """Structured success envelope."""

    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"status": "ok", "result": self.result}


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def error_invalid_arguments(
    message: str,
    *,
    command: list[str] | None = None,
    option: str | None = None,
) -> MachineError:
    details: dict[str, Any] = {}
    if option:
        details["option"] = option
    return MachineError(
        code=INVALID_ARGUMENTS,
        message=message,
        command=command or [],
        details=details,
    )


def error_invalid_path(
    message: str,
    *,
    command: list[str] | None = None,
    path: str | None = None,
) -> MachineError:
    details: dict[str, Any] = {}
    if path:
        details["path"] = path
    return MachineError(
        code=INVALID_PATH,
        message=message,
        command=command or [],
        details=details,
    )


def error_runtime(
    message: str,
    *,
    command: list[str] | None = None,
    exception_type: str | None = None,
) -> MachineError:
    details: dict[str, Any] = {}
    if exception_type:
        details["exception_type"] = exception_type
    return MachineError(
        code=RUNTIME_ERROR,
        message=message,
        command=command or [],
        details=details,
    )


def error_dependency_missing(
    message: str,
    *,
    command: list[str] | None = None,
    dependency: str | None = None,
) -> MachineError:
    details: dict[str, Any] = {}
    if dependency:
        details["dependency"] = dependency
    return MachineError(
        code=DEPENDENCY_MISSING,
        message=message,
        command=command or [],
        details=details,
    )


def error_unsupported_environment(
    message: str,
    *,
    command: list[str] | None = None,
) -> MachineError:
    return MachineError(
        code=UNSUPPORTED_ENVIRONMENT,
        message=message,
        command=command or [],
    )


def success(result: dict[str, Any] | None = None) -> MachineSuccess:
    return MachineSuccess(result=result or {})


def emit_success(result: dict[str, Any] | None = None) -> None:
    """Write a ``{"status": "ok", "result": …}`` envelope to stdout."""
    import json as _json

    sys.stdout.write(_json.dumps(success(result).to_dict(), indent=2))
    sys.stdout.write("\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Argv pre-scanning
# ---------------------------------------------------------------------------


def wants_json(argv: list[str]) -> bool:
    """Detect ``--json`` intent from raw argv before Click parses.

    Handles ``--json``, ``--json=true``, and the ``json_output`` form
    used by some subcommands.  This intentionally does *not* understand
    ``-f json`` (output format), which has different semantics.
    """
    for arg in argv:
        if arg == "--json":
            return True
        if arg.startswith("--json="):
            return True
    return False


def extract_command(argv: list[str]) -> list[str]:
    """Best-effort extraction of the subcommand path from raw argv."""
    parts: list[str] = []
    for arg in argv:
        if arg.startswith("-"):
            continue
        parts.append(arg)
    return parts


__all__ = [
    "MachineError",
    "MachineSuccess",
    "error_dependency_missing",
    "error_invalid_arguments",
    "error_invalid_path",
    "error_runtime",
    "error_unsupported_environment",
    "extract_command",
    "emit_success",
    "success",
    "wants_json",
    "INVALID_ARGUMENTS",
    "INVALID_PATH",
    "RUNTIME_ERROR",
    "DEPENDENCY_MISSING",
    "UNSUPPORTED_ENVIRONMENT",
]
