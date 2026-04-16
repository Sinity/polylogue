"""Machine-consumable CLI error and success envelopes.

When ``--json`` is passed, all CLI output — including failures — must be
valid JSON on stdout.  This module provides the stable envelope shapes
and a top-level exception handler that Click's default error path cannot
satisfy (Click writes plain text to stderr before command logic runs).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

INVALID_ARGUMENTS = "invalid_arguments"
INVALID_PATH = "invalid_path"
RUNTIME_ERROR = "runtime_error"
DEPENDENCY_MISSING = "dependency_missing"
UNSUPPORTED_ENVIRONMENT = "unsupported_environment"
NO_RESULTS = "no_results"


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


def error_no_results(
    message: str,
    *,
    command: list[str] | None = None,
    filters: list[str] | None = None,
) -> MachineError:
    details: dict[str, Any] = {}
    if filters:
        details["filters"] = filters
    return MachineError(
        code=NO_RESULTS,
        message=message,
        command=command or [],
        details=details,
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

    Handles explicit ``--json`` subcommand flags and the query-mode
    ``--format json`` form so machine callers still get structured
    failures when query execution aborts before normal rendering.
    """
    for index, arg in enumerate(argv):
        if arg == "--json":
            return True
        if arg.startswith("--json="):
            return True
        if arg == "--format" and index + 1 < len(argv) and argv[index + 1] == "json":
            return True
        if arg.startswith("--format=") and arg.split("=", 1)[1] == "json":
            return True
        if arg == "-f" and index + 1 < len(argv) and argv[index + 1] == "json":
            return True
    return False


def extract_command(argv: list[str]) -> list[str]:
    """Best-effort extraction of the subcommand path from raw argv."""
    flag_only_long = {
        "--json",
        "--plain",
        "--latest",
        "--reverse",
        "--stream",
        "--dialogue-only",
        "--has-tool-use",
        "--has-thinking",
        "--verbose",
        "--help",
        "-h",
        "-v",
        "-d",
    }
    value_long = {
        "--id",
        "--contains",
        "--exclude-text",
        "--retrieval-lane",
        "--provider",
        "--exclude-provider",
        "--tag",
        "--exclude-tag",
        "--title",
        "--path",
        "--action",
        "--exclude-action",
        "--action-sequence",
        "--action-text",
        "--tool",
        "--exclude-tool",
        "--similar",
        "--has",
        "--min-messages",
        "--max-messages",
        "--min-words",
        "--since",
        "--until",
        "--limit",
        "--by",
        "--fields",
        "--sort",
        "--sample",
        "--output",
        "--format",
        "--transform",
        "--set",
        "--add-tag",
        "--source",
        "--exec",
        "--webhook",
        "--target",
        "--schema-provider",
        "--artifact-provider",
        "--artifact-status",
        "--artifact-kind",
        "--artifact-limit",
        "--artifact-offset",
        "--schema-samples",
        "--schema-record-limit",
        "--schema-record-offset",
        "--transport",
        "--workspace",
        "--report-dir",
        "--snapshot",
        "--snapshot-from",
        "--capture",
        "--tier",
        "--only",
        "--skip",
        "--print-path",
    }
    value_short = {
        "-i",
        "-c",
        "-p",
        "-t",
        "-n",
        "-o",
        "-f",
    }
    parts: list[str] = []
    skip_values = 0
    for arg in argv:
        if skip_values:
            skip_values -= 1
            continue
        if arg.startswith("-"):
            if arg in flag_only_long:
                continue
            if arg in {"--set"}:
                skip_values = 2
                continue
            if arg in value_long or arg in value_short:
                skip_values = 1
                continue
            if arg.startswith("--") and "=" in arg:
                continue
            if arg.startswith("-") and len(arg) == 2 and arg not in {"-h", "-v", "-d"}:
                skip_values = 1
            continue
        parts.append(arg)
    return parts


__all__ = [
    "MachineError",
    "MachineSuccess",
    "error_dependency_missing",
    "error_invalid_arguments",
    "error_invalid_path",
    "error_no_results",
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
    "NO_RESULTS",
    "UNSUPPORTED_ENVIRONMENT",
]
