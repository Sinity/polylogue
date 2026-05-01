"""Machine-consumable CLI error and success envelopes."""

from __future__ import annotations

import sys
from collections.abc import Mapping

from polylogue.core.json import JSONDocument, require_json_document
from polylogue.surfaces.payloads import (
    MachineErrorPayload,
    MachineSuccessPayload,
)

# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

INVALID_ARGUMENTS = "invalid_arguments"
INVALID_PATH = "invalid_path"
RUNTIME_ERROR = "runtime_error"
DEPENDENCY_MISSING = "dependency_missing"
UNSUPPORTED_ENVIRONMENT = "unsupported_environment"
NO_RESULTS = "no_results"


class MachineError(MachineErrorPayload):
    """CLI-visible machine-error envelope."""


class MachineSuccess(MachineSuccessPayload):
    """CLI-visible machine-success envelope."""


def _normalize_result_payload(
    result: Mapping[str, object] | MachineSuccessPayload | None,
) -> JSONDocument:
    if result is None:
        return {}
    if isinstance(result, MachineSuccessPayload):
        return require_json_document(result.result, context="machine success result")
    return require_json_document(
        {str(key): value for key, value in result.items()},
        context="machine success result",
    )


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def error_invalid_arguments(
    message: str,
    *,
    command: list[str] | None = None,
    option: str | None = None,
) -> MachineError:
    details: JSONDocument = {}
    if option:
        details["option"] = option
    return MachineError(
        code=INVALID_ARGUMENTS,
        message=message,
        command=tuple(command or ()),
        details=details,
    )


def error_invalid_path(
    message: str,
    *,
    command: list[str] | None = None,
    path: str | None = None,
) -> MachineError:
    details: JSONDocument = {}
    if path:
        details["path"] = path
    return MachineError(
        code=INVALID_PATH,
        message=message,
        command=tuple(command or ()),
        details=details,
    )


def error_runtime(
    message: str,
    *,
    command: list[str] | None = None,
    exception_type: str | None = None,
) -> MachineError:
    details: JSONDocument = {}
    if exception_type:
        details["exception_type"] = exception_type
    return MachineError(
        code=RUNTIME_ERROR,
        message=message,
        command=tuple(command or ()),
        details=details,
    )


def error_dependency_missing(
    message: str,
    *,
    command: list[str] | None = None,
    dependency: str | None = None,
) -> MachineError:
    details: JSONDocument = {}
    if dependency:
        details["dependency"] = dependency
    return MachineError(
        code=DEPENDENCY_MISSING,
        message=message,
        command=tuple(command or ()),
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
        command=tuple(command or ()),
    )


def error_no_results(
    message: str,
    *,
    command: list[str] | None = None,
    filters: list[str] | None = None,
    diagnostics: JSONDocument | None = None,
) -> MachineError:
    details: JSONDocument = {}
    if filters:
        details["filters"] = list(filters)
    if diagnostics:
        details["diagnostics"] = diagnostics
    return MachineError(
        code=NO_RESULTS,
        message=message,
        command=tuple(command or ()),
        details=details,
    )


def success(result: Mapping[str, object] | MachineSuccess | None = None) -> MachineSuccess:
    return MachineSuccess(result=_normalize_result_payload(result))


def emit_success(result: Mapping[str, object] | MachineSuccess | None = None) -> None:
    """Write a ``{\"status\": \"ok\", \"result\": …}`` envelope to stdout."""
    sys.stdout.write(success(result).to_json(exclude_none=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Argv pre-scanning
# ---------------------------------------------------------------------------


def wants_json(argv: list[str]) -> bool:
    """Detect JSON machine-output intent from raw argv before Click parses."""
    for index, arg in enumerate(argv):
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
        "--referenced-path",
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
