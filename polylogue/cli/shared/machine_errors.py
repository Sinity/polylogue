"""Machine-consumable CLI error and success envelopes."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

from polylogue.core.json import JSONDocument, require_json_document


class MachineErrorEnvelope(TypedDict):
    """Serialized machine-error envelope with sparse optional keys."""

    status: Literal["error"]
    code: str
    message: str
    command: NotRequired[list[str]]
    details: NotRequired[JSONDocument]


class MachineSuccessEnvelope(TypedDict):
    """Serialized machine-success envelope."""

    status: Literal["ok"]
    result: JSONDocument


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

INVALID_ARGUMENTS = "invalid_arguments"
INVALID_PATH = "invalid_path"
RUNTIME_ERROR = "runtime_error"
DEPENDENCY_MISSING = "dependency_missing"
UNSUPPORTED_ENVIRONMENT = "unsupported_environment"
NO_RESULTS = "no_results"


@dataclass(frozen=True, slots=True)
class MachineError:
    """CLI-visible machine-error envelope."""

    code: str
    message: str
    command: tuple[str, ...] | list[str] = ()
    details: Mapping[str, object] = field(default_factory=dict)
    status: Literal["error"] = "error"

    def to_dict(self) -> MachineErrorEnvelope:
        payload: MachineErrorEnvelope = {
            "status": self.status,
            "code": self.code,
            "message": self.message,
        }
        if self.command:
            payload["command"] = list(self.command)
        if self.details:
            payload["details"] = require_json_document(dict(self.details), context="machine error details")
        return payload

    def to_json(self, *, exclude_none: bool = False) -> str:
        import json

        del exclude_none
        return json.dumps(self.to_dict(), indent=2)

    def emit(self, *, exit_code: int = 1) -> None:
        """Write the payload to stdout and exit."""
        sys.stdout.write(self.to_json(exclude_none=True))
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise SystemExit(exit_code)


@dataclass(frozen=True, slots=True)
class MachineSuccess:
    """CLI-visible machine-success envelope."""

    result: Mapping[str, object] = field(default_factory=dict)
    status: Literal["ok"] = "ok"

    def to_dict(self) -> MachineSuccessEnvelope:
        return {
            "status": self.status,
            "result": require_json_document(dict(self.result), context="machine success result"),
        }

    def to_json(self, *, exclude_none: bool = False) -> str:
        import json

        del exclude_none
        return json.dumps(self.to_dict(), indent=2)


def _normalize_result_payload(
    result: Mapping[str, object] | MachineSuccess | None,
) -> JSONDocument:
    if result is None:
        return {}
    if isinstance(result, MachineSuccess):
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
        "--origin",
        "--exclude-origin",
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
        "--set",
        "--add-tag",
        "--source",
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
        "--print-url",
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
