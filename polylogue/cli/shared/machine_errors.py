"""Machine-consumable CLI error and success envelopes."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

from polylogue.core.json import JSONDocument, require_json_document
from polylogue.surfaces.machine_envelope import (
    MachineSuccess,
    emit_success,
    success,
)
from polylogue.surfaces.outcome import OutcomeEnvelope, decide_outcome


class MachineErrorEnvelope(TypedDict):
    """Serialized machine-error envelope with sparse optional keys."""

    status: Literal["error"]
    code: str
    message: str
    command: NotRequired[list[str]]
    details: NotRequired[JSONDocument]
    outcome: NotRequired[JSONDocument]


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

INVALID_ARGUMENTS = "invalid_arguments"
INVALID_PATH = "invalid_path"
RUNTIME_ERROR = "runtime_error"
DEPENDENCY_MISSING = "dependency_missing"
UNSUPPORTED_ENVIRONMENT = "unsupported_environment"
NO_RESULTS = "no_results"
#: The resident daemon is the only executor and none is answering.
#:
#: Distinct from :data:`RUNTIME_ERROR` on purpose. Every daemon-absent CLI
#: refusal used to reach a machine caller as ``runtime_error`` -- the same code
#: an unparseable option value, a corrupt tier and an unexpected exception all
#: produce -- so a ``--format json`` client could not tell "start the daemon"
#: apart from "something went wrong", and the remedy that the terminal format
#: prints was not on the wire at all (polylogue-re6s3 AC4, polylogue-3eexy AC4).
DAEMON_REQUIRED = "daemon_required"


@dataclass(frozen=True, slots=True)
class MachineError:
    """CLI-visible machine-error envelope."""

    code: str
    message: str
    command: tuple[str, ...] | list[str] = ()
    details: Mapping[str, object] = field(default_factory=dict)
    outcome: OutcomeEnvelope | None = None
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
        if self.outcome is not None:
            payload["outcome"] = require_json_document(self.outcome.to_dict(), context="machine error outcome")
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


def error_daemon_required(
    message: str,
    *,
    command: list[str] | None = None,
    operation: str | None = None,
    archive_root: str | None = None,
) -> MachineError:
    """Build the machine envelope for a refusal that only a daemon can satisfy.

    ``operation`` and ``archive_root`` are carried in ``details`` rather than
    folded into the message: a client that wants to start the right daemon
    needs the archive it must serve as a field, not as prose it has to parse
    back out.
    """
    details: JSONDocument = {"remedy": "polylogued run"}
    if operation:
        details["operation"] = operation
    if archive_root:
        details["archive_root"] = archive_root
    return MachineError(
        code=DAEMON_REQUIRED,
        message=message,
        command=tuple(command or ()),
        details=details,
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
        outcome=decide_outcome(matched=0),
    )


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
        "--schema-origin",
        "--artifact-origin",
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
    "error_daemon_required",
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
    "DAEMON_REQUIRED",
    "INVALID_ARGUMENTS",
    "INVALID_PATH",
    "RUNTIME_ERROR",
    "DEPENDENCY_MISSING",
    "NO_RESULTS",
    "UNSUPPORTED_ENVIRONMENT",
]
