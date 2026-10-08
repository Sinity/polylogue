"""Parser and local export client for Antigravity session state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import stat as stat_module
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, MutableSequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Protocol
from urllib.request import ProxyHandler, Request, build_opener

from pydantic import ValidationError

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.archive.message.artifacts import classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, Provider, TitleSource
from polylogue.core.json import JSONDocument, dumps_bytes, loads
from polylogue.core.timestamps import iso_from_epoch_ms
from polylogue.sources.detection_projection import DetectorProjection
from polylogue.sources.tool_result_reasons import unknown_reason

from .base import (
    AdmissionDisposition,
    AdmissionOutcome,
    AdmissionRefusalReason,
    AdmissionUnit,
    AdmissionUnknownReason,
    ParseAccounting,
    ParsedContentBlock,
    ParsedFileEdit,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    human_authored_override,
    mark_last_occurrence_as_active_leaf,
    parser_admission,
    synthetic_message_id,
)

_TRAJECTORY_REQUIRED_META_COLUMNS = frozenset({"trajectory_id", "cascade_id"})
_TRAJECTORY_REQUIRED_STEP_COLUMNS = frozenset({"idx", "step_type", "step_format", "step_payload"})
_TRAJECTORY_DB_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
# Only reviewed source shapes may become normalized messages. Unknown future
# formats remain retained evidence instead of being guessed into prose.
_TRAJECTORY_SUPPORTED_STEP_FORMATS = frozenset({"v1", "json", "json_v1", "trajectory_v1"})
_TRAJECTORY_SUPPORTED_STEP_TYPES = frozenset(
    {
        "message",
        "user",
        "human",
        "prompt",
        "assistant",
        "planner",
        "planning",
        "plan",
        "plan_step",
        "terminal",
        "terminal_command",
        "command",
        "tool",
        "tool_call",
        "tool_use",
        "tool_result",
        "tool_output",
        "command_result",
        "file_edit",
        "edit",
    }
)


def trajectory_raw_id(source_path: Path | str, logical_revision: str, *, identity_path: Path | None = None) -> str:
    """Stable raw identity for one Antigravity trajectory-store revision."""
    coordinate = identity_path if identity_path is not None else Path(source_path).expanduser().resolve()
    identity = f"antigravity-trajectory\0{coordinate}\0{logical_revision}"
    return hashlib.sha256(identity.encode("utf-8", errors="surrogateescape")).hexdigest()


#: Language-server RPCs carry the CSRF token and go to loopback only; an
#: environment ``HTTP_PROXY`` does not bypass ``127.0.0.1`` on its own and
#: would hand the token to whoever runs the proxy.
_LOOPBACK_OPENER = build_opener(ProxyHandler({}))
_SEARCH_ENDPOINT = "/exa.language_server_pb.LanguageServerService/SearchConversations"
_MARKDOWN_ENDPOINT = "/exa.language_server_pb.LanguageServerService/ConvertTrajectoryToMarkdown"
_SECTION_RE = re.compile(r"^### (?P<title>User Input|Planner Response)\s*$", re.MULTILINE)

#: Socket budget for one probe or search request to the vendor HTTP surface.
_REQUEST_TIMEOUT_S = 10.0

#: Socket budget for one trajectory conversion. Conversion is the vendor's own
#: whole-trajectory work and its cost does not track the protobuf's size: on
#: this corpus the largest trajectories have spent 4-10s each while a larger
#: one finished in 0.06s. An expired conversion is a lost conversation, not a
#: retried probe, so the budget sits far above the observed cost -- and stays
#: finite, because the caller isolates one item's failure and must not be able
#: to block the rest of the corpus indefinitely.
_CONVERSION_TIMEOUT_S = 120.0

#: Sleep between readiness probes.
_READY_RETRY_SLEEP_S = 0.2

#: Readiness probes always run at least this many times. One probe can consume
#: the whole ``_REQUEST_TIMEOUT_S`` budget, so a deadline shorter than that
#: admits a single probe and the retry loop can never run.
_MIN_READY_ATTEMPTS = 3

#: Readiness deadline once the attempt floor is met. The vendor server answers
#: in ~2s on an idle host and has exceeded ``_REQUEST_TIMEOUT_S`` under load.
_STARTUP_TIMEOUT_S = 60.0

#: ``kind`` seed distinguishing a tool-activity message from a prose section.
_ACTIVITY_MESSAGE_KIND = "tool_activity"

#: Closed vocabulary of the tool-activity markers the language server renders
#: into a transcript, as ``(tool_name, pattern body, argument name)``.
#:
#: ``tool_name`` restates the marker's own phrasing rather than a vendor tool
#: identifier: the export renders activity, and the underlying tool name is not
#: on the wire. ``argument`` names the single value the marker preserves, or is
#: ``None`` where the export keeps no argument at all -- an absent argument
#: stays absent instead of being guessed from surrounding prose. A span no
#: entry matches is prose, which is what keeps the assistant's own italicised
#: sentences out of this vocabulary.
#:
#: A marker begins at the start of a line and ends at the end of one, but need
#: not be confined to a single line: an accepted command is rendered verbatim
#: inside backticks and heredocs run over many lines. Only that entry admits
#: newlines in its argument; every other argument is line-bounded so a marker
#: cannot swallow the prose that follows it.
_ACTIVITY_MARKERS: tuple[tuple[str, str, str | None], ...] = (
    ("viewed_file", r"\*Viewed \[[^\]\n]*\]\((?P<v>[^)\n]*)\)[ \t]*\*", "path"),
    ("listed_directory", r"\*Listed directory \[[^\]\n]*\]\((?P<v>[^)\n]*)\)[ \t]*\*", "path"),
    # ``[^`]*`` rather than ``.*?``: the command is rendered verbatim between a
    # single pair of backticks, so the argument cannot contain one. The lazy
    # wildcard matched the same strings but, combined with ``DOTALL`` and a
    # per-line start, rescanned the whole remaining body from every line that
    # opened a marker and never closed it -- quadratic time on an import whose
    # bytes an export controls. The character class fails at the first
    # backtick-or-end instead.
    ("accepted_command", r"\*User accepted the command `(?P<v>[^`]*)`[ \t]*\*", "command"),
    ("searched_web", r"\*Searched web for (?P<v>[^\n]*?)[ \t]*\*", "query"),
    ("read_url_content", r"\*Read URL content from (?P<v>[^\n]*?)[ \t]*\*", "url"),
    ("read_resource", r"\*Read resource from (?P<v>[^\n]*?)[ \t]*\*", "resource"),
    ("listed_resources", r"\*Listed resources from (?P<v>[^\n]*?)[ \t]*\*", "server"),
    ("edited_file", r"\*Edited relevant file[ \t]*\*", None),
    ("checked_command_status", r"\*Checked command status[ \t]*\*", None),
    ("grep_searched_codebase", r"\*Grep searched codebase[ \t]*\*", None),
    ("searched_filesystem", r"\*Searched filesystem[ \t]*\*", None),
    ("viewed_code_item", r"\*Viewed code item[ \t]*\*", None),
    ("viewed_content_chunk", r"\*Viewed content chunk[ \t]*\*", None),
)

#: ``_ACTIVITY_MARKERS`` as one ordered alternation over a whole section body,
#: so a marker spanning several lines is still one match.
_ACTIVITY_MARKER_RE = re.compile(
    "|".join(
        f"(?P<{tool_name}>^{pattern.replace('(?P<v>', f'(?P<v_{tool_name}>')}$)"
        for tool_name, pattern, _argument in _ACTIVITY_MARKERS
    ),
    re.MULTILINE | re.DOTALL,
)

#: Argument name per marker, keyed by the alternation's group name.
_ACTIVITY_ARGUMENTS: dict[str, str | None] = {
    tool_name: argument for tool_name, _pattern, argument in _ACTIVITY_MARKERS
}


class AntigravityExportError(RuntimeError):
    """Raised when Antigravity's local export surface cannot be queried."""


class AntigravityBinaryUnavailableError(AntigravityExportError):
    """Raised when the Antigravity language-server binary is not installed.

    This is a source coverage blocker for manifested conversation protobufs.
    Brain artifacts remain independently admitted as raw-only evidence.
    """


class AntigravityPartialExportError(AntigravityExportError):
    """Raised when the language-server export aborts mid-iteration.

    Distinct from a binary-absent condition: some sessions were already
    obtained before the failure, so the remainder is genuinely at risk of being
    dropped. Carries obtained-vs-expected counts so callers can surface the loss
    instead of silently truncating.
    """

    def __init__(self, message: str, *, obtained: int, expected: int) -> None:
        self.obtained = obtained
        self.expected = expected
        super().__init__(f"{message} (obtained {obtained} of {expected} sessions)")


class AntigravitySourceMutationError(AntigravityExportError):
    """Raised when a source item changes while read-only evidence is captured."""


class AntigravitySourceRole(StrEnum):
    """The admission role of one item below Antigravity's source root."""

    CONVERSATION_PROTOBUF = "conversation_protobuf"
    EXPORT_DOCUMENT = "export_document"
    BRAIN_DOCUMENT = "brain_document"
    METADATA_SIDECAR = "metadata_sidecar"
    UNKNOWN = "unknown"


class AntigravitySourceInspection(StrEnum):
    """How the census inspected one filesystem entry."""

    REGULAR = "regular"
    NON_REGULAR = "non_regular"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class AntigravityActivityMarker:
    """One tool call the language server rendered as a transcript marker line."""

    tool_name: str
    tool_input: dict[str, object] | None
    rendered: str


@dataclass(frozen=True, slots=True)
class AntigravitySourceClassification:
    """Positive source-role evidence shared by batch and resident routes."""

    role: AntigravitySourceRole
    parse_as_session: bool
    artifact_kind: ArtifactKind
    reason: str


@dataclass(frozen=True, slots=True)
class AntigravitySourceItem:
    """One immutable source-census item and its positive admission role."""

    path: Path
    relative_path: str
    classification: AntigravitySourceClassification | None
    inspection: AntigravitySourceInspection
    size_bytes: int
    content_sha256: str | None


@dataclass(frozen=True, slots=True)
class AntigravitySourceCensus:
    """Complete, read-only accounting for one Antigravity source root."""

    root: Path
    items: tuple[AntigravitySourceItem, ...]

    @property
    def counts(self) -> dict[AntigravitySourceRole, int]:
        counts = Counter(item.classification.role for item in self.items if item.classification is not None)
        return {role: counts.get(role, 0) for role in AntigravitySourceRole}

    @property
    def inspection_counts(self) -> dict[AntigravitySourceInspection, int]:
        counts = Counter(item.inspection for item in self.items)
        return {inspection: counts.get(inspection, 0) for inspection in AntigravitySourceInspection}

    @property
    def unknown_count(self) -> int:
        return self.counts[AntigravitySourceRole.UNKNOWN]

    @property
    def unexplained_items(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.items if item.classification is None)

    def assert_conserved(self) -> None:
        # Totality holds by construction: every census producer assigns a real
        # classification and classify_source_path is total, so no filesystem
        # state reaches either branch today. This is a guard against a future
        # partial classifier, not a measurement.
        if self.unexplained_items:
            raise ValueError("Antigravity source census contains unexplained items")
        if sum(self.counts.values()) != len(self.items):
            raise ValueError("Antigravity source census does not conserve its item denominator")


def census_source(root: Path) -> AntigravitySourceCensus:
    """Capture every filesystem entry below ``root`` and assign one source role.

    This is preparation evidence only. It does not open the archive or blob
    store, and a change during hashing is a visible source failure.
    """
    root = root.expanduser()
    items: list[AntigravitySourceItem] = []

    def record_unreadable(path: Path, detail: str) -> None:
        items.append(
            AntigravitySourceItem(
                path=path,
                relative_path=_relative_path(path, root),
                classification=AntigravitySourceClassification(
                    AntigravitySourceRole.UNKNOWN,
                    False,
                    ArtifactKind.UNKNOWN,
                    detail,
                ),
                inspection=AntigravitySourceInspection.UNREADABLE,
                size_bytes=0,
                content_sha256=None,
            )
        )

    from polylogue.sources.source_walk import layout_source_candidates
    from polylogue.sources.walk_faults import WalkRefusedError

    # The declared Antigravity layout bounds the census exactly as it bounds
    # admission. Links the layout places are still inspected, so they remain
    # accounted for as unsupported evidence while admission excludes them.
    try:
        candidates = layout_source_candidates("antigravity", root)
    except WalkRefusedError as exc:
        candidates = []
        for fault in exc.faults:
            record_unreadable(fault.path, f"source item is unreadable: {fault.detail}")
    for path in candidates:
        try:
            link_stat = path.lstat()
        except OSError as exc:
            record_unreadable(path, f"source item is unreadable: {exc}")
            continue
        if not stat_module.S_ISREG(link_stat.st_mode):
            items.append(
                AntigravitySourceItem(
                    path=path,
                    relative_path=_relative_path(path, root),
                    classification=AntigravitySourceClassification(
                        AntigravitySourceRole.UNKNOWN,
                        False,
                        ArtifactKind.UNKNOWN,
                        "non-regular Antigravity source item",
                    ),
                    inspection=AntigravitySourceInspection.NON_REGULAR,
                    size_bytes=link_stat.st_size,
                    content_sha256=None,
                )
            )
            continue
        try:
            before = path.stat()
            digest = _file_digest(path)
            classification = classify_source_path(path)
            after = path.stat()
        except OSError as exc:
            record_unreadable(path, f"source item is unreadable: {exc}")
            continue
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AntigravitySourceMutationError(f"Antigravity source changed during census: {path}")
        items.append(
            AntigravitySourceItem(
                path=path,
                relative_path=_relative_path(path, root),
                classification=classification,
                inspection=AntigravitySourceInspection.REGULAR,
                size_bytes=after.st_size,
                content_sha256=digest,
            )
        )
    census = AntigravitySourceCensus(root=root, items=tuple(items))
    census.assert_conserved()
    return census


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True, slots=True)
class AntigravityLanguageServerInfo:
    """Identity and capabilities established by the vendor adapter handshake."""

    binary_path: Path
    version: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AntigravityExportOutcome:
    """One conservation-bearing result for one manifested conversation."""

    source_path: Path
    cascade_id: str
    session: ParsedSession | None = None
    error: str | None = None
    converter: AntigravityLanguageServerInfo | None = None
    source_sha256: str | None = None

    @property
    def obtained(self) -> bool:
        return self.session is not None and self.error is None


def classify_source_path(source_path: str | Path, *, payload: object | None = None) -> AntigravitySourceClassification:
    """Classify every Antigravity path into a session, artifact, or unknown role."""
    path = Path(source_path)
    if path.suffix.lower() in _TRAJECTORY_DB_SUFFIXES and looks_like_trajectory_db_path(path):
        # Keep the established session role vocabulary for source-frontier
        # accounting.  The parser path (and not this compatibility role name)
        # distinguishes protobuf export from SQLite trajectory acquisition.
        return AntigravitySourceClassification(
            AntigravitySourceRole.CONVERSATION_PROTOBUF,
            True,
            ArtifactKind.SESSION_DOCUMENT,
            "verified Antigravity trajectory SQLite schema",
        )
    from polylogue.sources.origin_specs import artifact_rule_for_path

    rule = artifact_rule_for_path(Provider.ANTIGRAVITY, str(path))
    role_by_coverage = {
        "conversation_protobuf": AntigravitySourceRole.CONVERSATION_PROTOBUF,
        "brain_metadata_sidecar": AntigravitySourceRole.METADATA_SIDECAR,
        "brain_document": AntigravitySourceRole.BRAIN_DOCUMENT,
    }
    if rule is not None and rule.coverage_role in role_by_coverage:
        return AntigravitySourceClassification(
            role_by_coverage[rule.coverage_role],
            rule.parse_policy == "session",
            ArtifactKind(rule.kind),
            rule.fidelity_note,
        )
    if rule is None and path.suffix.lower() not in _TRAJECTORY_DB_SUFFIXES and path.suffix.lower() != ".zip":
        from polylogue.sources.origin_specs import recognize_json_source_class

        recognition = recognize_json_source_class(Provider.ANTIGRAVITY, path, payload=payload)
        if recognition is not None and recognition.source_class == "session":
            return AntigravitySourceClassification(
                AntigravitySourceRole.EXPORT_DOCUMENT,
                True,
                ArtifactKind.SESSION_DOCUMENT,
                recognition.reason,
            )
    return AntigravitySourceClassification(
        AntigravitySourceRole.UNKNOWN,
        False,
        ArtifactKind.UNKNOWN,
        "unrecognized Antigravity source item",
    )


@dataclass(frozen=True, slots=True)
class AntigravitySessionSummary:
    cascade_id: str
    title: str | None = None
    workspace_name: str | None = None
    snippet: str | None = None
    last_modified_time: str | None = None

    @classmethod
    def from_payload(cls, payload: JSONDocument) -> AntigravitySessionSummary | None:
        cascade_id = _string(payload.get("cascadeId"))
        if cascade_id is None:
            return None
        return cls(
            cascade_id=cascade_id,
            title=_string(payload.get("title")),
            workspace_name=_string(payload.get("workspaceName")),
            snippet=_string(payload.get("snippet")),
            last_modified_time=_string(payload.get("lastModifiedTime")),
        )


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> frozenset[str]:
    """Return a table's declared columns without trusting a filename.

    Antigravity has used several database names over its lifetime.  The
    trajectory schema, rather than ``.db`` suffixes or a machine-specific
    directory, is the admission identity.
    """
    try:
        from polylogue.sources.sqlite_export import readable_table_info

        return frozenset(str(row[1]) for row in readable_table_info(connection, table))
    except Exception as error:
        if _is_trajectory_storage_error(error):
            return frozenset()
        raise


def _trajectory_schema_matches(connection: sqlite3.Connection) -> bool:
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not _sqlite_columns(connection, "trajectory_meta") >= _TRAJECTORY_REQUIRED_META_COLUMNS:
        return False
    return "trajectory_meta" in tables and _sqlite_columns(connection, "steps") >= _TRAJECTORY_REQUIRED_STEP_COLUMNS


def looks_like_trajectory_db_path(path: Path, *, immutable: bool = False) -> bool:
    """Recognize the Antigravity trajectory store by its verified schema."""
    try:
        from polylogue.sources.sqlite_export import logical_source_context

        with logical_source_context(path, immutable=immutable) as connection:
            return _trajectory_schema_matches(connection)
    except Exception as error:
        if _is_trajectory_storage_error(error):
            return False
        raise


def _is_trajectory_storage_error(error: BaseException) -> bool:
    """Classify expected read/probe failures at the Antigravity adapter seam."""
    return isinstance(error, (OSError, sqlite3.Error, ValueError))


def _json_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return None
        if isinstance(decoded, dict):
            return {str(key): item for key, item in decoded.items()}
    return None


def _step_text(payload: Mapping[str, object]) -> str | None:
    for key in ("text", "content", "message", "body", "output", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (Mapping, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return None


def _step_timestamp(row: Mapping[str, object], payload: Mapping[str, object]) -> str | None:
    # Only source-declared timing is retained.  In particular, do not replace
    # an absent step timestamp with the database mtime or import time.
    for values in (payload, row):
        for key in ("timestamp", "occurred_at", "occurred_at_ms", "created_at", "createdAt", "updated_at"):
            value = values.get(key)
            if key == "occurred_at_ms":
                timestamp = iso_from_epoch_ms(value)
                if timestamp is not None:
                    return timestamp
            elif isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip():
                return str(value)
    return None


def _tool_outcome(
    payload: Mapping[str, object], row: Mapping[str, object]
) -> tuple[bool | None, int | None, str | None]:
    status = payload.get("status", row.get("status"))
    error = payload.get("error", row.get("error"))
    error_details = payload.get("error_details", row.get("error_details"))
    exit_code = next(
        (values[key] for values in (payload, row) for key in ("exit_code", "exitCode") if values.get(key) is not None),
        None,
    )
    if isinstance(exit_code, bool):
        exit_code = None
    if isinstance(exit_code, int) or (
        isinstance(exit_code, float) and math.isfinite(exit_code) and exit_code.is_integer()
    ):
        code = int(exit_code)
        return code != 0, code, None
    if isinstance(error, bool):
        return error, None, None
    if isinstance(error, (str, Mapping, list)) and error:
        return True, None, None
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {"ok", "success", "succeeded", "completed", "complete", "done"}:
            return False, None, None
        if normalized in {"error", "failed", "failure", "cancelled", "canceled", "aborted"}:
            return True, None, None
    if error_details not in (None, "", {}, []):
        return True, None, None
    return (
        None,
        None,
        unknown_reason(
            is_error=None,
            outcome_field_present=any(value is not None for value in (status, error, error_details, exit_code)),
        ),
    )


def _tool_input(payload: Mapping[str, object]) -> dict[str, object]:
    for key in ("input", "arguments", "args", "parameters"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return {str(name): item for name, item in value.items()}
        if isinstance(value, str):
            return {key: value}
    values = {
        key: payload[key] for key in ("command", "cmd", "path", "file_path", "query", "url", "patch") if key in payload
    }
    return values


def _file_edit(payload: Mapping[str, object]) -> ParsedFileEdit | None:
    """Return file-edit evidence only when the payload declares an edit result."""
    edit_fields = frozenset(
        {
            "old_string",
            "new_string",
            "structured_patch",
            "structuredPatch",
            "original_file",
            "replace_all",
            "user_modified",
            "oldString",
            "newString",
            "originalFile",
            "replaceAll",
            "userModified",
        }
    )
    if not edit_fields & payload.keys():
        return None
    replace_all = payload.get("replace_all", payload.get("replaceAll"))
    if not isinstance(replace_all, bool):
        replace_all = None
    user_modified = payload.get("user_modified", payload.get("userModified"))
    if not isinstance(user_modified, bool):
        user_modified = None
    structured_patch = payload.get("structured_patch", payload.get("structuredPatch"))
    return ParsedFileEdit(
        file_path=_string(payload.get("file_path") or payload.get("filePath") or payload.get("path")),
        structured_patch=structured_patch if isinstance(structured_patch, list) else None,
        original_file=_string(payload.get("original_file") or payload.get("originalFile")),
        old_string=_string(payload.get("old_string", payload.get("oldString"))),
        new_string=_string(payload.get("new_string", payload.get("newString"))),
        replace_all=replace_all,
        user_modified=user_modified,
    )


def _normalized_step_payload(row: Mapping[str, object]) -> dict[str, object] | None:
    payload = _json_mapping(row.get("step_payload"))
    if payload is None:
        return None
    # Some versions wrap the actual step under ``payload``.  Unwrap only a
    # mapping; opaque values remain refused rather than guessed into text.
    nested = payload.get("payload")
    if isinstance(nested, Mapping):
        return {
            **{key: value for key, value in payload.items() if key != "payload"},
            **{str(key): value for key, value in nested.items()},
        }
    return payload


def _trajectory_step_supported(step_type: str, step_format: str) -> bool:
    """Return whether the reviewed Antigravity step map knows this shape."""
    normalized_type = step_type.strip().lower().replace("-", "_")
    normalized_format = step_format.strip().lower().replace("-", "_")
    return (
        normalized_type in _TRAJECTORY_SUPPORTED_STEP_TYPES and normalized_format in _TRAJECTORY_SUPPORTED_STEP_FORMATS
    )


def _any_step_carries_a_key(connection: sqlite3.Connection, step_columns: Collection[str]) -> bool:
    """Whether any ``steps`` row carries a non-empty trajectory/cascade key."""
    key_columns = [name for name in ("trajectory_id", "cascade_id") if name in step_columns]
    if not key_columns:
        return False
    predicate = " OR ".join(f'"{name}" IS NOT NULL AND "{name}" != \'\'' for name in key_columns)
    row = connection.execute(f"SELECT EXISTS(SELECT 1 FROM steps WHERE {predicate})").fetchone()
    return bool(row[0])


def _parent_reference_id(reference: Mapping[str, object]) -> str | None:
    """Extract an asserted parent identity without joining on cascade names."""
    for key in ("parent_trajectory_id", "parent_cascade_id", "parent_session_id", "parent_id"):
        value = reference.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _event_payload_row(row: Mapping[str, object]) -> dict[str, object]:
    """Make SQLite row evidence safe for the canonical JSON event column."""
    payload: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, bytes):
            try:
                payload[str(key)] = value.decode("utf-8")
            except UnicodeDecodeError:
                payload[str(key)] = value.hex()
        else:
            payload[str(key)] = value
    return payload


def _step_tool_id(payload: Mapping[str, object]) -> str | None:
    """Return the call id a step declares on the wire, if any."""
    tool_id = payload.get("tool_id") or payload.get("toolId") or payload.get("call_id") or payload.get("callId")
    return str(tool_id) if tool_id is not None else None


def _answered_call_id(answerable_call: tuple[str, str] | None, result_tool_name: object) -> str | None:
    """Pair an ID-less result with the call it directly answers, or with none.

    A declared tool name that differs from the call's refutes the pairing.
    """
    if answerable_call is None:
        return None
    call_id, call_name = answerable_call
    if result_tool_name is not None and str(result_tool_name) != call_name:
        return None
    return call_id


def _trajectory_message(
    *,
    row: Mapping[str, object],
    payload: Mapping[str, object],
    position: int,
    step_ordinal: int,
    step_type: str,
    step_format: str,
    answerable_call: tuple[str, str] | None = None,
) -> ParsedMessage | None:
    """Materialize one supported step.

    ``answerable_call`` is the ``(tool_id, tool_name)`` of the ID-less call
    an ID-less result step may report: the caller passes it only when that
    call is the sole unanswered call immediately before this step.
    """
    if not _trajectory_step_supported(step_type, step_format):
        return None
    normalized_type = step_type.strip().lower().replace("-", "_")
    role_value = payload.get("role", row.get("role"))
    if isinstance(role_value, str):
        role = Role.normalize(role_value)
    elif normalized_type.startswith(("user", "human", "prompt")):
        role = Role.USER
    elif normalized_type in {"tool_result", "tool_output", "command_result"}:
        role = Role.TOOL
    else:
        role = Role.ASSISTANT
    # ``step_ordinal`` is the source ``idx``, not a count of materialized
    # messages: a malformed earlier step that later becomes materializable
    # must not renumber -- and so re-identify -- every unchanged message
    # after it.
    native_step_id = row.get("step_id") or row.get("id") or step_ordinal
    provider_message_id = f"{row.get('trajectory_id') or row.get('cascade_id') or 'trajectory'}:step:{native_step_id}"
    timestamp = _step_timestamp(row, payload)
    text = _step_text(payload)
    if normalized_type in {"plan", "plan_step", "planner", "planning"} and text is None:
        plan = payload.get("plan") or payload.get("steps") or payload.get("items")
        if plan is not None:
            text = json.dumps(plan, ensure_ascii=False, sort_keys=True)
    toolish = normalized_type in {
        "terminal",
        "terminal_command",
        "command",
        "tool",
        "tool_call",
        "tool_use",
        "tool_result",
        "tool_output",
        "command_result",
        "file_edit",
        "edit",
    }
    blocks: list[ParsedContentBlock] = []
    tool_name = payload.get("tool_name") or payload.get("toolName") or payload.get("name")
    tool_id = _step_tool_id(payload)
    if toolish and normalized_type in {"tool_result", "tool_output", "command_result"}:
        is_error, exit_code, outcome_unknown = _tool_outcome(payload, row)
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                text=text,
                tool_name=str(tool_name) if tool_name is not None else None,
                tool_id=tool_id if tool_id is not None else _answered_call_id(answerable_call, tool_name),
                is_error=is_error,
                exit_code=exit_code,
                outcome_unknown_reason=outcome_unknown,
                file_edit=_file_edit(payload),
            )
        )
        role = Role.TOOL
    elif toolish:
        if tool_name is None:
            tool_name = "terminal" if "command" in normalized_type or normalized_type == "terminal" else normalized_type
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                text=text,
                tool_name=str(tool_name),
                # An ID-less call still needs a block key: file_edits and the
                # use/result pairing are keyed by tool_id. The step's own
                # identity makes it deterministic across re-imports.
                tool_id=tool_id if tool_id is not None else f"{provider_message_id}:tool",
                tool_input=_tool_input(payload),
                file_edit=_file_edit(payload) if normalized_type in {"file_edit", "edit"} else None,
            )
        )
    elif text is not None:
        blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=text))
    if not blocks and text is None:
        return None
    message_type = MessageType.TOOL_USE if toolish else MessageType.MESSAGE
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=role,
        text=text,
        timestamp=timestamp,
        blocks=blocks,
        position=position,
        variant_index=0,
        is_active_path=True,
        # An explicit ``role`` is positive evidence about who authored the
        # content. Leaving it UNKNOWN excluded every imported Antigravity
        # prompt from human-authored/user-word and honest-cost accounting --
        # the same classification this module's Markdown parser applies.
        material_origin=human_authored_override(
            role,
            message_type,
            classify_material_origin(role=role, message_type=message_type, text=text),
        ),
    )


class _TrajectoryAccountingBuilder(Protocol):
    def append(self, outcome: AdmissionOutcome) -> None: ...

    def finish(self) -> ParseAccounting: ...


def _create_trajectory_parse_tables(connection: sqlite3.Connection) -> None:
    """Create disk-backed parser reductions in the caller-owned scratch DB."""
    for table in (
        "trajectory_native_ids",
        "trajectory_alias_claimants",
        "trajectory_summaries",
        "trajectory_parent_references",
        "trajectory_parent_ids",
    ):
        connection.execute(f"DROP TABLE IF EXISTS {table}")
    connection.execute("CREATE TABLE trajectory_native_ids (identity TEXT COLLATE BINARY PRIMARY KEY) WITHOUT ROWID")
    connection.execute(
        "CREATE TABLE trajectory_alias_claimants (alias TEXT COLLATE BINARY NOT NULL, "
        "canonical TEXT COLLATE BINARY NOT NULL, PRIMARY KEY(alias, canonical)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE trajectory_summaries (ordinal INTEGER PRIMARY KEY, identity TEXT COLLATE BINARY UNIQUE, "
        "row_json TEXT NOT NULL, matched INTEGER NOT NULL DEFAULT 0)"
    )
    connection.execute(
        "CREATE TABLE trajectory_parent_references (child TEXT COLLATE BINARY NOT NULL, ordinal INTEGER NOT NULL, "
        "row_json TEXT NOT NULL, parent TEXT COLLATE BINARY, PRIMARY KEY(child, ordinal)) WITHOUT ROWID"
    )
    connection.execute("CREATE TABLE trajectory_parent_ids (identity TEXT COLLATE BINARY PRIMARY KEY) WITHOUT ROWID")


def _event_json_row(row: Mapping[str, object]) -> str:
    return json.dumps(_event_payload_row(row), ensure_ascii=False, separators=(",", ":"))


def _load_trajectory_identity_evidence(
    source: sqlite3.Connection,
    grouping: sqlite3.Connection,
    *,
    meta_columns: Collection[str],
    summary_columns: Collection[str],
    parent_columns: Collection[str],
    check_cancelled: Callable[[], None],
) -> None:
    """Spill identity, summary, alias, and parent evidence before parsing steps."""
    if summary_columns:
        cursor = source.execute("SELECT * FROM conversation_summaries")
        try:
            for ordinal, row in enumerate(cursor):
                check_cancelled()
                key = next(
                    (
                        row[column]
                        for column in ("cascade_id", "trajectory_id")
                        if column in summary_columns and row[column] not in (None, "")
                    ),
                    None,
                )
                if key is None:
                    continue
                identity = str(key)
                grouping.execute("INSERT OR IGNORE INTO trajectory_native_ids VALUES (?)", (identity,))
                grouping.execute(
                    "INSERT INTO trajectory_summaries(ordinal, identity, row_json) VALUES (?, ?, ?) "
                    "ON CONFLICT(identity) DO UPDATE SET row_json=excluded.row_json",
                    (ordinal, identity, _event_json_row({str(k): row[k] for k in row.keys()})),  # noqa: SIM118
                )
        finally:
            cursor.close()
    if meta_columns:
        cursor = source.execute("SELECT * FROM trajectory_meta ORDER BY rowid")
        try:
            for row in cursor:
                check_cancelled()
                canonical = next(
                    (
                        str(row[name])
                        for name in ("trajectory_id", "cascade_id")
                        if name in meta_columns and row[name] not in (None, "")
                    ),
                    None,
                )
                for name in ("trajectory_id", "cascade_id"):
                    if name not in meta_columns or row[name] in (None, ""):
                        continue
                    alias = str(row[name])
                    grouping.execute("INSERT OR IGNORE INTO trajectory_native_ids VALUES (?)", (alias,))
                    if canonical is not None:
                        grouping.execute(
                            "INSERT OR IGNORE INTO trajectory_alias_claimants VALUES (?, ?)",
                            (alias, canonical),
                        )
        finally:
            cursor.close()
    if parent_columns:
        child_column = "cascade_id" if "cascade_id" in parent_columns else "trajectory_id"
        cursor = source.execute("SELECT * FROM parent_references")
        try:
            for ordinal, row in enumerate(cursor):
                check_cancelled()
                child_value = row[child_column]
                if child_value is None:
                    continue
                event_row = {str(key): row[key] for key in row.keys()}  # noqa: SIM118
                reference = _event_payload_row(event_row)
                parent = _parent_reference_id(reference)
                grouping.execute(
                    "INSERT INTO trajectory_parent_references VALUES (?, ?, ?, ?)",
                    (
                        str(child_value),
                        ordinal,
                        json.dumps(reference, ensure_ascii=False, separators=(",", ":")),
                        parent,
                    ),
                )
        finally:
            cursor.close()


def _row_identity(row: sqlite3.Row | None, column: str, columns: Collection[str]) -> str | None:
    if row is None or column not in columns or row[column] in (None, ""):
        return None
    return str(row[column])


def _trajectory_reserved(grouping: sqlite3.Connection, identity: str) -> bool:
    return (
        grouping.execute("SELECT 1 FROM trajectory_native_ids WHERE identity = ?", (identity,)).fetchone() is not None
    )


def _unused_trajectory_id(grouping: sqlite3.Connection, candidate: str) -> str:
    identity = candidate
    attempt = 0
    while _trajectory_reserved(grouping, identity):
        attempt += 1
        identity = f"{candidate}~{attempt}"
    return identity


def _trajectory_step_count(
    connection: sqlite3.Connection,
    step_columns: Collection[str],
    trajectory_id: str | None,
    cascade_id: str | None,
    meta_count: int,
) -> int:
    predicates: list[str] = []
    values: list[object] = []
    for column, value in (("trajectory_id", trajectory_id), ("cascade_id", cascade_id)):
        if column in step_columns and value is not None:
            predicates.append(f"{column} = ?")
            values.append(value)
    if predicates:
        count = int(
            connection.execute("SELECT COUNT(*) FROM steps WHERE " + " OR ".join(predicates), values).fetchone()[0]
        )
        if count:
            return count
    has_identity = bool({"trajectory_id", "cascade_id"}.intersection(step_columns))
    if meta_count == 1 and (not has_identity or not _any_step_carries_a_key(connection, step_columns)):
        return int(connection.execute("SELECT COUNT(*) FROM steps").fetchone()[0])
    return 0


def _matched_summary_key(grouping: sqlite3.Connection, cascade_id: str | None, trajectory_id: str | None) -> str | None:
    for identity in (cascade_id, trajectory_id):
        if (
            identity is not None
            and grouping.execute("SELECT 1 FROM trajectory_summaries WHERE identity = ?", (identity,)).fetchone()
        ):
            return identity
    return None


def _summary_row(grouping: sqlite3.Connection, identity: str) -> dict[str, object] | None:
    row = grouping.execute("SELECT row_json FROM trajectory_summaries WHERE identity = ?", (identity,)).fetchone()
    return json.loads(row[0]) if row else None


def _parent_reference_query(cascade_id: str | None, trajectory_id: str | None) -> tuple[str, tuple[str, ...]]:
    identities = [cascade_id] if cascade_id is not None else []
    if trajectory_id is not None and trajectory_id != cascade_id:
        identities.append(trajectory_id)
    if not identities:
        return "SELECT 0 WHERE 0", ()
    placeholders = ",".join("?" for _ in identities)
    order = "CASE child " + " ".join(f"WHEN ? THEN {i}" for i, _ in enumerate(identities)) + " ELSE 99 END"
    return (
        f"SELECT child, ordinal, row_json, parent FROM trajectory_parent_references WHERE child IN ({placeholders}) "
        f"ORDER BY {order}, ordinal",
        tuple(identities) + tuple(identities),
    )


def _settle_parent_references(
    grouping: sqlite3.Connection,
    cascade_id: str | None,
    trajectory_id: str | None,
    check_cancelled: Callable[[], None],
) -> tuple[str | None, int, bool, int]:
    grouping.execute("DELETE FROM trajectory_parent_ids")
    sql, params = _parent_reference_query(cascade_id, trajectory_id)
    cursor = grouping.execute(sql, params)
    count = 0
    try:
        for _child, _ordinal, _row_json, parent in cursor:
            check_cancelled()
            count += 1
            if parent is None:
                continue
            claimants = grouping.execute(
                "SELECT canonical FROM trajectory_alias_claimants WHERE alias = ? ORDER BY canonical", (parent,)
            )
            try:
                first = claimants.fetchone()
                if first is None:
                    grouping.execute("INSERT OR IGNORE INTO trajectory_parent_ids VALUES (?)", (parent,))
                else:
                    check_cancelled()
                    grouping.execute("INSERT OR IGNORE INTO trajectory_parent_ids VALUES (?)", first)
                    for claimant in claimants:
                        check_cancelled()
                        grouping.execute("INSERT OR IGNORE INTO trajectory_parent_ids VALUES (?)", claimant)
            finally:
                claimants.close()
    finally:
        cursor.close()
    id_count = int(grouping.execute("SELECT COUNT(*) FROM trajectory_parent_ids").fetchone()[0])
    parent_id = None
    if id_count == 1:
        parent_id = str(grouping.execute("SELECT identity FROM trajectory_parent_ids").fetchone()[0])
    observed = (
        parent_id is not None
        and grouping.execute("SELECT 1 FROM trajectory_native_ids WHERE identity = ?", (parent_id,)).fetchone()
        is not None
    )
    return parent_id, count, observed, id_count


def _streamed_array(
    grouping: sqlite3.Connection, values: Iterable[object], check_cancelled: Callable[[], None]
) -> object:
    from polylogue.sources.streamed_event_payload import SqliteJsonArrayWriter

    writer = SqliteJsonArrayWriter(grouping)
    for value in values:
        check_cancelled()
        writer.append(value)
    return writer.finish()


def _streamed_parent_ids(grouping: sqlite3.Connection, check_cancelled: Callable[[], None]) -> object:
    cursor = grouping.execute("SELECT identity FROM trajectory_parent_ids ORDER BY identity")
    try:
        return _streamed_array(grouping, (str(row[0]) for row in cursor), check_cancelled)
    finally:
        cursor.close()


def _streamed_parent_references(
    grouping: sqlite3.Connection,
    cascade_id: str | None,
    trajectory_id: str | None,
    check_cancelled: Callable[[], None],
) -> object:
    sql, params = _parent_reference_query(cascade_id, trajectory_id)
    cursor = grouping.execute(sql, params)
    try:
        return _streamed_array(grouping, (json.loads(row[2]) for row in cursor), check_cancelled)
    finally:
        cursor.close()


def _append_parent_reference_events(
    events: MutableSequence[ParsedSessionEvent],
    grouping: sqlite3.Connection,
    cascade_id: str | None,
    trajectory_id: str | None,
    parent_id: str | None,
    parent_observed: bool,
    parent_ids_count: int,
    check_cancelled: Callable[[], None],
) -> None:
    """Append the existing event payload shape backed by the shared spill."""
    parent_ids = _streamed_parent_ids(grouping, check_cancelled)
    references = _streamed_parent_references(grouping, cascade_id, trajectory_id, check_cancelled)
    events.append(
        ParsedSessionEvent(
            event_type="antigravity_parent_reference",
            payload={
                "references": references,
                "parent_provider_id": parent_id,
                "parent_provider_ids": parent_ids,
                "parent_observed": parent_observed,
            },
        )
    )
    if parent_ids_count > 1:
        events.append(
            ParsedSessionEvent(
                event_type="antigravity_ambiguous_parent_reference",
                payload={"parent_provider_ids": parent_ids},
            )
        )


def _trajectory_steps(
    connection: sqlite3.Connection,
    step_columns: Collection[str],
    trajectory_id: str | None,
    cascade_id: str | None,
    meta_count: int,
) -> Iterator[sqlite3.Row]:
    """Select native step rows without changing SQLite affinity or collation."""
    predicates: list[str] = []
    values: list[object] = []
    for column, value in (("trajectory_id", trajectory_id), ("cascade_id", cascade_id)):
        if column in step_columns and value is not None:
            predicates.append(f"{column} = ?")
            values.append(value)
    has_identity = bool({"trajectory_id", "cascade_id"}.intersection(step_columns))
    if predicates:
        cursor = connection.execute("SELECT * FROM steps WHERE " + " OR ".join(predicates) + " ORDER BY idx", values)
        try:
            first = cursor.fetchone()
            if first is not None:
                yield first
                yield from cursor
                return
        finally:
            cursor.close()

    if meta_count == 1 and (not has_identity or not _any_step_carries_a_key(connection, step_columns)):
        cursor = connection.execute("SELECT * FROM steps ORDER BY idx")
        try:
            yield from cursor
        finally:
            cursor.close()


def _inspect_trajectory_connection(
    connection: sqlite3.Connection,
    grouping: sqlite3.Connection,
    path: Path,
    *,
    preflight: bool,
) -> tuple[dict[str, object], int, bool]:
    """Count parser evidence, keeping identity reservations in private storage."""
    from polylogue.sources.dispatch import message_carries_authored_content
    from polylogue.sources.sqlite_export import LogicalExportError

    if not _trajectory_schema_matches(connection):
        raise LogicalExportError("Antigravity SQLite lacks the declared trajectory schema")
    meta_columns = _sqlite_columns(connection, "trajectory_meta")
    step_columns = _sqlite_columns(connection, "steps")
    summary_columns = _sqlite_columns(connection, "conversation_summaries")
    grouping.execute("CREATE TABLE trajectory_native_ids (identity TEXT COLLATE BINARY PRIMARY KEY) WITHOUT ROWID")
    grouping.execute(
        "CREATE TABLE trajectory_summaries (ordinal INTEGER PRIMARY KEY, identity TEXT COLLATE BINARY UNIQUE, "
        "matched INTEGER NOT NULL DEFAULT 0)"
    )

    def reserve(identity: str) -> None:
        grouping.execute("INSERT OR IGNORE INTO trajectory_native_ids VALUES (?)", (identity,))

    def reserved(identity: str) -> bool:
        return (
            grouping.execute("SELECT 1 FROM trajectory_native_ids WHERE identity = ?", (identity,)).fetchone()
            is not None
        )

    if summary_columns:
        for row in connection.execute("SELECT * FROM conversation_summaries"):
            key = next(
                (
                    row[column]
                    for column in ("cascade_id", "trajectory_id")
                    if column in summary_columns and row[column] not in (None, "")
                ),
                None,
            )
            if key is not None:
                identity = str(key)
                grouping.execute("INSERT OR IGNORE INTO trajectory_summaries (identity) VALUES (?)", (identity,))
                reserve(identity)
    meta_count = 0
    anonymous = 0
    for meta in connection.execute("SELECT * FROM trajectory_meta ORDER BY rowid"):
        meta_count += 1
        identities = [
            str(meta[column])
            for column in ("trajectory_id", "cascade_id")
            if column in meta_columns and meta[column] not in (None, "")
        ]
        anonymous += not identities
        for identity in identities:
            reserve(identity)
    if anonymous > 1:
        raise LogicalExportError(
            f"Antigravity SQLite holds {anonymous} trajectories with no trajectory or cascade id; "
            "no stable identity tells them apart"
        )
    fallback_id = path.stem
    effective_meta_count = meta_count or 1
    has_step_identity = bool({"trajectory_id", "cascade_id"}.intersection(step_columns))
    unattributed = (
        not has_step_identity
        and effective_meta_count > 1
        and bool(connection.execute("SELECT COUNT(*) FROM steps").fetchone()[0])
    )
    produced: dict[str, object] = {
        "sessions": 0,
        "messages": 0,
        "blocks": 0,
        "actions": 0,
        "raw_records": 0,
        "session_refs": [],
    }
    sessions = messages = blocks = actions = admitted = 0
    references: list[str] = []
    degraded = False
    meta_rows = connection.execute("SELECT * FROM trajectory_meta ORDER BY rowid") if meta_count else iter((None,))
    for meta in meta_rows:
        trajectory_id = (
            str(meta["trajectory_id"]) if meta is not None and meta["trajectory_id"] not in (None, "") else None
        )
        cascade_id = str(meta["cascade_id"]) if meta is not None and meta["cascade_id"] not in (None, "") else None
        row_fallback = fallback_id
        if trajectory_id is None and cascade_id is None and reserved(row_fallback):
            row_fallback = f"{fallback_id}:trajectory"
            attempt = 0
            while reserved(row_fallback):
                attempt += 1
                row_fallback = f"{fallback_id}:trajectory~{attempt}"
        native_id = trajectory_id or cascade_id or row_fallback
        own_messages = 0
        positive = False
        unsupported = False
        call_count = 0
        sole_call: tuple[str, str, bool] | None = None
        for ordinal, row in enumerate(
            _trajectory_steps(connection, step_columns, trajectory_id, cascade_id, effective_meta_count)
        ):
            row_columns = row.keys()
            row_map = {str(key): row[key] for key in row_columns}
            payload = _normalized_step_payload(row_map)
            previous_count, previous_call = call_count, sole_call
            call_count, sole_call = 0, None
            answerable_call = (
                (previous_call[0], previous_call[1])
                if previous_count == 1 and previous_call is not None and previous_call[2]
                else None
            )
            try:
                step_ordinal = int(row_map.get("idx", ordinal))
            except (TypeError, ValueError):
                step_ordinal = ordinal
            step_type = str(row_map.get("step_type") or "").strip().lower()
            step_format = str(row_map.get("step_format") or "").strip().lower()
            if payload is None or not _trajectory_step_supported(step_type, step_format):
                unsupported = True
                continue
            try:
                message = _trajectory_message(
                    row=row_map,
                    payload=payload,
                    position=own_messages,
                    step_ordinal=step_ordinal,
                    step_type=step_type,
                    step_format=step_format,
                    answerable_call=answerable_call,
                )
            except ValidationError:
                unsupported = True
                continue
            if message is None:
                unsupported = True
                continue
            own_messages += 1
            blocks += len(message.blocks)
            actions += sum(block.type is BlockType.TOOL_USE for block in message.blocks)
            positive |= message_carries_authored_content(message)
            call = message.blocks[0] if message.blocks and message.blocks[0].type is BlockType.TOOL_USE else None
            if call is not None and call.tool_id is not None:
                call_count = previous_count + 1
                if call_count == 1:
                    sole_call = (call.tool_id, call.tool_name or "", _step_tool_id(payload) is None)
        for summary_key in (cascade_id, trajectory_id):
            if summary_key is not None:
                cursor = grouping.execute(
                    "UPDATE trajectory_summaries SET matched = 1 WHERE identity = ?", (summary_key,)
                )
                if cursor.rowcount:
                    break
        sessions += 1
        messages += own_messages
        if not preflight:
            references.append(f"session:{Provider.ANTIGRAVITY.value}:{native_id}")
        admitted += positive if preflight else 0
        degraded |= unsupported or unattributed or (preflight and not positive)
    for row in grouping.execute("SELECT identity FROM trajectory_summaries WHERE matched = 0 ORDER BY ordinal"):
        sessions += 1
        if not preflight:
            references.append(f"session:{Provider.ANTIGRAVITY.value}:{row[0]}")
        degraded = True
    produced.update(
        sessions=sessions,
        messages=messages,
        blocks=blocks,
        actions=actions,
        raw_records=sessions,
        session_refs=references,
    )
    return produced, admitted, degraded


def parse_trajectory_db(
    path: Path,
    fallback_id: str | None = None,
    *,
    immutable: bool = False,
    grouping: sqlite3.Connection,
    message_sink_factory: Callable[[], MutableSequence[ParsedMessage]],
    event_sink_factory: Callable[[], MutableSequence[ParsedSessionEvent]],
    accounting_factory: Callable[[dict[AdmissionUnit, int]], _TrajectoryAccountingBuilder],
    check_cancelled: Callable[[], None] | None = None,
) -> Iterator[ParsedSession]:
    """Parse Antigravity's structured trajectory store through its read route.

    The parser is deliberately schema-first and fail-closed: a database with
    a familiar filename but no verified ``trajectory_meta``/``steps`` shape
    is not a conversation.  Unknown step formats remain session events and
    typed admission outcomes, so the writer can never report full coverage
    for a partially understood trajectory. The retained route requires
    caller-owned message, event, accounting, and grouping stores; it has no
    resident-list fallback.
    """
    from polylogue.sources.sqlite_export import logical_source_context

    with logical_source_context(path, immutable=immutable) as connection:
        yield from iter_trajectory_connection(
            connection,
            path,
            fallback_id=fallback_id,
            grouping=grouping,
            message_sink_factory=message_sink_factory,
            event_sink_factory=event_sink_factory,
            accounting_factory=accounting_factory,
            check_cancelled=check_cancelled,
        )


def iter_trajectory_connection(
    connection: sqlite3.Connection,
    path: Path,
    *,
    fallback_id: str | None,
    grouping: sqlite3.Connection,
    message_sink_factory: Callable[[], MutableSequence[ParsedMessage]],
    event_sink_factory: Callable[[], MutableSequence[ParsedSessionEvent]],
    accounting_factory: Callable[[dict[AdmissionUnit, int]], _TrajectoryAccountingBuilder],
    check_cancelled: Callable[[], None] | None = None,
) -> Iterator[ParsedSession]:
    """Parse one source transaction into caller-owned, disk-backed sinks.

    ``grouping`` is private scratch owned by the retained preparation. The
    message, event, and accounting factories are likewise caller-owned; this
    iterator never falls back to resident transcript or disposition lists.
    It yields a session only after all of that session's streams are complete.
    """
    from polylogue.sources.sqlite_export import LogicalExportError

    check_cancelled = check_cancelled or (lambda: None)
    connection.row_factory = sqlite3.Row
    check_cancelled()
    if not _trajectory_schema_matches(connection):
        raise LogicalExportError("Antigravity SQLite lacks the declared trajectory schema")
    meta_columns = _sqlite_columns(connection, "trajectory_meta")
    step_columns = _sqlite_columns(connection, "steps")
    summary_columns = _sqlite_columns(connection, "conversation_summaries")
    parent_columns = _sqlite_columns(connection, "parent_references")
    _create_trajectory_parse_tables(grouping)
    _load_trajectory_identity_evidence(
        connection,
        grouping,
        meta_columns=meta_columns,
        summary_columns=summary_columns,
        parent_columns=parent_columns,
        check_cancelled=check_cancelled,
    )
    meta_count = int(connection.execute("SELECT COUNT(*) FROM trajectory_meta").fetchone()[0])
    anonymous = int(
        connection.execute(
            "SELECT COUNT(*) FROM trajectory_meta WHERE "
            + " AND ".join(f'("{name}" IS NULL OR "{name}" = \'\')' for name in ("trajectory_id", "cascade_id"))
        ).fetchone()[0]
    )
    if anonymous > 1:
        raise LogicalExportError(
            f"Antigravity SQLite holds {anonymous} trajectories with no trajectory or cascade id; "
            "no stable identity tells them apart"
        )
    if not meta_count and not fallback_id:
        return
    effective_meta_count = meta_count or 1
    has_step_identity = bool({"trajectory_id", "cascade_id"}.intersection(step_columns))
    unattributed_count = (
        int(connection.execute("SELECT COUNT(*) FROM steps").fetchone()[0])
        if not has_step_identity and effective_meta_count > 1
        else 0
    )
    fallback_base = fallback_id or path.stem
    meta_cursor = connection.execute("SELECT * FROM trajectory_meta ORDER BY rowid") if meta_count else None
    try:
        meta_iter: Iterable[sqlite3.Row | None] = meta_cursor if meta_cursor is not None else (None,)
        for meta_index, meta in enumerate(meta_iter):
            check_cancelled()
            trajectory_id = _row_identity(meta, "trajectory_id", meta_columns)
            cascade_id = _row_identity(meta, "cascade_id", meta_columns)
            native_id = trajectory_id or cascade_id
            if native_id is None:
                native_id = fallback_base
                if _trajectory_reserved(grouping, native_id):
                    native_id = _unused_trajectory_id(grouping, f"{fallback_base}:trajectory")
            if not native_id:
                continue
            step_count = _trajectory_step_count(
                connection,
                step_columns,
                trajectory_id,
                cascade_id,
                effective_meta_count,
            )
            messages = message_sink_factory()
            events = event_sink_factory()
            accounting = accounting_factory({AdmissionUnit.PART: step_count})
            degraded_steps = False
            message_count = 0
            call_count = 0
            sole_call: tuple[str, str, bool] | None = None
            if unattributed_count:
                events.append(
                    ParsedSessionEvent(
                        event_type="antigravity_unattributed_steps",
                        payload={"step_count": unattributed_count, "meta_count": effective_meta_count},
                    )
                )
            for ordinal, row in enumerate(
                _trajectory_steps(connection, step_columns, trajectory_id, cascade_id, effective_meta_count)
            ):
                check_cancelled()
                row_map = {str(key): row[key] for key in row.keys()}  # noqa: SIM118
                payload = _normalized_step_payload(row_map)
                previous_call_count, previous_sole_call = call_count, sole_call
                call_count, sole_call = 0, None
                answerable_call = (
                    (previous_sole_call[0], previous_sole_call[1])
                    if previous_call_count == 1 and previous_sole_call is not None and previous_sole_call[2]
                    else None
                )
                try:
                    step_ordinal = int(row_map.get("idx", ordinal))
                except (TypeError, ValueError):
                    step_ordinal = ordinal
                step_type = str(row_map.get("step_type") or "").strip().lower()
                step_format = str(row_map.get("step_format") or "").strip().lower()
                outcome_key = f"step:{step_ordinal}"
                if payload is None:
                    accounting.append(
                        AdmissionOutcome(
                            unit=AdmissionUnit.PART,
                            ordinal=ordinal,
                            key=outcome_key,
                            disposition=AdmissionDisposition.TYPED_REFUSAL,
                            reason=AdmissionRefusalReason.MALFORMED,
                        )
                    )
                    degraded_steps = True
                    events.append(
                        ParsedSessionEvent(
                            event_type="antigravity_unsupported_step",
                            payload={
                                "idx": step_ordinal,
                                "step_type": step_type,
                                "step_format": step_format,
                                "reason": "malformed_payload",
                            },
                        )
                    )
                    continue
                if not _trajectory_step_supported(step_type, step_format):
                    accounting.append(
                        AdmissionOutcome(
                            unit=AdmissionUnit.PART,
                            ordinal=ordinal,
                            key=outcome_key,
                            disposition=AdmissionDisposition.TYPED_UNKNOWN,
                            reason=AdmissionUnknownReason.UNSUPPORTED_SHAPE,
                        )
                    )
                    degraded_steps = True
                    events.append(
                        ParsedSessionEvent(
                            event_type="antigravity_unsupported_step",
                            timestamp=_step_timestamp(row_map, payload),
                            payload={
                                "idx": step_ordinal,
                                "step_type": step_type,
                                "step_format": step_format,
                                "reason": "unsupported_step_format_or_type",
                                "payload": payload,
                            },
                        )
                    )
                    continue
                try:
                    message = _trajectory_message(
                        row=row_map,
                        payload=payload,
                        position=message_count,
                        step_ordinal=step_ordinal,
                        step_type=step_type,
                        step_format=step_format,
                        answerable_call=answerable_call,
                    )
                except ValidationError:
                    accounting.append(
                        AdmissionOutcome(
                            unit=AdmissionUnit.PART,
                            ordinal=ordinal,
                            key=outcome_key,
                            disposition=AdmissionDisposition.TYPED_REFUSAL,
                            reason=AdmissionRefusalReason.MALFORMED,
                        )
                    )
                    degraded_steps = True
                    events.append(
                        ParsedSessionEvent(
                            event_type="antigravity_unsupported_step",
                            payload={
                                "idx": step_ordinal,
                                "step_type": step_type,
                                "step_format": step_format,
                                "reason": "invalid_typed_step",
                            },
                        )
                    )
                    continue
                if message is None:
                    accounting.append(
                        AdmissionOutcome(
                            unit=AdmissionUnit.PART,
                            ordinal=ordinal,
                            key=outcome_key,
                            disposition=AdmissionDisposition.TYPED_UNKNOWN,
                            reason=AdmissionUnknownReason.UNSUPPORTED_SHAPE,
                        )
                    )
                    degraded_steps = True
                    events.append(
                        ParsedSessionEvent(
                            event_type="antigravity_unsupported_step",
                            timestamp=_step_timestamp(row_map, payload),
                            payload={
                                "idx": step_ordinal,
                                "step_type": step_type,
                                "step_format": step_format,
                                "payload": payload,
                            },
                        )
                    )
                    continue
                messages.append(message)
                message_count += 1
                call = message.blocks[0] if message.blocks and message.blocks[0].type is BlockType.TOOL_USE else None
                if call is not None and call.tool_id is not None:
                    call_count = previous_call_count + 1
                    if call_count == 1:
                        sole_call = (call.tool_id, call.tool_name or "", _step_tool_id(payload) is None)
                accounting.append(
                    AdmissionOutcome(
                        unit=AdmissionUnit.PART,
                        ordinal=ordinal,
                        key=outcome_key,
                        disposition=AdmissionDisposition.MATERIALIZED,
                    )
                )
            summary_key = _matched_summary_key(grouping, cascade_id, trajectory_id)
            title = None
            updated_at = None
            if summary_key is not None:
                summary = _summary_row(grouping, summary_key)
                if summary is not None:
                    grouping.execute("UPDATE trajectory_summaries SET matched = 1 WHERE identity = ?", (summary_key,))
                    for key in ("title", "name", "summary"):
                        if key in summary_columns and summary.get(key):
                            title = str(summary[key])
                            break
                    for key in ("last_modified_time", "updated_at", "updatedAt", "modified_at"):
                        if key in summary_columns and summary.get(key) is not None:
                            updated_at = str(summary[key])
                            break
                    events.append(
                        ParsedSessionEvent(
                            event_type="antigravity_conversation_summary",
                            timestamp=updated_at,
                            payload=summary,
                        )
                    )
            parent_id, parent_count, parent_observed, parent_ids_count = _settle_parent_references(
                grouping, cascade_id, trajectory_id, check_cancelled
            )
            if parent_count:
                _append_parent_reference_events(
                    events,
                    grouping,
                    cascade_id,
                    trajectory_id,
                    parent_id,
                    parent_observed,
                    parent_ids_count,
                    check_cancelled,
                )
                if parent_id is not None and not parent_observed:
                    events.append(
                        ParsedSessionEvent(
                            event_type="antigravity_unmatched_parent_reference",
                            payload={"parent_provider_id": parent_id},
                        )
                    )
            unit_accounting = accounting.finish()
            unit_accounting.assert_conserved()
            if not message_count:
                events.append(
                    ParsedSessionEvent(
                        event_type="antigravity_trajectory_empty",
                        payload={"step_count": step_count, "meta_index": meta_index},
                    )
                )
            ingest_flags = []
            if degraded_steps:
                ingest_flags.append("degraded:unsupported-trajectory-steps")
            if unattributed_count:
                ingest_flags.append("degraded:unattributed-trajectory-steps")
            session = ParsedSession.model_construct(
                source_name=Provider.ANTIGRAVITY,
                provider_session_id=native_id,
                provider_session_aliases=[
                    value for value in (trajectory_id, cascade_id) if value and value != native_id
                ],
                title=title,
                title_source=TitleSource.ORIGIN if title else None,
                updated_at=updated_at,
                messages=messages,
                session_events=events,
                unit_accounting=unit_accounting,
                active_leaf_message_provider_id=messages[-1].provider_message_id if message_count else None,
                attachments=[],
                parent_session_provider_id=parent_id,
                ingest_flags=ingest_flags,
                models_used=[],
                working_directories=[],
                pending_drafts=[],
                session_refs=[],
            )
            yield session
    finally:
        if meta_cursor is not None:
            meta_cursor.close()
    unmatched = grouping.execute(
        "SELECT identity, row_json FROM trajectory_summaries WHERE matched = 0 ORDER BY ordinal"
    )
    try:
        for summary_key, row_json in unmatched:
            check_cancelled()
            summary = json.loads(row_json)
            summary_time = next(
                (
                    str(summary[column])
                    for column in ("last_modified_time", "updated_at", "updatedAt", "modified_at")
                    if column in summary_columns and summary[column] is not None
                ),
                None,
            )
            events = event_sink_factory()
            events.append(
                ParsedSessionEvent(
                    event_type="antigravity_unmatched_summary",
                    timestamp=summary_time,
                    payload=summary,
                )
            )
            accounting = accounting_factory({})
            session = ParsedSession.model_construct(
                source_name=Provider.ANTIGRAVITY,
                provider_session_id=summary_key,
                title=None,
                title_source=None,
                updated_at=summary_time,
                messages=message_sink_factory(),
                session_events=events,
                unit_accounting=accounting.finish(),
                attachments=[],
                ingest_flags=["degraded:unmatched-trajectory-summary"],
                models_used=[],
                working_directories=[],
                pending_drafts=[],
                session_refs=[],
            )
            yield session
    finally:
        unmatched.close()


class _AntigravityLanguageServerExportClient(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]: ...

    def export_markdown(self, cascade_id: str) -> str: ...


class AntigravityLanguageServerClient:
    """Small client for Antigravity's own local language-server export API."""

    def __init__(
        self,
        root: Path,
        *,
        language_server_path: Path | None = None,
        startup_timeout_s: float = _STARTUP_TIMEOUT_S,
    ) -> None:
        self.root = root.expanduser()
        self.language_server_path = language_server_path
        self.startup_timeout_s = startup_timeout_s
        # The vendor server picks its own port (``-http_server_port=0``) and
        # publishes it in its discovery file, so no port is reserved and
        # released ahead of the child (the old TOCTOU). Every request carries a
        # per-run CSRF token; the server answers 401 without it, so another
        # local uid that finds the loopback port cannot search or export the
        # operator's conversations (polylogue-dahse).
        self.port: int | None = None
        self._csrf_token = secrets.token_urlsafe(32)
        self._process: subprocess.Popen[bytes] | None = None
        self.server_info: AntigravityLanguageServerInfo | None = None

    def __enter__(self) -> AntigravityLanguageServerClient:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def start(self) -> None:
        if self._process is not None:
            return
        binary = self.language_server_path or discover_language_server()
        if binary is None:
            raise AntigravityBinaryUnavailableError("Antigravity language_server_linux_x64 was not found")
        version = _discover_language_server_version(binary)

        cmd = [
            str(binary),
            "-standalone",
            "-persistent_mode",
            "-http_server_port=0",
            f"-csrf_token={self._csrf_token}",
            f"-gemini_dir={self.root.parent}",
            f"-app_data_dir={self.root.name}",
            "-override_ide_name=antigravity",
        ]
        before_launch = self._discovery_snapshot()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            self.port = self._await_discovered_port(before_launch=before_launch)
            self._wait_until_ready()
        except BaseException:
            # A start that does not complete (a refusal, a cancellation) must
            # not leave its child running; a retry would accumulate servers.
            self.close()
            raise
        self.server_info = AntigravityLanguageServerInfo(
            binary_path=binary,
            version=version,
            capabilities=("SearchConversations", "ConvertTrajectoryToMarkdown"),
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

    def search_sessions(self, *, limit: int = 10000, query: str = "") -> list[AntigravitySessionSummary]:
        payload = self._post(_SEARCH_ENDPOINT, {"query": query, "limit": limit})
        results = payload.get("results")
        if not isinstance(results, list):
            return []
        summaries: list[AntigravitySessionSummary] = []
        for item in results:
            if isinstance(item, dict):
                normalized = {str(key): value for key, value in item.items()}
                if summary := AntigravitySessionSummary.from_payload(normalized):
                    summaries.append(summary)
        return summaries

    def export_markdown(self, cascade_id: str) -> str:
        payload = self._post(_MARKDOWN_ENDPOINT, {"conversationId": cascade_id}, timeout=_CONVERSION_TIMEOUT_S)
        markdown = payload.get("markdown")
        if not isinstance(markdown, str) or not markdown:
            raise AntigravityExportError(f"Antigravity returned no markdown for cascade {cascade_id}")
        return markdown

    def _discovery_snapshot(self) -> dict[str, tuple[int, int, int, int]]:
        """``{name: (inode, mtime_ns, ctime_ns, size)}`` of the discovery files present now."""
        discovery_dir = self.root / "daemon"
        snapshot: dict[str, tuple[int, int, int, int]] = {}
        for candidate in discovery_dir.glob("ls_*.json") if discovery_dir.is_dir() else ():
            try:
                status = candidate.stat()
            except OSError:
                continue
            snapshot[candidate.name] = (status.st_ino, status.st_mtime_ns, status.st_ctime_ns, status.st_size)
        return snapshot

    def _await_discovered_port(self, *, before_launch: Mapping[str, tuple[int, int, int, int]]) -> int:
        """Read the port our own child published in its persistent-mode discovery file.

        The directory can also hold a discovery file from the operator's
        running IDE, so only the file naming this child's pid is accepted. A
        file left by a crashed server whose pid the child has since reused
        also names that pid, so a file is accepted only once it differs from
        the directory as it stood before this launch (new, replaced or
        rewritten); an unchanged one is watched until the child rewrites it.
        Comparing against that snapshot, not a clock cutoff, holds on a
        filesystem whose timestamps are coarser than the launch instant.

        There is no deadline: a slow child that is still alive is still
        starting. The child's exit ends the wait, and a cancellation of the
        caller ends it through ``start``, which then stops the child.
        """
        process = self._process
        if process is None:
            raise AntigravityExportError("Antigravity language server is not running")
        discovery_dir = self.root / "daemon"
        while True:
            if process.poll() is not None:
                raise AntigravityExportError(f"Antigravity language server exited with code {process.returncode}")
            for candidate in sorted(discovery_dir.glob("ls_*.json")) if discovery_dir.is_dir() else ():
                try:
                    status = candidate.stat()
                    fingerprint = (status.st_ino, status.st_mtime_ns, status.st_ctime_ns, status.st_size)
                    if before_launch.get(candidate.name) == fingerprint:
                        continue
                    published = loads(candidate.read_bytes())
                except (OSError, ValueError):
                    continue
                if not isinstance(published, dict) or published.get("pid") != process.pid:
                    continue
                port = published.get("httpPort")
                if isinstance(port, int) and not isinstance(port, bool) and port > 0:
                    return port
            time.sleep(_READY_RETRY_SLEEP_S)

    def _wait_until_ready(self) -> None:
        """Probe the vendor surface until it answers a search call.

        Bounded by an attempt floor as well as a deadline: one probe can spend
        the entire ``_REQUEST_TIMEOUT_S`` socket budget, so a deadline alone
        would let a single slow probe end readiness with no retry.
        """
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        attempts = 0
        while attempts < _MIN_READY_ATTEMPTS or time.monotonic() < deadline:
            attempts += 1
            process = self._process
            if process is not None and process.poll() is not None:
                raise AntigravityExportError(f"Antigravity language server exited with code {process.returncode}")
            try:
                self._post(_SEARCH_ENDPOINT, {"query": "", "limit": 1})
                return
            except AntigravityExportError as exc:
                last_error = exc
                time.sleep(_READY_RETRY_SLEEP_S)
        raise AntigravityExportError(
            f"Antigravity language server did not become ready after {attempts} probes: {last_error}"
        )

    def _post(self, endpoint: str, payload: JSONDocument, *, timeout: float | None = None) -> JSONDocument:
        if self.port is None:
            raise AntigravityExportError("Antigravity language server has not published its port")
        request = Request(
            f"http://127.0.0.1:{self.port}{endpoint}",
            data=dumps_bytes(payload),
            headers={"Content-Type": "application/json", "x-codeium-csrf-token": self._csrf_token},
            method="POST",
        )
        budget = _REQUEST_TIMEOUT_S if timeout is None else timeout
        try:
            with _LOOPBACK_OPENER.open(request, timeout=budget) as response:
                loaded = loads(response.read())
        except (OSError, TimeoutError, ValueError) as exc:
            raise AntigravityExportError(str(exc)) from exc
        if not isinstance(loaded, dict):
            raise AntigravityExportError(f"Antigravity endpoint {endpoint} returned non-object JSON")
        return {str(key): value for key, value in loaded.items()}


#: Root fields :func:`looks_like_markdown_export` reads.
MARKDOWN_EXPORT_SIGNATURE_FIELDS = frozenset({"source", "cascadeId", "markdown"})


def looks_like_markdown_export(payload: JSONDocument) -> bool:
    return (
        payload.get("source") == "antigravity_language_server"
        and isinstance(payload.get("cascadeId"), str)
        and isinstance(payload.get("markdown"), str)
    )


def _validate_language_server_markdown(markdown: str, cascade_id: str) -> None:
    """Reject a successful RPC response that is not a transcript export."""
    sections = list(_SECTION_RE.finditer(markdown))
    if not sections:
        raise AntigravityExportError(f"language server returned partial conversation export for cascade {cascade_id}")
    has_content = any(
        markdown[section.end() : (sections[index + 1].start() if index + 1 < len(sections) else len(markdown))].strip()
        for index, section in enumerate(sections)
    )
    if not has_content:
        raise AntigravityExportError(f"language server returned an empty conversation export for cascade {cascade_id}")


def markdown_export_payload(summary: AntigravitySessionSummary, markdown: str) -> JSONDocument:
    payload: JSONDocument = {
        "source": "antigravity_language_server",
        "cascadeId": summary.cascade_id,
        "markdown": markdown,
    }
    if summary.title:
        payload["title"] = summary.title
    if summary.workspace_name:
        payload["workspaceName"] = summary.workspace_name
    if summary.snippet:
        payload["snippet"] = summary.snippet
    if summary.last_modified_time:
        payload["lastModifiedTime"] = summary.last_modified_time
    return payload


@parser_admission("antigravity")
def parse_markdown_export_payload(payload: JSONDocument, fallback_id: str) -> ParsedSession:
    summary = AntigravitySessionSummary(
        cascade_id=_string(payload.get("cascadeId")) or fallback_id,
        title=_string(payload.get("title")),
        workspace_name=_string(payload.get("workspaceName")),
        snippet=_string(payload.get("snippet")),
        last_modified_time=_string(payload.get("lastModifiedTime")),
    )
    return parse_markdown_export(_string(payload.get("markdown")) or "", summary)


def parse_markdown_export(
    markdown: str,
    summary: AntigravitySessionSummary,
) -> ParsedSession:
    messages = _mark_active_leaf(_messages_from_markdown(markdown, summary.cascade_id))

    return ParsedSession(
        source_name=Provider.ANTIGRAVITY,
        provider_session_id=summary.cascade_id,
        title=summary.title,
        title_source=TitleSource.ORIGIN if summary.title else None,
        created_at=None,
        updated_at=summary.last_modified_time,
        messages=messages,
        active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
    )


def iter_language_server_exports(
    root: Path,
    *,
    client: _AntigravityLanguageServerExportClient | None = None,
    only_cascade_ids: frozenset[str] | None = None,
) -> Iterable[ParsedSession]:
    """Yield successful conversions, preserving the historical strict API."""
    outcomes = iter_language_server_export_results(root, client=client, only_cascade_ids=only_cascade_ids)
    expected = len(_conversation_pb_paths(root))
    if only_cascade_ids is not None:
        expected = sum(1 for path in _conversation_pb_paths(root) if path.stem in only_cascade_ids)
    for obtained, outcome in enumerate(outcomes):
        if not outcome.obtained:
            raise AntigravityPartialExportError(
                f"Antigravity export failed for cascade {outcome.cascade_id}: {outcome.error}",
                obtained=obtained,
                expected=expected,
            )
        assert outcome.session is not None
        yield outcome.session


def iter_language_server_export_results(
    root: Path,
    *,
    client: _AntigravityLanguageServerExportClient | None = None,
    only_cascade_ids: frozenset[str] | None = None,
    admit_path: Callable[[Path], bool] | None = None,
) -> Iterable[AntigravityExportOutcome]:
    """Yield one typed outcome for every manifested conversation protobuf.

    Conversion failures are isolated to their item so a poison trajectory
    cannot suppress unrelated progress. Startup and handshake failures remain
    raised because they invalidate the complete source route. ``admit_path``
    runs before each item's conversion (and before starting an owned server).
    Refused items remain unattempted; scheduling exceptions propagate unchanged.
    """
    owned_client = client is None
    runtime_client = client or AntigravityLanguageServerClient(root)
    try:
        pb_paths = _conversation_pb_paths(root)
        if only_cascade_ids is not None:
            pb_paths = [pb_path for pb_path in pb_paths if pb_path.stem in only_cascade_ids]
        if not pb_paths:
            return
        summaries_by_id: dict[str, AntigravitySessionSummary] | None = None
        seen_ids: set[str] = set()
        for pb_path in pb_paths:
            if admit_path is not None and not admit_path(pb_path):
                continue
            if summaries_by_id is None:
                if owned_client:
                    runtime_client.start()
                try:
                    summaries_by_id = {summary.cascade_id: summary for summary in runtime_client.search_sessions()}
                except Exception as exc:
                    raise AntigravityExportError(f"Antigravity SearchConversations handshake failed: {exc}") from exc
            cascade_id = pb_path.stem
            if cascade_id in seen_ids:
                yield AntigravityExportOutcome(
                    pb_path,
                    cascade_id,
                    error="duplicate conversation identity",
                    converter=getattr(runtime_client, "server_info", None),
                )
                continue
            seen_ids.add(cascade_id)
            try:
                before = _file_digest(pb_path)
                summary = summaries_by_id.get(cascade_id) or AntigravitySessionSummary(
                    cascade_id=cascade_id,
                    last_modified_time=_iso_mtime(pb_path),
                )
                markdown = runtime_client.export_markdown(cascade_id)
                _validate_language_server_markdown(markdown, cascade_id)
                session = parse_markdown_export_payload(markdown_export_payload(summary, markdown), cascade_id)
                if not session.messages or not any((message.text or "").strip() for message in session.messages):
                    raise AntigravityExportError("language server returned an empty or partial conversation export")
                after = _file_digest(pb_path)
                if before != after:
                    raise AntigravityExportError("conversation protobuf changed during conversion")
            except Exception as exc:
                yield AntigravityExportOutcome(
                    pb_path,
                    cascade_id,
                    error=str(exc),
                    converter=getattr(runtime_client, "server_info", None),
                )
            else:
                yield AntigravityExportOutcome(
                    pb_path,
                    cascade_id,
                    session=session,
                    converter=getattr(runtime_client, "server_info", None),
                    source_sha256=after,
                )
    finally:
        if owned_client:
            runtime_client.close()


def _conversation_pb_paths(root: Path) -> list[Path]:
    """List every production-discovered conversation trajectory."""
    from polylogue.sources.source_walk import layout_source_paths

    return [
        path
        for path in layout_source_paths("antigravity", root)
        if path.suffix.lower() == ".pb"
        and classify_source_path(path).role is AntigravitySourceRole.CONVERSATION_PROTOBUF
    ]


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iso_mtime(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()


def discover_language_server() -> Path | None:
    """Find the vendor binary on ``PATH``, in the system install, or in the Nix store."""
    if binary_path := shutil.which("language_server_linux_x64"):
        return Path(binary_path)
    if _SYSTEM_LANGUAGE_SERVER.is_file():
        return _SYSTEM_LANGUAGE_SERVER
    return _nix_store_language_server()


_NIX_STORE = Path("/nix/store")
#: Where the vendor's Linux package installs the bundled binary.
_SYSTEM_LANGUAGE_SERVER = Path(
    "/usr/share/antigravity/resources/app/extensions/antigravity/bin/language_server_linux_x64"
)
_PACKAGED_LANGUAGE_SERVER_PATHS = (
    "lib/antigravity/resources/app/extensions/antigravity/bin/language_server_linux_x64",
    "lib/antigravity-ide/resources/app/extensions/antigravity/bin/language_server_linux_x64",
)


def _nix_store_language_server() -> Path | None:
    """Find a packaged binary by streaming the store's top level once.

    The store holds hundreds of thousands of entries; one ``scandir`` pass
    filtered by name reads it without expanding a glob per pattern.
    """
    candidates: list[Path] = []
    try:
        with os.scandir(_NIX_STORE) as entries:
            for entry in entries:
                if "-antigravity-" not in entry.name:
                    continue
                for relative in _PACKAGED_LANGUAGE_SERVER_PATHS:
                    binary = Path(entry.path) / relative
                    if binary.is_file():
                        candidates.append(binary)
    except OSError:
        return None
    return sorted(candidates)[-1] if candidates else None


def _discover_language_server_version(binary: Path) -> str:
    """Read the vendor binary version before using its HTTP conversion API."""

    def compatible(version: str) -> str:
        if version.split(".", 1)[0] not in {"1", "2"}:
            raise AntigravityExportError(
                f"incompatible Antigravity language server version {version}; expected vendor 1.x or 2.x"
            )
        return version

    configured = os.environ.get("POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER_VERSION")
    if configured and configured.strip():
        return compatible(configured.strip())
    for flag in ("--version", "-version"):
        try:
            completed = subprocess.run(
                [str(binary), flag],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AntigravityExportError(f"could not handshake with language server {binary}: {exc}") from exc
        output = completed.stdout.decode("utf-8", errors="replace").strip()
        match = re.search(r"\b\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b", output)
        if match:
            return compatible(match.group(0))
    for candidate in (binary, binary.resolve()):
        package_match = re.search(r"antigravity-(\d+\.\d+\.\d+)", str(candidate))
        if package_match:
            return compatible(package_match.group(1))
    raise AntigravityExportError(f"language server {binary} did not report a compatible version")


def _activity_marker(match: re.Match[str]) -> AntigravityActivityMarker:
    """Build the marker one ``_ACTIVITY_MARKER_RE`` match stands for."""
    tool_name = next(name for name in _ACTIVITY_ARGUMENTS if match.group(name) is not None)
    argument = _ACTIVITY_ARGUMENTS[tool_name]
    tool_input: dict[str, object] | None = None
    if argument is not None:
        value = (match.group(f"v_{tool_name}") or "").strip()
        if value:
            tool_input = {argument: value}
    return AntigravityActivityMarker(tool_name=tool_name, tool_input=tool_input, rendered=match.group(0))


def _section_runs(body: str) -> list[str | tuple[AntigravityActivityMarker, ...]]:
    """Split one transcript section into alternating prose and activity runs.

    The export renders every tool call as an italic marker inside the section
    that follows the turn it belongs to -- including inside a ``User Input``
    section, where the tool activity is the agent's, not the operator's. Runs
    keep that boundary so authorship stays per message. Whitespace between two
    markers does not end their run; any other text does.
    """
    runs: list[str | tuple[AntigravityActivityMarker, ...]] = []
    markers: list[AntigravityActivityMarker] = []
    cursor = 0
    for match in _ACTIVITY_MARKER_RE.finditer(body):
        gap = body[cursor : match.start()].strip()
        if gap:
            if markers:
                runs.append(tuple(markers))
                markers = []
            runs.append(gap)
        markers.append(_activity_marker(match))
        cursor = match.end()
    if markers:
        runs.append(tuple(markers))
    tail = body[cursor:].strip()
    if tail:
        runs.append(tail)
    return runs


def _messages_from_markdown(markdown: str, cascade_id: str) -> list[ParsedMessage]:
    sections = list(_SECTION_RE.finditer(markdown))
    messages: list[ParsedMessage] = []
    activity_ordinal = 0
    for index, section in enumerate(sections):
        start = section.end()
        end = sections[index + 1].start() if index + 1 < len(sections) else len(markdown)
        heading = section.group("title")
        section_role = Role.USER if heading == "User Input" else Role.ASSISTANT
        for run in _section_runs(markdown[start:end]):
            if isinstance(run, str):
                messages.append(_prose_message(run, cascade_id, section_role, heading, len(messages)))
            else:
                messages.append(_activity_message(run, cascade_id, len(messages), activity_ordinal))
                activity_ordinal += 1

    if messages:
        return messages

    text = _strip_markdown_preamble(markdown)
    if not text:
        return []
    return [
        ParsedMessage(
            provider_message_id=synthetic_message_id(
                namespace=cascade_id,
                role=Role.ASSISTANT,
                text=text,
                timestamp=None,
                kind="export",
            ),
            role=Role.ASSISTANT,
            text=text,
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            position=0,
            variant_index=0,
            is_active_path=True,
        )
    ]


def _prose_message(
    text: str,
    cascade_id: str,
    role: Role,
    heading: str,
    position: int,
) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=synthetic_message_id(
            namespace=cascade_id,
            role=role,
            text=text,
            timestamp=None,
            kind=_message_kind(heading),
        ),
        role=role,
        text=text,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
        position=position,
        variant_index=0,
        is_active_path=True,
        # polylogue-gzgyl: an antigravity "User Input" section is
        # unambiguously a real human turn -- positive-evidence
        # override for the shared classify_material_origin
        # no-fallthrough (#2502).
        material_origin=human_authored_override(
            role,
            MessageType.MESSAGE,
            classify_material_origin(role=role, message_type=MessageType.MESSAGE, text=text),
        ),
    )


def _activity_message(
    markers: tuple[AntigravityActivityMarker, ...],
    cascade_id: str,
    position: int,
    ordinal: int,
) -> ParsedMessage:
    """Build the agent tool-activity message for one run of vendor markers.

    The marker phrasing is fully recoverable from ``tool_name`` plus
    ``tool_input``, so the message carries no prose: retaining the rendered
    line as text would count agent activity as authored words again.

    ``ordinal`` counts activity runs within the transcript and enters the
    identity seed. Two runs can render identical markers -- a lone ``*Edited
    relevant file*`` recurs throughout a real transcript -- and they are
    distinct events, so content alone cannot identify them.
    """
    rendered = "\n".join(marker.rendered for marker in markers)
    provider_message_id = synthetic_message_id(
        namespace=cascade_id,
        role=Role.ASSISTANT,
        text=rendered,
        timestamp=None,
        kind=f"{_ACTIVITY_MESSAGE_KIND}.{ordinal}",
    )
    blocks = [
        ParsedContentBlock(
            type=BlockType.TOOL_USE,
            tool_name=marker.tool_name,
            tool_id=f"{provider_message_id}.{block_index}",
            tool_input=marker.tool_input,
        )
        for block_index, marker in enumerate(markers)
    ]
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=Role.ASSISTANT,
        text=None,
        blocks=blocks,
        message_type=MessageType.TOOL_USE,
        position=position,
        variant_index=0,
        is_active_path=True,
        material_origin=classify_material_origin(
            role=Role.ASSISTANT,
            message_type=MessageType.TOOL_USE,
            text=None,
            block_types=tuple(block.type for block in blocks),
        ),
    )


def _mark_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    # bd polylogue-2hwl: delegate to the shared position-based helper --
    # flagging by provider_message_id equality (the previous approach here)
    # flags every message sharing the final message's id, not just the true
    # leaf, whenever a retried/regenerated section reuses that id.
    return mark_last_occurrence_as_active_leaf(messages)


def _strip_markdown_preamble(markdown: str) -> str:
    lines = markdown.splitlines()
    while lines and (lines[0].startswith("# ") or lines[0].startswith("Note:") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _message_kind(heading: str) -> str:
    return heading.lower().replace(" ", "_")


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "AntigravityActivityMarker",
    "AntigravityBinaryUnavailableError",
    "AntigravitySessionSummary",
    "AntigravityExportError",
    "AntigravityLanguageServerClient",
    "AntigravityPartialExportError",
    "AntigravityExportOutcome",
    "AntigravityLanguageServerInfo",
    "AntigravitySourceClassification",
    "AntigravitySourceRole",
    "AntigravitySourceInspection",
    "AntigravitySourceItem",
    "AntigravitySourceCensus",
    "AntigravitySourceMutationError",
    "census_source",
    "classify_source_path",
    "conversation_pb_paths",
    "discover_language_server",
    "iter_language_server_exports",
    "looks_like_markdown_export",
    "markdown_export_payload",
    "iter_language_server_export_results",
    "parse_markdown_export",
    "parse_markdown_export_payload",
]

# Public alias -- ``source_parsing.py`` needs the same disk-truth listing to
# locate the raw ``.pb`` bytes for blob snapshotting per exported session.
conversation_pb_paths = _conversation_pb_paths


def detection_projection() -> DetectorProjection:
    """Keep only the exact markdown-export signature fields."""
    return DetectorProjection(fields={name: DetectorProjection() for name in ("source", "cascadeId", "markdown")})
