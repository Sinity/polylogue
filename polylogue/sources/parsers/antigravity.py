"""Parser and local export client for Antigravity session state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import stat as stat_module
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from glob import glob
from pathlib import Path
from types import TracebackType
from typing import Protocol
from urllib.request import Request, urlopen

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.archive.message.artifacts import classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, Provider, TitleSource
from polylogue.core.json import JSONDocument, dumps_bytes, loads

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


def trajectory_raw_id(source_path: Path | str, logical_revision: str) -> str:
    """Stable raw identity for one Antigravity trajectory-store revision."""
    identity = f"antigravity-trajectory\0{Path(source_path).expanduser().resolve()}\0{logical_revision}"
    return hashlib.sha256(identity.encode("utf-8", errors="surrogateescape")).hexdigest()


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
    ("accepted_command", r"\*User accepted the command `(?P<v>.*?)`[ \t]*\*", "command"),
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

    def on_walk_error(error: OSError) -> None:
        path = Path(error.filename) if error.filename else root
        record_unreadable(path, f"source item is unreadable: {error}")

    from polylogue.sources.source_walk import _iter_source_entries

    # Use the same skip-directory traversal as production admission.  Census
    # still inspects non-regular files (including symlinks) so they remain
    # accounted for as unsupported evidence, while admission excludes them.
    for path in _iter_source_entries(root, onerror=on_walk_error):
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
                classification=classify_source_path(path),
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

    @property
    def obtained(self) -> bool:
        return self.session is not None and self.error is None


def classify_source_path(source_path: str | Path) -> AntigravitySourceClassification:
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
        quoted = '"' + table.replace('"', '""') + '"'
        return frozenset(str(row[1]) for row in connection.execute(f"PRAGMA table_info({quoted})"))
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
    if path.suffix.lower() not in _TRAJECTORY_DB_SUFFIXES:
        return False
    try:
        from polylogue.sources.sqlite_export import open_logical_source

        connection = open_logical_source(path, immutable=immutable)
        try:
            return _trajectory_schema_matches(connection)
        finally:
            connection.close()
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
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value)
    return None


def _tool_outcome(
    payload: Mapping[str, object], row: Mapping[str, object]
) -> tuple[bool | None, int | None, str | None]:
    status = payload.get("status", row.get("status"))
    error = payload.get("error", row.get("error"))
    error_details = payload.get("error_details", row.get("error_details"))
    exit_code = payload.get("exit_code", payload.get("exitCode"))
    if isinstance(exit_code, bool):
        exit_code = None
    if isinstance(exit_code, (int, float)):
        code = int(exit_code)
        return code != 0, code, None
    if isinstance(error, bool):
        return error, None, None
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {"ok", "success", "succeeded", "completed", "complete", "done"}:
            return False, None, None
        if normalized in {"error", "failed", "failure", "cancelled", "canceled", "aborted"}:
            return True, None, None
    if error_details not in (None, "", {}, []):
        return True, None, None
    return None, None, "unsupported_construct"


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


def _normalized_step_payload(row: Mapping[str, object]) -> dict[str, object] | None:
    payload = _json_mapping(row.get("step_payload"))
    if payload is None:
        return None
    # Some versions wrap the actual step under ``payload``.  Unwrap only a
    # mapping; opaque values remain refused rather than guessed into text.
    nested = payload.get("payload")
    if isinstance(nested, Mapping):
        return {str(key): value for key, value in nested.items()}
    return payload


def _trajectory_message(
    *,
    row: Mapping[str, object],
    payload: Mapping[str, object],
    position: int,
    step_type: str,
    step_format: str,
) -> ParsedMessage | None:
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
    native_step_id = row.get("step_id") or row.get("id") or position
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
    tool_id = payload.get("tool_id") or payload.get("toolId") or payload.get("call_id") or payload.get("callId")
    if toolish and normalized_type in {"tool_result", "tool_output", "command_result"}:
        is_error, exit_code, unknown_reason = _tool_outcome(payload, row)
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                text=text,
                tool_name=str(tool_name) if tool_name is not None else None,
                tool_id=str(tool_id) if tool_id is not None else None,
                is_error=is_error,
                exit_code=exit_code,
                outcome_unknown_reason=unknown_reason,
            )
        )
        role = Role.TOOL
    elif toolish:
        if tool_name is None:
            tool_name = "terminal" if "command" in normalized_type or normalized_type == "terminal" else normalized_type
        old_string = _string(payload.get("old_string"))
        new_string = _string(payload.get("new_string"))
        replace_all = payload.get("replace_all")
        if not isinstance(replace_all, bool):
            replace_all = None
        blocks.append(
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                text=text,
                tool_name=str(tool_name),
                tool_id=str(tool_id) if tool_id is not None else None,
                tool_input=_tool_input(payload),
                file_edit=(
                    ParsedFileEdit(
                        file_path=(
                            str(payload.get("file_path") or payload.get("filePath") or payload.get("path"))
                            if payload.get("file_path") or payload.get("filePath") or payload.get("path")
                            else None
                        ),
                        old_string=old_string,
                        new_string=new_string,
                        replace_all=replace_all,
                    )
                    if normalized_type in {"file_edit", "edit"}
                    else None
                ),
            )
        )
    elif text is not None:
        blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=text))
    if not blocks and text is None:
        return None
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=role,
        text=text,
        timestamp=timestamp,
        blocks=blocks,
        position=position,
        variant_index=0,
        is_active_path=True,
    )


def parse_trajectory_db(
    path: Path,
    fallback_id: str | None = None,
    *,
    immutable: bool = False,
) -> Iterator[ParsedSession]:
    """Parse Antigravity's structured trajectory store through its read route.

    The parser is deliberately schema-first and fail-closed: a database with
    a familiar filename but no verified ``trajectory_meta``/``steps`` shape
    is not a conversation.  Unknown step formats remain session events and
    typed admission outcomes, so the writer can never report full coverage
    for a partially understood trajectory.
    """
    from polylogue.sources.sqlite_export import LogicalExportError, open_logical_source

    connection = open_logical_source(path, immutable=immutable)
    connection.row_factory = sqlite3.Row
    try:
        if not _trajectory_schema_matches(connection):
            raise LogicalExportError("Antigravity SQLite lacks the declared trajectory schema")
        meta_columns = _sqlite_columns(connection, "trajectory_meta")
        step_columns = _sqlite_columns(connection, "steps")
        summary_columns = _sqlite_columns(connection, "conversation_summaries")
        parent_columns = _sqlite_columns(connection, "parent_references")
        summaries: dict[str, sqlite3.Row] = {}
        if summary_columns:
            for row in connection.execute("SELECT * FROM conversation_summaries"):
                key = (
                    row["cascade_id"]
                    if "cascade_id" in summary_columns
                    else row["trajectory_id"]
                    if "trajectory_id" in summary_columns
                    else None
                )
                if key is not None:
                    summaries[str(key)] = row
        parent_refs: dict[str, list[dict[str, object]]] = {}
        if parent_columns:
            for row in connection.execute("SELECT * FROM parent_references"):
                child = (
                    row["cascade_id"]
                    if "cascade_id" in parent_columns
                    else row["trajectory_id"]
                    if "trajectory_id" in parent_columns
                    else None
                )
                if child is not None:
                    parent_refs.setdefault(str(child), []).append(
                        {str(key): value for key, value in zip(row.keys(), row, strict=True)}
                    )
        meta_query = "SELECT * FROM trajectory_meta ORDER BY rowid"
        for meta in connection.execute(meta_query):
            trajectory_id = (
                str(meta["trajectory_id"])
                if "trajectory_id" in meta_columns and meta["trajectory_id"] not in (None, "")
                else None
            )
            cascade_id = (
                str(meta["cascade_id"])
                if "cascade_id" in meta_columns and meta["cascade_id"] not in (None, "")
                else None
            )
            native_id = trajectory_id or cascade_id or fallback_id
            if not native_id:
                continue
            if "trajectory_id" in step_columns or "cascade_id" in step_columns:
                predicates: list[str] = []
                values: list[object] = []
                if "trajectory_id" in step_columns and trajectory_id is not None:
                    predicates.append("trajectory_id = ?")
                    values.append(trajectory_id)
                if "cascade_id" in step_columns and cascade_id is not None:
                    predicates.append("cascade_id = ?")
                    values.append(cascade_id)
                steps = (
                    connection.execute(
                        "SELECT * FROM steps WHERE " + " OR ".join(predicates) + " ORDER BY idx",
                        values,
                    ).fetchall()
                    if predicates
                    else []
                )
            else:
                steps = []
            # A store may key steps by a single native id not named in the
            # meta row.  The verified schema still permits that shape.
            if not steps:
                steps = connection.execute("SELECT * FROM steps ORDER BY idx").fetchall()
            messages: list[ParsedMessage] = []
            outcomes: list[AdmissionOutcome] = []
            events: list[ParsedSessionEvent] = []
            for ordinal, row in enumerate(steps):
                row_columns = row.keys()
                row_map = {str(key): row[key] for key in row_columns}
                payload = _normalized_step_payload(row_map)
                idx = row_map.get("idx", ordinal)
                try:
                    step_ordinal = int(idx)
                except (TypeError, ValueError):
                    step_ordinal = ordinal
                step_type = str(row_map.get("step_type") or "").strip().lower()
                step_format = str(row_map.get("step_format") or "").strip().lower()
                key = f"step:{step_ordinal}"
                if payload is None:
                    outcomes.append(
                        AdmissionOutcome(
                            unit=AdmissionUnit.PART,
                            ordinal=ordinal,
                            key=key,
                            disposition=AdmissionDisposition.TYPED_REFUSAL,
                            reason=AdmissionRefusalReason.MALFORMED,
                        )
                    )
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
                message = _trajectory_message(
                    row=row_map, payload=payload, position=len(messages), step_type=step_type, step_format=step_format
                )
                if message is None:
                    outcomes.append(
                        AdmissionOutcome(
                            unit=AdmissionUnit.PART,
                            ordinal=ordinal,
                            key=key,
                            disposition=AdmissionDisposition.TYPED_UNKNOWN,
                            reason=AdmissionUnknownReason.UNSUPPORTED_SHAPE,
                        )
                    )
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
                outcomes.append(
                    AdmissionOutcome(
                        unit=AdmissionUnit.PART, ordinal=ordinal, key=key, disposition=AdmissionDisposition.MATERIALIZED
                    )
                )
            summary = summaries.get(cascade_id or "")
            title = None
            updated_at = None
            if summary is not None:
                for key in ("title", "name", "summary"):
                    if key in summary_columns and summary[key]:
                        title = str(summary[key])
                        break
                for key in ("last_modified_time", "updated_at", "updatedAt", "modified_at"):
                    if key in summary_columns and summary[key] is not None:
                        updated_at = str(summary[key])
                        break
            if parent_refs.get(cascade_id or ""):
                events.append(
                    ParsedSessionEvent(
                        event_type="antigravity_parent_reference", payload={"references": parent_refs[cascade_id or ""]}
                    )
                )
            accounting = ParseAccounting(expected={AdmissionUnit.PART: len(steps)}, outcomes=outcomes)
            accounting.assert_conserved()
            if not messages and steps:
                events.append(
                    ParsedSessionEvent(event_type="antigravity_trajectory_empty", payload={"step_count": len(steps)})
                )
            yield_session = ParsedSession(
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
                unit_accounting=accounting,
                active_leaf_message_provider_id=messages[-1].provider_message_id if messages else None,
            )
            yield_session = yield_session.model_copy(
                update={"ingest_flags": ["degraded:unsupported-trajectory-steps"]}
                if any(outcome.disposition is not AdmissionDisposition.MATERIALIZED for outcome in outcomes)
                else {}
            )
            yield yield_session
    finally:
        connection.close()


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
        self.port = _free_local_port()
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
            f"-http_server_port={self.port}",
            f"-gemini_dir={self.root.parent}",
            f"-app_data_dir={self.root.name}",
            "-override_ide_name=antigravity",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._wait_until_ready()
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
        request = Request(
            f"http://127.0.0.1:{self.port}{endpoint}",
            data=dumps_bytes(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        budget = _REQUEST_TIMEOUT_S if timeout is None else timeout
        try:
            with urlopen(request, timeout=budget) as response:
                loaded = loads(response.read())
        except (OSError, TimeoutError, ValueError) as exc:
            raise AntigravityExportError(str(exc)) from exc
        if not isinstance(loaded, dict):
            raise AntigravityExportError(f"Antigravity endpoint {endpoint} returned non-object JSON")
        return {str(key): value for key, value in loaded.items()}


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
) -> Iterable[AntigravityExportOutcome]:
    """Yield one typed outcome for every manifested conversation protobuf.

    Conversion failures are isolated to their item so a poison trajectory
    cannot suppress unrelated progress. Startup and handshake failures remain
    raised because they invalidate the complete source route.
    """
    owned_client = client is None
    runtime_client = client or AntigravityLanguageServerClient(root)
    try:
        if owned_client:
            runtime_client.start()
        pb_paths = _conversation_pb_paths(root)
        if only_cascade_ids is not None:
            pb_paths = [pb_path for pb_path in pb_paths if pb_path.stem in only_cascade_ids]
        if not pb_paths:
            return
        try:
            summaries_by_id = {summary.cascade_id: summary for summary in runtime_client.search_sessions()}
        except Exception as exc:
            raise AntigravityExportError(f"Antigravity SearchConversations handshake failed: {exc}") from exc
        seen_ids: set[str] = set()
        for pb_path in pb_paths:
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
                )
    finally:
        if owned_client:
            runtime_client.close()


def _conversation_pb_paths(root: Path) -> list[Path]:
    """List every production-discovered conversation trajectory."""
    from polylogue.sources.source_walk import _walk_source_paths

    return [
        path
        for path in _walk_source_paths(root, provider=Provider.ANTIGRAVITY)
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
    from polylogue.config import load_polylogue_config

    configured_path = load_polylogue_config().antigravity_language_server
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.is_file():
            return path

    if binary_path := shutil.which("language_server_linux_x64"):
        return Path(binary_path)

    candidates = sorted(
        Path(match)
        for match in glob(
            "/nix/store/*-antigravity-*/lib/antigravity/resources/app/extensions/antigravity/bin/language_server_linux_x64"
        )
    )
    return candidates[-1] if candidates else None


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


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
