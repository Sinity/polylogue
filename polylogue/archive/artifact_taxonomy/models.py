"""Artifact taxonomy data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from polylogue.core.enums import Provider


class ArtifactKind(StrEnum):
    SESSION_DOCUMENT = "session_document"
    SESSION_RECORD_STREAM = "session_record_stream"
    AGENT_SIDECAR_META = "agent_sidecar_meta"
    WORKFLOW_RUN_SNAPSHOT = "workflow_run_snapshot"
    WORKFLOW_JOURNAL = "workflow_journal"
    AGENT_TRANSCRIPT = "agent_transcript"
    ADOPT_MANIFEST = "adopt_manifest"
    COORDINATOR_SESSION_STREAM = "coordinator_session_stream"
    TODO_SNAPSHOT = "todo_snapshot"
    SESSION_INDEX = "session_index"
    BRIDGE_POINTER = "bridge_pointer"
    # polylogue-omsw: Claude Code persists tool-result overflow content to
    # ``<session>/tool-results/<name>.<ext>`` (see
    # ``sources/live/tool_result_sidecars.py``'s module docstring) -- the
    # sidecar join mechanism reads it FROM disk to attach to its owning
    # ``tool_result`` block, it is never independent conversation content.
    # A tool call's own output can coincidentally reproduce a genuine
    # session-document shape byte-for-byte (verified live: a
    # ``tool-results/*.txt`` file whose content was a real claude.ai export
    # document -- some prior turn's tool call had fetched and dumped one --
    # classified as SESSION_DOCUMENT/parse_as_session=True under the old
    # content-only rules), so content heuristics alone can never refuse this
    # family; only a path rule can, mirroring ``AGENT_SIDECAR_META`` /
    # ``WORKFLOW_JOURNAL``.
    TOOL_RESULT_SIDECAR = "tool_result_sidecar"
    # polylogue-omsw: Claude Code writes a session-uuid-named ``.jsonl`` file
    # under ``projects/<proj>/`` for pure filesystem-checkpoint activity that
    # never carried a single chat turn -- every record's ``type`` is
    # ``file-history-snapshot``/``progress``. The path alone is
    # indistinguishable from a genuine ``coordinator_session_stream`` (same
    # ``projects/<proj>/<uuid>.jsonl`` shape), so the ``OriginArtifactRule``
    # path match wins first and would otherwise mark this content
    # ``parse_as_session=True``. ``classify_artifact`` (content-aware) checks
    # the decoded record types and overrides to this sidecar kind when every
    # record is one of the known non-conversational envelope types --
    # mirroring ``archive/raw_materialization.py``'s
    # ``parsed_non_session_artifact_reason``, which already performs the
    # equivalent check post-parse for archive-debt/backlog reporting.
    FILE_HISTORY_SNAPSHOT = "file_history_snapshot"
    METADATA_DOCUMENT = "metadata_document"
    HOOK_EVENT = "hook_event"
    # polylogue-hbtj2: a raw payload whose magic bytes are a recognized
    # binary container with no dedicated, content-verified session parser
    # for this exact shape -- refused as session content at the earliest
    # detection point (``core.binary_signatures.detect_binary_signature``),
    # never handed to a session/JSON parser. ``BINARY_DATABASE`` covers
    # SQLite-shaped payloads specifically (the concrete miscapture the audit
    # found: Hermes/Codex state databases swept into ``raw_sessions``
    # without content verification); ``BINARY_DOCUMENT`` covers the other
    # recognized binary formats in the same registry (PNG/JPEG/PDF/gzip).
    BINARY_DATABASE = "binary_database"
    BINARY_DOCUMENT = "binary_document"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArtifactClassification:
    """Heuristic classification of a raw payload or document sample."""

    provider: Provider
    kind: ArtifactKind
    parse_as_session: bool
    schema_eligible: bool
    default_priority: int
    reason: str

    @property
    def cohort(self) -> str:
        return self.kind.value
