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
