"""Artifact taxonomy data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from polylogue.core.enums import Provider


class ArtifactKind(StrEnum):
    SESSION_DOCUMENT = "session_document"
    SESSION_RECORD_STREAM = "session_record_stream"
    SUBAGENT_SESSION_STREAM = "subagent_session_stream"
    AGENT_SIDECAR_META = "agent_sidecar_meta"
    WORKFLOW_RUN_SNAPSHOT = "workflow_run_snapshot"
    WORKFLOW_JOURNAL = "workflow_journal"
    AGENT_TRANSCRIPT = "agent_transcript"
    ADOPT_MANIFEST = "adopt_manifest"
    SESSION_INDEX = "session_index"
    BRIDGE_POINTER = "bridge_pointer"
    METADATA_DOCUMENT = "metadata_document"
    HOOK_EVENT = "hook_event"
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
