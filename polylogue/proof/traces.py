"""Observable trace primitives for proof evidence.

Trace evidence compares semantic operations across surfaces. It intentionally
does not compare raw rendered output strings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Self

from polylogue.lib.json import JSONDocument, JSONValue, require_json_document


class ObservableEventName(str, Enum):
    """Stable event nouns shared by proof evidence and runtime diagnostics."""

    READ_ARCHIVE = "ReadArchive"
    APPLY_FILTER = "ApplyFilter"
    RETURN_ROWS = "ReturnRows"
    EMIT_CONVERSATION = "EmitConversation"
    EMIT_ERROR = "EmitError"


@dataclass(frozen=True, slots=True)
class HappensBeforeEdge:
    """Partial-order edge between two observable trace events."""

    before_event_id: str
    after_event_id: str
    relation: str = "total_order"

    def to_payload(self) -> JSONDocument:
        return {
            "before_event_id": self.before_event_id,
            "after_event_id": self.after_event_id,
            "relation": self.relation,
        }


@dataclass(frozen=True, slots=True)
class ObservableTraceEvent:
    """One normalized observable operation emitted by a surface adapter."""

    event_id: str
    name: ObservableEventName
    payload: JSONDocument
    surface: str
    subject_id: str | None = None
    claim_id: str | None = None
    operation: str | None = None
    artifact_node: str | None = None

    def semantic_key(self) -> tuple[str, str]:
        return (self.name.value, _json_fingerprint(self.payload))

    def to_payload(self) -> JSONDocument:
        return require_json_document(
            {
                "event_id": self.event_id,
                "name": self.name.value,
                "payload": self.payload,
                "surface": self.surface,
                "subject_id": self.subject_id,
                "claim_id": self.claim_id,
                "operation": self.operation,
                "artifact_node": self.artifact_node,
            },
            context="observable trace event",
        )


@dataclass(frozen=True, slots=True)
class ObservableTrace:
    """Ordered trace with room for partial-order evidence."""

    surface: str
    events: tuple[ObservableTraceEvent, ...]
    happens_before: tuple[HappensBeforeEdge, ...]

    @classmethod
    def ordered(cls, *, surface: str, events: tuple[ObservableTraceEvent, ...]) -> Self:
        edges = tuple(
            HappensBeforeEdge(before_event_id=before.event_id, after_event_id=after.event_id)
            for before, after in zip(events, events[1:], strict=False)
        )
        return cls(surface=surface, events=events, happens_before=edges)

    def event_names(self) -> tuple[str, ...]:
        return tuple(event.name.value for event in self.events)

    def semantic_signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(event.semantic_key() for event in self.events)

    def to_payload(self) -> JSONDocument:
        return require_json_document(
            {
                "surface": self.surface,
                "events": [event.to_payload() for event in self.events],
                "happens_before": [edge.to_payload() for edge in self.happens_before],
            },
            context="observable trace",
        )


@dataclass(frozen=True, slots=True)
class ObservableDiagnosticMapping:
    """Mapping from an existing diagnostic shape into observable event nouns."""

    diagnostic_name: str
    source: str
    event_name: ObservableEventName
    payload_contract: JSONDocument
    subject_id: str
    operation: str
    artifact_node: str
    claim_id: str | None = None

    def to_payload(self) -> JSONDocument:
        return require_json_document(
            {
                "diagnostic_name": self.diagnostic_name,
                "source": self.source,
                "event_name": self.event_name.value,
                "payload_contract": self.payload_contract,
                "subject_id": self.subject_id,
                "operation": self.operation,
                "artifact_node": self.artifact_node,
                "claim_id": self.claim_id,
            },
            context="observable diagnostic mapping",
        )


def ids_hash(ids: tuple[str, ...]) -> str:
    """Hash returned row IDs without exposing the full result list as the comparator."""
    payload = json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_filter_query_trace(
    *,
    surface: str,
    provider: str,
    ids: tuple[str, ...],
    subject_id: str,
    claim_id: str | None = None,
) -> ObservableTrace:
    """Build the first query trace: read archive, apply provider filter, return rows."""
    operation = "query-conversations"
    events = (
        ObservableTraceEvent(
            event_id=f"{surface}:read-archive",
            name=ObservableEventName.READ_ARCHIVE,
            payload={"operation": operation, "archive_scope": "seeded"},
            surface=surface,
            subject_id=subject_id,
            claim_id=claim_id,
            operation=operation,
            artifact_node="archive_conversation_rows",
        ),
        ObservableTraceEvent(
            event_id=f"{surface}:apply-provider-filter",
            name=ObservableEventName.APPLY_FILTER,
            payload={"filter": "provider", "provider": provider},
            surface=surface,
            subject_id=subject_id,
            claim_id=claim_id,
            operation=operation,
            artifact_node="conversation_query_results",
        ),
        ObservableTraceEvent(
            event_id=f"{surface}:return-rows",
            name=ObservableEventName.RETURN_ROWS,
            payload={"count": len(ids), "ids_hash": ids_hash(ids)},
            surface=surface,
            subject_id=subject_id,
            claim_id=claim_id,
            operation=operation,
            artifact_node="conversation_query_results",
        ),
    )
    return ObservableTrace.ordered(surface=surface, events=events)


def pipeline_probe_archive_subset_mapping(
    *, subject_id: str, claim_id: str | None = None
) -> ObservableDiagnosticMapping:
    """Map archive-subset pipeline-probe diagnostics to the observable vocabulary."""
    return ObservableDiagnosticMapping(
        diagnostic_name="pipeline-probe.archive-subset.sample",
        source="devtools.pipeline_probe.ProbeSummary.sample",
        event_name=ObservableEventName.READ_ARCHIVE,
        payload_contract={
            "input_mode": "archive-subset",
            "selected_count": "sample.selected_count",
            "provider_counts": "sample.provider_counts",
        },
        subject_id=subject_id,
        claim_id=claim_id,
        operation="acquire-raw-conversations",
        artifact_node="source_payload_stream",
    )


def trace_signature_hash(trace: ObservableTrace) -> str:
    payload: tuple[tuple[str, str], ...] = trace.semantic_signature()
    return _json_fingerprint(payload)


def _json_fingerprint(value: JSONValue | tuple[object, ...]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "HappensBeforeEdge",
    "ObservableDiagnosticMapping",
    "ObservableEventName",
    "ObservableTrace",
    "ObservableTraceEvent",
    "ids_hash",
    "pipeline_probe_archive_subset_mapping",
    "provider_filter_query_trace",
    "trace_signature_hash",
]
