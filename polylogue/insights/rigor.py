"""Per-product insight rigor contract matrix (#1275).

Each insight product declared in :mod:`polylogue.insights.archive` has a
``RigorContract`` row here. The contract is the durable, machine-readable
description of how to read a row of that product — which fields carry
direct evidence from the source archive, which fields carry probabilistic
inference, what fallback markers callers should consult, what readiness
semantics apply, and which consumer-facing fields are stable surface
contract.

The contract drives:

- ``polylogue insights audit`` (CLI): per-product rigor profile rollup
  with evidence/inference/fallback marker coverage, stale-version count,
  and confidence distribution.
- documentation: the rendered matrix lives at
  ``docs/insights-rigor-matrix.md`` (regenerated from this module).
- consumer self-discovery: future MCP and API surfaces can query the
  matrix instead of reading prose docs.

This module is contract-only; it does not query the archive. The audit
runner lives in :mod:`polylogue.insights.audit`.
"""

from __future__ import annotations

from collections.abc import Sequence

from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.storage.runtime.store_constants import (
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
)


class RigorVersionField(ArchiveInsightModel):
    """One materialization version that a row carries."""

    name: str
    current_version: int
    payload_path: tuple[str, ...] = ()


class RigorContract(ArchiveInsightModel):
    """Per-product rigor contract matrix entry.

    Fields:
        insight_name: matches the registry ``InsightType.name``.
        display_name: human label for tables and reports.
        evidence_payload: dotted path to the evidence payload field on the
            insight item (e.g. ``("evidence",)``). Empty tuple means the
            product is aggregate-only with no direct evidence payload.
        inference_payload: dotted path to the inference payload field.
            Empty tuple means the product is evidence-only or aggregate.
        fallback_markers: payload-dotted paths whose truthy value flags
            a row as having taken a fallback rather than a fully-grounded
            inference (e.g. ``("inference", "fallback_inference")``).
        confidence_field: payload-dotted path to a ``[0, 1]`` confidence
            score, when the product carries one. Empty tuple means no
            confidence score is exposed.
        readiness_semantics: short prose describing how callers should
            decide whether a row is consumable.
        consumer_fields: stable surface fields that consumers may rely
            on. Listed for documentation and contract review; not
            enforced at runtime.
        version_fields: materialization version fields the row carries.
            Used by the audit runner to count stale-version rows.
        notes: optional free-form notes (deprecated fields, transition
            anchors, etc.).
    """

    insight_name: str
    display_name: str
    evidence_payload: tuple[str, ...] = ()
    inference_payload: tuple[str, ...] = ()
    fallback_markers: tuple[tuple[str, ...], ...] = ()
    confidence_field: tuple[str, ...] = ()
    readiness_semantics: str = ""
    consumer_fields: tuple[str, ...] = ()
    version_fields: tuple[RigorVersionField, ...] = ()
    notes: str = ""


_RIGOR_MATRIX: tuple[RigorContract, ...] = (
    RigorContract(
        insight_name="session_profiles",
        display_name="Session Profiles",
        evidence_payload=("evidence",),
        inference_payload=("inference",),
        fallback_markers=(("enrichment", "fallback_reasons"),),
        confidence_field=("enrichment", "confidence"),
        readiness_semantics=(
            "Evidence payload is fully grounded in archive counts and timestamps. "
            "Inference payload is probabilistic — consult ``inference.support_level`` "
            "and ``inference.engaged_duration_source`` for grounding. "
            "Enrichment payload is also probabilistic and carries intent/outcome "
            "summaries plus ``enrichment.support_level`` / ``enrichment.confidence``. "
            "Profiles missing an ``inference`` payload should be treated as "
            "evidence-only."
        ),
        consumer_fields=(
            "session_id",
            "source_name",
            "title",
            "semantic_tier",
            "evidence",
            "inference",
            "enrichment",
            "provenance",
            "inference_provenance",
            "enrichment_provenance",
        ),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
            RigorVersionField(name="inference_version", current_version=SESSION_INFERENCE_VERSION),
            RigorVersionField(name="enrichment_version", current_version=SESSION_ENRICHMENT_VERSION),
        ),
    ),
    RigorContract(
        insight_name="session_work_events",
        display_name="Work Events",
        evidence_payload=("evidence",),
        inference_payload=("inference",),
        fallback_markers=(("inference", "fallback_inference"),),
        confidence_field=("inference", "confidence"),
        readiness_semantics=(
            "Evidence payload describes the message-range and timing footprint "
            "of the event. Inference payload carries heuristic label/summary; rows with "
            "``inference.fallback_inference == True`` were emitted by the "
            "heuristic fallback and should be treated as low-rigor."
        ),
        consumer_fields=(
            "event_id",
            "session_id",
            "source_name",
            "event_index",
            "evidence",
            "inference",
        ),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
            RigorVersionField(name="inference_version", current_version=SESSION_INFERENCE_VERSION),
        ),
    ),
    RigorContract(
        insight_name="session_phases",
        display_name="Session Phases",
        evidence_payload=("evidence",),
        inference_payload=("inference",),
        fallback_markers=(("inference", "fallback_inference"),),
        confidence_field=("inference", "confidence"),
        readiness_semantics=(
            "Evidence payload describes the phase's message-range timing and "
            "tool counts. Inference payload carries the phase-kind classification "
            "with a confidence score; ``inference.fallback_inference`` flags "
            "heuristic fallback rows."
        ),
        consumer_fields=(
            "phase_id",
            "session_id",
            "source_name",
            "phase_index",
            "evidence",
            "inference",
        ),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
            RigorVersionField(name="inference_version", current_version=SESSION_INFERENCE_VERSION),
        ),
    ),
    RigorContract(
        insight_name="threads",
        display_name="Work Threads",
        evidence_payload=("thread",),
        inference_payload=(),
        fallback_markers=(),
        confidence_field=(),
        readiness_semantics=(
            "Thread payload is a deterministic rollup over session "
            "parent/child links; there is no probabilistic inference layer. "
            "Rigor is governed by the underlying parent-link evidence."
        ),
        consumer_fields=("thread_id", "root_id", "dominant_repo", "thread"),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
        ),
    ),
    RigorContract(
        insight_name="session_tag_rollups",
        display_name="Session Tag Rollups",
        evidence_payload=(),
        inference_payload=(),
        fallback_markers=(),
        confidence_field=(),
        readiness_semantics=(
            "Tag rollups aggregate explicit and auto tag counts. Auto-tag "
            "rows derive from probabilistic enrichment; explicit-tag rows "
            "are direct evidence. Inspect ``explicit_count`` vs "
            "``auto_count`` for rigor."
        ),
        consumer_fields=(
            "tag",
            "session_count",
            "explicit_count",
            "auto_count",
            "provider_breakdown",
            "repo_breakdown",
        ),
        version_fields=(
            RigorVersionField(name="materializer_version", current_version=SESSION_INSIGHT_MATERIALIZER_VERSION),
        ),
    ),
)


def list_rigor_contracts() -> tuple[RigorContract, ...]:
    """Return the immutable per-product rigor contract matrix."""

    return _RIGOR_MATRIX


def get_rigor_contract(insight_name: str) -> RigorContract | None:
    """Lookup a rigor contract by its insight registry name."""

    for contract in _RIGOR_MATRIX:
        if contract.insight_name == insight_name:
            return contract
    return None


def rigor_contract_names() -> tuple[str, ...]:
    """Return the insight names covered by the matrix in declaration order."""

    return tuple(contract.insight_name for contract in _RIGOR_MATRIX)


def resolve_payload(obj: object, path: Sequence[str]) -> object | None:
    """Walk a dotted attribute/key path against an insight item.

    The path can mix model attributes and dict keys. Returns ``None`` when
    any segment is missing or the result itself is ``None``.
    """

    current: object | None = obj
    for segment in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(segment)
            continue
        current = getattr(current, segment, None)
    return current


__all__ = [
    "RigorContract",
    "RigorVersionField",
    "get_rigor_contract",
    "list_rigor_contracts",
    "resolve_payload",
    "rigor_contract_names",
]
