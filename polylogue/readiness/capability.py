"""Canonical capability-readiness DTOs and adapters.

This module is the convergence point for #1832: existing archive, daemon,
embedding, insight, and operation status types can map into one small product
capability vocabulary without rewriting every surface in the same PR.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polylogue.core.json import JSONDocument, json_document
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.operations.operation_contract import OperationStatus
from polylogue.storage.repair import ArchiveDebtStatus


class CapabilityReadinessState(str, Enum):
    """Shared product-readiness vocabulary across status surfaces."""

    READY = "ready"
    REBUILDING = "rebuilding"
    STALE = "stale"
    MISSING = "missing"
    BLOCKED = "blocked"
    DEGRADED = "degraded"
    POISONED = "poisoned"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ComponentReadiness:
    """Machine-readable readiness for one product capability/component."""

    component: str
    state: CapabilityReadinessState
    scope: str = "archive"
    summary: str = ""
    last_success: str | None = None
    last_attempt: str | None = None
    counts: Mapping[str, int | float | bool | None] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()
    repair_hint: str | None = None
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> JSONDocument:
        payload: dict[str, object] = {
            "component": self.component,
            "scope": self.scope,
            "state": self.state.value,
            "summary": self.summary,
            "last_success": self.last_success,
            "last_attempt": self.last_attempt,
            "counts": dict(self.counts),
            "caveats": list(self.caveats),
            "repair_hint": self.repair_hint,
            "evidence_refs": list(self.evidence_refs),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return json_document(payload)


LEGACY_READINESS_SOURCE_TYPES: tuple[str, ...] = (
    "OutcomeStatus",
    "OperationStatus",
    "ReadinessReport",
    "InsightReadinessReport",
    "InsightReadinessEntry",
    "SessionInsightStatusSnapshot",
    "ArchiveDebtStatus",
    "EmbeddingStatusPayload",
    "DerivedModelStatus",
    "CatchupStatus",
)


def component_from_outcome_check(check: OutcomeCheck, *, scope: str = "archive") -> ComponentReadiness:
    state = {
        OutcomeStatus.OK: CapabilityReadinessState.READY,
        OutcomeStatus.WARNING: CapabilityReadinessState.DEGRADED,
        OutcomeStatus.ERROR: CapabilityReadinessState.BLOCKED,
        OutcomeStatus.SKIP: CapabilityReadinessState.MISSING,
    }[check.status]
    counts: dict[str, int] = {"count": check.count}
    counts.update(check.breakdown)
    return ComponentReadiness(
        component=check.name,
        scope=scope,
        state=state,
        summary=check.summary,
        counts=counts,
        caveats=tuple(check.details),
    )


def component_from_derived_model(status: DerivedModelStatus, *, scope: str = "derived_model") -> ComponentReadiness:
    if status.ready:
        state = CapabilityReadinessState.READY
    elif status.invalidated_reason is not None or status.stale_rows:
        state = CapabilityReadinessState.STALE
    elif status.source_documents == 0 and status.source_rows == 0 and status.materialized_rows == 0:
        state = CapabilityReadinessState.MISSING
    else:
        state = CapabilityReadinessState.DEGRADED

    caveats: list[str] = []
    if status.invalidated_reason is not None:
        caveats.append(f"invalidated:{status.invalidated_reason.value}")
    if status.matches_version is False:
        caveats.append("materializer_version_mismatch")

    return ComponentReadiness(
        component=status.name,
        scope=scope,
        state=state,
        summary=status.detail,
        counts={
            "source_documents": status.source_documents,
            "materialized_documents": status.materialized_documents,
            "source_rows": status.source_rows,
            "materialized_rows": status.materialized_rows,
            "pending_documents": status.pending_documents,
            "pending_rows": status.pending_rows,
            "stale_rows": status.stale_rows,
            "orphan_rows": status.orphan_rows,
            "missing_provenance_rows": status.missing_provenance_rows,
        },
        caveats=tuple(caveats),
    )


def component_from_archive_debt(status: ArchiveDebtStatus, *, scope: str = "archive_debt") -> ComponentReadiness:
    if status.healthy:
        state = CapabilityReadinessState.READY
    elif status.skipped:
        state = CapabilityReadinessState.BLOCKED
    else:
        state = CapabilityReadinessState.DEGRADED
    return ComponentReadiness(
        component=status.name,
        scope=scope,
        state=state,
        summary=status.detail,
        counts={"issue_count": status.issue_count, "destructive": status.destructive},
        repair_hint=status.maintenance_target,
    )


def component_from_raw_materialization_readiness(readiness: Mapping[str, Any] | None) -> ComponentReadiness:
    payload = readiness or {}
    available = bool(payload.get("available", False))
    total = int(payload.get("total") or 0)
    critical = int(payload.get("critical") or 0)
    warning = int(payload.get("warning") or 0)
    actionable = int(payload.get("actionable") or 0)
    blocked = int(payload.get("blocked") or 0)
    classified = int(payload.get("classified") or 0)
    affected_total = int(payload.get("affected_total") or 0)
    affected_actionable = int(payload.get("affected_actionable") or 0)
    affected_open = int(payload.get("affected_open") or 0)
    affected_classified = int(payload.get("affected_classified") or 0)
    unchecked = int(payload.get("unchecked") or 0)
    affected_unchecked = int(payload.get("affected_unchecked") or 0)
    lost_source_evidence_count = int(payload.get("lost_source_evidence_count") or 0)
    raw_artifact_count = int(payload.get("raw_artifact_count") or 0)
    materialized_raw_artifact_count = int(payload.get("materialized_raw_artifact_count") or 0)
    archive_session_count = int(payload.get("archive_session_count") or 0)
    join_gap_count = int(payload.get("join_gap_count") or total)
    if not available:
        state = CapabilityReadinessState.UNKNOWN
        summary = "unknown"
    elif lost_source_evidence_count > 0:
        state = CapabilityReadinessState.BLOCKED
        summary = "source evidence missing"
    elif total == 0:
        state = CapabilityReadinessState.READY
        summary = "ready"
    elif blocked > 0:
        state = CapabilityReadinessState.BLOCKED
        summary = "raw evidence blocked"
    elif critical > 0:
        state = CapabilityReadinessState.POISONED
        summary = "raw evidence not materialized"
    elif warning > 0 or actionable > 0 or affected_actionable > 0:
        state = CapabilityReadinessState.STALE
        summary = "raw evidence pending materialization"
    elif unchecked > 0 or affected_unchecked > 0:
        state = CapabilityReadinessState.DEGRADED
        summary = "raw/index join gaps need classification"
    elif classified > 0 or affected_classified > 0:
        state = CapabilityReadinessState.READY
        summary = "raw evidence classified; no materialization debt"
    else:
        state = CapabilityReadinessState.DEGRADED
        summary = "raw evidence classified as non-actionable"
    caveats: tuple[str, ...]
    if state is CapabilityReadinessState.READY and (classified > 0 or affected_classified > 0):
        caveats = ("raw_index_join_gaps_classified_not_materialization_debt",)
    elif state is CapabilityReadinessState.DEGRADED and (unchecked > 0 or affected_unchecked > 0):
        caveats = ("raw_index_join_gaps_unclassified_by_fast_readiness",)
    elif state is CapabilityReadinessState.DEGRADED:
        caveats = ("raw_index_join_gaps_classified_not_replayable",)
    elif state is CapabilityReadinessState.BLOCKED:
        caveats = ("lost_source_evidence",) if lost_source_evidence_count > 0 else ("raw_materialization_blocked",)
    elif state is CapabilityReadinessState.POISONED:
        caveats = ("raw_materialization_critical_debt",)
    else:
        caveats = ()
    return ComponentReadiness(
        component="raw_materialization",
        scope="archive",
        state=state,
        summary=summary,
        counts={
            "total": total,
            "critical": critical,
            "warning": warning,
            "actionable": actionable,
            "blocked": blocked,
            "classified": classified,
            "affected_total": affected_total,
            "affected_actionable": affected_actionable,
            "affected_open": affected_open,
            "affected_classified": affected_classified,
            "unchecked": unchecked,
            "affected_unchecked": affected_unchecked,
            "lost_source_evidence_count": lost_source_evidence_count,
            "raw_artifact_count": raw_artifact_count,
            "materialized_raw_artifact_count": materialized_raw_artifact_count,
            "archive_session_count": archive_session_count,
            "join_gap_count": join_gap_count,
        },
        caveats=caveats,
        metadata={
            "category_counts": dict(payload.get("category_counts") or {}),
            "source_family_counts": dict(payload.get("source_family_counts") or {}),
            "lost_source_evidence_samples": list(payload.get("lost_source_evidence_samples") or []),
        },
        repair_hint=(
            None
            if state == CapabilityReadinessState.READY
            else "restore exact raw artifact"
            if lost_source_evidence_count > 0
            else "polylogued run"
        ),
    )


def component_from_raw_frontier_integrity(payload: Mapping[str, Any] | None) -> ComponentReadiness:
    """Map the polylogue-yla8.7 raw-frontier-integrity projection into the shared DTO.

    ``payload`` is the ``RawFrontierIntegrity`` status model dumped to a dict
    (or the direct-fallback CLI path's equivalent plain dict) — both surfaces
    share this one mapping so component semantics cannot drift between the
    daemon-serving and no-daemon direct paths.
    """
    data = payload or {}
    overall = str(data.get("overall_status") or "unknown")
    if overall == "unknown":
        state = CapabilityReadinessState.UNKNOWN
        summary = "raw frontier authority unavailable"
    elif overall == "violated":
        state = CapabilityReadinessState.POISONED
        summary = "raw frontier integrity violated"
    else:
        state = CapabilityReadinessState.READY
        summary = "ready"
    caveats: list[str] = []
    for key in ("broken_head_status", "missing_source_raw_status", "cursor_ahead_status"):
        value = str(data.get(key) or "unknown")
        if value != "healthy":
            caveats.append(f"{key}:{value}")
    return ComponentReadiness(
        component="raw_frontier_integrity",
        scope="archive",
        state=state,
        summary=summary,
        counts={
            "broken_head_count": int(data.get("broken_head_count") or 0),
            "broken_head_checked_count": int(data.get("broken_head_checked_count") or 0),
            "missing_source_raw_count": int(data.get("missing_source_raw_count") or 0),
            "cursor_ahead_count": int(data.get("cursor_ahead_count") or 0),
            "cursor_ahead_checked_count": int(data.get("cursor_ahead_checked_count") or 0),
            "cursor_head_comparison_count": int(data.get("cursor_head_comparison_count") or 0),
            "cursor_ahead_comparison_count": int(data.get("cursor_ahead_comparison_count") or 0),
            "cursor_authority_gap_count": int(data.get("cursor_authority_gap_count") or 0),
        },
        caveats=tuple(caveats),
        repair_hint=None if state == CapabilityReadinessState.READY else "polylogue ops status --full",
    )


def raw_frontier_integrity_is_proven_healthy(payload: object) -> bool:
    """Return whether a complete frontier projection proves every check healthy."""

    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("available") is True
        and payload.get("overall_status") == "healthy"
        and payload.get("broken_head_status") == "healthy"
        and payload.get("missing_source_raw_status") == "healthy"
        and payload.get("cursor_ahead_status") == "healthy"
    )


def normalize_raw_frontier_status_payload(
    payload: Mapping[str, Any],
    *,
    snapshot_state: str | None = None,
) -> dict[str, Any]:
    """Fail closed at cached/presentation boundaries lacking fresh authority.

    Rich status is cached for request-time boundedness. A legacy, partial, or
    stale cache entry must not retain a top-level green bit after the canonical
    frontier projection becomes unavailable. Proven violations remain visible
    when stale; formerly healthy or unknown snapshots become explicit unknown.
    The raw-frontier component and converged claim are replaced from the same
    normalized mapping so adapters cannot drift.
    """

    from polylogue.storage.raw_retention import (
        raw_frontier_integrity_summary,
        unknown_raw_frontier_integrity_projection,
    )

    normalized: dict[str, Any] = dict(payload)
    if snapshot_state is None:
        snapshot = payload.get("status_snapshot")
        snapshot_state = str(snapshot.get("state") or "") if isinstance(snapshot, Mapping) else None

    raw_frontier = payload.get("raw_frontier_integrity")
    frontier = dict(raw_frontier) if isinstance(raw_frontier, Mapping) else None
    overall = str(frontier.get("overall_status") or "") if frontier is not None else ""
    stale_or_minimal = bool(snapshot_state and snapshot_state != "fresh")

    reason: str | None = None
    if frontier is None:
        reason = "status payload omitted a valid raw frontier integrity projection"
    elif overall not in {"healthy", "unknown", "violated"}:
        reason = "status payload contains a malformed raw frontier integrity projection"
    elif stale_or_minimal and overall != "violated":
        reason = f"status snapshot is {snapshot_state}; fresh raw frontier authority is unavailable"
    elif overall == "healthy" and not raw_frontier_integrity_is_proven_healthy(frontier):
        reason = "status payload contains an incomplete healthy raw frontier integrity projection"

    if reason is not None:
        frontier = unknown_raw_frontier_integrity_projection(reason).to_dict()
    elif frontier is not None and overall == "unknown":
        explicit_unknown = unknown_raw_frontier_integrity_projection(raw_frontier_integrity_summary(frontier)).to_dict()
        explicit_unknown.update(frontier)
        explicit_unknown["available"] = False
        explicit_unknown["overall_status"] = "unknown"
        frontier = explicit_unknown
    elif frontier is not None and overall == "violated":
        explicit_violation = unknown_raw_frontier_integrity_projection(
            raw_frontier_integrity_summary(frontier)
        ).to_dict()
        explicit_violation.update(frontier)
        explicit_violation["overall_status"] = "violated"
        if stale_or_minimal:
            explicit_violation["available"] = False
        frontier = explicit_violation

    assert frontier is not None
    normalized["raw_frontier_integrity"] = frontier
    proven_healthy = raw_frontier_integrity_is_proven_healthy(frontier) and not stale_or_minimal
    normalized["ok"] = bool(payload.get("ok", payload.get("daemon_liveness"))) and proven_healthy

    existing_components = payload.get("component_readiness")
    components = dict(existing_components) if isinstance(existing_components, Mapping) else {}
    components["raw_frontier_integrity"] = component_from_raw_frontier_integrity(frontier).to_dict()
    normalized["component_readiness"] = components

    existing_guard = payload.get("claim_guard")
    if isinstance(existing_guard, Mapping):
        guard = {
            str(key): dict(value) if isinstance(value, Mapping) else value for key, value in existing_guard.items()
        }
        converged = guard.get("converged")
        if not proven_healthy and isinstance(converged, dict) and bool(converged.get("value")):
            converged["value"] = False
            converged["reason"] = raw_frontier_integrity_summary(frontier)
            converged["signal"] = (
                "raw_frontier_integrity (fresh accepted raw chains and ingest cursors proven consistent)"
            )
        normalized["claim_guard"] = guard

    return normalized


def component_from_embedding_payload(payload: Mapping[str, Any]) -> ComponentReadiness:
    if not bool(payload.get("config_enabled")):
        state = CapabilityReadinessState.MISSING
    elif not bool(payload.get("has_voyage_api_key")):
        state = CapabilityReadinessState.BLOCKED
    elif int(payload.get("failure_count") or 0) > 0:
        state = CapabilityReadinessState.DEGRADED
    elif str(payload.get("freshness_status") or "") == "stale" or int(payload.get("stale_messages") or 0) > 0:
        state = CapabilityReadinessState.STALE
    elif bool(payload.get("retrieval_ready")):
        state = CapabilityReadinessState.READY
    else:
        state = CapabilityReadinessState.REBUILDING

    pending_messages_exact = bool(payload.get("pending_messages_exact"))
    pending_messages = int(payload.get("pending_messages") or 0) if pending_messages_exact else None

    return ComponentReadiness(
        component="embeddings",
        scope="semantic",
        state=state,
        summary=str(payload.get("status") or ""),
        counts={
            "total_sessions": int(payload.get("total_sessions") or 0),
            "embedded_sessions": int(payload.get("embedded_sessions") or 0),
            "embedded_messages": int(payload.get("embedded_messages") or 0),
            "pending_sessions": int(payload.get("pending_sessions") or 0),
            "pending_messages": pending_messages,
            "pending_messages_exact": pending_messages_exact,
            "stale_messages": int(payload.get("stale_messages") or 0),
            "failure_count": int(payload.get("failure_count") or 0),
            "retrieval_ready": bool(payload.get("retrieval_ready")),
        },
        repair_hint=_embedding_repair_hint(payload.get("next_action")),
    )


def component_from_assertion_substrate(
    *,
    table_exists: bool,
    assertion_count: int = 0,
    target_count: int = 0,
    active_count: int = 0,
    error: str | None = None,
    overlay_audit: Mapping[str, object] | None = None,
) -> ComponentReadiness:
    """Map the user-tier assertion substrate into the shared DTO."""

    caveats: tuple[str, ...]
    if error:
        state = CapabilityReadinessState.DEGRADED
        summary = error
        caveats = (error,)
    elif table_exists:
        state = CapabilityReadinessState.READY
        summary = "ready"
        caveats = ()
    else:
        state = CapabilityReadinessState.MISSING
        summary = "assertions table missing"
        caveats = ("assertions_table_missing",)

    return ComponentReadiness(
        component="assertions",
        scope="user",
        state=state,
        summary=summary,
        counts={
            "assertion_count": assertion_count,
            "target_count": target_count,
            "active_count": active_count,
        },
        caveats=caveats,
        repair_hint=None if table_exists and not error else "polylogue ops maintenance archive-init --yes",
        evidence_refs=("user.db:assertions",) if table_exists else (),
        metadata={"overlay_audit": dict(overlay_audit)} if overlay_audit else {},
    )


def component_from_transform_registry(
    *,
    transform_count: int,
    session_count: int | None,
    session_digest_transform_version: int | None = None,
    error: str | None = None,
) -> ComponentReadiness:
    """Map deterministic transform availability into the shared DTO."""

    if error:
        state = CapabilityReadinessState.BLOCKED
        summary = error
        repair_hint = None
    elif transform_count <= 0:
        state = CapabilityReadinessState.MISSING
        summary = "no transforms registered"
        repair_hint = None
    elif session_count is None or session_count <= 0:
        state = CapabilityReadinessState.MISSING
        summary = "no sessions"
        repair_hint = "polylogue import --demo"
    else:
        state = CapabilityReadinessState.READY
        summary = "ready"
        repair_hint = None

    counts: dict[str, int | float | bool] = {
        "transform_count": transform_count,
    }
    if session_count is not None:
        counts["session_count"] = session_count
    if session_digest_transform_version is not None:
        counts["session_digest_transform_version"] = session_digest_transform_version

    return ComponentReadiness(
        component="transforms",
        scope="session-analysis",
        state=state,
        summary=summary,
        counts=counts,
        caveats=(error,) if error else (),
        repair_hint=repair_hint,
        evidence_refs=("transform_registry",) if transform_count > 0 else (),
    )


def component_from_archive_surface(
    component: str,
    surface: Mapping[str, Any],
    *,
    scope: str = "archive",
    repair_hint: str | None = None,
) -> ComponentReadiness:
    """Map direct archive-readiness surfaces into the shared DTO."""
    ready = surface.get("ready")
    blockers = _string_tuple(surface.get("blockers", ()))
    evidence = surface.get("evidence", {})
    counts = _numeric_mapping(evidence) if isinstance(evidence, Mapping) else {}

    if ready is True:
        state = CapabilityReadinessState.READY
    elif ready is False and blockers:
        state = CapabilityReadinessState.STALE
    elif ready is False:
        state = CapabilityReadinessState.DEGRADED
    else:
        state = CapabilityReadinessState.UNKNOWN

    return ComponentReadiness(
        component=component,
        scope=scope,
        state=state,
        summary=", ".join(blockers) if blockers else ("ready" if ready is True else "unknown"),
        counts=counts,
        caveats=blockers,
        repair_hint=repair_hint if blockers else None,
    )


def component_from_insight_entry(entry: Any, *, scope: str = "insights") -> ComponentReadiness:
    verdict = str(getattr(entry, "verdict", "unknown"))
    state = {
        "ready": CapabilityReadinessState.READY,
        "partial": CapabilityReadinessState.DEGRADED,
        "empty": CapabilityReadinessState.MISSING,
        "missing": CapabilityReadinessState.MISSING,
        "stale": CapabilityReadinessState.STALE,
        "incompatible": CapabilityReadinessState.POISONED,
        "degraded": CapabilityReadinessState.DEGRADED,
        "unknown": CapabilityReadinessState.UNKNOWN,
    }.get(verdict, CapabilityReadinessState.UNKNOWN)
    return ComponentReadiness(
        component=str(getattr(entry, "insight_name", "unknown")),
        scope=scope,
        state=state,
        summary=str(getattr(entry, "display_name", "")),
        counts={
            "row_count": int(getattr(entry, "row_count", 0) or 0),
            "missing_count": int(getattr(entry, "missing_count", 0) or 0),
            "stale_count": int(getattr(entry, "stale_count", 0) or 0),
            "orphan_count": int(getattr(entry, "orphan_count", 0) or 0),
            "incompatible_count": int(getattr(entry, "incompatible_count", 0) or 0),
            "degraded_count": int(getattr(entry, "degraded_count", 0) or 0),
        },
        repair_hint=str(getattr(entry, "repair_command", "") or "") or None,
        evidence_refs=_string_tuple(getattr(entry, "evidence", ())),
    )


def component_from_operation_status(
    status: OperationStatus | str,
    *,
    component: str,
    scope: str = "operation",
    summary: str = "",
) -> ComponentReadiness:
    operation_status = status if isinstance(status, OperationStatus) else OperationStatus(str(status))
    state = {
        OperationStatus.ACCEPTED: CapabilityReadinessState.REBUILDING,
        OperationStatus.PENDING: CapabilityReadinessState.REBUILDING,
        OperationStatus.RUNNING: CapabilityReadinessState.REBUILDING,
        OperationStatus.COMPLETED: CapabilityReadinessState.READY,
        OperationStatus.REJECTED: CapabilityReadinessState.BLOCKED,
        OperationStatus.FAILED: CapabilityReadinessState.BLOCKED,
    }[operation_status]
    return ComponentReadiness(component=component, scope=scope, state=state, summary=summary)


def component_from_catchup_status(status: Any, *, component: str = "daemon_catchup") -> ComponentReadiness:
    mode = str(getattr(status, "mode", "unknown"))
    failed_count = int(getattr(status, "failed_file_count", 0) or 0)
    if failed_count > 0:
        state = CapabilityReadinessState.DEGRADED
    elif mode in {"running", "catching_up", "active", "importing"}:
        state = CapabilityReadinessState.REBUILDING
    elif mode == "idle":
        state = CapabilityReadinessState.READY
    else:
        state = CapabilityReadinessState.UNKNOWN
    return ComponentReadiness(
        component=component,
        scope="daemon",
        state=state,
        summary=mode,
        counts={
            "queued_file_count": int(getattr(status, "queued_file_count", 0) or 0),
            "needed_file_count": int(getattr(status, "needed_file_count", 0) or 0),
            "succeeded_file_count": int(getattr(status, "succeeded_file_count", 0) or 0),
            "failed_file_count": failed_count,
        },
        last_attempt=str(getattr(status, "current_phase", "") or "") or None,
    )


def _embedding_repair_hint(next_action: object) -> str | None:
    if not isinstance(next_action, Mapping):
        return None
    if str(next_action.get("code") or "") == "ready":
        return None
    command = next_action.get("command")
    return str(command) if command else None


def _string_tuple(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    if not isinstance(values, Sequence):
        return ()
    return tuple(str(value) for value in values)


def _numeric_mapping(values: Mapping[str, Any]) -> dict[str, int | float | bool]:
    result: dict[str, int | float | bool] = {}
    for key, value in values.items():
        if isinstance(value, bool | int | float):
            result[str(key)] = value
    return result


__all__ = [
    "CapabilityReadinessState",
    "ComponentReadiness",
    "LEGACY_READINESS_SOURCE_TYPES",
    "component_from_archive_debt",
    "component_from_archive_surface",
    "component_from_assertion_substrate",
    "component_from_catchup_status",
    "component_from_derived_model",
    "component_from_embedding_payload",
    "component_from_insight_entry",
    "component_from_operation_status",
    "component_from_outcome_check",
    "component_from_raw_frontier_integrity",
    "component_from_raw_materialization_readiness",
    "component_from_transform_registry",
    "normalize_raw_frontier_status_payload",
    "raw_frontier_integrity_is_proven_healthy",
]
