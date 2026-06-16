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
    counts: Mapping[str, int | float | bool] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()
    repair_hint: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
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
        )


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
        repair_hint=None if table_exists and not error else "polylogue maintenance archive-init --yes",
        evidence_refs=("user.db:assertions",) if table_exists else (),
    )


def component_from_transform_registry(
    *,
    transform_count: int,
    session_count: int,
    recovery_transform_version: int | None = None,
) -> ComponentReadiness:
    """Map deterministic transform availability into the shared DTO."""

    if transform_count <= 0:
        state = CapabilityReadinessState.MISSING
        summary = "no transforms registered"
        repair_hint = None
    elif session_count <= 0:
        state = CapabilityReadinessState.MISSING
        summary = "no sessions"
        repair_hint = "polylogue import --demo"
    else:
        state = CapabilityReadinessState.READY
        summary = "ready"
        repair_hint = None

    counts: dict[str, int | float | bool] = {
        "transform_count": transform_count,
        "session_count": session_count,
    }
    if recovery_transform_version is not None:
        counts["recovery_transform_version"] = recovery_transform_version

    return ComponentReadiness(
        component="transforms",
        scope="recovery",
        state=state,
        summary=summary,
        counts=counts,
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
    "component_from_transform_registry",
]
