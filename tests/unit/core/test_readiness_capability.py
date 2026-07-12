from __future__ import annotations

from types import SimpleNamespace

from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.maintenance.models import DerivedModelStatus, MaintenanceCategory
from polylogue.operations.operation_contract import OperationStatus
from polylogue.readiness import (
    LEGACY_READINESS_SOURCE_TYPES,
    CapabilityReadinessState,
    ComponentReadiness,
    ReadinessReport,
    component_from_archive_debt,
    component_from_archive_surface,
    component_from_assertion_substrate,
    component_from_catchup_status,
    component_from_derived_model,
    component_from_embedding_payload,
    component_from_insight_entry,
    component_from_operation_status,
    component_from_outcome_check,
    component_from_raw_frontier_integrity,
    component_from_raw_materialization_readiness,
    component_from_transform_registry,
)
from polylogue.storage.repair import ArchiveDebtStatus


def test_capability_readiness_state_matches_issue_vocabulary() -> None:
    assert {state.value for state in CapabilityReadinessState} == {
        "ready",
        "rebuilding",
        "stale",
        "missing",
        "blocked",
        "degraded",
        "poisoned",
        "unknown",
    }


def test_component_readiness_serializes_machine_shape() -> None:
    payload = ComponentReadiness(
        component="embeddings",
        scope="semantic",
        state=CapabilityReadinessState.STALE,
        summary="needs catch-up",
        counts={"pending_messages": 3},
        caveats=("daemon stage disabled",),
        repair_hint="polylogue ops embed backfill",
        evidence_refs=("embedding_status:latest",),
    ).to_dict()

    assert payload == {
        "component": "embeddings",
        "scope": "semantic",
        "state": "stale",
        "summary": "needs catch-up",
        "last_success": None,
        "last_attempt": None,
        "counts": {"pending_messages": 3},
        "caveats": ["daemon stage disabled"],
        "repair_hint": "polylogue ops embed backfill",
        "evidence_refs": ["embedding_status:latest"],
    }


def test_known_legacy_readiness_sources_are_tracked() -> None:
    assert set(LEGACY_READINESS_SOURCE_TYPES) == {
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
    }


def test_archive_convergence_hoists_materialization_progress_counts() -> None:
    report = ReadinessReport(
        raw_materialization_readiness={
            "available": True,
            "classification": "not_run",
            "raw_artifact_count": 10,
            "materialized_raw_artifact_count": 7,
            "archive_session_count": 8,
            "join_gap_count": 3,
            "total": 3,
            "unchecked": 3,
            "affected_unchecked": 3,
        }
    )

    convergence = report.archive_convergence

    assert convergence["checked"] is True
    assert convergence["converging"] is True
    assert convergence["materialization_ready"] is False
    assert convergence["materialization_progress"] == {
        "raw_artifact_count": 10,
        "materialized_raw_artifact_count": 7,
        "archive_session_count": 8,
        "join_gap_count": 3,
    }


def test_runtime_only_readiness_does_not_claim_archive_convergence() -> None:
    report = ReadinessReport()

    convergence = report.archive_convergence

    assert convergence["checked"] is False
    assert convergence["converging"] is False
    assert convergence["materialization_ready"] is False
    assert convergence["raw_frontier_integrity"] == {}


def test_outcome_checks_map_to_capability_states() -> None:
    warning = component_from_outcome_check(
        OutcomeCheck("database", OutcomeStatus.WARNING, summary="busy", count=2, details=["retry later"])
    )
    error = component_from_outcome_check(OutcomeCheck("schema", OutcomeStatus.ERROR, summary="mismatch"))

    assert warning.state is CapabilityReadinessState.DEGRADED
    assert warning.counts["count"] == 2
    assert warning.caveats == ("retry later",)
    assert error.state is CapabilityReadinessState.BLOCKED


def test_derived_model_status_maps_stale_and_missing() -> None:
    stale = component_from_derived_model(
        DerivedModelStatus(
            name="messages_fts",
            ready=False,
            detail="stale rows",
            source_rows=10,
            materialized_rows=8,
            stale_rows=2,
        )
    )
    missing = component_from_derived_model(DerivedModelStatus(name="embeddings", ready=False, detail="no source"))

    assert stale.state is CapabilityReadinessState.STALE
    assert stale.counts["stale_rows"] == 2
    assert missing.state is CapabilityReadinessState.MISSING


def test_archive_debt_status_maps_debt_to_degraded() -> None:
    status = ArchiveDebtStatus(
        name="orphaned_messages",
        category=MaintenanceCategory.ARCHIVE_CLEANUP,
        destructive=False,
        issue_count=4,
        detail="orphans found",
        maintenance_target="orphaned_messages",
    )

    component = component_from_archive_debt(status)

    assert component.state is CapabilityReadinessState.DEGRADED
    assert component.counts == {"issue_count": 4, "destructive": False}
    assert component.repair_hint == "orphaned_messages"


def test_raw_materialization_readiness_maps_actionable_debt_to_stale() -> None:
    component = component_from_raw_materialization_readiness(
        {
            "available": True,
            "total": 1,
            "warning": 1,
            "actionable": 1,
            "affected_total": 4,
            "affected_actionable": 4,
            "category_counts": {"parsed-without-session": 4},
        }
    )

    assert component.state is CapabilityReadinessState.STALE
    assert component.summary == "raw evidence pending materialization"
    assert component.counts["affected_actionable"] == 4
    assert component.caveats == ()
    assert component.repair_hint == "polylogued run"


def test_raw_materialization_readiness_maps_classified_info_debt_to_ready_with_caveat() -> None:
    component = component_from_raw_materialization_readiness(
        {
            "available": True,
            "total": 2,
            "critical": 0,
            "warning": 0,
            "actionable": 0,
            "blocked": 0,
            "classified": 2,
            "affected_total": 276,
            "affected_actionable": 0,
            "affected_open": 0,
            "affected_classified": 276,
            "category_counts": {"materialized-alias": 47, "parsed-non-session-artifact": 229},
            "source_family_counts": {"claude-code-session": 272, "codex-session": 4},
        }
    )

    assert component.state is CapabilityReadinessState.READY
    assert component.summary == "raw evidence classified; no materialization debt"
    assert component.counts["affected_classified"] == 276
    assert component.caveats == ("raw_index_join_gaps_classified_not_materialization_debt",)
    assert component.metadata["category_counts"] == {"materialized-alias": 47, "parsed-non-session-artifact": 229}
    assert component.repair_hint is None


def test_raw_materialization_readiness_maps_unchecked_join_gaps_to_degraded() -> None:
    component = component_from_raw_materialization_readiness(
        {
            "available": True,
            "classification": "not_run",
            "total": 3,
            "raw_artifact_count": 10,
            "materialized_raw_artifact_count": 7,
            "archive_session_count": 8,
            "join_gap_count": 3,
            "unchecked": 3,
            "affected_total": 3,
            "affected_unchecked": 3,
            "category_counts": {"raw_id_join_gap": 3},
        }
    )

    assert component.state is CapabilityReadinessState.DEGRADED
    assert component.summary == "raw/index join gaps need classification"
    assert component.counts["affected_unchecked"] == 3
    assert component.counts["raw_artifact_count"] == 10
    assert component.counts["materialized_raw_artifact_count"] == 7
    assert component.counts["archive_session_count"] == 8
    assert component.counts["join_gap_count"] == 3
    assert component.caveats == ("raw_index_join_gaps_unclassified_by_fast_readiness",)
    assert component.repair_hint == "polylogued run"


def test_raw_materialization_readiness_maps_lost_source_evidence_to_blocked() -> None:
    sample = {
        "session_id": "codex-session:native-1",
        "origin": "codex-session",
        "native_id": "native-1",
        "missing_raw_id": "raw-missing",
        "evidence_status": "lost_source_evidence",
    }

    component = component_from_raw_materialization_readiness(
        {
            "available": True,
            "total": 0,
            "lost_source_evidence_count": 1,
            "lost_source_evidence_samples": [sample],
        }
    )

    assert component.state is CapabilityReadinessState.BLOCKED
    assert component.summary == "source evidence missing"
    assert component.counts["lost_source_evidence_count"] == 1
    assert component.caveats == ("lost_source_evidence",)
    assert component.repair_hint == "restore exact raw artifact"
    assert component.metadata["lost_source_evidence_samples"] == [sample]


def test_raw_frontier_integrity_maps_healthy_to_ready() -> None:
    component = component_from_raw_frontier_integrity(
        {
            "overall_status": "healthy",
            "broken_head_status": "healthy",
            "broken_head_count": 0,
            "broken_head_checked_count": 3,
            "missing_source_raw_status": "healthy",
            "missing_source_raw_count": 0,
            "cursor_ahead_status": "healthy",
            "cursor_ahead_count": 0,
            "cursor_ahead_checked_count": 3,
            "cursor_head_comparison_count": 4,
            "cursor_ahead_comparison_count": 0,
            "cursor_authority_gap_count": 0,
        }
    )

    assert component.component == "raw_frontier_integrity"
    assert component.state is CapabilityReadinessState.READY
    assert component.summary == "ready"
    assert component.caveats == ()
    assert component.repair_hint is None
    assert component.counts["broken_head_checked_count"] == 3
    assert component.counts["cursor_head_comparison_count"] == 4
    assert component.counts["cursor_ahead_comparison_count"] == 0
    assert component.counts["cursor_authority_gap_count"] == 0


def test_raw_frontier_integrity_maps_violated_broken_head_to_poisoned_with_caveat() -> None:
    component = component_from_raw_frontier_integrity(
        {
            "overall_status": "violated",
            "broken_head_status": "violated",
            "broken_head_count": 1,
            "broken_head_checked_count": 2,
            "missing_source_raw_status": "healthy",
            "missing_source_raw_count": 0,
            "cursor_ahead_status": "healthy",
            "cursor_ahead_count": 0,
        }
    )

    assert component.state is CapabilityReadinessState.POISONED
    assert component.summary == "raw frontier integrity violated"
    assert component.caveats == ("broken_head_status:violated",)
    assert component.counts["broken_head_count"] == 1
    assert component.repair_hint == "polylogue ops status --full"


def test_raw_frontier_integrity_maps_unavailable_authority_to_unknown_never_ready() -> None:
    """Bead AC: unavailable authority must never render as ready/healthy."""
    component = component_from_raw_frontier_integrity(
        {
            "overall_status": "unknown",
            "broken_head_status": "unknown",
            "broken_head_reason": "index tier is unavailable",
            "cursor_ahead_status": "unknown",
            "missing_source_raw_status": "healthy",
        }
    )

    assert component.state is CapabilityReadinessState.UNKNOWN
    assert component.summary == "raw frontier authority unavailable"
    assert "broken_head_status:unknown" in component.caveats
    assert "cursor_ahead_status:unknown" in component.caveats
    assert "missing_source_raw_status" not in "".join(component.caveats)


def test_raw_frontier_integrity_maps_known_violation_with_unknown_sibling_to_poisoned() -> None:
    component = component_from_raw_frontier_integrity(
        {
            "available": False,
            "overall_status": "violated",
            "broken_head_status": "violated",
            "broken_head_count": 1,
            "missing_source_raw_status": "healthy",
            "cursor_ahead_status": "unknown",
            "cursor_authority_gap_count": 1,
        }
    )

    assert component.state is CapabilityReadinessState.POISONED
    assert component.counts["cursor_authority_gap_count"] == 1
    assert component.caveats == ("broken_head_status:violated", "cursor_ahead_status:unknown")


def test_raw_frontier_integrity_maps_none_payload_to_unknown() -> None:
    component = component_from_raw_frontier_integrity(None)

    assert component.state is CapabilityReadinessState.UNKNOWN
    assert component.counts["broken_head_count"] == 0


def test_embedding_payload_maps_missing_blocked_stale_and_ready() -> None:
    base = {
        "config_enabled": True,
        "has_voyage_api_key": True,
        "status": "ready",
        "total_sessions": 2,
        "embedded_sessions": 2,
        "embedded_messages": 5,
        "pending_sessions": 0,
        "pending_messages": None,
        "pending_messages_exact": False,
        "stale_messages": 0,
        "failure_count": 0,
        "retrieval_ready": True,
        "freshness_status": "fresh",
        "next_action": {"command": None},
    }

    assert component_from_embedding_payload({**base, "config_enabled": False}).state is CapabilityReadinessState.MISSING
    assert (
        component_from_embedding_payload({**base, "has_voyage_api_key": False}).state
        is CapabilityReadinessState.BLOCKED
    )
    assert component_from_embedding_payload({**base, "stale_messages": 1}).state is CapabilityReadinessState.STALE
    ready = component_from_embedding_payload(base)
    assert ready.state is CapabilityReadinessState.READY
    assert ready.counts["pending_messages"] is None
    assert ready.counts["pending_messages_exact"] is False
    exact = component_from_embedding_payload({**base, "pending_messages": 7, "pending_messages_exact": True})
    assert exact.counts["pending_messages"] == 7
    assert exact.counts["pending_messages_exact"] is True


def test_archive_surface_maps_search_mismatch_to_stale_component() -> None:
    component = component_from_archive_surface(
        "search",
        {
            "ready": False,
            "blockers": ["messages_fts_row_mismatch"],
            "evidence": {"text_block_count": 10, "messages_fts_count": 8},
        },
        scope="lexical",
        repair_hint="polylogued run",
    )

    assert component.state is CapabilityReadinessState.STALE
    assert component.scope == "lexical"
    assert component.counts == {"text_block_count": 10, "messages_fts_count": 8}
    assert component.caveats == ("messages_fts_row_mismatch",)
    assert component.repair_hint == "polylogued run"


def test_assertion_substrate_maps_missing_ready_and_error_states() -> None:
    missing = component_from_assertion_substrate(table_exists=False)
    ready = component_from_assertion_substrate(
        table_exists=True,
        assertion_count=4,
        target_count=2,
        active_count=3,
        overlay_audit={"surfaces": []},
    )
    busy = component_from_assertion_substrate(table_exists=True, error="database is locked")

    assert missing.state is CapabilityReadinessState.MISSING
    assert missing.caveats == ("assertions_table_missing",)
    assert ready.state is CapabilityReadinessState.READY
    assert ready.scope == "user"
    assert ready.counts == {"assertion_count": 4, "target_count": 2, "active_count": 3}
    assert ready.evidence_refs == ("user.db:assertions",)
    assert ready.metadata == {"overlay_audit": {"surfaces": []}}
    assert ready.to_dict()["metadata"] == {"overlay_audit": {"surfaces": []}}
    assert busy.state is CapabilityReadinessState.DEGRADED
    assert busy.caveats == ("database is locked",)


def test_transform_registry_maps_registry_and_session_availability() -> None:
    ready = component_from_transform_registry(
        transform_count=1,
        session_count=2,
        session_digest_transform_version=1,
    )
    no_sessions = component_from_transform_registry(transform_count=1, session_count=0)
    no_registry = component_from_transform_registry(transform_count=0, session_count=2)
    blocked = component_from_transform_registry(
        transform_count=1,
        session_count=None,
        error="database is locked",
    )

    assert ready.state is CapabilityReadinessState.READY
    assert ready.scope == "session-analysis"
    assert ready.counts == {
        "transform_count": 1,
        "session_count": 2,
        "session_digest_transform_version": 1,
    }
    assert ready.evidence_refs == ("transform_registry",)
    assert no_sessions.state is CapabilityReadinessState.MISSING
    assert no_sessions.repair_hint == "polylogue import --demo"
    assert no_registry.state is CapabilityReadinessState.MISSING
    assert blocked.state is CapabilityReadinessState.BLOCKED
    assert blocked.summary == "database is locked"
    assert blocked.caveats == ("database is locked",)
    assert blocked.repair_hint is None
    assert "session_count" not in blocked.counts


def test_insight_entry_operation_and_catchup_adapters() -> None:
    insight = component_from_insight_entry(
        SimpleNamespace(
            insight_name="session_profiles",
            display_name="Session Profiles",
            verdict="incompatible",
            row_count=3,
            repair_command="polylogue ops maintenance repair session-insights",
            evidence=("session_insight_status",),
        )
    )
    operation = component_from_operation_status(OperationStatus.RUNNING, component="demo_import")
    catchup = component_from_catchup_status(SimpleNamespace(mode="idle", failed_file_count=0, succeeded_file_count=7))

    assert insight.state is CapabilityReadinessState.POISONED
    assert insight.evidence_refs == ("session_insight_status",)
    assert operation.state is CapabilityReadinessState.REBUILDING
    assert catchup.state is CapabilityReadinessState.READY
    assert catchup.counts["succeeded_file_count"] == 7
