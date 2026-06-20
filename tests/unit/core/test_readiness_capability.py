from __future__ import annotations

from types import SimpleNamespace

from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.maintenance.models import DerivedModelStatus, MaintenanceCategory
from polylogue.operations.operation_contract import OperationStatus
from polylogue.readiness import (
    LEGACY_READINESS_SOURCE_TYPES,
    CapabilityReadinessState,
    ComponentReadiness,
    component_from_archive_debt,
    component_from_archive_surface,
    component_from_assertion_substrate,
    component_from_catchup_status,
    component_from_derived_model,
    component_from_embedding_payload,
    component_from_insight_entry,
    component_from_operation_status,
    component_from_outcome_check,
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


def test_embedding_payload_maps_missing_blocked_stale_and_ready() -> None:
    base = {
        "config_enabled": True,
        "has_voyage_api_key": True,
        "status": "ready",
        "total_sessions": 2,
        "embedded_sessions": 2,
        "embedded_messages": 5,
        "pending_sessions": 0,
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
    assert component_from_embedding_payload(base).state is CapabilityReadinessState.READY


def test_archive_surface_maps_search_mismatch_to_stale_component() -> None:
    component = component_from_archive_surface(
        "search",
        {
            "ready": False,
            "blockers": ["messages_fts_row_mismatch"],
            "evidence": {"text_block_count": 10, "messages_fts_count": 8},
        },
        scope="lexical",
        repair_hint="polylogue ops maintenance run --target dangling_fts",
    )

    assert component.state is CapabilityReadinessState.STALE
    assert component.scope == "lexical"
    assert component.counts == {"text_block_count": 10, "messages_fts_count": 8}
    assert component.caveats == ("messages_fts_row_mismatch",)
    assert component.repair_hint == "polylogue ops maintenance run --target dangling_fts"


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
        recovery_transform_version=1,
    )
    no_sessions = component_from_transform_registry(transform_count=1, session_count=0)
    no_registry = component_from_transform_registry(transform_count=0, session_count=2)
    blocked = component_from_transform_registry(
        transform_count=1,
        session_count=None,
        error="database is locked",
    )

    assert ready.state is CapabilityReadinessState.READY
    assert ready.scope == "recovery"
    assert ready.counts == {
        "transform_count": 1,
        "session_count": 2,
        "recovery_transform_version": 1,
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
