"""Pinned archive-status producer for daemon read operations.

This is deliberately a product boundary: it derives the direct-status
payload from the ``ArchiveStore`` reader selected by the operation runtime.
It never resolves an archive root or opens another SQLite connection.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

if TYPE_CHECKING:
    from polylogue.config import Config, PolylogueConfig
    from polylogue.storage.embeddings.status_payload import EmbeddingStatusSettings
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


_TIER_STATUS_TABLES: dict[str, tuple[str, ...]] = {
    "source": ("raw_sessions", "raw_artifacts", "raw_membership_census", "raw_session_memberships"),
    "index": ("sessions", "messages", "blocks", "messages_fts", "session_profiles", "work_events"),
    "embeddings": ("embedding_status", "message_embeddings_meta", "embedding_failures"),
    "user": ("assertions", "settings", "annotation_schemas"),
    "audit": ("mutation_previews", "mutation_authorizations", "mutation_attempts"),
    "ops": ("ingest_cursor", "ingest_attempts", "convergence_debt", "schema_drift_samples", "embedding_catchup_runs"),
}


def produce_direct_status(
    *,
    archive: ArchiveStore,
    now_ms: int,
    config: Config | PolylogueConfig | None = None,
    include_archive_readiness: bool = False,
) -> dict[str, object]:
    """Return the legacy direct-status shape from one supplied snapshot."""

    from polylogue.readiness.capability import (
        component_from_archive_surface,
        component_from_raw_frontier_integrity,
        component_from_raw_materialization_readiness,
        normalize_raw_frontier_status_payload,
    )
    from polylogue.readiness.claim_guard import derive_claim_guard
    from polylogue.storage.archive_readiness import archive_readiness_status_from_connections

    index_conn = archive.index_connection
    source_conn = _source_connection(archive)
    ops_conn = _attached_connection(index_conn, "ops_tier")
    archive_stats = archive.stats().to_dict()
    from polylogue.storage.embeddings.status_payload import (
        embedding_status_payload_from_connections,
        embedding_status_settings_from_config,
    )

    settings = embedding_status_settings_from_config(config)

    embedding_status = embedding_status_payload_from_connections(
        index_conn,
        config=config,
        include_detail=False,
    ) or _unavailable_embedding_status(settings)
    archive_stats.update(
        {
            "embedded_sessions": embedding_status["embedded_sessions"],
            "embedded_messages": embedding_status["embedded_messages"],
            "pending_embedding_sessions": embedding_status["pending_sessions"],
            "stale_embedding_messages": embedding_status["stale_messages"],
            "messages_missing_embedding_provenance": embedding_status["messages_missing_provenance"],
            "embedding_models": embedding_status["embedding_models"],
            "embedding_dimensions": embedding_status["embedding_dimensions"],
            "retrieval_ready": embedding_status["retrieval_ready"],
            "embedding_readiness_status": embedding_status["freshness_status"],
        }
    )
    tiers = _archive_tiers(archive, index_conn)
    materialization = _raw_materialization_status(index_conn, archive_root=archive.archive_root)
    archive_readiness = (
        archive_readiness_status_from_connections(
            index_conn,
            source_conn,
            raw_materialization_readiness=materialization,
        )
        if include_archive_readiness
        else {
            "checked": False,
            "reason": "direct_status_default_skips_exact_archive_readiness",
            "surfaces": {},
        }
    )
    frontier = _frontier_status(source_conn, index_conn, ops_conn, materialization)
    raw_failures = _raw_failure_status(source_conn)
    workload = _ops_workload_status(ops_conn, now_ms=now_ms)
    convergence = _convergence_status(ops_conn, now_ms=now_ms)
    schema_drift = _schema_drift_status(ops_conn, now_ms=now_ms)
    replay_backlog = (
        _raw_replay_backlog_status(index_conn, archive_root=archive.archive_root)
        if source_conn is not None
        else {"available": False, "reason": "source_tier_missing"}
    )
    components = _components(
        index_conn=index_conn,
        archive_readiness=archive_readiness,
        materialization=materialization,
        frontier=frontier,
        embedding_status=embedding_status,
        component_from_archive_surface=component_from_archive_surface,
        component_from_raw_materialization_readiness=component_from_raw_materialization_readiness,
        component_from_raw_frontier_integrity=component_from_raw_frontier_integrity,
    )
    missing_tiers = [name for name, info in tiers.items() if not info["exists"]]
    mismatched_tiers = [name for name, info in tiers.items() if info["exists"] and info["version_status"] != "ok"]
    raw_component = components.get("raw_materialization", {})
    frontier_component = components.get("raw_frontier_integrity", {})
    search_component = components.get("search", {})
    claim_guard = derive_claim_guard(
        archive_schema_ready=not missing_tiers and not mismatched_tiers,
        schema_mismatches=mismatched_tiers,
        missing_tiers=missing_tiers,
        raw_materialization_ready=_raw_ready(materialization),
        raw_materialization_summary=str(raw_component.get("summary", "unknown")),
        raw_frontier_integrity_ready=frontier_component.get("state") == "ready",
        raw_frontier_integrity_summary=str(frontier_component.get("summary", "unknown")),
        search_ready=search_component.get("state") == "ready",
        search_summary=str(search_component.get("summary", "unknown")),
        active_writer=not bool(workload.get("available"))
        or bool(workload.get("actively_ingesting"))
        or bool(workload.get("running_count")),
        active_writer_summary=(
            "ingest workload inspection unavailable; cannot rule out a concurrent archive writer"
            if not workload.get("available")
            else f"{workload.get('running_count', 0)} live ingest attempt(s) running"
            if workload.get("actively_ingesting")
            else ""
        ),
        convergence_debt_available=bool(convergence.get("available")),
        convergence_debt_pending=bool(convergence.get("failed_count") or convergence.get("deferred_count")),
        convergence_debt_summary=str(convergence.get("error") or "no pending convergence debt"),
    ).to_dict()
    payload: dict[str, object] = {
        "ok": _status_ok(components, raw_failures),
        "daemon_liveness": False,
        "archive_root": str(archive.archive_root),
        "active_archive_root": str(archive.archive_root),
        "active_archive_root_matches_configured": True,
        "db_exists": "index" not in archive.operation_degraded_components,
        "active_db_path": str(archive.index_db_path),
        "config_exists": config is not None,
        "config_path": None,
        "archive_tiers": tiers,
        "sinex_publication": _not_observed("sinex runtime mode not supplied"),
        "sqlite_maintenance": _sqlite_maintenance(index_conn),
        "ingest_workload": workload,
        "convergence": convergence,
        "schema_drift": schema_drift,
        "raw_replay_backlog": replay_backlog,
        "archive_readiness": archive_readiness,
        "assertion_candidate_queue": _not_observed(
            "judgment scheduler runtime/config not supplied", pending_count=None
        ),
        "raw_materialization_readiness": materialization,
        "raw_frontier_integrity": frontier,
        "embedding_status": embedding_status,
        "component_readiness": components,
        "claim_guard": claim_guard,
        "next_action": "runtime diagnostics not observed from pinned archive",
        "diagnostic": {"kind": "not_observed", "reason": "CLI first-run diagnostics excluded from pinned status"},
        "archive_stats": archive_stats,
    }
    payload.update(archive_stats)
    payload.update(raw_failures)
    return normalize_raw_frontier_status_payload(payload, snapshot_state="pinned")


def produce_operation_status(
    *,
    archive: ArchiveStore,
    now_ms: int,
    config: Config | PolylogueConfig | None = None,
    runtime_status: Mapping[str, object] | None = None,
    include_archive_readiness: bool = False,
) -> dict[str, object]:
    """Combine pinned archive evidence with separately observed runtime facts.

    Legacy daemon-only projections remain visible for existing consumers, but
    their original capture metadata and field ownership travel with them.
    They never replace evidence read from the executing archive snapshot.
    """

    pinned = produce_direct_status(
        archive=archive,
        now_ms=now_ms,
        config=config,
        include_archive_readiness=include_archive_readiness,
    )
    if runtime_status is None:
        return pinned
    runtime_only = {
        "daemon_liveness",
        "sinex_publication",
        "assertion_candidate_queue",
        "next_action",
        "diagnostic",
    }
    pinned_fields = set(pinned) - runtime_only
    result = dict(runtime_status)
    result.update({key: value for key, value in pinned.items() if key not in runtime_only})
    for key in runtime_only:
        result[key] = runtime_status.get(key, pinned[key])
    result["ok"] = bool(pinned["ok"]) and bool(runtime_status.get("daemon_liveness"))
    result["status_observations"] = {
        "archive": {
            "state": "pinned",
            "observed_at_ms": now_ms,
            "fields": sorted(pinned_fields),
            "schema_versions": dict(archive.operation_schema_versions),
        },
        "runtime": {
            "state": "separately_observed",
            "checked_at": runtime_status.get("checked_at"),
            "status_snapshot": runtime_status.get("status_snapshot"),
            "fields": sorted(set(runtime_status) - pinned_fields),
        },
    }
    return result


def _source_connection(archive: ArchiveStore) -> sqlite3.Connection | None:
    if "source" in archive.operation_degraded_components:
        return None
    return archive.source_connection


def _attached_connection(conn: sqlite3.Connection, schema: str) -> sqlite3.Connection | None:
    aliases = {str(row[1]) for row in conn.execute("PRAGMA database_list").fetchall()}
    return conn if schema in aliases else None


def _table_exists(conn: sqlite3.Connection | None, table: str, *, schema: str = "main") -> bool:
    if conn is None:
        return False
    return (
        conn.execute(
            f"SELECT 1 FROM {schema}.sqlite_schema WHERE type IN ('table', 'view') AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _count(conn: sqlite3.Connection | None, sql: str, params: tuple[object, ...] = ()) -> int:
    if conn is None:
        return 0
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _archive_tiers(archive: ArchiveStore, conn: sqlite3.Connection) -> dict[str, dict[str, object]]:
    aliases = {str(row[1]) for row in conn.execute("PRAGMA database_list").fetchall()}
    result: dict[str, dict[str, object]] = {}
    for tier in ArchiveTier:
        alias = "main" if tier is ArchiveTier.INDEX else f"{tier.value}_tier"
        exists = alias in aliases
        version = archive.operation_schema_versions.get(tier.value) if exists else None
        expected = ARCHIVE_VERSION_BY_TIER[tier]
        table_counts = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {alias}.{table}").fetchone()[0] or 0)
            for table in _TIER_STATUS_TABLES[tier.value]
            if exists and _table_exists(conn, table, schema=alias)
        }
        result[tier.value] = {
            "path": str(archive.archive_root / f"{tier.value}.db"),
            "exists": exists,
            "size_bytes": None,
            "expected_user_version": expected,
            "user_version": version,
            "version_status": "ok" if version == expected else "missing" if not exists else "mismatch",
            "table_counts": table_counts,
            "table_count_precision": dict.fromkeys(table_counts, "exact"),
            "file_metadata": {
                "state": "not_observed",
                "reason": "operation snapshot pins SQLite evidence, not filesystem size/WAL metadata",
            },
        }
    return result


def _raw_materialization_status(index_conn: sqlite3.Connection, *, archive_root: Path) -> dict[str, object]:
    from polylogue.storage.archive_readiness import raw_materialization_readiness_from_pinned_index

    return raw_materialization_readiness_from_pinned_index(index_conn, archive_root=archive_root)


def _frontier_status(
    source_conn: sqlite3.Connection | None,
    index_conn: sqlite3.Connection,
    ops_conn: sqlite3.Connection | None,
    materialization: Mapping[str, object],
) -> dict[str, object]:
    if source_conn is None or ops_conn is None:
        return {"available": False, "overall_status": "unknown", "reason": "source or ops tier unavailable"}
    from polylogue.storage.raw_retention import (
        RawFrontierIntegrityProjection,
        combine_raw_frontier_integrity_statuses,
        missing_source_raw_integrity_status,
        raw_frontier_integrity_snapshot_from_connections,
    )

    snapshot = raw_frontier_integrity_snapshot_from_connections(source_conn, index_conn=index_conn, ops_conn=ops_conn)
    missing_status, missing_count, missing_samples, missing_reason = missing_source_raw_integrity_status(
        materialization
    )
    statuses = (snapshot.broken_head_status, missing_status, snapshot.cursor_ahead_status)
    return RawFrontierIntegrityProjection(
        available="unknown" not in statuses,
        overall_status=combine_raw_frontier_integrity_statuses(*statuses),
        broken_head_status=snapshot.broken_head_status,
        broken_head_count=snapshot.broken_head_count,
        broken_head_checked_count=snapshot.broken_head_checked_count,
        broken_head_samples=snapshot.broken_head_samples,
        broken_head_reason=snapshot.broken_head_reason,
        missing_source_raw_status=missing_status,
        missing_source_raw_count=missing_count,
        missing_source_raw_samples=missing_samples,
        missing_source_raw_reason=missing_reason,
        cursor_ahead_status=snapshot.cursor_ahead_status,
        cursor_ahead_count=snapshot.cursor_ahead_count,
        cursor_ahead_checked_count=snapshot.cursor_ahead_checked_count,
        cursor_head_comparison_count=snapshot.cursor_head_comparison_count,
        cursor_ahead_comparison_count=snapshot.cursor_ahead_comparison_count,
        cursor_ahead_samples=snapshot.cursor_ahead_samples,
        cursor_authority_gap_count=snapshot.cursor_authority_gap_count,
        cursor_authority_gap_samples=snapshot.cursor_authority_gap_samples,
        cursor_authority_deferred_count=snapshot.cursor_authority_deferred_count,
        cursor_ahead_reason=snapshot.cursor_ahead_reason,
    ).to_dict()


def _raw_failure_status(source_conn: sqlite3.Connection | None) -> dict[str, object]:
    from polylogue.operations.status_workload import raw_failure_status_from_connection

    return raw_failure_status_from_connection(source_conn)


def _unavailable_embedding_status(settings: EmbeddingStatusSettings) -> dict[str, object]:
    """Complete explicit-unknown payload when an embeddings tier was not pinned."""

    return {
        "config_enabled": settings.config_enabled,
        "has_voyage_api_key": settings.has_voyage_api_key,
        "daemon_stage_enabled": None,
        "configured_model": settings.configured_model,
        "configured_dimension": settings.configured_dimension,
        "monthly_cost_cap_usd": settings.monthly_cost_cap_usd,
        "status": "unavailable",
        "total_sessions": 0,
        "embedded_sessions": 0,
        "blocked_sessions": 0,
        "embedded_messages": 0,
        "pending_sessions": 0,
        "pending_messages": None,
        "pending_messages_exact": False,
        "candidate_prose_messages": None,
        "candidate_prose_messages_exact": False,
        "embedding_coverage_percent": 0.0,
        "embedding_coverage_basis": "sessions",
        "message_coverage_percent": None,
        "retrieval_ready": False,
        "freshness_status": "unavailable",
        "stale_messages": 0,
        "messages_missing_provenance": 0,
        "oldest_embedded_at": None,
        "newest_embedded_at": None,
        "embedding_models": {},
        "embedding_dimensions": {},
        "retrieval_bands": {},
        "failure_count": 0,
        "terminal_failure_count": 0,
        "retryable_failure_count": 0,
        "failure_details": [],
        "total_estimated_cost_usd": None,
        "latest_catchup_run": None,
        "latest_material_catchup_run": None,
        "next_action": {"code": "not_observed", "reason": "embeddings tier unavailable from pinned reader"},
    }


def _ops_workload_status(conn: sqlite3.Connection | None, *, now_ms: int) -> dict[str, object]:
    from polylogue.operations.status_workload import ops_workload_status_from_connection

    return ops_workload_status_from_connection(conn, now_ms=now_ms)


def _convergence_status(conn: sqlite3.Connection | None, *, now_ms: int) -> dict[str, object]:
    from polylogue.operations.status_workload import convergence_status_from_connection

    return convergence_status_from_connection(conn, now_ms=now_ms)


def _schema_drift_status(conn: sqlite3.Connection | None, *, now_ms: int) -> dict[str, object]:
    from polylogue.analysis.schema_drift import schema_drift_status_from_connection

    return schema_drift_status_from_connection(conn, now_ms=now_ms)


def _raw_replay_backlog_status(index_conn: sqlite3.Connection, *, archive_root: Path) -> dict[str, object]:
    from polylogue.storage.raw_convergence import raw_materialization_replay_backlog_from_pinned_index

    return raw_materialization_replay_backlog_from_pinned_index(index_conn, archive_root=archive_root, limit=5)


def _components(
    *,
    index_conn: sqlite3.Connection,
    archive_readiness: Mapping[str, object],
    materialization: Mapping[str, object],
    frontier: Mapping[str, object],
    embedding_status: Mapping[str, object],
    component_from_archive_surface: Any,
    component_from_raw_materialization_readiness: Any,
    component_from_raw_frontier_integrity: Any,
) -> dict[str, dict[str, object]]:
    from polylogue.analysis.transforms import SESSION_DIGEST_TRANSFORM_VERSION, TRANSFORM_REGISTRY
    from polylogue.readiness.capability import (
        CapabilityReadinessState,
        ComponentReadiness,
        component_from_assertion_substrate,
        component_from_embedding_payload,
        component_from_transform_registry,
    )
    from polylogue.storage.sqlite.archive_tiers.user_audit import audit_user_overlay_storage

    components: dict[str, dict[str, object]] = {}
    surfaces = archive_readiness.get("surfaces")
    if isinstance(surfaces, Mapping):
        for name, surface in surfaces.items():
            if isinstance(surface, Mapping):
                components[str(name)] = component_from_archive_surface(str(name), surface, scope="archive").to_dict()
    material = component_from_raw_materialization_readiness(materialization).to_dict()
    components[str(material["component"])] = material
    frontier_component = component_from_raw_frontier_integrity(frontier).to_dict()
    components[str(frontier_component["component"])] = frontier_component
    components["embeddings"] = component_from_embedding_payload(embedding_status).to_dict()
    has_user = _attached_connection(index_conn, "user_tier") is not None
    has_assertions = has_user and _table_exists(index_conn, "assertions", schema="user_tier")
    assertion_counts = (
        index_conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT target_ref), "
            "COALESCE(SUM(status IS NULL OR status IN ('active', 'candidate')), 0) FROM user_tier.assertions"
        ).fetchone()
        if has_assertions
        else (0, 0, 0)
    )
    components["assertions"] = component_from_assertion_substrate(
        table_exists=has_assertions,
        assertion_count=int(assertion_counts[0]),
        target_count=int(assertion_counts[1]),
        active_count=int(assertion_counts[2]),
        overlay_audit=audit_user_overlay_storage(index_conn, schema="user_tier").to_dict() if has_assertions else None,
    ).to_dict()
    if archive_readiness.get("checked") is False:
        reason = str(archive_readiness.get("reason") or "archive_readiness_unchecked")
        components["transforms"] = ComponentReadiness(
            component="transforms",
            scope="session-analysis",
            state=CapabilityReadinessState.UNKNOWN,
            summary=reason,
            counts={
                "transform_count": len(TRANSFORM_REGISTRY),
                "session_digest_transform_version": SESSION_DIGEST_TRANSFORM_VERSION,
            },
            caveats=(reason,),
            evidence_refs=("transform_registry",),
        ).to_dict()
    else:
        counts = archive_readiness.get("counts")
        components["transforms"] = component_from_transform_registry(
            transform_count=len(TRANSFORM_REGISTRY),
            session_count=int(counts.get("session_count") or 0) if isinstance(counts, Mapping) else 0,
            session_digest_transform_version=SESSION_DIGEST_TRANSFORM_VERSION,
        ).to_dict()
    return components


def _sqlite_maintenance(conn: sqlite3.Connection) -> dict[str, object]:
    aliases = {str(row[1]) for row in conn.execute("PRAGMA database_list").fetchall()}
    tiers: dict[str, dict[str, object]] = {}
    with_planner_stats: list[str] = []
    for tier in ArchiveTier:
        alias = "main" if tier is ArchiveTier.INDEX else f"{tier.value}_tier"
        if alias not in aliases:
            tiers[tier.value] = {
                "exists": False,
                "wal_bytes": None,
                "sqlite_stat1_rows": 0,
                "planner_stats_present": False,
            }
            continue
        rows = (
            int(conn.execute(f"SELECT COUNT(*) FROM {alias}.sqlite_stat1").fetchone()[0] or 0)
            if _table_exists(conn, "sqlite_stat1", schema=alias)
            else 0
        )
        if rows:
            with_planner_stats.append(tier.value)
        tiers[tier.value] = {
            "exists": True,
            "wal_bytes": None,
            "sqlite_stat1_rows": rows,
            "planner_stats_present": rows > 0,
            "wal_metadata": {
                "state": "not_observed",
                "reason": "filesystem sidecar metadata is outside pinned SQLite snapshot",
            },
        }
    return {
        "archive_root": None,
        "total_wal_bytes": None,
        "tiers_with_planner_stats": with_planner_stats,
        "tiers": tiers,
    }


def _raw_ready(value: Mapping[str, object]) -> bool:
    from polylogue.storage.archive_readiness import raw_materialization_ready

    return raw_materialization_ready(value)


def _not_observed(reason: str, **extra: object) -> dict[str, object]:
    return {"state": "not_observed", "reason": reason, **extra}


def _status_ok(components: Mapping[str, Mapping[str, object]], raw_failures: Mapping[str, object]) -> bool:
    if not raw_failures.get("raw_failure_lifecycle_available") or raw_failures.get("raw_unexplained_failures"):
        return False
    required_missing = {"archive_sessions", "raw_materialization", "search", "transforms", "assertions"}
    required_known = {"raw_frontier_integrity"}
    if any(name not in components for name in required_known):
        return False
    for name, readiness in components.items():
        state = str(readiness.get("state") or "unknown")
        if state in {"blocked", "poisoned", "stale", "degraded"}:
            return False
        if state == "unknown" and name in required_known:
            return False
        if state == "missing" and name in required_missing:
            return False
    return True
