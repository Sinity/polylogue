"""Status command — query daemon health and archive state."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_BUILTIN_DAEMON_URL = "http://127.0.0.1:8766"
# Bare `polylogue` uses this as a quick probe before falling back to SQLite.
_FAST_TIMEOUT_S = 1.0
# Explicit status is still an operator command; a busy local daemon should fall
# through to bounded read-only SQLite status instead of hiding the archive.
_FULL_TIMEOUT_S = 3.0
_ARCHIVE_TIER_TARGETS: tuple[str, ...] = (
    "source.db",
    "index.db",
    "embeddings.db",
    "user.db",
    "ops.db",
)

_ARCHIVE_COMPONENT_SCOPES: dict[str, str] = {
    "archive_sessions": "archive",
    "raw_artifacts": "source",
    "search": "lexical",
    "session_profiles": "insights",
    "timeline_work_events": "insights",
    "timeline_phases": "insights",
    "threads": "insights",
    "tool_usage": "actions",
    "latency_profiles": "insights",
}

_ARCHIVE_COMPONENT_REPAIR_HINTS: dict[str, str] = {
    "search": "polylogue ops maintenance run --target dangling_fts",
    "session_profiles": "polylogue ops maintenance run --target session_insights",
    "timeline_work_events": "polylogue ops maintenance run --target session_insights",
    "timeline_phases": "polylogue ops maintenance run --target session_insights",
    "threads": "polylogue ops maintenance run --target session_insights",
    "latency_profiles": "polylogue ops maintenance run --target session_insights",
}


def _default_daemon_url() -> str:
    """Resolve the default daemon URL.

    Honours ``POLYLOGUE_DAEMON_URL`` so test fixtures can route the CLI to an
    unreachable address and avoid contacting an operator-host ``polylogued``
    listening at the built-in default (#1325).
    """
    override = os.environ.get("POLYLOGUE_DAEMON_URL")
    if override:
        return override
    return _BUILTIN_DAEMON_URL


def _fast_count(conn: Any, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _fast_fts_doc_count(conn: Any) -> int:
    # Exact FTS coverage is no longer a direct-status fallback concern.
    # Counting the FTS shadow table can fault gigabytes of pages during
    # catch-up; daemon status carries the maintained freshness state.
    return 0


def _daemon_live(daemon_url: str, *, timeout: float) -> bool:
    try:
        req = Request(
            f"{daemon_url}/healthz/live",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 500
    except (OSError, ValueError):
        return False


def _table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _archive_index_path(db: Any) -> Any | None:
    index_db = db if getattr(db, "name", None) == "index.db" else db.with_name("index.db")
    return index_db if index_db.exists() else None


def _active_status_db(db: Any) -> Any | None:
    if isinstance(db, Path):
        try:
            from polylogue.paths import active_index_db_path

            active_db = active_index_db_path()
            if active_db.exists():
                return active_db
        except Exception:
            pass
    index_db = _archive_index_path(db)
    if index_db is not None:
        return index_db
    return db if db.exists() else index_db


def _archive_tier_files(root: Path) -> dict[str, Path]:
    return {
        "source": root / "source.db",
        "index": root / "index.db",
        "embeddings": root / "embeddings.db",
        "user": root / "user.db",
        "ops": root / "ops.db",
    }


_ARCHIVE_TIER_TABLES: dict[str, tuple[str, ...]] = {
    "source": ("raw_sessions", "blob_refs", "raw_artifacts", "raw_hook_events", "history_sidecars"),
    "index": (
        "sessions",
        "messages",
        "blocks",
        "actions",
        "session_profiles",
        "session_work_events",
        "session_phases",
        "threads",
        "thread_sessions",
        "insight_materialization",
    ),
    "embeddings": ("embeddings_meta", "embedding_status"),
    "user": (
        "marks",
        "annotations",
        "corrections",
        "suppressions",
        "session_tags",
        "session_metadata",
        "saved_views",
        "recall_packs",
        "workspaces",
        "blackboard_notes",
    ),
    "ops": (
        "ingest_cursor",
        "ingest_attempts",
        "convergence_debt",
        "cursor_lag_samples",
        "daemon_stage_events",
        "daemon_events",
        "embedding_catchup_runs",
        "otlp_spans",
        "otlp_telemetry",
    ),
}


_ARCHIVE_TIER_ENUM: dict[str, ArchiveTier] = {tier.value: tier for tier in ArchiveTier}


_ARCHIVE_FACADE_ROUTES: dict[str, tuple[str, str, str]] = {
    "add_mark": ("archive_routed", "user", "writes user marks through user.db"),
    "add_tag": ("archive_routed", "user", "writes user tags through user.db"),
    "bulk_get_messages": ("archive_routed", "index", "reads messages from index.db"),
    "bulk_tag_sessions": ("archive_routed", "user", "writes user tags through user.db"),
    "clear_corrections": ("archive_routed", "user", "clears user corrections through user.db"),
    "close": ("not_archive_runtime", "none", "resource lifecycle method"),
    "cost_outlook": ("archive_routed", "index", "uses archive-routed session cost insight reads"),
    "count_sessions": ("archive_routed", "index", "counts sessions from index.db"),
    "create_recall_pack": ("archive_routed", "user", "writes recall packs through user.db"),
    "delete_annotation": ("archive_routed", "user", "deletes annotations through user.db"),
    "delete_session": ("archive_routed", "index", "delegates to archive-routed safe delete"),
    "delete_session_safe": ("archive_routed", "index", "deletes index rows while preserving user.db overlays"),
    "delete_correction": ("archive_routed", "user", "deletes corrections through user.db"),
    "delete_metadata": ("archive_routed", "user", "deletes user metadata through user.db"),
    "delete_recall_pack": ("archive_routed", "user", "deletes recall packs through user.db"),
    "delete_view": ("archive_routed", "user", "deletes saved views through user.db"),
    "delete_workspace": ("archive_routed", "user", "deletes workspaces through user.db"),
    "diagnose_query_miss": ("archive_direct", "index", "explains an empty index query result from index.db probes"),
    "export_insight_bundle": ("archive_routed", "index", "exports from insight tables when active"),
    "facets": ("archive_routed", "index", "computes scoped/global facets from index.db"),
    "find_resume_candidates": ("archive_routed", "index", "uses archive-routed resume operations"),
    "find_stuck_session_latency_profile_insights": ("archive_routed", "index", "reads latency profiles from index.db"),
    "explain_query_expression": ("archive_routed", "index", "explains query DSL parsing and lowering"),
    "get_actions": ("archive_direct", "index", "derives actions from index.db content blocks"),
    "get_actions_batch": (
        "archive_direct",
        "index",
        "batch-derives actions from index.db content blocks",
    ),
    "get_ancestors": ("archive_routed", "index", "reads session topology from index.db"),
    "get_annotation": ("archive_routed", "user", "reads annotations through user.db"),
    "get_session": ("archive_routed", "index", "reads session envelopes from index.db"),
    "get_session_stats": ("archive_routed", "index", "reads stats from index.db summaries"),
    "get_session_summary": ("archive_routed", "index", "reads summaries from index.db"),
    "get_sessions": ("archive_routed", "index", "delegates each read to the archive-routed session reader"),
    "get_descendants": ("archive_routed", "index", "reads session topology from index.db"),
    "get_index_status": ("archive_direct", "index", "reads block-FTS existence and doc count from index.db"),
    "get_logical_session": ("archive_routed", "index", "reads session topology from index.db"),
    "get_messages_paginated": ("archive_routed", "index", "reads messages from index.db session envelopes"),
    "get_metadata": ("archive_routed", "user", "reads user metadata through user.db"),
    "get_raw_artifacts_for_session": ("archive_routed", "source", "reads raw artifacts through source.db"),
    "get_recall_pack": ("archive_routed", "user", "reads recall packs through user.db"),
    "get_session_insight_status": ("archive_routed", "index", "reads insight readiness from index.db"),
    "get_session_latency_profile_insight": ("archive_routed", "index", "reads latency profiles from index.db"),
    "get_session_phase_insights": ("archive_routed", "index", "reads phases from index.db"),
    "get_session_profile_insight": ("archive_routed", "index", "reads profiles from index.db"),
    "get_session_profile_record": ("archive_routed", "index", "reads session profile record from index.db"),
    "get_session_topology": ("archive_routed", "index", "reads session topology from index.db"),
    "get_session_tree": ("archive_routed", "index", "reads session tree from index.db"),
    "get_session_work_event_insights": ("archive_routed", "index", "reads work events from index.db"),
    "get_siblings": ("archive_routed", "index", "reads session topology from index.db"),
    "get_stats_by": ("archive_direct", "index", "groups session counts from index.db"),
    "get_thread": ("archive_routed", "index", "reads session topology from index.db"),
    "get_view": ("archive_routed", "user", "reads saved views through user.db"),
    "get_view_by_name": ("archive_routed", "user", "reads saved views through user.db"),
    "get_thread_insight": ("archive_routed", "index", "reads work threads from index.db"),
    "get_workspace": ("archive_routed", "user", "reads workspaces through user.db"),
    "health_check": ("archive_routed", "index", "returns archive tier/index readiness"),
    "insight_readiness_report": ("archive_routed", "index", "reads insight readiness from index.db"),
    "insight_rigor_audit": ("archive_routed", "index", "reads insight rigor from index.db"),
    "list_annotations": ("archive_routed", "user", "reads annotations through user.db"),
    "list_assertion_claims": ("archive_routed", "user", "reads assertion claims through user.db"),
    "list_assertion_claim_payloads": ("archive_routed", "user", "reads assertion claim payload DTOs through user.db"),
    "list_blackboard_notes": ("archive_routed", "user", "reads blackboard notes through user.db"),
    "list_archive_coverage_insights": ("archive_routed", "index", "reads coverage insights from index.db"),
    "list_archive_debt_insights": ("archive_routed", "index", "reads archive debt projection from index.db"),
    "list_read_view_profiles": ("archive_routed", "index", "lists executable read-view profiles"),
    "list_sessions": ("archive_routed", "index", "reads full sessions from index.db"),
    "list_sessions_for_spec": ("archive_direct", "index", "runs a query spec directly against index.db"),
    "list_corrections": ("archive_routed", "user", "reads corrections through user.db"),
    "list_cost_rollup_insights": ("archive_routed", "index", "reads cost rollups from index.db"),
    "list_marks": ("archive_routed", "user", "reads marks through user.db"),
    "list_recall_packs": ("archive_routed", "user", "reads recall packs through user.db"),
    "list_session_cost_insights": ("archive_routed", "index", "reads session costs from index.db"),
    "list_session_latency_profile_insights": ("archive_routed", "index", "reads latency profiles from index.db"),
    "list_session_phase_insights": ("archive_routed", "index", "reads phases from index.db"),
    "list_session_profile_insights": ("archive_routed", "index", "reads profiles from index.db"),
    "list_session_tag_rollup_insights": ("archive_routed", "index", "reads tag rollups from index.db"),
    "list_session_work_event_insights": ("archive_routed", "index", "reads work events from index.db"),
    "list_summaries": ("archive_direct", "index", "reads session summaries from index.db without hydration"),
    "list_tags": ("archive_routed", "user", "reads user tag counts through user.db"),
    "list_tool_usage_insights": ("archive_routed", "index", "reads tool usage insights from index.db"),
    "list_views": ("archive_routed", "user", "reads saved views through user.db"),
    "list_thread_insights": ("archive_routed", "index", "reads work threads from index.db"),
    "list_workspaces": ("archive_routed", "user", "reads workspaces through user.db"),
    "neighbor_candidates": ("archive_routed", "index", "discovers neighbors from index.db"),
    "neighbor_candidate_payloads": ("archive_routed", "index", "builds neighbor DTOs from index.db"),
    "parse_file": ("archive_routed", "source", "writes source.db and index.db directly"),
    "post_blackboard_note": ("archive_routed", "user", "writes blackboard notes through user.db"),
    "parse_sources": ("archive_routed", "source", "writes source.db and index.db directly"),
    "query_units": ("archive_routed", "index", "queries terminal archive units from index.db"),
    "query_completions": ("archive_routed", "index", "returns query DSL completions from index.db metadata"),
    "query_sessions": ("archive_routed", "index", "queries summaries from index.db"),
    "rebuild_index": ("archive_routed", "index", "rebuilds messages_fts from index.db"),
    "rebuild_insights": ("archive_routed", "index", "rebuilds insight tables"),
    "resolve_ref": ("archive_routed", "index", "resolves public refs through bounded archive read payloads"),
    "record_correction": ("archive_routed", "user", "writes corrections through user.db"),
    "context_pack_payload": ("archive_routed", "index", "builds context-pack DTOs from archive-routed reads"),
    "context_preamble_payload": ("archive_routed", "index", "builds context preamble DTOs from archive-routed reads"),
    "recovery_digest": ("archive_routed", "index", "builds recovery digests from archive-routed session reads"),
    "recovery_report": ("archive_routed", "index", "renders recovery reports from archive-routed session reads"),
    "recovery_read_payload": (
        "archive_routed",
        "index",
        "renders recovery read DTOs from archive-routed session reads",
    ),
    "recovery_work_packet": (
        "archive_routed",
        "index",
        "renders recovery work packets from archive-routed session reads",
    ),
    "remove_mark": ("archive_routed", "user", "writes user marks through user.db"),
    "remove_tag": ("archive_routed", "user", "writes user tags through user.db"),
    "resume_brief": ("archive_routed", "index", "builds resume briefs from archive-routed reads"),
    "save_annotation": ("archive_routed", "user", "writes annotations through user.db"),
    "save_view": ("archive_routed", "user", "writes saved views through user.db"),
    "save_workspace": ("archive_routed", "user", "writes workspaces through user.db"),
    "archive_count_sessions": ("archive_direct", "index", "current archive helper"),
    "archive_get_session": ("archive_direct", "index", "current archive helper"),
    "archive_list_sessions": ("archive_direct", "index", "current archive helper"),
    "archive_search_sessions": ("archive_direct", "index", "current archive helper"),
    "search": ("archive_routed", "index", "searches index.db block FTS"),
    "search_session_hits": ("archive_direct", "index", "projects FTS/hybrid search hits from index.db"),
    "search_envelope": ("archive_routed", "index", "builds envelopes from index.db"),
    "session_correlation_payload": (
        "archive_routed",
        "index",
        "builds git/GitHub correlation DTOs from index.db sessions",
    ),
    "set_metadata": ("archive_routed", "user", "writes user metadata through user.db"),
    "stats": ("archive_routed", "index", "reads archive stats from index.db"),
    "storage_stats": ("archive_direct", "index", "reads lightweight archive counts from index.db"),
    "update_index": ("archive_direct", "index", "rebuilds the block-FTS index against index.db blocks"),
    "update_metadata": ("archive_routed", "user", "delegates to archive-routed metadata writes"),
}


_ARCHIVE_CLI_ROUTES: dict[str, tuple[str, str, str]] = {
    "user-state.blackboard.list": ("archive_direct", "user", "reads blackboard notes through active-root user.db"),
    "user-state.blackboard.post": ("archive_direct", "user", "writes blackboard notes through active-root user.db"),
    "reset.session": (
        "archive_direct",
        "user",
        "writes suppressions to active-root user.db and deletes rebuildable active-root index rows",
    ),
    "reset.database": ("archive_direct", "source", "deletes active-root tier files and sidecars"),
    "reset.source": (
        "archive_direct",
        "source",
        "resolves source paths through active-root source.db/index.db and writes user-tier suppressions",
    ),
}


def _archive_facade_route_status() -> dict[str, Any]:
    route_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    routes: dict[str, dict[str, str]] = {}
    for method, (route, tier, detail) in sorted(_ARCHIVE_FACADE_ROUTES.items()):
        route_counts[route] = route_counts.get(route, 0) + 1
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        routes[method] = {"route": route, "tier": tier, "detail": detail}
    unsupported = [method for method, info in routes.items() if info["route"] == "unsupported"]
    archive_ready_count = route_counts.get("archive_routed", 0) + route_counts.get("archive_direct", 0)
    return {
        "checked": True,
        "source": "static_facade_route_catalog",
        "total_method_count": len(routes),
        "archive_ready_method_count": archive_ready_count,
        "unsupported_method_count": len(unsupported),
        "route_counts": route_counts,
        "tier_counts": tier_counts,
        "unsupported_methods": unsupported,
        "routes": routes,
    }


def _archive_cli_route_status() -> dict[str, Any]:
    route_counts: dict[str, int] = {}
    tier_counts: dict[str, int] = {}
    routes: dict[str, dict[str, str]] = {}
    for command, (route, tier, detail) in sorted(_ARCHIVE_CLI_ROUTES.items()):
        route_counts[route] = route_counts.get(route, 0) + 1
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        routes[command] = {"route": route, "tier": tier, "detail": detail}
    unsupported = [command for command, info in routes.items() if info["route"] == "unsupported"]
    archive_ready_count = route_counts.get("archive_routed", 0) + route_counts.get("archive_direct", 0)
    return {
        "checked": True,
        "source": "static_cli_route_catalog",
        "total_command_count": len(routes),
        "archive_ready_command_count": archive_ready_count,
        "unsupported_command_count": len(unsupported),
        "route_counts": route_counts,
        "tier_counts": tier_counts,
        "unsupported_commands": unsupported,
        "routes": routes,
    }


def _archive_runtime_path_status() -> dict[str, Any]:
    facade_status = _archive_facade_route_status()
    cli_status = _archive_cli_route_status()
    routes = facade_status["routes"]
    unsupported_primary_methods = [
        method
        for method, info in routes.items()
        if info["route"] == "unsupported" and info["tier"] in {"source", "index"}
    ]
    final_shape_blockers = [
        {
            "method": method,
            "tier": routes[method]["tier"],
            "current_primary_store": "unavailable",
            "required_primary_store": "archive_file_set",
            "detail": routes[method]["detail"],
        }
        for method in unsupported_primary_methods
    ]
    ingest_primary_methods = [
        method for method in ("parse_file", "parse_sources") if method in unsupported_primary_methods
    ]
    index_primary_methods = [method for method in ("rebuild_index",) if method in unsupported_primary_methods]
    ingest_write_mode = "archive" if not ingest_primary_methods else "unsupported"
    return {
        "checked": True,
        "source": "static_facade_route_catalog",
        "archive_runtime_ready": not final_shape_blockers,
        "primary_ingest_store": "archive_file_set" if not ingest_primary_methods else "unavailable",
        "ingest_write_mode": ingest_write_mode,
        "archive_ingest_write_targets": ["source.db", "index.db"] if not ingest_primary_methods else [],
        "archive_tier_targets": list(_ARCHIVE_TIER_TARGETS),
        "facade_tier_route_counts": facade_status["tier_counts"],
        "cli_tier_route_counts": cli_status["tier_counts"],
        "index_rebuild_store": "index_db" if not index_primary_methods else "unavailable",
        "unsupported_primary_method_count": len(unsupported_primary_methods),
        "unsupported_primary_methods": unsupported_primary_methods,
        "final_shape_blockers": final_shape_blockers,
    }


def _archive_tier_status(root: Path) -> dict[str, dict[str, Any]]:
    return {tier: _archive_one_tier_status(tier, path) for tier, path in _archive_tier_files(root).items()}


def _archive_one_tier_status(tier: str, path: Path) -> dict[str, Any]:
    expected_version = ARCHIVE_VERSION_BY_TIER[_ARCHIVE_TIER_ENUM[tier]]
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "expected_user_version": expected_version,
        "user_version": None,
        "version_status": "missing",
        "table_counts": {},
    }
    if not path.exists():
        return status

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            user_version = int(row[0] or 0) if row is not None else 0
            status["user_version"] = user_version
            status["version_status"] = "ok" if user_version == expected_version else "mismatch"
            status["table_counts"] = _archive_table_counts(conn, _ARCHIVE_TIER_TABLES[tier])
        finally:
            conn.close()
    except sqlite3.Error as exc:
        status["version_status"] = "error"
        status["error"] = str(exc)
    return status


def _archive_table_counts(conn: sqlite3.Connection, table_names: Sequence[str]) -> dict[str, int]:
    return {
        table: _fast_count(conn, f"SELECT COUNT(*) FROM {table}") for table in table_names if _table_exists(conn, table)
    }


def _archive_readiness_status(root: Path) -> dict[str, Any]:
    index_db = root / "index.db"
    source_db = root / "source.db"
    if not index_db.exists():
        return {"checked": False, "reason": "missing_index_tier", "surfaces": {}}

    try:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            if not _table_exists(conn, "sessions"):
                return {"checked": False, "reason": "missing_sessions_table", "surfaces": {}}
            source_check_available = source_db.exists()
            source_conn: sqlite3.Connection | None = None
            try:
                if source_check_available:
                    source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
                    source_check_available = _table_exists(source_conn, "raw_sessions")
                counts = _archive_readiness_counts(
                    conn,
                    source_conn=source_conn,
                    source_check_available=source_check_available,
                )
            finally:
                if source_conn is not None:
                    source_conn.close()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"checked": False, "reason": str(exc), "surfaces": {}}

    surfaces = _archive_status_surfaces(counts, source_check_available=source_check_available)
    ready_count = sum(1 for info in surfaces.values() if info["ready"] is True)
    blocked_count = sum(1 for info in surfaces.values() if info["ready"] is not True)
    return {
        "checked": True,
        "reason": None,
        "source_check_available": source_check_available,
        "ready_surface_count": ready_count,
        "blocked_surface_count": blocked_count,
        "total_surface_count": len(surfaces),
        "counts": counts,
        "surfaces": surfaces,
    }


def _archive_readiness_counts(
    conn: sqlite3.Connection,
    *,
    source_conn: sqlite3.Connection | None,
    source_check_available: bool,
) -> dict[str, int]:
    session_count = _fast_count(conn, "SELECT COUNT(*) FROM sessions")
    raw_link_count = (
        _fast_count(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL")
        if _column_exists(conn, "sessions", "raw_id")
        else 0
    )
    missing_raw_session_count = 0
    if source_check_available and source_conn is not None and _column_exists(conn, "sessions", "raw_id"):
        raw_ids = {
            str(row[0])
            for row in source_conn.execute("SELECT raw_id FROM raw_sessions").fetchall()
            if row[0] is not None
        }
        missing_raw_session_count = sum(
            1
            for row in conn.execute("SELECT raw_id FROM sessions WHERE raw_id IS NOT NULL").fetchall()
            if str(row[0]) not in raw_ids
        )
    return {
        "session_count": session_count,
        "raw_link_count": raw_link_count,
        "missing_raw_session_count": missing_raw_session_count,
        "message_count": _fast_count(conn, "SELECT COUNT(*) FROM messages") if _table_exists(conn, "messages") else 0,
        "text_block_count": _fast_count(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
        if _table_exists(conn, "blocks")
        else 0,
        "messages_fts_count": _fast_count(conn, "SELECT COUNT(*) FROM messages_fts")
        if _table_exists(conn, "messages_fts")
        else 0,
        "profile_row_count": _fast_count(conn, "SELECT COUNT(*) FROM session_profiles")
        if _table_exists(conn, "session_profiles")
        else 0,
        "missing_profile_row_count": _archive_missing_rows(conn, "session_profiles", "session_id"),
        "work_event_row_count": _fast_count(conn, "SELECT COUNT(*) FROM session_work_events")
        if _table_exists(conn, "session_work_events")
        else 0,
        "phase_row_count": _fast_count(conn, "SELECT COUNT(*) FROM session_phases")
        if _table_exists(conn, "session_phases")
        else 0,
        "thread_count": _fast_count(conn, "SELECT COUNT(*) FROM threads") if _table_exists(conn, "threads") else 0,
        "action_count": _fast_count(conn, "SELECT COUNT(*) FROM actions") if _table_exists(conn, "actions") else 0,
        "missing_session_profile_materialization": _archive_missing_materialization(conn, "session_profile"),
        "missing_work_events_materialization": _archive_missing_materialization(conn, "work_events"),
        "missing_phases_materialization": _archive_missing_materialization(conn, "phases"),
        "missing_thread_materialization": _archive_missing_materialization(conn, "thread"),
        "missing_latency_materialization": _archive_missing_materialization(conn, "latency"),
    }


def _archive_missing_rows(conn: sqlite3.Connection, table: str, column: str) -> int:
    if not _table_exists(conn, table):
        return _fast_count(conn, "SELECT COUNT(*) FROM sessions")
    return _fast_count(
        conn,
        f"""
        SELECT COUNT(*)
        FROM sessions AS s
        WHERE NOT EXISTS (
            SELECT 1 FROM {table} AS t WHERE t.{column} = s.session_id
        )
        """,
    )


def _archive_missing_materialization(conn: sqlite3.Connection, insight_type: str) -> int:
    if not _table_exists(conn, "insight_materialization"):
        return _fast_count(conn, "SELECT COUNT(*) FROM sessions")
    return _fast_count(
        conn,
        """
        SELECT COUNT(*)
        FROM sessions AS s
        WHERE NOT EXISTS (
            SELECT 1
            FROM insight_materialization AS m
            WHERE m.insight_type = ? AND m.session_id = s.session_id
        )
        """,
        (insight_type,),
    )


def _archive_status_surfaces(counts: dict[str, int], *, source_check_available: bool) -> dict[str, dict[str, Any]]:
    def surface(*, ready: bool | None, blockers: list[str], evidence: dict[str, int | bool]) -> dict[str, Any]:
        return {"ready": ready, "blockers": blockers, "evidence": evidence}

    raw_blockers: list[str] = []
    raw_ready: bool | None
    if not source_check_available:
        raw_ready = None
        raw_blockers.append("source_tier_unavailable")
    elif counts["missing_raw_session_count"]:
        raw_ready = False
        raw_blockers.append("missing_source_raw_sessions")
    else:
        raw_ready = True

    search_blockers = (
        ["messages_fts_row_mismatch"] if counts["text_block_count"] != counts["messages_fts_count"] else []
    )
    profile_blockers: list[str] = []
    if counts["missing_profile_row_count"]:
        profile_blockers.append("missing_profile_rows")
    if counts["missing_session_profile_materialization"]:
        profile_blockers.append("missing_session_profile_materialization")

    def materialized(name: str) -> tuple[bool, list[str]]:
        key = f"missing_{name}_materialization"
        missing = counts[key]
        return (missing == 0, [] if missing == 0 else [key])

    work_ready, work_blockers = materialized("work_events")
    phase_ready, phase_blockers = materialized("phases")
    thread_ready, thread_blockers = materialized("thread")
    latency_ready, latency_blockers = materialized("latency")

    return {
        "archive_sessions": surface(
            ready=True,
            blockers=[],
            evidence={"session_count": counts["session_count"], "message_count": counts["message_count"]},
        ),
        "raw_artifacts": surface(
            ready=raw_ready,
            blockers=raw_blockers,
            evidence={
                "source_check_available": source_check_available,
                "raw_link_count": counts["raw_link_count"],
                "missing_raw_session_count": counts["missing_raw_session_count"],
            },
        ),
        "search": surface(
            ready=not search_blockers,
            blockers=search_blockers,
            evidence={
                "text_block_count": counts["text_block_count"],
                "messages_fts_count": counts["messages_fts_count"],
            },
        ),
        "session_profiles": surface(
            ready=not profile_blockers,
            blockers=profile_blockers,
            evidence={
                "profile_row_count": counts["profile_row_count"],
                "missing_profile_row_count": counts["missing_profile_row_count"],
                "missing_materialization_count": counts["missing_session_profile_materialization"],
            },
        ),
        "timeline_work_events": surface(
            ready=work_ready,
            blockers=work_blockers,
            evidence={
                "work_event_row_count": counts["work_event_row_count"],
                "missing_materialization_count": counts["missing_work_events_materialization"],
            },
        ),
        "timeline_phases": surface(
            ready=phase_ready,
            blockers=phase_blockers,
            evidence={
                "phase_row_count": counts["phase_row_count"],
                "missing_materialization_count": counts["missing_phases_materialization"],
            },
        ),
        "threads": surface(
            ready=thread_ready,
            blockers=thread_blockers,
            evidence={
                "thread_count": counts["thread_count"],
                "missing_materialization_count": counts["missing_thread_materialization"],
            },
        ),
        "tool_usage": surface(ready=True, blockers=[], evidence={"action_count": counts["action_count"]}),
        "latency_profiles": surface(
            ready=latency_ready,
            blockers=latency_blockers,
            evidence={"missing_materialization_count": counts["missing_latency_materialization"]},
        ),
    }


def _direct_archive_counts(conn: Any) -> dict[str, int]:
    if _table_exists(conn, "sessions"):
        messages = (
            _fast_count(conn, "SELECT COALESCE(SUM(message_count), 0) FROM sessions")
            if _column_exists(conn, "sessions", "message_count")
            else (_fast_count(conn, "SELECT COUNT(*) FROM messages") if _table_exists(conn, "messages") else 0)
        )
        return {
            "sessions": _fast_count(conn, "SELECT COUNT(*) FROM sessions"),
            "messages": messages,
            "raw_records": _archive_source_raw_count(conn),
        }
    return {"sessions": 0, "messages": 0, "raw_records": 0}


def _archive_source_raw_count(conn: Any) -> int:
    if _table_exists(conn, "raw_sessions"):
        return _fast_count(conn, "SELECT COUNT(*) FROM raw_sessions")
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except Exception:
        return 0
    if row is None or len(row) < 3 or not row[2]:
        return 0
    source_db = Path(str(row[2])).with_name("source.db")
    if not source_db.exists():
        return 0
    try:
        source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            if not _table_exists(source_conn, "raw_sessions"):
                return 0
            return _fast_count(source_conn, "SELECT COUNT(*) FROM raw_sessions")
        finally:
            source_conn.close()
    except sqlite3.Error:
        return 0


def _column_exists(conn: Any, table_name: str, column_name: str) -> bool:
    return any(str(row[1]) == column_name for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall())


@click.command("status")
@click.option(
    "--daemon-url",
    default=_default_daemon_url,
    show_default=True,
    help="Daemon API URL (env: POLYLOGUE_DAEMON_URL).",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json"]),
    default=None,
    help="Output format (json for machine-readable).",
)
@click.option(
    "--json",
    "json_alias",
    is_flag=True,
    default=False,
    help="Alias for ``--format json``. Matches the sibling commands' ``--json`` flag (#1612).",
)
@click.pass_obj
def status_command(
    env: AppEnv,
    daemon_url: str,
    output_format: str | None,
    json_alias: bool,
) -> None:
    """Show daemon and archive health.

    Queries the running polylogued daemon for archive status: daemon
    liveness, ingestion progress, FTS coverage, insight freshness, and
    component health. Read-only — does not modify state.
    """
    if json_alias and output_format is None:
        output_format = "json"
    try:
        req = Request(
            f"{daemon_url}/api/status",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urlopen(req, timeout=_FULL_TIMEOUT_S) as resp:
            result = json.loads(resp.read())
    except (OSError, ValueError):
        # ValueError covers malformed URLs (urllib raises before any I/O).
        if output_format == "json":
            if _daemon_live(daemon_url, timeout=_FAST_TIMEOUT_S):
                _show_daemon_status_unavailable_json(env)
            else:
                _show_direct_json(env)
        else:
            if _daemon_live(daemon_url, timeout=_FAST_TIMEOUT_S):
                _show_daemon_status_unavailable(env)
            else:
                _show_direct_status(env)
        return

    if output_format == "json":
        _show_status_json(env, result)
    else:
        _show_daemon_status(env, result)


def show_fast_status(env: AppEnv, *, daemon_url: str | None = None) -> None:
    """Fast bare-invocation status: try daemon, fall back to local SQLite.

    Called from ``polylogue`` with no args. Uses a short HTTP timeout
    and bounded SQLite queries to stay under 2 seconds.
    """
    resolved_url = daemon_url if daemon_url is not None else _default_daemon_url()
    try:
        req = Request(
            f"{resolved_url}/api/status",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urlopen(req, timeout=_FAST_TIMEOUT_S) as resp:
            result = json.loads(resp.read())
        _show_daemon_status(env, result, compact=True)
    except OSError:
        if _daemon_live(resolved_url, timeout=_FAST_TIMEOUT_S):
            _show_daemon_status_unavailable(env, compact=True)
        else:
            _show_direct_status(env, compact=True)


def _show_daemon_status(env: AppEnv, status: dict[str, Any], *, compact: bool = False) -> None:
    """Render daemon status from the real DaemonStatus payload."""
    liveness = status.get("daemon_liveness", False)
    liveness_color = "green" if liveness else "yellow"
    liveness_text = "running" if liveness else "degraded"
    env.ui.console.print(f"\n[bold {liveness_color}]Daemon: {liveness_text}[/bold {liveness_color}]")

    # Component state
    components = status.get("component_state", {})
    if isinstance(components, dict):
        for name in ("watcher", "api", "browser_capture"):
            comp = components.get(name, {})
            if isinstance(comp, dict) and comp:
                state = comp.get("state", "unknown")
                state_color = {"running": "green", "degraded": "yellow", "stopped": "red", "disabled": "dim"}.get(
                    state, "white"
                )
                desc = comp.get("description", "")
                env.ui.console.print(f"  [{state_color}]●[/{state_color}] {name}: {desc}")

    # Live ingest. The status payload carries LiveIngestAttemptSummary, which
    # exposes running_count + per-attempt worker progress in `recent` — it has
    # no top-level completed_count/total_count, so the previous keys always read
    # 0 and this section never rendered (#1743 follow-up).
    live = status.get("live_ingest_attempts", {})
    if isinstance(live, dict):
        running = int(live.get("running_count", 0) or 0)
        recent = live.get("recent", [])
        recent_states = [r for r in recent if isinstance(r, dict)] if isinstance(recent, list) else []
        files_done = sum(int(r.get("worker_completed_count") or 0) for r in recent_states)
        files_total = sum(int(r.get("worker_total_count") or 0) for r in recent_states)
        if running or files_total:
            parts: list[str] = []
            if running:
                parts.append(f"{running} running")
            if files_total:
                parts.append(f"{files_done}/{files_total} files")
            for label, key in (("stale", "stale_running_count"), ("stuck", "stuck_running_count")):
                count = int(live.get(key, 0) or 0)
                if count:
                    parts.append(f"{count} {label}")
            if parts:
                env.ui.console.print(f"  Ingest: {', '.join(parts)}")

    # FTS
    fts = status.get("fts_readiness", {})
    if isinstance(fts, dict):
        pct = float(fts.get("coverage_pct", 100 if fts.get("messages_ready") else 0))
        fts_color = "green" if fts.get("messages_ready") else "yellow"
        env.ui.console.print(f"  FTS: [{fts_color}]{pct:.1f}% indexed[/{fts_color}]")

    # Sizes
    db_bytes = status.get("db_size_bytes", 0)
    disk_free = status.get("disk_free_bytes", 0)
    if db_bytes:
        env.ui.console.print(f"  DB: {_fmt_bytes(db_bytes)}  Free: {_fmt_bytes(disk_free)}")

    # Raw failures
    raw_parse = status.get("raw_parse_failures", 0)
    raw_val = status.get("raw_validation_failures", 0)
    raw_quarantined = status.get("raw_quarantined", 0)
    total_raw = (raw_parse or 0) + (raw_val or 0)
    if total_raw > 0:
        fail_color = "red" if total_raw > 10 else "yellow"
        env.ui.console.print(
            f"  Raw failures: [{fail_color}]{total_raw} total ({raw_quarantined} quarantined)"
            f" [{fail_color}]({raw_parse} parse + {raw_val} validation)[/{fail_color}]"
        )

    if not compact:
        checked = status.get("checked_at", "")
        if checked:
            env.ui.console.print(f"\n  [dim]Checked: {checked}[/dim]")


def _show_status_json(env: AppEnv, status: dict[str, Any]) -> None:
    """Machine-readable JSON status output."""
    env.ui.console.print(json.dumps(status, indent=2, default=str))


def _show_daemon_status_unavailable_json(env: AppEnv) -> None:
    payload = {
        "daemon_liveness": True,
        "status_snapshot": {
            "state": "unavailable",
            "reason": "api_status_timeout",
        },
    }
    env.ui.console.print(json.dumps(payload, indent=2, default=str))


def _show_daemon_status_unavailable(env: AppEnv, *, compact: bool = False) -> None:
    env.ui.console.print("\n[bold yellow]Daemon: running[/bold yellow]")
    env.ui.console.print("  Status snapshot: [yellow]unavailable[/yellow]")
    if not compact:
        env.ui.console.print("  [dim]/api/status did not answer within the bounded CLI timeout.[/dim]")


def _show_direct_json(env: AppEnv) -> None:
    """Machine-readable JSON fallback when daemon is not running."""
    from polylogue.cli.commands.init import starter_config_path
    from polylogue.cli.commands.status_diagnostics import (
        diagnose_first_run,
        diagnostic_payload,
    )
    from polylogue.paths import archive_root, db_path

    db = db_path()
    root = archive_root()
    config_path = starter_config_path()
    diag = diagnose_first_run(daemon_alive=False)
    active_db = _active_status_db(db)
    active_root = active_db.parent if active_db is not None and active_db.name == "index.db" else root
    archive_readiness = _archive_readiness_status(active_root)
    payload: dict[str, Any] = {
        "daemon_liveness": False,
        "archive_root": str(root),
        "active_archive_root": str(active_root),
        "active_archive_root_matches_configured": active_root == root,
        "db_exists": db.exists(),
        "active_db_path": None,
        "config_exists": config_path.exists(),
        "config_path": str(config_path),
        "archive_tiers": _archive_tier_status(active_root),
        "archive_readiness": archive_readiness,
        "archive_facade_routes": _archive_facade_route_status(),
        "archive_cli_routes": _archive_cli_route_status(),
        "archive_runtime_paths": _archive_runtime_path_status(),
        "component_readiness": _direct_component_readiness(
            env,
            active_root=active_root,
            archive_readiness=archive_readiness,
        ),
        "next_action": diag.next_action,
        "diagnostic": diagnostic_payload(diag),
    }
    if active_db is not None and active_db.exists():
        payload["active_db_path"] = str(active_db)
        try:
            from polylogue.storage.sqlite.connection_profile import open_readonly_connection

            conn = open_readonly_connection(active_db)
            try:
                payload.update(_direct_archive_counts(conn))
                payload["db_exists"] = True
            finally:
                conn.close()
        except Exception as exc:
            payload["error"] = str(exc)
    env.ui.console.print(json.dumps(payload, indent=2, default=str))


def _direct_component_readiness(
    env: AppEnv,
    *,
    active_root: Path,
    archive_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return additive component readiness for direct status JSON."""
    components: dict[str, Any] = {}
    if archive_readiness is not None:
        try:
            from polylogue.readiness.capability import component_from_archive_surface

            surfaces = archive_readiness.get("surfaces") or {}
            if isinstance(surfaces, dict):
                for component, scope in _ARCHIVE_COMPONENT_SCOPES.items():
                    surface = surfaces.get(component)
                    if not isinstance(surface, dict):
                        continue
                    readiness = component_from_archive_surface(
                        component,
                        surface,
                        scope=scope,
                        repair_hint=_ARCHIVE_COMPONENT_REPAIR_HINTS.get(component),
                    )
                    components[readiness.component] = readiness.to_dict()
        except Exception:
            pass
    try:
        from polylogue.readiness.capability import component_from_embedding_payload
        from polylogue.storage.embeddings.status_payload import embedding_status_payload

        embedding_payload = embedding_status_payload(env, include_retrieval_bands=False)
        embedding = component_from_embedding_payload(embedding_payload)
        components[embedding.component] = embedding.to_dict()
    except Exception:
        pass
    try:
        assertions = _direct_assertion_component(active_root)
        components[assertions["component"]] = assertions
    except Exception:
        pass
    try:
        transforms = _direct_transform_component(archive_readiness)
        components[transforms["component"]] = transforms
    except Exception:
        pass
    return components


def _direct_assertion_component(active_root: Path) -> dict[str, Any]:
    from polylogue.readiness.capability import component_from_assertion_substrate
    from polylogue.storage.sqlite.archive_tiers.user_audit import audit_user_overlay_storage

    user_db = active_root / "user.db"
    if not user_db.exists():
        return component_from_assertion_substrate(table_exists=False).to_dict()

    try:
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
        try:
            table_exists = _table_exists(conn, "assertions")
            if not table_exists:
                return component_from_assertion_substrate(table_exists=False).to_dict()
            component = component_from_assertion_substrate(
                table_exists=True,
                assertion_count=_fast_count(conn, "SELECT COUNT(*) FROM assertions"),
                target_count=_fast_count(conn, "SELECT COUNT(DISTINCT target_ref) FROM assertions"),
                active_count=_fast_count(
                    conn,
                    """
                    SELECT COUNT(*)
                    FROM assertions
                    WHERE status IS NULL OR status IN ('active', 'candidate')
                    """,
                ),
                overlay_audit=audit_user_overlay_storage(conn).to_dict(),
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        component = component_from_assertion_substrate(table_exists=True, error=str(exc))
    return component.to_dict()


def _direct_transform_component(archive_readiness: dict[str, Any] | None) -> dict[str, Any]:
    from polylogue.insights.transforms import RECOVERY_TRANSFORM_VERSION, TRANSFORM_REGISTRY
    from polylogue.readiness.capability import component_from_transform_registry

    if isinstance(archive_readiness, dict) and archive_readiness.get("checked") is False:
        reason = str(archive_readiness.get("reason") or "archive_readiness_unchecked")
        return component_from_transform_registry(
            transform_count=len(TRANSFORM_REGISTRY),
            session_count=None,
            recovery_transform_version=RECOVERY_TRANSFORM_VERSION,
            error=reason,
        ).to_dict()

    counts = archive_readiness.get("counts") if isinstance(archive_readiness, dict) else None
    session_count = int(counts.get("session_count") or 0) if isinstance(counts, dict) else 0
    return component_from_transform_registry(
        transform_count=len(TRANSFORM_REGISTRY),
        session_count=session_count,
        recovery_transform_version=RECOVERY_TRANSFORM_VERSION,
    ).to_dict()


def _render_diagnostic(env: AppEnv, diag: Any) -> None:
    """Render a ``StatusDiagnostic`` with rich tags but no traceback."""
    color = "red" if diag.kind in {"schema_mismatch", "locked_db", "unknown_db_error"} else "yellow"
    env.ui.console.print(f"\n[{color}]{diag.headline}[/{color}]")
    if diag.detail:
        env.ui.console.print(f"  {diag.detail}")


def _render_direct_embedding_status(env: AppEnv, payload: dict[str, Any]) -> None:
    """Render bounded embedding readiness in direct SQLite fallback status."""
    if int(payload.get("total_sessions", 0) or 0) <= 0:
        return

    status = str(payload.get("status", "unknown"))
    freshness = str(payload.get("freshness_status", status))
    retrieval_ready = bool(payload.get("retrieval_ready", False))
    embedded_messages = int(payload.get("embedded_messages", 0) or 0)
    embedded_sessions = int(payload.get("embedded_sessions", 0) or 0)
    total_sessions = int(payload.get("total_sessions", 0) or 0)
    pending_sessions = int(payload.get("pending_sessions", 0) or 0)
    coverage = float(payload.get("embedding_coverage_percent", 0.0) or 0.0)
    stale_messages = int(payload.get("stale_messages", 0) or 0)
    failure_count = int(payload.get("failure_count", 0) or 0)

    color = "green" if retrieval_ready and freshness != "stale" else "yellow" if embedded_messages else "dim"
    ready_text = "ready" if retrieval_ready else "not ready"
    line = (
        f"  Embeddings: [{color}]{status}/{freshness}, {ready_text}; "
        f"{embedded_messages:,} msgs, {embedded_sessions:,}/{total_sessions:,} convs "
        f"({coverage:.1f}%), {pending_sessions:,} pending convs"
    )
    if stale_messages:
        line += f", {stale_messages:,} stale msgs"
    line += f"[/{color}]"
    env.ui.console.print(line)

    if failure_count:
        env.ui.console.print(f"  Embedding failures: [yellow]{failure_count:,}[/yellow]")

    latest = payload.get("latest_catchup_run")
    if isinstance(latest, dict):
        processed = int(latest.get("processed_sessions", 0) or 0)
        planned = int(latest.get("planned_sessions", 0) or 0)
        embedded = int(latest.get("embedded_messages", 0) or 0)
        errors = int(latest.get("error_count", 0) or 0)
        env.ui.console.print(
            "  Embedding catch-up: "
            f"{latest.get('status', 'unknown')}, {processed:,}/{planned:,} convs, "
            f"{embedded:,} msgs embedded, {errors:,} errors"
        )


def _show_direct_status(env: AppEnv, *, compact: bool = False) -> None:
    """Fallback status when daemon is not running."""
    from polylogue.cli.commands.status_diagnostics import diagnose_first_run
    from polylogue.paths import archive_root, db_path

    db = db_path()
    active_db = _active_status_db(db)
    if active_db is None or not active_db.exists():
        diag = diagnose_first_run(daemon_alive=False)
        _render_diagnostic(env, diag)
        return

    # Pre-flight: detect schema mismatch / locked db / stale pidfile
    # before attempting row counts. Short-circuits with actionable text
    # rather than a Python traceback (#1263).
    if db.exists() and active_db == db:
        diag = diagnose_first_run(daemon_alive=False)
        if diag.kind in {"schema_mismatch", "locked_db", "stale_pidfile"}:
            _render_diagnostic(env, diag)
            return

    try:
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        conn = open_readonly_connection(active_db)
        try:
            counts = _direct_archive_counts(conn)
            convs = counts["sessions"]
            msgs = counts["messages"]
            raw = counts["raw_records"]
            fts = _fast_fts_doc_count(conn)
        finally:
            conn.close()

        env.ui.console.print("\n[bold]Archive (daemon not running)[/bold]")
        env.ui.console.print(f"  Database: {active_db.name}")
        if active_db.name == "index.db":
            active_root = active_db.parent
            tiers = _archive_tier_status(active_root)
            present = ", ".join(tier for tier, info in tiers.items() if info["exists"])
            missing = ", ".join(tier for tier, info in tiers.items() if not info["exists"])
            tier_line = f"  Schema tiers: present={present or 'none'}"
            if missing:
                tier_line += f"; missing={missing}"
            env.ui.console.print(tier_line)
            tier_detail = _archive_tier_detail_line(tiers)
            if tier_detail:
                env.ui.console.print(f"  Archive tier detail: {tier_detail}")
            _render_archive_readiness(env, _archive_readiness_status(active_root))
            _render_archive_facade_routes(env, _archive_facade_route_status())
            _render_archive_cli_routes(env, _archive_cli_route_status())
            _render_archive_runtime_paths(env, _archive_runtime_path_status())
        env.ui.console.print(f"  Sessions: {convs:,}")
        env.ui.console.print(f"  Messages: {msgs:,}")
        env.ui.console.print(f"  Raw records: {raw:,}")
        if fts:
            fts_pct = 100 * fts / msgs if msgs else 100
            fts_color = "green" if fts_pct > 99 else "yellow"
            env.ui.console.print(f"  FTS indexed: [{fts_color}]{fts_pct:.1f}%[/{fts_color}]")
        else:
            env.ui.console.print("  FTS indexed: [dim]daemon status unavailable[/dim]")

        try:
            from polylogue.storage.embeddings.status_payload import embedding_status_payload

            ep = embedding_status_payload(env, include_retrieval_bands=False)
            _render_direct_embedding_status(env, dict(ep))
        except Exception:
            pass
        # When the archive is empty (no ingest has run yet), surface the
        # most relevant first-run diagnostic so the operator knows what to
        # do next — typically `no_sources` or `no_daemon` (#1263).
        if msgs == 0 and convs == 0:
            from polylogue.cli.commands.status_diagnostics import diagnose_first_run

            diag = diagnose_first_run(daemon_alive=False)
            if diag.kind in {"no_sources", "no_daemon", "missing_optional_dep"}:
                _render_diagnostic(env, diag)
                return

        if not compact:
            env.ui.console.print("\n  [dim]Run [bold]polylogued run[/bold] to start the daemon.[/dim]")
    except Exception:
        env.ui.console.print(f"\n[yellow]Archive exists at {archive_root()} but could not be queried.[/yellow]")


def _render_archive_readiness(env: AppEnv, readiness: dict[str, Any]) -> None:
    if not readiness.get("checked"):
        reason = readiness.get("reason") or "unknown"
        env.ui.console.print(f"  Archive readiness: [yellow]unchecked ({reason})[/yellow]")
        return

    ready = int(readiness.get("ready_surface_count", 0) or 0)
    total = int(readiness.get("total_surface_count", 0) or 0)
    blocked = int(readiness.get("blocked_surface_count", 0) or 0)
    color = "green" if blocked == 0 else "yellow"
    env.ui.console.print(f"  Archive surfaces: [{color}]{ready}/{total} ready, {blocked} blocked[/{color}]")
    surfaces = readiness.get("surfaces") or {}
    blocked_surfaces = [(name, info) for name, info in surfaces.items() if info.get("ready") is not True]
    for name, info in blocked_surfaces[:5]:
        blockers = ", ".join(info.get("blockers") or ["unknown"])
        env.ui.console.print(f"    {name}: {blockers}")
    if len(blocked_surfaces) > 5:
        env.ui.console.print(f"    +{len(blocked_surfaces) - 5} more blocked surfaces")


def _render_archive_facade_routes(env: AppEnv, routing: dict[str, Any]) -> None:
    total = int(routing.get("total_method_count", 0) or 0)
    archive_ready = int(routing.get("archive_ready_method_count", 0) or 0)
    unsupported = int(routing.get("unsupported_method_count", 0) or 0)
    color = "green" if unsupported == 0 else "yellow"
    env.ui.console.print(
        f"  Facade routes: [{color}]{archive_ready}/{total} archive-ready, {unsupported} unsupported[/{color}]"
    )
    methods = list(routing.get("unsupported_methods") or [])
    routes = routing.get("routes") or {}
    for method in methods[:5]:
        detail = routes.get(method, {}).get("detail", "unsupported route")
        env.ui.console.print(f"    {method}: {detail}")
    if len(methods) > 5:
        env.ui.console.print(f"    +{len(methods) - 5} more unsupported methods")


def _render_archive_cli_routes(env: AppEnv, routing: dict[str, Any]) -> None:
    total = int(routing.get("total_command_count", 0) or 0)
    archive_ready = int(routing.get("archive_ready_command_count", 0) or 0)
    unsupported = int(routing.get("unsupported_command_count", 0) or 0)
    color = "green" if unsupported == 0 else "yellow"
    env.ui.console.print(
        f"  CLI routes: [{color}]{archive_ready}/{total} archive-ready, {unsupported} unsupported[/{color}]"
    )
    commands = list(routing.get("unsupported_commands") or [])
    routes = routing.get("routes") or {}
    for command in commands[:5]:
        detail = routes.get(command, {}).get("detail", "unsupported route")
        env.ui.console.print(f"    {command}: {detail}")
    if len(commands) > 5:
        env.ui.console.print(f"    +{len(commands) - 5} more unsupported commands")


def _render_archive_runtime_paths(env: AppEnv, status: dict[str, Any]) -> None:
    ready = bool(status.get("archive_runtime_ready", False))
    color = "green" if ready else "yellow"
    mode = status.get("ingest_write_mode", "unknown")
    targets = list(status.get("archive_ingest_write_targets") or [])
    target_suffix = f" -> {','.join(str(target) for target in targets)}" if targets else ""
    archive_tiers = ",".join(str(tier) for tier in status.get("archive_tier_targets") or [])
    archive_tier_suffix = f", tiers={archive_tiers}" if archive_tiers else ""
    index_store = status.get("index_rebuild_store", "unknown")
    blockers = list(status.get("final_shape_blockers") or [])
    env.ui.console.print(
        f"  Archive runtime paths: [{color}]ingest={mode}{target_suffix}, rebuild_index={index_store}; "
        f"{len(blockers)} blockers{archive_tier_suffix}[/{color}]"
    )
    facade_counts = status.get("facade_tier_route_counts") or {}
    cli_counts = status.get("cli_tier_route_counts") or {}
    if facade_counts or cli_counts:
        env.ui.console.print(
            "  Archive route ownership: "
            f"facade={_archive_route_count_summary(facade_counts)}; "
            f"cli={_archive_route_count_summary(cli_counts)}"
        )
    for blocker in blockers[:5]:
        method = blocker.get("method", "unknown")
        current = blocker.get("current_primary_store", "unknown")
        required = blocker.get("required_primary_store", "unknown")
        env.ui.console.print(f"    {method}: {current} -> {required}")
    if len(blockers) > 5:
        env.ui.console.print(f"    +{len(blockers) - 5} more runtime blockers")


def _archive_route_count_summary(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "none"
    parts = [f"{tier}:{counts[tier]}" for tier in sorted(counts)]
    return ",".join(parts)


def _archive_tier_detail_line(tiers: dict[str, dict[str, Any]]) -> str:
    details: list[str] = []
    for tier, info in tiers.items():
        if not info.get("exists"):
            continue
        version = info.get("user_version")
        expected = info.get("expected_user_version")
        status = info.get("version_status")
        counts = info.get("table_counts") or {}
        primary_count = _archive_primary_tier_count(tier, counts)
        count_text = f", {primary_count[0]}={primary_count[1]:,}" if primary_count is not None else ""
        details.append(f"{tier} v{version}/{expected} {status}{count_text}")
    return "; ".join(details)


def _archive_primary_tier_count(tier: str, counts: dict[str, int]) -> tuple[str, int] | None:
    primary_tables = {
        "source": "raw_sessions",
        "index": "sessions",
        "embeddings": "embedding_status",
        "user": "annotations",
        "ops": "ingest_attempts",
    }
    table = primary_tables.get(tier)
    if table is None or table not in counts:
        return None
    return table, counts[table]


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{n / 1_000:.0f} KB"


__all__ = ["status_command", "show_fast_status"]
