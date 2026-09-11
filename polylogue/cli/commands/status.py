"""Status command — query daemon health and archive state."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from polylogue.cli.shared.types import AppEnv


def normalize_raw_frontier_status_payload(*args: Any, **kwargs: Any) -> Any:
    from polylogue.readiness.capability import normalize_raw_frontier_status_payload as implementation

    return implementation(*args, **kwargs)


def raw_frontier_integrity_is_proven_healthy(*args: Any, **kwargs: Any) -> Any:
    from polylogue.readiness.capability import raw_frontier_integrity_is_proven_healthy as implementation

    return implementation(*args, **kwargs)


def raw_frontier_integrity_summary(*args: Any, **kwargs: Any) -> Any:
    from polylogue.readiness.capability import raw_frontier_integrity_summary as implementation

    return implementation(*args, **kwargs)


def status_snapshot_has_fresh_provenance(*args: Any, **kwargs: Any) -> Any:
    from polylogue.readiness.capability import status_snapshot_has_fresh_provenance as implementation

    return implementation(*args, **kwargs)


def archive_file_set_root(*args: Any, **kwargs: Any) -> Any:
    from polylogue.storage.archive_identity import archive_file_set_root as implementation

    return implementation(*args, **kwargs)


_BUILTIN_DAEMON_URL = "http://127.0.0.1:8766"


def _default_daemon_url() -> str:
    """Resolve the default daemon URL through the layered config resolver.

    Honours ``daemon.url`` / ``POLYLOGUE_DAEMON_URL`` (site TOML -> user TOML
    -> env -> CLI) so test fixtures can route the CLI to an unreachable
    address and avoid contacting an operator-host ``polylogued`` listening at
    the built-in default (#1325).
    """
    from polylogue.config import load_polylogue_config

    return load_polylogue_config().daemon_url or _BUILTIN_DAEMON_URL


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _status_operation_result(
    env: AppEnv,
    *,
    daemon_url: str | None = None,
    include_archive_readiness: bool = False,
) -> Any:
    """Use the configured machine endpoint or the same pinned direct reader."""
    from polylogue.cli.operation_kernel import OperationKernelError, configured_read_operation
    from polylogue.cli.shared.helpers import load_effective_config

    if daemon_url not in (None, _BUILTIN_DAEMON_URL):
        raise OperationKernelError("machine status uses the configured archive's Unix socket, not a daemon URL")
    config = load_effective_config(env)
    return configured_read_operation(
        config,
        "status",
        {"include_archive_readiness": include_archive_readiness},
        daemon_disabled=bool(getattr(env, "no_daemon", False)),
    )


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
@click.option(
    "--full",
    "full_payload",
    is_flag=True,
    default=False,
    help="Emit full daemon/direct JSON details. Does not run exact archive-readiness probes.",
)
@click.option(
    "--exact-archive-readiness",
    is_flag=True,
    default=False,
    help="Run expensive exact archive-readiness probes in direct SQLite fallback.",
)
@click.option(
    "--source",
    "source_path",
    type=click.Path(path_type=Path),
    help="Inspect one exact source path from cursor evidence through searchability.",
)
@click.option(
    "--strict-source",
    is_flag=True,
    default=False,
    help="Fail when --source is degraded, incomplete, or has unsafe evidence reads.",
)
@click.pass_obj
def status_command(
    env: AppEnv,
    daemon_url: str,
    output_format: str | None,
    json_alias: bool,
    full_payload: bool,
    exact_archive_readiness: bool,
    source_path: Path | None = None,
    strict_source: bool = False,
) -> None:
    """Show daemon and archive health.

    Root ``polylogue status`` and ``polylogue ops status`` are the same
    read-only operational surface. It queries the running polylogued daemon
    for liveness, ingestion progress, FTS coverage, insight freshness, and
    component health; when the daemon is unavailable it falls back to bounded
    local archive checks.
    """
    if json_alias and output_format is None:
        output_format = "json"
    if source_path is not None:
        # A named-source status call is intentionally direct and bounded. Do
        # not ask the daemon for aggregate status first: the incident was an
        # excluded path hidden by aggregate coverage.
        from polylogue.archive.query.source_freshness import (
            NamedSourceOperationalState,
            NamedSourceStage,
            project_named_source_freshness,
        )
        from polylogue.archive.query.source_freshness_surfaces import (
            render_source_freshness_status,
        )
        from polylogue.cli.shared.helpers import load_effective_config

        if not source_path.is_absolute():
            raise click.UsageError("--source must be an absolute exact source path")
        config = load_effective_config(env)
        # polylogue-yla8.1 split-root contract: config.db_path always names a
        # concrete index.db (explicit override or resolved active generation).
        archive_root = archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)
        freshness = project_named_source_freshness(archive_root, source_path)
        if output_format == "json":
            click.echo(json.dumps(freshness.to_dict(), sort_keys=True))
        else:
            click.echo(render_source_freshness_status(freshness))
        unsafe = bool(freshness.receipt.unsafe_scan_rejections or freshness.errors)
        incomplete = (
            freshness.operational_state is NamedSourceOperationalState.DEGRADED
            or freshness.stage is not NamedSourceStage.SEARCHABLE
        )
        if unsafe:
            raise click.exceptions.Exit(3)
        if strict_source and incomplete:
            raise click.exceptions.Exit(2)
        return
    from polylogue.operations.route_observation import observe_route
    from polylogue.paths import archive_root as _resolve_archive_root

    try:
        observed_archive_root: Path | None = _resolve_archive_root()
    except Exception:
        observed_archive_root = None

    with observe_route(
        archive_root=observed_archive_root,
        surface="cli",
        route="cli.status",
        verb="full" if full_payload else "compact",
    ) as obs:
        from polylogue.cli.operation_kernel import OperationKernelError

        try:
            operation_result = _status_operation_result(
                env,
                daemon_url=daemon_url,
                include_archive_readiness=exact_archive_readiness,
            )
        except OperationKernelError:
            obs.attributes["daemon_reachable"] = True
            obs.daemon_path = "daemon"
            if output_format == "json":
                _show_daemon_status_unavailable_json(env)
            else:
                _show_daemon_status_unavailable(env, compact=not full_payload)
            raise click.exceptions.Exit(1) from None
        mode = operation_result.authority.get("mode")
        obs.attributes["daemon_reachable"] = mode == "daemon"
        obs.daemon_path = str(mode)
        status = operation_result.value
        status_ok = (
            _show_status_json(env, status, full=full_payload or exact_archive_readiness)
            if output_format == "json"
            else _render_direct_status_payload(env, status, compact=not full_payload)
            if mode == "direct"
            else _show_daemon_status(env, status)
        )
        if not status_ok:
            raise click.exceptions.Exit(1)
    return


def show_fast_status(env: AppEnv, *, daemon_url: str | None = None) -> None:
    """Fast bare-invocation status: try daemon, fall back to local SQLite.

    Called from ``polylogue`` with no args. Uses a short HTTP timeout
    and bounded SQLite queries to stay under 2 seconds.
    """
    del daemon_url
    from polylogue.cli.operation_kernel import OperationKernelError

    try:
        result = _status_operation_result(env)
    except OperationKernelError:
        _show_daemon_status_unavailable(env, compact=True)
        return
    if result.authority.get("mode") == "direct":
        _render_direct_status_payload(env, result.value, compact=True)
    else:
        _show_daemon_status(env, result.value, compact=True)


def _show_daemon_status(env: AppEnv, status: dict[str, Any], *, compact: bool = False) -> bool:
    """Render daemon status from the real DaemonStatus payload."""
    status = normalize_raw_frontier_status_payload(status, require_fresh_snapshot=True)
    liveness = status.get("daemon_liveness", False)
    overall_ok = _status_ok(status, require_fresh_snapshot=True)
    liveness_color = "green" if liveness and overall_ok else "yellow"
    liveness_text = "running" if liveness and overall_ok else "running; status degraded" if liveness else "degraded"
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

    # Runtime (polylogue-dcz5): free-threaded (3.14t+) vs GIL-enabled.
    gil_enabled = status.get("gil_enabled")
    if gil_enabled is not None:
        mode_text = "free-threaded (GIL disabled)" if gil_enabled is False else "GIL enabled"
        mode_color = "green" if gil_enabled is False else "dim"
        env.ui.console.print(f"  Runtime: [{mode_color}]{mode_text}[/{mode_color}]")

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

    convergence = status.get("convergence", {})
    if isinstance(convergence, dict):
        failed_count = int(convergence.get("failed_count", 0) or 0)
        deferred_count = int(convergence.get("deferred_count", 0) or 0)
        retry_due_count = int(convergence.get("retry_due_count", 0) or 0)
        if failed_count or deferred_count:
            env.ui.console.print(
                f"  Convergence debt: {failed_count} failed, {deferred_count} deferred, {retry_due_count} retry due"
            )

    # FTS
    fts = status.get("fts_readiness", {})
    if isinstance(fts, dict):
        fts_color = "green" if fts.get("messages_ready") else "yellow"
        raw_pct = fts.get("coverage_pct")
        if raw_pct is None:
            # An unmeasured coverage_pct must never be silently rendered as
            # a fabricated percentage (polylogue-roax) -- say plainly that
            # coverage is unknown rather than defaulting to 100%/0% based on
            # the boolean readiness flag alone.
            env.ui.console.print(f"  FTS: [{fts_color}]coverage unknown[/{fts_color}]")
        else:
            pct = _safe_float(raw_pct, default=0.0)
            env.ui.console.print(f"  FTS: [{fts_color}]{pct:.1f}% indexed[/{fts_color}]")

    raw_frontier = status.get("raw_frontier_integrity")
    if isinstance(raw_frontier, dict):
        _render_raw_frontier_integrity(env, raw_frontier)

    assertion_candidate_queue = status.get("assertion_candidate_queue")
    if isinstance(assertion_candidate_queue, dict):
        _render_assertion_candidate_queue(env, assertion_candidate_queue)

    sinex_publication = status.get("sinex_publication")
    if isinstance(sinex_publication, dict):
        _render_sinex_publication(env, sinex_publication)

    # Sizes
    db_bytes = status.get("db_size_bytes", 0)
    disk_free = status.get("disk_free_bytes", 0)
    if db_bytes:
        env.ui.console.print(f"  DB: {_fmt_bytes(db_bytes)}  Free: {_fmt_bytes(disk_free)}")

    raw_replay_backlog = status.get("raw_replay_backlog")
    if isinstance(raw_replay_backlog, dict):
        _render_raw_replay_backlog(env, raw_replay_backlog)

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

    if not _raw_failure_lifecycle_is_healthy(status):
        lifecycle_state = str(status.get("raw_failure_lifecycle_state") or "unavailable")
        lifecycle_reason = str(status.get("raw_failure_lifecycle_reason") or "source.db evidence is unavailable")
        if total_raw == 0 or lifecycle_state in {"unavailable", "blocked"}:
            env.ui.console.print(f"  Raw failure lifecycle: [{lifecycle_state}] {lifecycle_reason}")

    if not compact:
        checked = status.get("checked_at", "")
        if checked:
            env.ui.console.print(f"\n  [dim]Checked: {checked}[/dim]")
    return overall_ok


def _show_status_json(env: AppEnv, status: dict[str, Any], *, full: bool = False) -> bool:
    """Machine-readable JSON status output."""
    source = "direct" if status.get("daemon_liveness") is False else "daemon"
    normalized = normalize_raw_frontier_status_payload(
        status,
        snapshot_state="pinned" if source == "direct" else None,
        require_fresh_snapshot=source == "daemon",
    )
    payload = normalized if full else _compact_status_payload(normalized, source=source)
    env.ui.console.print(json.dumps(payload, indent=2, default=str))
    return _status_ok(normalized, require_fresh_snapshot=source == "daemon")


def _render_direct_status_payload(env: AppEnv, status: dict[str, Any], *, compact: bool = False) -> bool:
    """Render the already executed reader result without opening any archive."""
    env.ui.console.print("\n[bold]Archive (pinned direct snapshot)[/bold]")
    env.ui.console.print(f"  Sessions: {_safe_int(status.get('total_sessions')):,}")
    env.ui.console.print(f"  Messages: {_safe_int(status.get('total_messages')):,}")
    source_tier = status.get("archive_tiers", {}).get("source", {})
    if source_tier.get("exists"):
        env.ui.console.print(f"  Raw records: {_safe_int(source_tier.get('table_counts', {}).get('raw_sessions')):,}")
    _render_ingest_workload(env, status.get("ingest_workload", {}))
    convergence = status.get("convergence", {})
    if isinstance(convergence, dict):
        if convergence.get("available"):
            env.ui.console.print(
                "  Convergence debt: "
                f"{_safe_int(convergence.get('failed_count'))} failed, "
                f"{_safe_int(convergence.get('deferred_count'))} deferred, "
                f"{_safe_int(convergence.get('retry_due_count'))} retry due"
            )
        else:
            env.ui.console.print("  Convergence debt: unavailable")
    _render_schema_drift_status(env, status.get("schema_drift", {}))
    _render_raw_frontier_integrity(env, status.get("raw_frontier_integrity", {}))
    _render_direct_embedding_status(env, status.get("embedding_status", {}))
    if not compact:
        tier_detail = _archive_tier_detail_line(status.get("archive_tiers", {}))
        if tier_detail:
            env.ui.console.print(f"  Tiers: {tier_detail}")
        _render_sqlite_maintenance(env, status.get("sqlite_maintenance", {}))
        _render_raw_replay_backlog(env, status.get("raw_replay_backlog", {}))
        _render_archive_readiness(env, status.get("archive_readiness", {}))
        _render_assertion_candidate_queue(env, status.get("assertion_candidate_queue", {}))
        _render_sinex_publication(env, status.get("sinex_publication", {}))
    return _status_ok(status)


def _compact_status_payload(status: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Return the operator-facing status JSON without debug-heavy subtrees."""
    status = normalize_raw_frontier_status_payload(
        status,
        snapshot_state="live" if source == "direct" else None,
        require_fresh_snapshot=source == "daemon",
    )
    payload: dict[str, Any] = {
        "ok": _status_ok(status),
        "source": source,
        "daemon_liveness": bool(status.get("daemon_liveness", False)),
        "full_status_command": "polylogue ops status --json --full",
    }
    for key in (
        "checked_at",
        "archive_root",
        "active_archive_root",
        "active_archive_root_matches_configured",
        "db_exists",
        "active_db_path",
        "config_exists",
        "config_path",
        "sessions",
        "messages",
        "raw_records",
        "unidentified_artifacts",
        "next_action",
        "gil_enabled",
    ):
        if key in status:
            payload[key] = status[key]

    status_snapshot = status.get("status_snapshot")
    if isinstance(status_snapshot, dict):
        payload["status_snapshot"] = status_snapshot

    component_readiness = status.get("component_readiness")
    if isinstance(component_readiness, dict):
        payload["component_readiness"] = component_readiness

    claim_guard = status.get("claim_guard")
    if isinstance(claim_guard, dict):
        payload["claim_guard"] = claim_guard

    archive_tiers = status.get("archive_tiers")
    if isinstance(archive_tiers, dict):
        payload["archive_tiers"] = archive_tiers

    sqlite_maintenance = status.get("sqlite_maintenance")
    if isinstance(sqlite_maintenance, dict):
        payload["sqlite_maintenance"] = sqlite_maintenance

    storage = _compact_storage_status(status)
    if storage:
        payload["storage"] = storage

    sinex_publication = status.get("sinex_publication")
    if isinstance(sinex_publication, dict):
        payload["sinex_publication"] = sinex_publication

    ingest = _compact_ingest_status(status)
    if ingest:
        payload["ingest"] = ingest

    archive_debt = _compact_archive_debt_status(status.get("archive_debt"))
    if archive_debt:
        payload["archive_debt"] = archive_debt

    assertion_candidate_queue = status.get("assertion_candidate_queue")
    if isinstance(assertion_candidate_queue, dict):
        payload["assertion_candidate_queue"] = assertion_candidate_queue

    sinex_publication = status.get("sinex_publication")
    if isinstance(sinex_publication, dict):
        payload["sinex_publication"] = sinex_publication

    raw_materialization = _compact_mapping_without(
        status.get("raw_materialization_readiness"),
        {"sampled_rows"},
    )
    if raw_materialization:
        payload["raw_materialization_readiness"] = raw_materialization

    raw_frontier_integrity = _compact_mapping_without(
        status.get("raw_frontier_integrity"),
        {
            "broken_head_samples",
            "missing_source_raw_samples",
            "cursor_ahead_samples",
            "cursor_authority_gap_samples",
        },
    )
    if raw_frontier_integrity:
        payload["raw_frontier_integrity"] = raw_frontier_integrity

    raw_replay_backlog = _compact_mapping_without(
        status.get("raw_replay_backlog"),
        {"source_path_summary"},
    )
    if raw_replay_backlog:
        payload["raw_replay_backlog"] = raw_replay_backlog

    embedding_readiness = _compact_mapping_without(
        status.get("embedding_readiness"),
        {"embedding_latest_catchup_run"},
    )
    if embedding_readiness:
        payload["embedding_readiness"] = embedding_readiness

    raw_failures = _compact_raw_failure_status(status)
    if raw_failures:
        payload["raw_failures"] = raw_failures

    if "diagnostic" in status:
        payload["diagnostic"] = status["diagnostic"]

    return payload


def _status_ok(status: dict[str, Any], *, require_fresh_snapshot: bool = False) -> bool:
    """Preserve existing status health while requiring proven raw authority."""

    ok = bool(status.get("ok", status.get("daemon_liveness")))
    snapshot = status.get("status_snapshot")
    if require_fresh_snapshot and not status_snapshot_has_fresh_provenance(status):
        return False
    if isinstance(snapshot, dict) and snapshot.get("state") not in {None, "fresh"}:
        return False
    if not _raw_failure_lifecycle_is_healthy(status):
        return False
    return ok and raw_frontier_integrity_is_proven_healthy(status.get("raw_frontier_integrity"))


def _render_raw_frontier_integrity(env: AppEnv, integrity: dict[str, Any]) -> None:
    overall = str(integrity.get("overall_status") or "unknown")
    color = {"healthy": "green", "violated": "red", "unknown": "yellow"}.get(overall, "yellow")
    summary = raw_frontier_integrity_summary(integrity)
    detail = "" if summary == "ready" else f" — {summary}"
    env.ui.console.print(f"  Raw frontier: [{color}]{overall}[/{color}]{detail}")


def _compact_mapping_without(value: Any, heavy_keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if key not in heavy_keys}


def _compact_storage_status(status: dict[str, Any]) -> dict[str, Any]:
    storage: dict[str, Any] = {}
    archive_storage = status.get("archive_storage")
    if isinstance(archive_storage, dict):
        for key in (
            "state",
            "active_store",
            "active_tier_role",
            "present_tiers",
            "missing_tiers",
            "blockers",
            "identity",
            "identity_conflicts",
        ):
            if key in archive_storage:
                storage[key] = archive_storage[key]
    for key in (
        "db_path",
        "db_size_bytes",
        "wal_size_bytes",
        "blob_dir_size_bytes",
        "disk_free_bytes",
        "active_db_path",
    ):
        if key in status:
            storage[key] = status[key]
    return storage


def _compact_ingest_status(status: dict[str, Any]) -> dict[str, Any]:
    live = status.get("live_ingest_attempts")
    if isinstance(live, dict):
        ingest: dict[str, Any] = {}
        for key in (
            "total",
            "running_count",
            "completed_count",
            "failed_count",
            "stale_running_count",
            "stuck_running_count",
        ):
            if key in live:
                ingest[key] = live[key]
        recent = live.get("recent")
        if isinstance(recent, list) and recent:
            latest = recent[0]
            if isinstance(latest, dict):
                ingest["latest"] = {
                    key: latest[key]
                    for key in (
                        "attempt_id",
                        "status",
                        "stage",
                        "started_at",
                        "updated_at",
                        "completed_at",
                        "worker_completed_count",
                        "worker_total_count",
                    )
                    if key in latest
                }
        return ingest
    workload = status.get("ingest_workload")
    if isinstance(workload, dict):
        return workload
    return {}


def _compact_archive_debt_status(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {key: item for key, item in value.items() if key != "rows"}
    if value.get("rows") is not None:
        rows = value.get("rows")
        compact["row_count"] = len(rows) if isinstance(rows, list) else None
    return compact


def _compact_raw_failure_status(status: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "parse": "raw_parse_failures",
        "validation": "raw_validation_failures",
        "quarantined": "raw_quarantined",
        "deferred_retryable": "raw_deferred_failures",
        "terminal_rejections": "raw_terminal_rejections",
        "unexplained": "raw_unexplained_failures",
        "detection_warnings": "raw_detection_warnings",
        "lifecycle_available": "raw_failure_lifecycle_available",
        "lifecycle_state": "raw_failure_lifecycle_state",
        "lifecycle_reason": "raw_failure_lifecycle_reason",
    }
    failures = {label: status[key] for label, key in keys.items() if key in status}
    samples = status.get("raw_failure_samples")
    if isinstance(samples, list):
        failures["sample_count"] = len(samples)
    return failures


def _raw_failure_lifecycle_is_healthy(status: dict[str, Any]) -> bool:
    """Require explicit, clean source-tier lifecycle evidence for green status."""
    return (
        status.get("raw_failure_lifecycle_available") is True
        and status.get("raw_failure_lifecycle_state") == "healthy"
        and status.get("raw_parse_failures") == 0
        and status.get("raw_validation_failures") == 0
        and status.get("raw_unexplained_failures") == 0
    )


def _show_daemon_status_unavailable_json(env: AppEnv) -> None:
    payload = {
        "daemon_liveness": True,
        "status_snapshot": {
            "state": "unavailable",
            "reason": "api_status_timeout",
        },
    }
    env.ui.console.print(json.dumps(_compact_status_payload(payload, source="daemon"), indent=2, default=str))


def _show_daemon_status_unavailable(env: AppEnv, *, compact: bool = False) -> None:
    env.ui.console.print("\n[bold yellow]Daemon: running[/bold yellow]")
    env.ui.console.print("  Status snapshot: [yellow]unavailable[/yellow]")
    if not compact:
        env.ui.console.print("  [dim]/api/status did not answer within the bounded CLI timeout.[/dim]")


def _render_ingest_workload(env: AppEnv, workload: dict[str, Any]) -> None:
    """Render live ingest-workload state derived from ops.db."""
    if not workload.get("available"):
        reason = str(workload.get("reason") or "workload evidence is unavailable")
        env.ui.console.print(f"  Ingest workload: [yellow]unavailable[/yellow] — {reason}")
        return
    tput = workload.get("throughput") or {}
    files = int(tput.get("files", 0) or 0)
    rate = float(tput.get("files_per_second", 0.0) or 0.0)
    window = int(tput.get("window_minutes", 0) or 0)
    running = int(workload.get("running_count", 0) or 0)
    active = bool(workload.get("actively_ingesting"))
    debt = workload.get("debt") or {}
    debt_total = int(debt.get("total", 0) or 0)
    # Nothing live or recent to show — stay quiet.
    if not active and running == 0 and files == 0 and debt_total == 0:
        return

    state_color = "green" if active else "yellow" if running else "dim"
    state_text = "ingesting" if active else "idle (stale running attempt)" if running else "idle"
    env.ui.console.print(f"  Ingest workload: [{state_color}]{state_text}[/{state_color}]")

    if running:
        parts: list[str] = []
        for entry in (workload.get("running") or [])[:3]:
            phase = entry.get("phase") or "unknown"
            hb = entry.get("heartbeat_age_ms")
            parts.append(f"{phase} ({int(hb) // 1000}s ago)" if hb is not None else str(phase))
        env.ui.console.print(f"    in-flight: {running} attempt(s) — {', '.join(parts)}")

    if files or rate:
        env.ui.console.print(
            f"    throughput (last {window}m): {files:,} files, "
            f"{int(tput.get('batches', 0) or 0)} batches, {rate:.2f} files/s"
        )

    cursor = workload.get("cursor") or {}
    if cursor:
        line = f"    coverage: {int(cursor.get('tracked', 0) or 0):,} files tracked"
        retry = int(cursor.get("retry_pending", 0) or 0)
        excluded = int(cursor.get("excluded", 0) or 0)
        if retry:
            line += f", {retry} retry-pending"
        if excluded:
            line += f", {excluded} excluded"
        env.ui.console.print(line)

    if debt_total:
        detail = ", ".join(f"{status}={count}" for status, count in (debt.get("by_status") or {}).items())
        env.ui.console.print(f"    convergence debt: [yellow]{debt_total}[/yellow] ({detail})")


def _render_schema_drift_status(env: AppEnv, drift: dict[str, Any]) -> None:
    """Render windowed format-drift rates: 'origin X: N% ... carry unseen shapes'.

    Follow-up action always points at schema generate/promote (#polylogue-da1)
    -- the sentinel only detects; it never repairs the schema package.
    """
    if not drift.get("available"):
        return
    origins = drift.get("origins") or []
    if not isinstance(origins, list) or not origins:
        return
    since_ms = int(drift.get("since_ms", 0) or 0)
    since_date = datetime.fromtimestamp(since_ms / 1000, tz=UTC).date().isoformat() if since_ms else "unknown"
    noteworthy = [item for item in origins if isinstance(item, dict) and item.get("severity") != "ok"]
    if not noteworthy:
        return
    env.ui.console.print("  Format drift sentinel:")
    for item in sorted(noteworthy, key=lambda entry: float(entry.get("risky_rate", 0.0) or 0.0), reverse=True):
        origin = item.get("origin") or "unknown"
        total = int(item.get("total", 0) or 0)
        risky_rate = float(item.get("risky_rate", 0.0) or 0.0)
        severity = str(item.get("severity") or "ok")
        color = "red" if severity == "error" else "yellow"
        examples = item.get("example_native_ids") or []
        example_text = f" e.g. {', '.join(str(x) for x in examples[:3])}" if examples else ""
        env.ui.console.print(
            f"    [{color}]origin {origin}: {risky_rate:.0%} of {total} records since {since_date} "
            f"carry unseen shapes[/{color}]{example_text}"
        )
    env.ui.console.print("    next action: devtools schema generate/promote (detection only)")


def _render_raw_replay_backlog(env: AppEnv, backlog: dict[str, Any]) -> None:
    """Render weighted raw materialization replay backlog."""
    if not backlog.get("available"):
        return
    candidates = _safe_int(backlog.get("candidate_count"))
    missing = _safe_int(backlog.get("missing_blob_count"))
    if candidates <= 0 and missing <= 0:
        return
    total_bytes = _safe_int(backlog.get("total_blob_bytes"))
    max_bytes = _safe_int(backlog.get("max_blob_bytes"))
    oversized = _safe_int(backlog.get("oversized_count"))
    line = f"  Raw replay backlog: [yellow]{candidates:,} raw row(s), {_fmt_bytes(total_bytes)} pending"
    if max_bytes:
        line += f"; largest {_fmt_bytes(max_bytes)}"
    if missing:
        line += f"; {missing:,} missing blob(s)"
    if oversized:
        line += f"; {oversized:,} oversized"
    line += "[/yellow]"
    env.ui.console.print(line)
    block_reason = backlog.get("execution_block_reason")
    if backlog.get("execution_blocked") and isinstance(block_reason, str) and block_reason:
        env.ui.console.print(f"    [yellow]{block_reason}[/yellow]")

    origins = backlog.get("origin_summary")
    if isinstance(origins, list) and origins:
        parts: list[str] = []
        for item in origins[:3]:
            if not isinstance(item, dict):
                continue
            origin = item.get("origin") or "unknown"
            raw_count = _safe_int(item.get("raw_count"))
            blob_bytes = _safe_int(item.get("total_blob_bytes"))
            parts.append(f"{origin}={raw_count:,}/{_fmt_bytes(blob_bytes)}")
        if parts:
            env.ui.console.print(f"    weighted by origin: {', '.join(parts)}")


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
    material = payload.get("latest_material_catchup_run")
    if isinstance(material, dict) and (not isinstance(latest, dict) or material.get("run_id") != latest.get("run_id")):
        processed = int(material.get("processed_sessions", 0) or 0)
        planned = int(material.get("planned_sessions", 0) or 0)
        embedded = int(material.get("embedded_messages", 0) or 0)
        errors = int(material.get("error_count", 0) or 0)
        env.ui.console.print(
            "  Embedding material catch-up: "
            f"{material.get('status', 'unknown')}, {processed:,}/{planned:,} convs, "
            f"{embedded:,} msgs embedded, {errors:,} errors"
        )


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


def _render_assertion_candidate_queue(env: AppEnv, queue: dict[str, Any]) -> None:
    """Render the same queue-health product used by root judge and daemon JSON."""

    state = str(queue.get("state") or "unavailable")
    pending = _safe_int(queue.get("pending_count"))
    color = "green" if state == "healthy-empty" else "yellow"
    if state in {"producer-stalled", "scheduler-stalled", "parked-pending", "stale-pending", "unavailable"}:
        color = "red"
    line = f"  Assertion candidate queue: [{color}]{state}, {pending} pending[/{color}]"
    oldest_age = queue.get("oldest_pending_age_ms")
    if isinstance(oldest_age, int | float):
        line += f", oldest={float(oldest_age) / (24 * 60 * 60 * 1000):.1f}d"
    env.ui.console.print(line)
    receipt_status = queue.get("judgment_scheduler_receipt_status")
    if receipt_status is not None:
        receipt_details = [str(receipt_status)]
        receipt_at_ms = queue.get("judgment_scheduler_receipt_at_ms")
        if isinstance(receipt_at_ms, int | float) and not isinstance(receipt_at_ms, bool):
            with suppress(OverflowError, OSError, ValueError):
                receipt_timestamp = datetime.fromtimestamp(float(receipt_at_ms) / 1000, tz=UTC).isoformat()
                receipt_details.append(f"at={receipt_timestamp}")
        receipt_age_ms = queue.get("judgment_scheduler_receipt_age_ms")
        if isinstance(receipt_age_ms, int | float) and not isinstance(receipt_age_ms, bool):
            receipt_age_s = float(receipt_age_ms) / 1000
            if receipt_age_s >= 24 * 60 * 60:
                receipt_details.append(f"age={receipt_age_s / (24 * 60 * 60):.1f}d")
            else:
                receipt_details.append(f"age={receipt_age_s:.1f}s")
        receipt_reason = queue.get("judgment_scheduler_receipt_reason")
        if receipt_reason:
            receipt_details.append(f"reason={receipt_reason}")
        retryable = queue.get("judgment_scheduler_receipt_retryable")
        if isinstance(retryable, bool):
            receipt_details.append(f"retryable={retryable}")
        retry_route = queue.get("judgment_scheduler_receipt_retry_route")
        if retry_route is not None:
            receipt_details.append(f"route={retry_route}")
        batch_limit = queue.get("judgment_scheduler_receipt_batch_limit")
        if isinstance(batch_limit, int) and not isinstance(batch_limit, bool):
            receipt_details.append(f"batch={batch_limit}")
        counter_parts = []
        for name in ("considered", "accepted", "rejected", "escalated", "idempotent", "failed"):
            value = queue.get(f"judgment_scheduler_receipt_{name}")
            if value is not None:
                counter_parts.append(f"{name}={value}")
        if counter_parts:
            receipt_details.append("counts=" + ",".join(counter_parts))
        degraded = queue.get("judgment_scheduler_receipt_persistence_degraded")
        recovered = queue.get("judgment_scheduler_receipt_persistence_recovered")
        if isinstance(degraded, bool) or isinstance(recovered, bool):
            receipt_details.append(f"persistence_degraded={degraded};persistence_recovered={recovered}")
        env.ui.console.print(f"    judgment scheduler receipt: {', '.join(receipt_details)}")


def _render_sinex_publication(env: AppEnv, status: dict[str, Any]) -> None:
    """Render durable Sinex publication lag without exposing payload details."""
    mode = str(status.get("mode") or "off")
    if status.get("state") == "unavailable":
        env.ui.console.print(f"  Sinex publication: [yellow]unavailable ({status.get('reason', 'unknown')})[/yellow]")
        return
    lag = _safe_int(status.get("active_lag"))
    blocking = _safe_int(status.get("blocking"))
    color = "green" if lag == 0 and blocking == 0 else "yellow"
    if blocking:
        color = "red"
    env.ui.console.print(
        f"  Sinex publication: [{color}]{mode}, lag={lag}, blocking={blocking}, "
        f"retry_due={_safe_int(status.get('retry_due'))}[/{color}]"
    )


def _render_sqlite_maintenance(env: AppEnv, status: dict[str, Any]) -> None:
    wal_bytes = _safe_int(status.get("total_wal_bytes"))
    planner_tiers = [str(tier) for tier in status.get("tiers_with_planner_stats") or []]
    planner_text = ",".join(planner_tiers) if planner_tiers else "none"
    color = "green" if wal_bytes == 0 and planner_tiers else "yellow"
    env.ui.console.print(
        f"  SQLite maintenance: [{color}]WAL {_fmt_bytes(wal_bytes)}, planner stats={planner_text}[/{color}]"
    )


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
        "user": "assertions",
        "audit": "mutation_attempts",
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
