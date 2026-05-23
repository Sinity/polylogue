"""Status command — query daemon health and archive state."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen

import click

from polylogue.cli.shared.types import AppEnv

_BUILTIN_DAEMON_URL = "http://127.0.0.1:8766"
# Bare `polylogue` uses this as a quick probe before falling back to SQLite.
_FAST_TIMEOUT_S = 1.0
# Explicit status is still an operator command; a busy local daemon should fall
# through to bounded read-only SQLite status instead of hiding the archive.
_FULL_TIMEOUT_S = 3.0


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


# Backwards-compatible name retained for the (rare) external import.
_DEFAULT_DAEMON_URL = _BUILTIN_DAEMON_URL


def _fast_count(conn: Any, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _fast_fts_doc_count(conn: Any) -> int:
    try:
        return _fast_count(conn, "SELECT COUNT(*) FROM messages_fts_docsize")
    except Exception:
        return 0


def _table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _fast_message_count(conn: Any) -> int:
    if _table_exists(conn, "conversation_stats"):
        return _fast_count(conn, "SELECT COALESCE(SUM(message_count), 0) FROM conversation_stats")
    return _fast_count(conn, "SELECT COUNT(*) FROM messages")


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
@click.pass_obj
def status_command(
    env: AppEnv,
    daemon_url: str,
    output_format: str | None,
) -> None:
    """Show daemon and archive health.

    Queries the running polylogued daemon for archive status: daemon
    liveness, ingestion progress, FTS coverage, insight freshness, and
    component health. Read-only — does not modify state.
    """
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
            _show_direct_json(env)
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

    # Live ingest
    live = status.get("live_ingest_attempts", {})
    if isinstance(live, dict):
        completed = live.get("completed_count", 0)
        total = live.get("total_count", 0)
        in_flight = live.get("worker_in_flight_count", 0)
        if total:
            parts = [f"{completed}/{total} done"]
            if in_flight:
                parts.append(f"+{in_flight} in-flight")
            env.ui.console.print(f"  Ingest: {', '.join(parts)}")

    # FTS
    fts = status.get("fts_readiness", {})
    if isinstance(fts, dict):
        pct = float(fts.get("coverage_pct", 100 if fts.get("messages_ready") else 0))
        fts_color = "green" if fts.get("messages_ready") and fts.get("action_events_ready") else "yellow"
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


def _show_direct_json(env: AppEnv) -> None:
    """Machine-readable JSON fallback when daemon is not running."""
    from polylogue.cli.commands.init import starter_config_path
    from polylogue.cli.commands.status_diagnostics import (
        diagnose_first_run,
        diagnostic_payload,
    )
    from polylogue.paths import archive_root, db_path

    db = db_path()
    config_path = starter_config_path()
    diag = diagnose_first_run(daemon_alive=False)
    payload: dict[str, Any] = {
        "daemon_liveness": False,
        "archive_root": str(archive_root()),
        "db_exists": db.exists(),
        "config_exists": config_path.exists(),
        "config_path": str(config_path),
        "next_action": diag.next_action,
        "diagnostic": diagnostic_payload(diag),
    }
    if db.exists():
        try:
            from polylogue.storage.sqlite.connection_profile import open_readonly_connection

            conn = open_readonly_connection(db)
            try:
                payload["conversations"] = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
                payload["messages"] = _fast_message_count(conn)
                payload["raw_records"] = conn.execute("SELECT COUNT(*) FROM raw_conversations").fetchone()[0]
            finally:
                conn.close()
        except Exception as exc:
            payload["error"] = str(exc)
    env.ui.console.print(json.dumps(payload, indent=2, default=str))


def _render_diagnostic(env: AppEnv, diag: Any) -> None:
    """Render a ``StatusDiagnostic`` with rich tags but no traceback."""
    color = "red" if diag.kind in {"schema_mismatch", "locked_db", "unknown_db_error"} else "yellow"
    env.ui.console.print(f"\n[{color}]{diag.headline}[/{color}]")
    if diag.detail:
        env.ui.console.print(f"  {diag.detail}")


def _show_direct_status(env: AppEnv, *, compact: bool = False) -> None:
    """Fallback status when daemon is not running."""
    from polylogue.cli.commands.status_diagnostics import diagnose_first_run
    from polylogue.paths import archive_root, db_path

    db = db_path()
    if not db.exists():
        diag = diagnose_first_run(daemon_alive=False)
        _render_diagnostic(env, diag)
        return

    # Pre-flight: detect schema mismatch / locked db / stale pidfile
    # before attempting row counts. Short-circuits with actionable text
    # rather than a Python traceback (#1263).
    diag = diagnose_first_run(daemon_alive=False)
    if diag.kind in {"schema_mismatch", "locked_db", "stale_pidfile"}:
        _render_diagnostic(env, diag)
        return

    try:
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        conn = open_readonly_connection(db)
        try:
            convs = _fast_count(conn, "SELECT COUNT(*) FROM conversations")
            msgs = _fast_message_count(conn)
            raw = _fast_count(conn, "SELECT COUNT(*) FROM raw_conversations")
            fts = _fast_fts_doc_count(conn)
        finally:
            conn.close()

        env.ui.console.print("\n[bold]Archive (daemon not running)[/bold]")
        env.ui.console.print(f"  Conversations: {convs:,}")
        env.ui.console.print(f"  Messages: {msgs:,}")
        env.ui.console.print(f"  Raw records: {raw:,}")
        fts_pct = 100 * fts / msgs if msgs else 100
        fts_color = "green" if fts_pct > 99 else "yellow"
        env.ui.console.print(f"  FTS indexed: [{fts_color}]{fts_pct:.1f}%[/{fts_color}]")

        try:
            from polylogue.storage.embeddings.status_payload import embedding_status_payload

            ep = embedding_status_payload(env, include_retrieval_bands=False)
            if ep["total_conversations"] > 0:
                emb_pct = ep["embedding_coverage_percent"]
                emb_color = "green" if ep["retrieval_ready"] else "dim"
                env.ui.console.print(
                    f"  Embeddings: [{emb_color}]{ep['embedded_messages']:,} msgs, "
                    f"{ep['embedded_conversations']:,}/{ep['total_conversations']:,} convs "
                    f"({emb_pct:.1f}%)[/{emb_color}]"
                )
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


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{n / 1_000:.0f} KB"


__all__ = ["status_command", "show_fast_status"]
