"""Status command — query daemon health and archive state."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

import click

from polylogue.cli.shared.types import AppEnv

_DEFAULT_DAEMON_URL = "http://127.0.0.1:8766"
_FAST_TIMEOUT_S = 1.0
_FULL_TIMEOUT_S = 5.0


@click.command("status")
@click.option(
    "--daemon-url",
    default=_DEFAULT_DAEMON_URL,
    show_default=True,
    help="Daemon API URL.",
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
    req = Request(
        f"{daemon_url}/api/status",
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urlopen(req, timeout=_FULL_TIMEOUT_S) as resp:
            result = json.loads(resp.read())
    except OSError:
        if output_format == "json":
            _show_direct_json(env)
        else:
            _show_direct_status(env)
        return

    if output_format == "json":
        _show_status_json(env, result)
    else:
        _show_daemon_status(env, result)


def show_fast_status(env: AppEnv, *, daemon_url: str = _DEFAULT_DAEMON_URL) -> None:
    """Fast bare-invocation status: try daemon, fall back to local SQLite.

    Called from ``polylogue`` with no args. Uses a short HTTP timeout
    and bounded SQLite queries to stay under 2 seconds.
    """
    try:
        req = Request(
            f"{daemon_url}/api/status",
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
        pct = fts.get("coverage_pct", 0)
        fts_color = "green" if pct > 99 else "yellow"
        env.ui.console.print(f"  FTS: [{fts_color}]{pct:.1f}% indexed[/{fts_color}]")

    # Sizes
    db_bytes = status.get("db_size_bytes", 0)
    disk_free = status.get("disk_free_bytes", 0)
    if db_bytes:
        env.ui.console.print(f"  DB: {_fmt_bytes(db_bytes)}  Free: {_fmt_bytes(disk_free)}")

    if not compact:
        checked = status.get("checked_at", "")
        if checked:
            env.ui.console.print(f"\n  [dim]Checked: {checked}[/dim]")


def _show_status_json(env: AppEnv, status: dict[str, Any]) -> None:
    """Machine-readable JSON status output."""
    env.ui.console.print(json.dumps(status, indent=2, default=str))


def _show_direct_json(env: AppEnv) -> None:
    """Machine-readable JSON fallback when daemon is not running."""
    from polylogue.paths import archive_root, db_path

    db = db_path()
    payload: dict[str, Any] = {
        "daemon_liveness": False,
        "archive_root": str(archive_root()),
        "db_exists": db.exists(),
    }
    if db.exists():
        try:
            from polylogue.storage.sqlite.connection_profile import open_connection

            conn = open_connection(db)
            try:
                payload["conversations"] = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
                payload["messages"] = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                payload["raw_records"] = conn.execute("SELECT COUNT(*) FROM raw_conversations").fetchone()[0]
            finally:
                conn.close()
        except Exception as exc:
            payload["error"] = str(exc)
    env.ui.console.print(json.dumps(payload, indent=2, default=str))


def _show_direct_status(env: AppEnv, *, compact: bool = False) -> None:
    """Fallback status when daemon is not running."""
    from polylogue.paths import archive_root, db_path

    db = db_path()
    if not db.exists():
        env.ui.console.print(
            "\n[yellow]No archive found.[/yellow] Start the daemon with [bold]polylogued run[/bold] to begin ingestion."
        )
        return

    try:
        from polylogue.storage.sqlite.connection_profile import open_connection

        conn = open_connection(db)
        try:
            convs = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            raw = conn.execute("SELECT COUNT(*) FROM raw_conversations").fetchone()[0]
            fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        finally:
            conn.close()

        env.ui.console.print("\n[bold]Archive (daemon not running)[/bold]")
        env.ui.console.print(f"  Conversations: {convs:,}")
        env.ui.console.print(f"  Messages: {msgs:,}")
        env.ui.console.print(f"  Raw records: {raw:,}")
        fts_pct = 100 * fts / msgs if msgs else 100
        fts_color = "green" if fts_pct > 99 else "yellow"
        env.ui.console.print(f"  FTS indexed: [{fts_color}]{fts_pct:.1f}%[/{fts_color}]")
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
