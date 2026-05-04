"""Status command — query daemon health and archive state."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

import click

from polylogue.cli.shared.types import AppEnv

_DEFAULT_DAEMON_URL = "http://127.0.0.1:8766"


@click.command("status")
@click.option(
    "--daemon-url",
    default=_DEFAULT_DAEMON_URL,
    show_default=True,
    help="Daemon API URL.",
)
@click.pass_obj
def status_command(
    env: AppEnv,
    daemon_url: str,
) -> None:
    """Show daemon and archive health.

    Queries the running polylogued daemon for archive status: daemon
    uptime, ingestion progress, FTS coverage, insight freshness, and
    component health. Read-only — does not modify state.
    """
    req = Request(
        f"{daemon_url}/api/status",
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
    except OSError:
        # Daemon not running — fall back to direct archive check
        _show_direct_status(env)
        return

    _show_daemon_status(env, result)


def _show_daemon_status(env: AppEnv, status: dict) -> None:
    overall = status.get("overall", "unknown")
    color = {"ok": "green", "warn": "yellow", "err": "red", "idle": "blue"}.get(overall, "white")
    env.ui.console.print(f"\n[bold {color}]Status: {overall}[/bold {color}]")

    pid = status.get("pid")
    uptime = status.get("uptime_s")
    if pid:
        env.ui.console.print(f"  Daemon PID: {pid}")
    if uptime is not None:
        env.ui.console.print(f"  Uptime: {uptime:.0f}s")

    components = status.get("components", [])
    if components:
        env.ui.console.print("\n[bold]Components:[/bold]")
        for c in components:
            state_color = {
                "ok": "green",
                "warn": "yellow",
                "err": "red",
                "stale": "yellow",
                "idle": "blue",
                "disabled": "dim",
            }.get(c.get("state"), "white")
            env.ui.console.print(f"  [{state_color}]●[/{state_color}] {c['name']}: {c.get('summary', '')}")

    env.ui.console.print(f"\n  Archive: {status.get('archive_display_path', 'unknown')}")
    version = status.get("version")
    if version:
        env.ui.console.print(f"  Version: {version}")


def _show_direct_status(env: AppEnv) -> None:
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
    except Exception:
        env.ui.console.print(f"\n[yellow]Archive exists at {archive_root()} but could not be queried.[/yellow]")


__all__ = ["status_command"]
