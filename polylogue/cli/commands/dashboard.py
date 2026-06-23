"""Dashboard command."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

import click

from polylogue.cli.shared.types import AppEnv


@click.command("dashboard")
@click.option(
    "--status",
    "status_only",
    is_flag=True,
    help="Print dashboard launch/readiness evidence without starting the TUI.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format for --status and launch evidence.",
)
@click.pass_obj
def dashboard_command(env: AppEnv, status_only: bool, output_format: str) -> None:
    """Launch the terminal dashboard TUI with explicit runtime evidence."""
    evidence = _dashboard_launch_evidence()
    if status_only:
        _emit_dashboard_evidence(evidence, output_format=output_format)
        return
    if output_format == "json":
        raise click.UsageError("dashboard --format json requires --status.")
    _emit_dashboard_evidence(evidence, output_format="text")
    from polylogue.ui.tui.app import PolylogueApp

    app = PolylogueApp(polylogue=env.polylogue)
    app.run()


def _dashboard_launch_evidence() -> dict[str, object]:
    daemon_url = os.environ.get("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:8766")
    status: dict[str, object] = {
        "surface": "terminal_tui",
        "launches": "Textual dashboard in the current terminal",
        "daemon_api_url": daemon_url,
        "web_reader_url": daemon_url,
        "daemon_api_reachable": False,
        "failure_reason": None,
    }
    try:
        req = Request(f"{daemon_url}/api/status", method="GET")
        with urlopen(req, timeout=0.5) as resp:
            resp.read(1)
        status["daemon_api_reachable"] = True
    except Exception as exc:
        status["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return status


def _emit_dashboard_evidence(evidence: dict[str, object], *, output_format: str) -> None:
    if output_format == "json":
        click.echo(json.dumps(evidence, indent=2, sort_keys=True))
        return
    click.echo("Dashboard surface: terminal TUI")
    click.echo(f"Daemon API: {evidence['daemon_api_url']}")
    click.echo(f"Web reader: {evidence['web_reader_url']}")
    if evidence["daemon_api_reachable"]:
        click.echo("Readiness: daemon API reachable")
    else:
        click.echo(f"Readiness: degraded ({evidence['failure_reason']})")
    click.echo("Launching Textual dashboard in this terminal.")


__all__ = ["dashboard_command"]
