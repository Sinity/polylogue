"""Dashboard command."""

from __future__ import annotations

import json

import click

from polylogue.cli.daemon_probe import daemon_serving_probe
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
    evidence = _dashboard_launch_evidence(env)
    if status_only:
        _emit_dashboard_evidence(evidence, output_format=output_format)
        return
    if output_format == "json":
        raise click.UsageError("dashboard --format json requires --status.")
    _emit_dashboard_evidence(evidence, output_format="text")
    from polylogue.ui.tui.app import PolylogueApp

    app = PolylogueApp(polylogue=env.polylogue)
    app.run()


def _dashboard_launch_evidence(env: AppEnv) -> dict[str, object]:
    from polylogue.cli.shared.helpers import load_effective_config
    from polylogue.config import load_polylogue_config

    daemon_url = load_polylogue_config().daemon_url or "http://127.0.0.1:8766"
    status: dict[str, object] = {
        "surface": "terminal_tui",
        "launches": "Textual dashboard in the current terminal",
        "daemon_api_url": daemon_url,
        "reader_surface": "terminal_tui",
        "web_reader_url": None,
        "web_reader_launch_attempted": False,
        "daemon_api_reachable": False,
        "failure_reason": None,
    }
    # Reachability is read off the probe's authority, not off the call
    # succeeding: the ``status`` operation falls back to a direct in-process
    # read, which would otherwise report "daemon reachable" with no daemon
    # running at all. See polylogue.cli.daemon_probe.
    served_by_daemon, failure_reason = daemon_serving_probe(load_effective_config(env))
    status["daemon_api_reachable"] = served_by_daemon
    status["failure_reason"] = failure_reason
    return status


def _emit_dashboard_evidence(evidence: dict[str, object], *, output_format: str) -> None:
    if output_format == "json":
        click.echo(json.dumps(evidence, indent=2, sort_keys=True))
        return
    click.echo("Dashboard surface: terminal TUI (Textual)")
    click.echo(f"Daemon status-operation probe (configured API URL: {evidence['daemon_api_url']})")
    click.echo("Web reader launch: not attempted by this command")
    if evidence["daemon_api_reachable"]:
        click.echo("Readiness: daemon API reachable")
    else:
        click.echo(f"Readiness: degraded ({evidence['failure_reason']})")
        click.echo("Prerequisite: start the daemon with `polylogued run` for live ingestion and API-backed reads.")
        click.echo("The dashboard below still reads the archive directly, without the daemon.")
        click.echo("CLI fallback: `polylogue find QUERY then read` needs no daemon and no TUI at all.")
    click.echo("Launching Textual dashboard in this terminal.")


__all__ = ["dashboard_command"]
