"""Daemon commands, loading runtime implementations only when selected."""

from __future__ import annotations

from importlib import import_module

import click

from polylogue.core.json import JSONDocument, dumps, json_document

_COMMANDS = {
    "api": ("polylogue.daemon.api_auth", "api_command", "Run the Polylogue HTTP API server."),
    "browser-capture": (
        "polylogue.daemon.browser_capture",
        "browser_capture_command",
        "Run the browser capture receiver.",
    ),
    "health": ("polylogue.daemon.cli", "health_command", "Run tiered daemon health checks."),
    "run": ("polylogue.daemon.cli", "run_command", "Run configured long-lived daemon components."),
    "watch": ("polylogue.daemon.cli", "watch_command", "Watch source directories and ingest new sessions live."),
}


class DaemonCommandGroup(click.Group):
    """Register daemon command implementations without loading other commands."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(self.commands) | set(_COMMANDS))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is None and cmd_name in _COMMANDS:
            module, attribute, _help = _COMMANDS[cmd_name]
            command = getattr(import_module(module), attribute)
            self.add_command(command, cmd_name)
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = []
        for name in self.list_commands(ctx):
            command = self.commands.get(name)
            help_text = command.get_short_help_str() if command is not None else _COMMANDS[name][2]
            rows.append((name, help_text))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


def _show_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    from polylogue.version import POLYLOGUE_VERSION

    click.echo(f"polylogued, version {POLYLOGUE_VERSION}")
    ctx.exit()


@click.group(cls=DaemonCommandGroup, help="Run long-lived Polylogue local services.")
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_show_version,
    help="Show the version and exit.",
)
def main() -> None:
    from polylogue.runtime import require_free_threaded_runtime

    require_free_threaded_runtime(consumer="polylogued")


def _live_daemon_status_payload() -> JSONDocument:
    """Read the daemon's status once over its peer-verified machine socket."""
    from polylogue.cli.operation_kernel import (
        OperationKernelError,
        OperationRequest,
        OperationUnavailableError,
        dispatch,
    )
    from polylogue.config import load_polylogue_config

    try:
        result = dispatch(load_polylogue_config(), OperationRequest("status", {}), daemon_only=True)
    except OperationKernelError as exc:
        reason = (
            "daemon_absent"
            if isinstance(exc, OperationUnavailableError)
            else getattr(exc, "code", "status_read_failed")
        )
        snapshot: JSONDocument = {"state": "unavailable", "reason": reason, "detail": str(exc)}
        if request_id := getattr(exc, "request_id", None):
            snapshot["request_id"] = request_id
        return {
            "daemon": "polylogued",
            "ok": False,
            "daemon_liveness": False if isinstance(exc, OperationUnavailableError) else None,
            "status_snapshot": snapshot,
        }
    return json_document(result.value)


@main.command("status", help="Show configured daemon component status.")
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def status_command(output_format: str | None) -> None:
    payload = _live_daemon_status_payload()
    if output_format == "json":
        click.echo(dumps(payload))
    elif "status_snapshot" in payload and payload.get("ok") is False and "archive_storage" not in payload:
        snapshot = payload["status_snapshot"]
        if not isinstance(snapshot, dict):
            raise click.ClickException("daemon returned invalid status metadata")
        click.echo(f"Polylogue daemon: {snapshot['state']}")
        for key, label in (("reason", "Reason"), ("detail", "Detail"), ("request_id", "Request")):
            if value := snapshot.get(key):
                click.echo(f"{label}: {value}")
    else:
        from polylogue.daemon.status import format_daemon_status_lines

        for line in format_daemon_status_lines(payload):
            click.echo(line)
    if payload.get("ok") is not True:
        raise SystemExit(1)
