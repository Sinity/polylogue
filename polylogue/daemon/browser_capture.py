"""Daemon commands for the local browser-capture receiver."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import click
from pydantic import ValidationError

from polylogue.browser_capture.models import BrowserPostCommandRequest, BrowserPostProvider, BrowserPostTarget
from polylogue.browser_capture.receiver import (
    BROWSER_POST_ENABLED_ENV,
    BrowserPostDisabledError,
    browser_post_enabled,
    enqueue_post_command,
)
from polylogue.browser_capture.server import make_server
from polylogue.core.json import dumps
from polylogue.daemon.status import browser_capture_status_payload


@click.group("browser-capture")
def browser_capture_command() -> None:
    """Run and inspect the browser-capture receiver."""


@browser_capture_command.command("status")
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def status_command(spool_path: Path | None, output_format: str | None) -> None:
    """Show receiver configuration and capture-spool target."""
    payload = browser_capture_status_payload(spool_path, include_spool_path=True)
    if output_format == "json":
        click.echo(dumps(payload))
        return
    click.echo("Browser capture receiver")
    click.echo(f"Spool: {'ready' if payload.get('spool_ready') else 'unavailable'}")
    origins = payload.get("allowed_origins", [])
    origin_text = ", ".join(str(item) for item in origins) if isinstance(origins, list) else str(origins)
    click.echo(f"Allowed origins: {origin_text}")


@browser_capture_command.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
def serve_command(host: str, port: int, spool_path: Path | None) -> None:
    """Run the local browser-capture receiver."""
    server = make_server(host, port, spool_path=spool_path)
    click.echo(f"Listening on http://{host}:{port}")
    click.echo(f"Writing captures to {server.config.spool_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopping browser capture receiver")
    finally:
        server.server_close()


@browser_capture_command.command("post")
@click.option("--provider", type=click.Choice(list(get_args(BrowserPostProvider))), required=True)
@click.option(
    "--conversation-id",
    "conversation_id",
    default="new",
    show_default=True,
    help='Provider-native conversation id, or "new" to request a fresh thread.',
)
@click.option("--project-ref", "project_ref", default=None, help="Optional provider project/workspace ref.")
@click.option("--text", required=True, help="Prompt text to post into the conversation.")
@click.option("--command-id", "command_id", default=None, help="Optional explicit command id.")
@click.option(
    "--submit/--no-submit",
    "submit",
    default=False,
    show_default=True,
    help="Request the extension to actually click send. Default is a dry-run that fills the composer only.",
)
@click.option("--spool", "spool_path", type=click.Path(path_type=Path), default=None)
@click.option("--format", "output_format", type=click.Choice(["json"]), default=None, help="Output format.")
def post_command(
    provider: str,
    conversation_id: str,
    project_ref: str | None,
    text: str,
    command_id: str | None,
    submit: bool,
    spool_path: Path | None,
    output_format: str | None,
) -> None:
    """Enqueue an outbound post command for the extension to deliver.

    Safety: refused unless POLYLOGUE_BROWSER_POST_ENABLED=1 is set. The command
    is only queued here; the running receiver serves it to the extension via
    GET /v1/post-commands.
    """
    try:
        request = BrowserPostCommandRequest(
            provider=provider,  # type: ignore[arg-type]
            target=BrowserPostTarget(conversation_id=conversation_id, project_ref=project_ref),
            text=text,
            command_id=command_id,
            submit=submit,
        )
        command = enqueue_post_command(request, spool_path=spool_path)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from None
    except BrowserPostDisabledError:
        raise click.ClickException(
            f"Outbound posting is disabled. Set {BROWSER_POST_ENABLED_ENV}=1 to enable (default OFF safety guard)."
        ) from None
    payload = command.model_dump(mode="json", exclude_none=True)
    if output_format == "json":
        click.echo(dumps(payload))
        return
    click.echo(f"Queued post command {command.command_id}")
    click.echo(f"Provider: {command.provider}  target: {command.target.conversation_id}")
    click.echo(f"Submit (click send): {command.submit}")
    if not browser_post_enabled():  # pragma: no cover - defensive; enqueue would have raised
        click.echo(f"WARNING: {BROWSER_POST_ENABLED_ENV} is not set")


__all__ = ["browser_capture_command", "post_command", "serve_command", "status_command"]
