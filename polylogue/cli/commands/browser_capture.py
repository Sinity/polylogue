"""Browser-capture receiver CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig, receiver_status_payload
from polylogue.browser_capture.server import make_server
from polylogue.lib.json import dumps


@click.group("browser-capture")
def browser_capture_command() -> None:
    """Receive browser-extension captures into the normal inbox."""


@browser_capture_command.command("status")
@click.option("--inbox", "inbox_path", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True)
def status_command(inbox_path: Path | None, as_json: bool) -> None:
    """Show receiver configuration and inbox target."""
    config = BrowserCaptureReceiverConfig.default()
    if inbox_path is not None:
        config = BrowserCaptureReceiverConfig(inbox_path=inbox_path, allowed_origins=config.allowed_origins)
    payload = receiver_status_payload(config)
    if as_json:
        click.echo(dumps(payload))
        return
    click.echo("Browser capture receiver")
    click.echo(f"Inbox: {payload['inbox_path']}")
    origins = payload.get("allowed_origins", [])
    origin_text = ", ".join(str(item) for item in origins) if isinstance(origins, list) else str(origins)
    click.echo(f"Allowed origins: {origin_text}")


@browser_capture_command.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--inbox", "inbox_path", type=click.Path(path_type=Path), default=None)
def serve_command(host: str, port: int, inbox_path: Path | None) -> None:
    """Run the local browser-capture receiver."""
    server = make_server(host, port, inbox_path=inbox_path)
    click.echo(f"Listening on http://{host}:{port}")
    click.echo(f"Writing captures to {server.config.inbox_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopping browser capture receiver")
    finally:
        server.server_close()


__all__ = ["browser_capture_command"]
