"""Serve command."""

from __future__ import annotations

import os

import click

from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.types import AppEnv


@click.command("serve")
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
@click.pass_obj
def serve_command(env: AppEnv, host: str, port: int) -> None:
    """Start the Semantic API server."""
    try:
        import uvicorn
    except ImportError:
        fail("serve", "uvicorn not installed. Run with [server] extras.")

    try:
        config = load_effective_config(env)
        # Ensure server process sees the correct paths
        if config.archive_root:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(config.archive_root)

        ui = env.ui
        ui.console.print(f"Starting server...")
        ui.console.print(f"  Web UI:  [bold blue]http://{host}:{port}/[/bold blue]")
        ui.console.print(f"  API:     [bold blue]http://{host}:{port}/api/docs[/bold blue]")
        uvicorn.run("polylogue.server.app:app", host=host, port=port, log_level="info")
    except Exception as exc:
        fail("serve", str(exc))
