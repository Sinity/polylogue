"""Dashboard command."""

from __future__ import annotations

import click

from polylogue.cli.types import AppEnv


@click.command("dashboard")
@click.pass_obj
def dashboard_command(env: AppEnv) -> None:
    """Launch the dashboard TUI."""
    from polylogue.ui.tui.app import PolylogueApp

    app = PolylogueApp(repository=env.repository)
    app.run()


__all__ = ["dashboard_command"]
