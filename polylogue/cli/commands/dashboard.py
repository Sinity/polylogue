from __future__ import annotations

import click

from polylogue.cli.types import AppEnv
from polylogue.config import get_config


@click.command("dashboard")
@click.pass_obj
def dashboard_command(env: AppEnv) -> None:
    """Launch the Mission Control TUI dashboard."""
    from polylogue.ui.tui.app import PolylogueApp

    config = get_config()
    app = PolylogueApp(config=config)
    app.run()


__all__ = ["dashboard_command"]
