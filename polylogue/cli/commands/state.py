"""State management commands."""

from __future__ import annotations

import click

from polylogue.cli.helpers import fail, source_state_path
from polylogue.cli.types import AppEnv
from polylogue.storage.db import default_db_path, open_connection


@click.group("state")
def state_command() -> None:
    """State management commands."""


@state_command.command("reset")
@click.option("--db/--no-db", "reset_db", default=True, show_default=True, help="Reset the local SQLite DB")
@click.option("--last-source", is_flag=True, help="Clear the stored last-source selection")
@click.option("--all", "reset_all", is_flag=True, help="Reset DB and last-source state")
@click.option("--force", is_flag=True, help="Skip confirmation prompts")
@click.pass_obj
def state_reset(
    env: AppEnv,
    reset_db: bool,
    last_source: bool,
    reset_all: bool,
    force: bool,
) -> None:
    if reset_all:
        reset_db = True
        last_source = True
    if not reset_db and not last_source:
        fail("state reset", "Nothing to reset; use --db, --last-source, or --all.")
    if env.ui.plain and not force:
        fail("state reset", "--force is required in plain mode.")
    if not force and not env.ui.plain:
        prompt = "Reset local state? This removes the SQLite DB and/or last-source selection."
        if not env.ui.confirm(prompt, default=False):
            env.ui.console.print("Reset cancelled.")
            return

    if reset_db:
        db_path = default_db_path()
        if db_path.exists():
            db_path.unlink()
        with open_connection(db_path):
            pass
        env.ui.console.print(f"Reset DB: {db_path}")
    if last_source:
        state_path = source_state_path()
        if state_path.exists():
            state_path.unlink()
        env.ui.console.print("Cleared last-source selection.")
