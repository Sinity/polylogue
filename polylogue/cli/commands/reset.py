"""Reset command for clearing database and state."""

from __future__ import annotations

import shutil

import click

from polylogue.cli.helpers import fail
from polylogue.cli.types import AppEnv
from polylogue.paths import CACHE_HOME, STATE_HOME
from polylogue.storage.db import default_db_path


@click.command("reset")
@click.option("--database", is_flag=True, help="Delete the SQLite database")
@click.option("--cache", is_flag=True, help="Delete cached data")
@click.option("--state", is_flag=True, help="Delete state files")
@click.option("--all", "reset_all", is_flag=True, help="Reset everything")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.pass_obj
def reset_command(
    env: AppEnv,
    database: bool,
    cache: bool,
    state: bool,
    reset_all: bool,
    force: bool,
) -> None:
    """Reset database, cache, or state files.

    By default, requires explicit flags to specify what to reset.
    Use --all to reset everything.
    """
    if reset_all:
        database = cache = state = True

    if not (database or cache or state):
        fail("reset", "Specify --database, --cache, --state, or --all")

    targets = []
    if database:
        db_path = default_db_path()
        if db_path.exists():
            targets.append(("database", db_path))
    if cache and CACHE_HOME.exists():
        targets.append(("cache", CACHE_HOME))
    if state and STATE_HOME.exists():
        targets.append(("state", STATE_HOME))

    if not targets:
        env.ui.console.print("Nothing to reset (no files exist).")
        return

    # Show what will be deleted
    lines = [f"  {name}: {path}" for name, path in targets]
    env.ui.summary("Will delete", lines)

    # Confirm unless --force
    if not force:
        if env.ui.plain:
            env.ui.console.print("Use --force to confirm deletion.")
            return
        if not env.ui.confirm("Delete these files?", default=False):
            env.ui.console.print("Reset cancelled.")
            return

    # Perform deletion
    deleted = 0
    for name, path in targets:
        try:
            if path.is_file():
                path.unlink()
                deleted += 1
                env.ui.console.print(f"  Deleted {name}: {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                deleted += 1
                env.ui.console.print(f"  Deleted {name}: {path}")
        except OSError as exc:
            env.ui.console.print(f"  Failed to delete {name}: {exc}")

    env.ui.console.print(f"\nReset complete: {deleted} item(s) deleted.")
