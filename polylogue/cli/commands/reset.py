"""Reset command for clearing database and state."""

from __future__ import annotations

import shutil

import click

from polylogue.cli.helpers import fail
from polylogue.cli.types import AppEnv
from polylogue.paths import CACHE_HOME, DATA_HOME, DB_PATH, DRIVE_TOKEN_PATH, RENDER_ROOT, STATE_HOME


@click.command("reset")
@click.option("--database", is_flag=True, help="Delete the SQLite database")
@click.option("--assets", is_flag=True, help="Delete archived assets/attachments")
@click.option("--render", is_flag=True, help="Delete rendered conversations (Markdown/HTML)")
@click.option("--cache", is_flag=True, help="Delete search indexes and cache")
@click.option("--auth", is_flag=True, help="Delete Google Drive OAuth tokens")
@click.option("--all", "reset_all", is_flag=True, help="Reset everything")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_obj
def reset_command(
    env: AppEnv,
    database: bool,
    assets: bool,
    render: bool,
    cache: bool,
    auth: bool,
    reset_all: bool,
    yes: bool,
) -> None:
    """Reset database, assets, rendered outputs, or auth state.

    By default, requires explicit flags to specify what to reset.
    Use --all to reset everything.
    """
    if reset_all:
        database = assets = render = cache = auth = True

    if not (database or assets or render or cache or auth):
        fail(
            "reset",
            "Specify at least one target (e.g., --database, --assets, --render, --cache, --auth) or use --all",
        )

    targets = []
    if database and DB_PATH.exists():
        targets.append(("database", DB_PATH))
        # Also clean up WAL/SHM files and health cache alongside the database
        for suffix in (".db-wal", ".db-shm"):
            wal_path = DB_PATH.with_suffix(suffix)
            if wal_path.exists():
                targets.append((f"database {suffix}", wal_path))
        health_path = DATA_HOME / "health.json"
        if health_path.exists():
            targets.append(("health cache", health_path))
    if assets:
        assets_dir = DATA_HOME / "assets"
        if assets_dir.exists():
            targets.append(("assets", assets_dir))
    if render and RENDER_ROOT.exists():
        targets.append(("render results", RENDER_ROOT))
    if cache and CACHE_HOME.exists():
        targets.append(("cache/indexes", CACHE_HOME))
    if auth and DRIVE_TOKEN_PATH.exists():
        targets.append(("OAuth token", DRIVE_TOKEN_PATH))
    if reset_all:
        # Clean up run history and last-source state
        runs_dir = DATA_HOME / "runs"
        if runs_dir.exists():
            targets.append(("run history", runs_dir))
        last_source = STATE_HOME / "last-source.json"
        if last_source.exists():
            targets.append(("last-source state", last_source))

    if not targets:
        env.ui.console.print("Nothing to reset (no files exist for selected targets).")
        return

    # Show what will be deleted
    lines = [f"  {name}: {path}" for name, path in targets]
    env.ui.summary("Will delete", lines)

    # Confirm unless --yes
    if not yes:
        if env.ui.plain:
            env.ui.console.print("Use --yes to confirm deletion.")
            return
        if not env.ui.confirm("Delete these files/directories?", default=False):
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
