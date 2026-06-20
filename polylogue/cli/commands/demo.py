"""Deterministic demo archive commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.demo import render_demo_script, seed_demo_archive, verify_demo_archive
from polylogue.paths import archive_root


@click.group("demo")
def demo_command() -> None:
    """Seed and verify the deterministic local demo archive."""


@demo_command.command("seed")
@click.option(
    "--root",
    "root",
    type=click.Path(path_type=Path),
    default=None,
    help="Archive root to seed. Defaults to POLYLOGUE_ARCHIVE_ROOT.",
)
@click.option("--force", is_flag=True, help="Replace the generated demo source directory before seeding.")
@click.option("--with-overlays", is_flag=True, help="Seed deterministic user overlays after archive ingest.")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def seed_command(
    env: AppEnv,
    root: Path | None,
    force: bool,
    with_overlays: bool,
    output_format: str,
) -> None:
    """Create a ready-to-query deterministic demo archive."""

    resolved_root = (root or archive_root()).expanduser().resolve()
    result = asyncio.run(seed_demo_archive(resolved_root, force=force, with_overlays=with_overlays))
    payload = result.to_payload()
    if output_format == "json":
        click.echo(json.dumps(payload, sort_keys=True))
        return
    env.ui.console.print(
        "[bold green]Demo archive ready[/bold green]\n"
        f"  Archive root: {result.archive_root}\n"
        f"  Source root:  {result.source_root}\n"
        f"  Sessions:     {result.session_count}\n"
        f"  Messages:     {result.message_count}\n"
        f"  Overlays:     {'yes' if result.overlays_seeded else 'no'}\n"
        "  Verify:       polylogue demo verify"
    )


@demo_command.command("verify")
@click.option(
    "--root",
    "root",
    type=click.Path(path_type=Path),
    default=None,
    help="Archive root to verify. Defaults to POLYLOGUE_ARCHIVE_ROOT.",
)
@click.option("--require-overlays", is_flag=True, help="Fail unless deterministic demo overlays are present.")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def verify_command(
    env: AppEnv,
    root: Path | None,
    require_overlays: bool,
    output_format: str,
) -> None:
    """Verify semantic facts for the deterministic demo archive."""

    resolved_root = (root or archive_root()).expanduser().resolve()
    result = verify_demo_archive(resolved_root, require_overlays=require_overlays)
    payload = result.to_payload()
    if output_format == "json":
        click.echo(json.dumps(payload, sort_keys=True))
    else:
        status = "[bold green]ok[/bold green]" if result.ok else "[bold red]failed[/bold red]"
        env.ui.console.print(
            f"Demo archive verification: {status}\n"
            f"  Archive root: {result.archive_root}\n"
            f"  Sessions:     {result.session_count}\n"
            f"  Messages:     {result.message_count}\n"
            f"  Query hits:   {len(result.query_hits)}\n"
            f"  Overlays:     {'yes' if result.overlays_present else 'no'}"
        )
        for problem in result.problems:
            env.ui.console.print(f"  - {problem}")
    if not result.ok:
        raise click.ClickException("demo archive verification failed")


@demo_command.command("script")
@click.option(
    "--root",
    "root",
    type=click.Path(path_type=Path),
    default=None,
    help="Archive root to embed in the script. Defaults to POLYLOGUE_ARCHIVE_ROOT.",
)
@click.option("--shell", type=click.Choice(["bash"]), default="bash", show_default=True)
def script_command(root: Path | None, shell: str) -> None:
    """Print a copy-pastable demo command script."""

    resolved_root = (root or archive_root()).expanduser().resolve()
    click.echo(render_demo_script(resolved_root, shell=shell), nl=False)


__all__ = ["demo_command"]
