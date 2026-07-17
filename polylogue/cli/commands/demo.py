"""Deterministic demo archive commands."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.core.errors import DatabaseError
from polylogue.demo import (
    inspect_completion_claims,
    inspect_demo_receipts,
    render_completion_claims,
    render_demo_receipts,
    render_demo_script,
    run_demo_tour,
    seed_demo_archive,
    verify_demo_archive,
)
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


@demo_command.command("receipts")
@click.option(
    "--root",
    "root",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Archive root to inspect. When neither --root nor POLYLOGUE_ARCHIVE_ROOT is set, "
        "the command seeds ./polylogue-receipts-demo/archive first."
    ),
)
@click.option(
    "--seed/--no-seed",
    default=None,
    help="Seed the deterministic archive before inspection; defaults to yes only for the self-contained path.",
)
@click.option(
    "--force/--no-force",
    default=True,
    show_default=True,
    help="Replace the self-contained demo archive when seeding.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.option(
    "--completion-claims-only",
    is_flag=True,
    help="Inspect only the completion-claim cohort; suitable for an operator-owned archive.",
)
@click.option(
    "--compact",
    is_flag=True,
    help="Keep plain output to the claim, test outcomes, and anti-grep control.",
)
def receipts_command(
    root: Path | None,
    seed: bool | None,
    force: bool,
    output_format: str,
    completion_claims_only: bool,
    compact: bool,
) -> None:
    """Compare a demo assistant claim with structural tool evidence."""

    has_configured_root = root is not None or "POLYLOGUE_ARCHIVE_ROOT" in os.environ
    if root is not None:
        resolved_root = root.expanduser().resolve()
    elif has_configured_root:
        resolved_root = archive_root().expanduser().resolve()
    else:
        resolved_root = (Path.cwd() / "polylogue-receipts-demo" / "archive").resolve()

    should_seed = (not has_configured_root) if seed is None else seed
    if should_seed:
        asyncio.run(seed_demo_archive(resolved_root, force=force, with_overlays=False))

    if completion_claims_only:
        try:
            completion_claims = inspect_completion_claims(resolved_root)
        except (DatabaseError, OSError, sqlite3.Error) as exc:
            raise click.ClickException(f"completion-claim evidence unavailable: {exc}") from exc
        if output_format == "json":
            click.echo(json.dumps(completion_claims.to_payload(), sort_keys=True))
        else:
            click.echo(render_completion_claims(completion_claims), nl=False)
        return

    result = inspect_demo_receipts(resolved_root)
    if output_format == "json":
        payload = result.to_payload()
        payload["seeded_for_command"] = should_seed
        click.echo(json.dumps(payload, sort_keys=True))
    else:
        archive_label = None if has_configured_root else "<demo-archive>"
        click.echo(
            render_demo_receipts(
                result,
                archive_label=archive_label,
                compact=compact,
            ),
            nl=False,
        )
    if not result.ok:
        raise click.ClickException("demo receipts verification failed")


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


@demo_command.command("tour")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("polylogue-demo-tour"),
    show_default=True,
    help="Directory where the tour archive, transcript, report, and recording tape are written.",
)
@click.option(
    "--root",
    "root",
    type=click.Path(path_type=Path),
    default=None,
    help="Archive root to seed. Defaults to <out-dir>/archive.",
)
@click.option("--force/--no-force", default=True, show_default=True, help="Replace the output directory first.")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
)
@click.pass_obj
def tour_command(
    env: AppEnv,
    out_dir: Path,
    root: Path | None,
    force: bool,
    output_format: str,
) -> None:
    """Run a one-command public demo tour and write shareable artifacts."""

    result = run_demo_tour(output_dir=out_dir, archive_root=root, force=force)
    payload = result.to_payload()
    if output_format == "json":
        click.echo(json.dumps(payload, sort_keys=True))
    else:
        status = "[bold green]passed[/bold green]" if result.ok else "[bold red]failed[/bold red]"
        env.ui.console.print(
            f"Polylogue demo tour: {status}\n"
            f"  First result: {result.first_result_s:.3f}s\n"
            f"  Full tour:    {result.total_duration_s:.3f}s\n"
            f"  Archive root: {result.archive_root}\n"
            f"  Report:       {result.report_markdown_path}\n"
            f"  Transcript:   {result.transcript_path}\n"
            f"  Recording:    {result.recording_tape_path}"
        )
        for problem in result.problems:
            env.ui.console.print(f"  - {problem}")
    if not result.ok:
        raise click.ClickException("demo tour failed")


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
