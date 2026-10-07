"""Deterministic demo archive commands."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.core.errors import DatabaseError
from polylogue.demo import (
    DemoSeedResult,
    DemoSeedTargetUnsafeError,
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


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _require_scratch_target(target: Path, *, purpose: str) -> Path:
    """Return ``target`` resolved, refusing any overlap with the configured archive.

    Demo seeding is the only CLI route that owns archive tiers offline, and it
    owns only a scratch root no daemon serves. The configured archive is
    written solely through ``polylogued`` -- even when it is still empty, which
    is exactly when the content-based guard in :mod:`polylogue.demo.seed`
    cannot tell it from a fresh scratch root. ``purpose`` names what the
    target is (an archive root or the tour's output directory that
    ``--force`` deletes first).
    """

    from polylogue.cli.shared.helpers import DaemonRequiredError

    resolved = target.expanduser().resolve()
    configured = archive_root().expanduser().resolve()
    if _overlaps(resolved, configured):
        raise DaemonRequiredError(
            f"the demo {purpose} {resolved} overlaps the configured archive {configured}, which only "
            "`polylogued run` writes. Point the demo at a separate scratch --root, or seed the configured "
            "archive through the daemon with `polylogue import --demo`",
            operation="maintenance.demo.augment",
            archive_root=configured,
        )
    return resolved


def _seed_demo_archive(
    env: AppEnv,
    target: Path,
    *,
    force: bool,
    with_overlays: bool,
) -> DemoSeedResult:
    """Seed an isolated synthetic scratch root through canonical live acquisition."""
    del env
    from polylogue.cli.shared.helpers import DaemonRequiredError
    from polylogue.demo.seed import _archive_root_has_real_content, _archive_root_is_demo_owned

    resolved_root = _require_scratch_target(target, purpose="archive root")
    if _archive_root_has_real_content(resolved_root) and not _archive_root_is_demo_owned(resolved_root):
        raise DaemonRequiredError(
            "demo seed cannot acquire into an existing real archive locally; start `polylogued run` "
            "and use the daemon's `ingest` operation followed by `maintenance.demo.augment`",
            operation="ingest",
            archive_root=resolved_root,
        )
    return asyncio.run(seed_demo_archive(resolved_root, force=force, with_overlays=with_overlays))


@click.group("demo")
def demo_command() -> None:
    """Seed and verify the deterministic local demo archive."""


@demo_command.command("seed")
@click.option(
    "--root",
    "root",
    type=click.Path(path_type=Path),
    required=True,
    help="Scratch archive root to seed. Must not be, contain, or lie inside the configured archive.",
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
    root: Path,
    force: bool,
    with_overlays: bool,
    output_format: str,
) -> None:
    """Create a ready-to-query deterministic demo archive in a scratch root."""

    try:
        result = _seed_demo_archive(env, root, force=force, with_overlays=with_overlays)
    except DemoSeedTargetUnsafeError as exc:
        raise click.ClickException(str(exc)) from exc
    payload = result.to_payload()
    if output_format == "json":
        click.echo(json.dumps(payload, sort_keys=True))
        return
    healed_note = (
        f"  Self-healed: {', '.join(result.healed_tiers)} (stale schema moved aside and rebuilt)\n"
        if result.healed_tiers
        else ""
    )
    env.ui.console.print(
        "[bold green]Demo archive ready[/bold green]\n"
        f"  Archive root: {result.archive_root}\n"
        f"  Source root:  {result.source_root}\n"
        f"  Sessions:     {result.session_count}\n"
        f"  Messages:     {result.message_count}\n"
        f"  Overlays:     {'yes' if result.overlays_seeded else 'no'}\n"
        f"{healed_note}"
        "  Verify:       polylogue demo verify"
    )


@demo_command.command("receipts")
@click.option(
    "--root",
    "root",
    type=click.Path(path_type=Path),
    required=True,
    help=("Scratch archive root to inspect or seed. Must be explicit and must not overlap the configured archive."),
)
@click.option(
    "--seed/--no-seed",
    default=None,
    help="Seed the scratch archive before inspection; defaults to no with an explicit root.",
)
@click.option(
    "--force/--no-force",
    default=True,
    show_default=True,
    help="Replace the generated demo source directory when seeding.",
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
    help="Inspect only the completion-claim cohort in the scratch archive.",
)
@click.option(
    "--compact",
    is_flag=True,
    help="Keep plain output to the claim, test outcomes, and anti-grep control.",
)
def receipts_command(
    root: Path,
    seed: bool | None,
    force: bool,
    output_format: str,
    completion_claims_only: bool,
    compact: bool,
) -> None:
    """Compare a demo assistant claim with structural tool evidence."""

    from polylogue.cli.shared.helpers import DaemonRequiredError

    try:
        resolved_root = _require_scratch_target(root, purpose="archive root")
    except DaemonRequiredError as exc:
        if output_format != "json":
            raise
        from polylogue.cli.shared.machine_errors import error_daemon_required

        error_daemon_required(
            str(exc),
            command=["demo", "receipts"],
            operation=exc.operation,
            archive_root=exc.archive_root,
        ).emit()

    should_seed = False if seed is None else seed
    if should_seed:
        try:
            _seed_demo_archive(AppEnv(), resolved_root, force=force, with_overlays=False)
        except DemoSeedTargetUnsafeError as exc:
            raise click.ClickException(str(exc)) from exc

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
        archive_label = "<demo-archive>"
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
        if output_format == "json":
            click.get_current_context().exit(1)
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
    required=True,
    help="Explicit scratch archive root to seed. Must not overlap the configured archive.",
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
    root: Path,
    force: bool,
    output_format: str,
) -> None:
    """Run a one-command public demo tour and write shareable artifacts."""

    # Checked before the tour starts: ``--force`` deletes the output directory
    # first, so an out-dir holding the configured archive would be destroyed
    # long before the seed's own target check could refuse.
    _require_scratch_target(out_dir, purpose="tour output directory")
    scratch_root = _require_scratch_target(root, purpose="archive root")
    result = run_demo_tour(
        output_dir=out_dir,
        archive_root=scratch_root,
        force=force,
        seed_archive=lambda target, **options: _seed_demo_archive(env, target, **options),
    )
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
        if output_format == "json":
            click.get_current_context().exit(1)
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
