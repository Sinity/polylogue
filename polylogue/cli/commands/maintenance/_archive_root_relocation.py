"""``maintenance archive-root-relocation``: explicit offline root rebinding."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from polylogue.operations.archive_root_relocation import (
    ArchiveRootRelocationError,
    RelocationPostMoveWitness,
    apply_archive_root_relocation,
    load_archive_root_relocation_plan,
    prepare_archive_root_relocation,
    write_archive_root_relocation_plan,
)
from polylogue.operations.durable_change_train import ArchiveOwnershipError, acquire_durable_archive_ownership
from polylogue.paths import archive_root


@click.group("archive-root-relocation")
def archive_root_relocation_command() -> None:
    """Move one complete archive root without changing any SQLite rows.

    This accepts only an inode-preserving move with a previously verified
    full-evidence backup. It is an offline transition, never startup repair.
    """


@archive_root_relocation_command.command("plan")
@click.option("--old-root", required=True, type=click.Path(path_type=Path))
@click.option("--backup-manifest", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--post-move-witness", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def archive_root_relocation_plan_command(
    old_root: Path,
    backup_manifest: Path,
    post_move_witness: Path | None,
    output: Path,
    output_format: str,
) -> None:
    """Record a strict, read-only relocation plan for the configured destination."""
    from polylogue.cli.commands.maintenance._migrate_tier import _require_stopped_daemon

    root = archive_root()
    try:
        with acquire_durable_archive_ownership(root, owner_id=f"archive-root-relocation-plan:{os.getpid()}"):
            stopped = _require_stopped_daemon(root)
            witness = None
            if post_move_witness is not None:
                try:
                    witness = RelocationPostMoveWitness.model_validate_json(
                        post_move_witness.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError) as exc:
                    raise ArchiveRootRelocationError("archive-root relocation post-move witness is invalid") from exc
            plan = prepare_archive_root_relocation(
                old_root=old_root,
                new_root=root,
                backup_manifest=backup_manifest,
                stopped_daemon_evidence_ref=stopped,
                single_writer_evidence_ref="proof:archive-ownership-lock",
                post_move_witness=witness,
            )
            write_archive_root_relocation_plan(plan, output)
    except (ArchiveOwnershipError, ArchiveRootRelocationError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        click.echo(f"Wrote inode-preserving archive-root relocation plan: {output}")


@archive_root_relocation_command.command("apply")
@click.option("--plan", "plan_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--authorize", required=True)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def archive_root_relocation_apply_command(plan_path: Path, authorize: str, output_format: str) -> None:
    """Apply the plan to durable trains and sealed index-generation topology."""
    root = archive_root()
    try:
        plan = load_archive_root_relocation_plan(plan_path)
        result = apply_archive_root_relocation(
            root=root,
            plan=plan,
            authorization=authorize,
        )
    except (ArchiveRootRelocationError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        click.echo(f"Archive-root relocation {result.state}: {result.receipt_path}")
