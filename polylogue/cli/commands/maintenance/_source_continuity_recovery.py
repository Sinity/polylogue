"""Offline bridge for one authenticated historical source mutation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from polylogue.operations.archive_root_relocation import RelocationPostMoveWitness
from polylogue.operations.durable_change_train import ArchiveOwnershipError, acquire_durable_archive_ownership
from polylogue.operations.historical_source_continuity_recovery import (
    HistoricalSourceContinuityRecoveryError,
    apply_historical_source_continuity_recovery,
    load_historical_source_continuity_recovery_plan,
    prepare_historical_source_continuity_recovery,
    write_historical_source_continuity_recovery_plan,
)
from polylogue.paths import archive_root


@click.group("source-continuity-recovery")
def source_continuity_recovery_command() -> None:
    """Recover one pre-#3868 liveness receipt with independently attested evidence."""


@source_continuity_recovery_command.command("plan")
@click.option("--old-root", required=True, type=click.Path(path_type=Path))
@click.option("--mutation-receipt", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--pre-backup-manifest", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--post-backup-manifest", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--post-move-witness", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def source_continuity_recovery_plan_command(
    old_root: Path,
    mutation_receipt: Path,
    pre_backup_manifest: Path,
    post_backup_manifest: Path,
    post_move_witness: Path | None,
    output: Path,
    output_format: str,
) -> None:
    """Seal a read-only bridge plan; it never writes SQLite or sidecars."""
    from polylogue.cli.commands.maintenance._migrate_tier import _require_stopped_daemon

    root = archive_root()
    try:
        with acquire_durable_archive_ownership(
            root, owner_id=f"historical-source-continuity-recovery-plan:{os.getpid()}"
        ):
            stopped = _require_stopped_daemon(root)
            witness = None
            if post_move_witness is not None:
                try:
                    witness = RelocationPostMoveWitness.model_validate_json(
                        post_move_witness.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError) as exc:
                    raise HistoricalSourceContinuityRecoveryError(
                        "historical continuity post-move witness is invalid"
                    ) from exc
            plan = prepare_historical_source_continuity_recovery(
                old_root=old_root,
                new_root=root,
                mutation_receipt=mutation_receipt,
                pre_backup_manifest=pre_backup_manifest,
                post_backup_manifest=post_backup_manifest,
                stopped_daemon_evidence_ref=stopped,
                single_writer_evidence_ref="proof:archive-ownership-lock",
                post_move_witness=witness,
            )
            write_historical_source_continuity_recovery_plan(plan, output)
    except (ArchiveOwnershipError, HistoricalSourceContinuityRecoveryError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True)
        if output_format == "json"
        else f"Wrote historical source continuity recovery plan: {output}"
    )


@source_continuity_recovery_command.command("apply")
@click.option("--plan", "plan_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--authorize", required=True)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def source_continuity_recovery_apply_command(plan_path: Path, authorize: str, output_format: str) -> None:
    """CAS-revise only the released current source train and retained receipts."""
    from polylogue.cli.commands.maintenance._migrate_tier import _require_stopped_daemon

    root = archive_root()
    try:
        plan = load_historical_source_continuity_recovery_plan(plan_path)
        with acquire_durable_archive_ownership(
            root, owner_id=f"historical-source-continuity-recovery-apply:{os.getpid()}"
        ):
            stopped = _require_stopped_daemon(root)
            result = apply_historical_source_continuity_recovery(
                root=root,
                plan=plan,
                authorization=authorize,
                stopped_daemon_evidence_ref=stopped,
                single_writer_evidence_ref="proof:archive-ownership-lock",
            )
    except (ArchiveOwnershipError, HistoricalSourceContinuityRecoveryError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
        if output_format == "json"
        else f"Historical source continuity recovery {result.state}: {result.receipt_path}"
    )
