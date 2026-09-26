"""Backup CLI command."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.cli.shared.types import AppEnv
from polylogue.daemon.backup import BACKUP_PROFILES, BackupProfile, BackupResult, backup_archive, format_backup_result
from polylogue.logging import configure_logging


@click.command("backup", help="Back up the Polylogue archive durability tiers.")
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write the backup into.",
)
@click.option(
    "--check",
    "check_only",
    is_flag=True,
    default=False,
    help="Verify backup prerequisites without creating a backup.",
)
@click.option(
    "--verify",
    "verify",
    is_flag=True,
    default=False,
    help="Restore the finished backup into a scratch directory and run smoke checks.",
)
@click.option(
    "--profile",
    type=click.Choice(BACKUP_PROFILES),
    default="rebuildable_cache_exclude",
    show_default=True,
    help="Named backup profile controlling copied archive tiers.",
)
@click.option("--format", "output_format", type=click.Choice(("plain", "json")), default="plain")
@click.pass_obj
def backup_command(
    env: AppEnv,
    output_dir: Path,
    check_only: bool,
    verify: bool,
    profile: BackupProfile,
    output_format: str,
) -> None:
    """Back up the Polylogue archive.

    Backups copy the precious source/user/embeddings tiers
    plus referenced blobs, and intentionally omit rebuildable index.db and
    disposable ops.db.

    Use --check to verify prerequisites (disk space, DB readability)
    without creating a backup.
    """
    configure_logging()
    if check_only:
        result = backup_archive(
            output_dir=output_dir,
            check_only=True,
            verify=verify,
            profile=profile,
            archive_root_path=env.config.archive_root,
        )
    else:
        from polylogue.cli.archive_query import submit_cli_mutation
        from polylogue.cli.operation_kernel import OperationFailedError

        try:
            completed = submit_cli_mutation(
                env,
                "maintenance.backup",
                {"output_dir": str(output_dir), "check_only": False, "verify": verify, "profile": profile},
            )
        except click.ClickException as exc:
            failure = exc.__cause__
            if not isinstance(failure, OperationFailedError) or failure.code != "backup_failed":
                raise
            payload = failure.data.get("backup_result")
            if not isinstance(payload, dict):
                raise
            result = BackupResult.model_validate(payload)
        else:
            result = BackupResult.model_validate(completed.get("result"))
    if output_format == "json":
        import json

        click.echo(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        if not result.ok:
            raise SystemExit(1)
        return
    for line in format_backup_result(result):
        click.echo(line)
    if not result.ok:
        raise SystemExit(1)


__all__ = ["backup_command"]
