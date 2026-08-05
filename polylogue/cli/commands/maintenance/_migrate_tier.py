"""``maintenance migrate-tier``: apply additive migrations for one durable archive tier.

Unlike its sibling maintenance commands, this one's own ``--help`` still
imports ``polylogue.storage.sqlite.archive_tiers.types``: ``click.Choice(...)``
on the ``tier`` argument needs the real ``DURABLE_MIGRATION_TIERS`` value at
decoration time to render the valid choices, and that constant is derived
from ``ArchiveTier``. Historically that single import forced the whole
``archive_tiers`` package's eager DDL chain (every tier's DDL module) plus the
parent ``polylogue.storage.sqlite`` package's eager ``SQLiteBackend`` import
-- ~1s of pure import tax for one enum. polylogue-h1wt made both of those
parent/package inits lazy (PEP 562 ``__getattr__``), so this command's
``--help`` now pays only ``ArchiveTier``'s own weight (~100ms, in the same
required-gate budget as every sibling command).
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import click

from polylogue.operations.durable_change_train import (
    ArchiveOwnershipError,
    acquire_durable_archive_ownership,
    execute_durable_change_train,
)
from polylogue.paths import archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS, MigrationError


def _daemon_pidfile_is_live(pidfile: Path) -> bool:
    """Return whether the archive pidfile names a live polylogued process."""
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return b"polylogued" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return False


def _require_stopped_daemon(root: Path) -> str:
    """Refuse before opening SQLite when the daemon still owns the archive."""
    pidfile = root / "daemon.pid"
    if _daemon_pidfile_is_live(pidfile):
        raise MigrationError(f"durable migration requires the daemon to be stopped; live pidfile: {pidfile}")
    return "proof:daemon-stopped"


@click.command("migrate-tier")
@click.argument("tier", type=click.Choice(tuple(sorted(tier.value for tier in DURABLE_MIGRATION_TIERS))))
@click.option(
    "--backup-manifest",
    required=False,
    type=click.Path(path_type=Path, exists=True),
    help="Verified backup manifest. Required only when a selected migration changes existing durable data.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def migrate_tier_command(tier: str, backup_manifest: Path | None, output_format: str) -> None:
    """Apply additive migrations for one durable archive tier.

    Derived tiers are intentionally excluded from this command; rebuild or
    blue-green replace those from source evidence instead.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

    archive_tier = ArchiveTier(tier)
    spec = ARCHIVE_TIER_SPECS[archive_tier]
    path = archive_root() / spec.filename
    stopped_daemon_evidence_ref: str | None = None
    try:
        with acquire_durable_archive_ownership(path.parent, owner_id=f"migrate-tier:{os.getpid()}") as archive_owner:
            stopped_daemon_evidence_ref = _require_stopped_daemon(path.parent)
            execution = execute_durable_change_train(
                path.parent,
                archive_tier,
                backup_manifest=backup_manifest,
                daemon_stopped_evidence_ref=stopped_daemon_evidence_ref,
                single_writer_evidence_ref="proof:archive-ownership-lock",
                release_archive_ownership=archive_owner.release,
            )
    except (sqlite3.Error, MigrationError, ArchiveOwnershipError) as exc:
        if output_format == "json":
            click.echo(
                json.dumps(
                    {
                        "ok": False,
                        "tier": tier,
                        "path": str(path),
                        "backup_manifest": str(backup_manifest) if backup_manifest is not None else None,
                        "stopped_daemon_evidence_ref": stopped_daemon_evidence_ref,
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            click.echo(f"Migration blocked for {tier}: {exc}", err=True)
        raise SystemExit(1) from exc

    result = execution.migration_result
    payload = {
        "ok": True,
        "tier": tier,
        "path": str(path),
        "backup_manifest": str(backup_manifest) if backup_manifest is not None else None,
        "stopped_daemon_evidence_ref": stopped_daemon_evidence_ref,
        "train_manifest": str(execution.manifest_path) if execution.manifest_path is not None else None,
        "train_state": execution.train.state.value if execution.train is not None else None,
        "backup_receipt": str(result.backup_receipt)
        if result is not None and result.backup_receipt is not None
        else None,
        "from_version": result.from_version if result is not None else None,
        "to_version": result.to_version if result is not None else None,
        "applied_versions": list(result.applied_versions) if result is not None else [],
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if result is None:
        click.echo(f"No pending durable migration for {tier}.")
        return
    applied = ", ".join(str(version) for version in result.applied_versions) or "none"
    click.echo(
        f"Migrated {tier}: {result.from_version} -> {result.to_version} "
        f"(applied: {applied}; receipt: {result.backup_receipt})"
    )
