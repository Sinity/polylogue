"""``maintenance migrate-tier``: apply additive migrations for one durable archive tier.

Unlike its sibling maintenance commands, this one's own ``--help`` cannot
avoid the ``polylogue.storage.sqlite.archive_tiers`` package's import cost:
``click.Choice(...)`` on the ``tier`` argument needs the real
``DURABLE_MIGRATION_TIERS`` value at decoration time to render the valid
choices, and that constant is only derived from ``ArchiveTier``
(``archive_tiers.types``), which -- via the package's own ``__init__.py`` --
eagerly imports every tier's DDL module. This is the same root cause
tracked as a separate, deeper follow-up (the ``archive_tiers`` package-init
import-weight audit); polylogue-sod7's own scope stops at this module
boundary rather than attempting that larger fix here.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

import click

from polylogue.paths import archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS, MigrationError


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
    from polylogue.storage.sqlite.migration_runner import migrate_archive_tier

    archive_tier = ArchiveTier(tier)
    spec = ARCHIVE_TIER_SPECS[archive_tier]
    path = archive_root() / spec.filename
    try:
        with contextlib.closing(sqlite3.connect(path)) as conn:
            result = migrate_archive_tier(conn, archive_tier, backup_manifest=backup_manifest)
    except (sqlite3.Error, MigrationError) as exc:
        if output_format == "json":
            click.echo(
                json.dumps(
                    {
                        "ok": False,
                        "tier": tier,
                        "path": str(path),
                        "backup_manifest": str(backup_manifest) if backup_manifest is not None else None,
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            click.echo(f"Migration blocked for {tier}: {exc}", err=True)
        raise SystemExit(1) from exc

    payload = {
        "ok": True,
        "tier": tier,
        "path": str(path),
        "backup_manifest": str(backup_manifest) if backup_manifest is not None else None,
        "backup_receipt": str(result.backup_receipt) if result.backup_receipt is not None else None,
        "from_version": result.from_version,
        "to_version": result.to_version,
        "applied_versions": list(result.applied_versions),
    }
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    applied = ", ".join(str(version) for version in result.applied_versions) or "none"
    click.echo(
        f"Migrated {tier}: {result.from_version} -> {result.to_version} "
        f"(applied: {applied}; receipt: {result.backup_receipt})"
    )
