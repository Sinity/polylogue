"""``maintenance backup-plan``: inspect archive backup boundaries without copying data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from polylogue.paths import archive_root

_BACKUP_PROFILES: tuple[dict[str, object], ...] = (
    {
        "name": "full_evidence",
        "include": ["source.db", "index.db", "embeddings.db", "user.db", "blob/", "ops.db optional"],
        "exclude": ["*-wal after checkpoint", "*-shm after checkpoint"],
        "use_case": "fastest complete restore with raw evidence, read models, vectors, and overlays",
    },
    {
        "name": "user_overlays",
        "include": ["user.db", "user-owned referenced blobs"],
        "exclude": ["index.db", "ops.db", "rebuildable derived models"],
        "use_case": "protect irreplaceable human and agent state before resets or schema rebuilds",
    },
    {
        "name": "rebuildable_cache_exclude",
        "include": ["source.db", "user.db", "blob/", "embeddings.db optional"],
        "exclude": ["index.db", "ops.db", "derived/cache artifacts"],
        "use_case": "smaller backup that can rebuild parsed and indexed data locally",
    },
    {
        "name": "diagnostics_bundle",
        "include": ["ops.db", "backup-plan json", "workload diagnostics json", "logs", "readonly status outputs"],
        "exclude": ["private raw blobs unless explicitly needed"],
        "use_case": "incident triage without over-sharing archive contents",
    },
)


@click.command("backup-plan")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def backup_plan_command(output_format: str) -> None:
    """Inspect archive backup boundaries without copying data.

    Reports the backup class and presence of each archive tier, blob-store
    boundary, and the named backup profiles documented for operators.
    """
    import json

    root = archive_root()
    payload = _backup_plan_payload(root)

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_backup_plain_plan(payload)


def _backup_plan_payload(root: Path) -> dict[str, Any]:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    backup_policies: dict[ArchiveTier, dict[str, object]] = {
        ArchiveTier.SOURCE: {
            "backup_class": "critical",
            "policy": "back_up",
            "restore_role": "raw acquisition evidence and rebuild root",
        },
        ArchiveTier.INDEX: {
            "backup_class": "warm_cache",
            "policy": "optional_full_evidence",
            "restore_role": "parsed sessions, search indexes, graph rows, and derived read models",
        },
        ArchiveTier.EMBEDDINGS: {
            "backup_class": "expensive_rebuild",
            "policy": "back_up_when_present",
            "restore_role": "vector rows and embedding catch-up state",
        },
        ArchiveTier.USER: {
            "backup_class": "critical",
            "policy": "always_back_up",
            "restore_role": "human and agent overlays",
        },
        ArchiveTier.OPS: {
            "backup_class": "diagnostic",
            "policy": "diagnostics_only",
            "restore_role": "daemon cursors, convergence debt, and runtime telemetry",
        },
    }

    tier_payloads = []
    wal_warnings: list[str] = []
    for tier in ArchiveTier:
        spec = ARCHIVE_TIER_SPECS[tier]
        path = root / spec.filename
        wal_path = path.with_name(f"{path.name}-wal")
        shm_path = path.with_name(f"{path.name}-shm")
        wal_present = wal_path.exists()
        if wal_present:
            wal_warnings.append(f"{spec.filename}-wal is present; checkpoint before copying {spec.filename}")
        policy = backup_policies[tier]
        tier_payloads.append(
            {
                "tier": tier.value,
                "filename": spec.filename,
                "path": str(path),
                "present": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "durability": spec.durability,
                "expected_user_version": spec.version,
                "backup_required": spec.backup_required,
                "backup_class": policy["backup_class"],
                "backup_policy": policy["policy"],
                "restore_role": policy["restore_role"],
                "wal_present": wal_present,
                "shm_present": shm_path.exists(),
                "checkpoint_recommended": wal_present,
            }
        )

    blob_root = root / "blob"
    return {
        "ok": True,
        "mode": "backup_plan",
        "archive_root": str(root),
        "tiers": tier_payloads,
        "blob_store": {
            "path": str(blob_root),
            "present": blob_root.exists(),
            "backup_policy": "back_up_referenced_blobs_with_source_and_user_tiers",
            "gc_safety_boundary": "source.db raw references plus the gc_generations age floor are authoritative",
        },
        "profiles": list(_BACKUP_PROFILES),
        "warnings": wal_warnings,
        "mutates": False,
    }


def _render_backup_plain_plan(payload: dict[str, Any]) -> None:
    click.echo("Archive backup plan")
    click.echo(f"Archive root: {payload['archive_root']}")
    click.echo("")
    click.echo("Tiers:")
    for tier in payload["tiers"]:
        assert isinstance(tier, dict)
        status = "present" if tier["present"] else "missing"
        checkpoint = " checkpoint-before-copy" if tier["checkpoint_recommended"] else ""
        click.echo(f"  {tier['filename']}: {tier['backup_class']} policy={tier['backup_policy']} {status}{checkpoint}")
    click.echo("")
    click.echo("Profiles:")
    for profile in payload["profiles"]:
        assert isinstance(profile, dict)
        click.echo(f"  {profile['name']}: {profile['use_case']}")
    warnings = payload["warnings"]
    if warnings:
        click.echo("")
        click.echo("Warnings:")
        for warning in warnings:
            click.echo(f"  {warning}")
