"""``maintenance archive-plan`` / ``archive-init``: archive file-set readiness and creation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from polylogue.paths import archive_root

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive_init import ArchiveInitResult
    from polylogue.storage.sqlite.archive_tiers.archive_plan import ArchiveInitPlan


@click.command("archive-plan")
@click.option(
    "--replace-existing",
    is_flag=True,
    help="Classify existing archive tier files as replaceable instead of blocking.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def archive_plan_command(replace_existing: bool, output_format: str) -> None:
    """Inspect readiness for the archive file set.

    Read-only. The command reports the planned source/index/embeddings/user/ops
    target files, required backups, and blockers before archive initialization.
    """
    from polylogue.storage.sqlite.archive_tiers.archive_plan import build_archive_init_plan

    plan = build_archive_init_plan(
        archive_root=archive_root(),
        replace_existing=replace_existing,
    )

    if output_format == "json":
        click.echo(json.dumps(_archive_plan_payload(plan), indent=2, sort_keys=True))
        return

    _render_archive_plain_plan(plan)


def _archive_plan_payload(plan: ArchiveInitPlan) -> dict[str, object]:
    return {
        "ready": plan.ready,
        "archive_root": str(plan.archive_root),
        "blockers": list(plan.blockers),
        "tiers": [
            {
                "tier": tier_plan.tier.value,
                "path": str(tier_plan.path),
                "durability": tier_plan.durability,
                "exists": tier_plan.exists,
                "user_version": tier_plan.user_version,
                "expected_user_version": tier_plan.expected_user_version,
                "backup_required": tier_plan.backup_required,
                "action": tier_plan.action.value,
                "backup_path": str(tier_plan.backup_path) if tier_plan.backup_path is not None else None,
                "blockers": list(tier_plan.blockers),
            }
            for tier_plan in plan.tiers
        ],
    }


@click.command("archive-init")
@click.option(
    "--replace-existing",
    is_flag=True,
    help="Back up and replace existing durable archive tier files; recreate disposable ops state.",
)
@click.option("--yes", "-y", is_flag=True, help="Actually create the archive tier files.")
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format.",
)
def archive_init_command(replace_existing: bool, yes: bool, output_format: str) -> None:
    """Initialize the archive file set after explicit confirmation.

    This command backs up any durable archive targets it is allowed to replace,
    then creates fresh source, index, embeddings, user, and ops databases.
    Ingest and read surfaces populate and consume the archive.
    """
    from polylogue.storage.sqlite.archive_tiers.archive_init import (
        ArchiveInitBlockedError,
        initialize_archive_tier_files_from_plan,
    )
    from polylogue.storage.sqlite.archive_tiers.archive_plan import build_archive_init_plan

    plan = build_archive_init_plan(
        archive_root=archive_root(),
        replace_existing=replace_existing,
    )
    if not yes:
        if output_format == "json":
            payload = _archive_plan_payload(plan)
            payload["executed"] = False
            payload["next_action"] = "rerun with --yes to initialize the archive tier file set"
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        click.echo("Dry run only. Use --yes to initialize the archive tier file set.")
        click.echo("")
        _render_archive_plain_plan(plan)
        return

    try:
        result = initialize_archive_tier_files_from_plan(plan)
    except ArchiveInitBlockedError as exc:
        if output_format == "json":
            payload = _archive_plan_payload(plan)
            payload["executed"] = False
            payload["error"] = str(exc)
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _render_archive_plain_plan(plan)
            click.echo(f"\nBlocked: {exc}", err=True)
        raise SystemExit(1) from exc

    if output_format == "json":
        payload = _archive_init_payload(result)
        payload["executed"] = True
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("Initialized:")
    for tier_result in result.tier_results:
        backup = f" backup={tier_result.backup_path}" if tier_result.backup_path is not None else ""
        click.echo(f"  {tier_result.tier}: {tier_result.action.value} {tier_result.path}{backup}")


def _render_archive_plain_plan(plan: ArchiveInitPlan) -> None:
    status = "ready" if plan.ready else "blocked"
    click.echo(f"Archive tier initialization: {status}")
    click.echo(f"Archive root: {plan.archive_root}")
    click.echo("")
    click.echo("Targets:")
    for tier_plan in plan.tiers:
        backup = f" backup={tier_plan.backup_path}" if tier_plan.backup_path is not None else ""
        version = f" version={tier_plan.user_version}" if tier_plan.user_version is not None else ""
        click.echo(
            f"  {tier_plan.tier.value}: {tier_plan.action.value} {tier_plan.path}"
            f" durability={tier_plan.durability} expected_version={tier_plan.expected_user_version}"
            f"{version}{backup}"
        )
        for blocker in tier_plan.blockers:
            click.echo(f"    blocker: {blocker}")
    if plan.blockers:
        click.echo("")
        click.echo("Blockers:")
        for blocker in plan.blockers:
            click.echo(f"  {blocker}")


def _archive_init_payload(result: ArchiveInitResult) -> dict[str, object]:
    return {
        "tiers": [
            {
                "tier": tier_result.tier,
                "path": str(tier_result.path),
                "action": tier_result.action.value,
                "backup_path": str(tier_result.backup_path) if tier_result.backup_path is not None else None,
                "initialized": tier_result.initialized,
            }
            for tier_result in result.tier_results
        ],
    }
