"""``maintenance blob-disposition``: compile or consume one disposition plan."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group("blob-disposition")
def blob_disposition_group() -> None:
    """Plan and consume the physical blob namespace disposition."""


@blob_disposition_group.command("plan")
@click.option(
    "--archive-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False, readable=True),
    required=True,
    help="Archive root whose blob namespace is planned.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Destination for the immutable plan artifact.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def blob_disposition_plan_command(archive_root: Path, output: Path, output_format: str) -> None:
    """Compile a read-only, zero-unknown disposition plan. Never mutates."""
    from polylogue.maintenance.blob_disposition import compile_disposition_plan, resolve_disposition_roots

    _, hook_sources, capture_spool = resolve_disposition_roots(archive_root)
    try:
        plan = compile_disposition_plan(
            archive_root=archive_root,
            blob_root=archive_root / "blob",
            source_db=archive_root / "source.db",
            hook_spool_sources=hook_sources,
            browser_capture_spool=capture_spool,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    summary = {
        "plan": str(output),
        "digest": plan.digest(),
        "accepted": plan.accepted,
        "counts": plan.counts,
        "bytes_by_disposition": plan.bytes_by_disposition,
        "denominator": plan.denominator.to_dict(),
    }
    if output_format == "json":
        click.echo(json.dumps(summary, sort_keys=True))
        return
    click.echo(f"Disposition plan: {output}")
    click.echo(f"Digest: {plan.digest()}")
    click.echo(f"Accepted (zero unresolved): {plan.accepted}")
    click.echo(f"Counts: {json.dumps(plan.counts, sort_keys=True)}")
    click.echo("Read-only: true")


@blob_disposition_group.command("restore")
@click.option(
    "--archive-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False, readable=True),
    required=True,
    help="Archive root the plan was compiled from.",
)
@click.option(
    "--plan",
    "plan_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    required=True,
    help="The plan naming the sole-copy carriers to restore.",
)
@click.option("--authorized-digest", required=True, help="Digest of the reviewed plan.")
@click.option(
    "--receipt",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Destination for the restoration receipt.",
)
@click.option(
    "--active",
    is_flag=True,
    default=False,
    help="Perform the restorations. Without it the run is a dry rehearsal.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def blob_disposition_restore_command(
    archive_root: Path,
    plan_path: Path,
    authorized_digest: str,
    receipt: Path,
    active: bool,
    output_format: str,
) -> None:
    """Restore sole-copy carriers into their ordinary spool. Deletes nothing.

    Restoration is additive, so it does not wait on the whole plan reaching
    zero unresolved: withholding it would leave the only carrier of wanted
    material unpreserved while unrelated objects are still being classified.
    """
    from polylogue.maintenance.blob_disposition import (
        BlobDispositionPlan,
        build_disposition_context,
        resolve_disposition_roots,
    )
    from polylogue.maintenance.blob_disposition_apply import (
        TOOL_VERSION,
        DispositionApplyReceipt,
        restore_plan_members,
        write_receipt,
    )

    hooks_root, hook_sources, capture_spool = resolve_disposition_roots(archive_root)
    try:
        plan = BlobDispositionPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
        if plan.digest() != authorized_digest:
            raise click.ClickException("authorized digest does not match the plan")
        context = build_disposition_context(
            archive_root=archive_root,
            blob_root=archive_root / "blob",
            source_db=archive_root / "source.db",
            hook_spool_sources=hook_sources,
            browser_capture_spool=capture_spool,
        )
        results = restore_plan_members(
            plan,
            context=context,
            hook_spool_root=hooks_root,
            browser_capture_spool=capture_spool,
            dry_run=not active,
        )
        result = DispositionApplyReceipt(
            tool_version=TOOL_VERSION,
            plan_digest=plan.digest(),
            archive_root=plan.archive_root,
            blob_root=plan.blob_root,
            dry_run=not active,
            results=results,
        )
        write_receipt(receipt, result)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(json.dumps({"receipt": str(receipt), "ok": result.ok, "counts": result.counts}, sort_keys=True))
    else:
        click.echo(f"Restoration receipt: {receipt}")
        click.echo(f"Dry run: {not active}")
        click.echo(f"Counts: {json.dumps(result.counts, sort_keys=True)}")
    if not result.ok:
        raise SystemExit(1)


@blob_disposition_group.command("apply")
@click.option(
    "--archive-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False, readable=True),
    required=True,
    help="Archive root the authorized plan was compiled from.",
)
@click.option(
    "--plan",
    "plan_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    required=True,
    help="The exact accepted plan artifact.",
)
@click.option("--authorized-digest", required=True, help="Digest of the independently accepted plan.")
@click.option(
    "--receipt",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Destination for the complete before/after receipt.",
)
@click.option(
    "--active",
    is_flag=True,
    default=False,
    help="Perform the authorized effects. Without it the run is a dry rehearsal.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
def blob_disposition_apply_command(
    archive_root: Path,
    plan_path: Path,
    authorized_digest: str,
    receipt: Path,
    active: bool,
    output_format: str,
) -> None:
    """Restore sole copies, then delete proven-redundant objects."""
    from polylogue.config import Config
    from polylogue.maintenance.blob_disposition import (
        BlobDispositionPlan,
        build_disposition_context,
        resolve_disposition_roots,
    )
    from polylogue.maintenance.blob_disposition_apply import apply_disposition_plan, write_receipt
    from polylogue.maintenance.offline_guard import offline_writer_block_reason
    from polylogue.paths import render_root

    hooks_root, hook_sources, capture_spool = resolve_disposition_roots(archive_root)
    try:
        plan = BlobDispositionPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
        context = build_disposition_context(
            archive_root=archive_root,
            blob_root=archive_root / "blob",
            source_db=archive_root / "source.db",
            hook_spool_sources=hook_sources,
            browser_capture_spool=capture_spool,
        )
        block_reason = offline_writer_block_reason(
            Config(archive_root=archive_root, render_root=render_root(), sources=[])
        )
        result = apply_disposition_plan(
            plan,
            context=context,
            authorized_digest=authorized_digest,
            source_db=archive_root / "source.db",
            index_db=archive_root / "index.db",
            hook_spool_root=hooks_root,
            browser_capture_spool=capture_spool,
            writer_block_reason=block_reason,
            dry_run=not active,
        )
        write_receipt(receipt, result)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    summary = {"receipt": str(receipt), "ok": result.ok, "counts": result.counts, "blockers": list(result.blockers)}
    if output_format == "json":
        click.echo(json.dumps(summary, sort_keys=True))
    else:
        click.echo(f"Disposition receipt: {receipt}")
        click.echo(f"Dry run: {not active}")
        click.echo(f"Counts: {json.dumps(result.counts, sort_keys=True)}")
        for blocker in result.blockers:
            click.echo(f"Blocked: {blocker}")
    if not result.ok:
        raise SystemExit(1)


__all__ = [
    "blob_disposition_apply_command",
    "blob_disposition_group",
    "blob_disposition_plan_command",
    "blob_disposition_restore_command",
]
