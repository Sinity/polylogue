"""``maintenance run-preview``: read-only dry run of a maintenance replay.

Read-only twin of ``maintenance run`` (:mod:`polylogue.cli.commands.maintenance._run`).
Shares the same target catalog, resume/operation-id, and scope-filter surface,
and calls the identical :func:`polylogue.maintenance.replay.execute_replay`
machinery -- resumable per-target checkpointing included -- with ``dry_run``
fixed to ``True`` instead of exposed as a flag. This command never mutates the
archive; it exercises the same repair-simulation code path each target's
``execute`` step would take, which is why it is a distinct verb rather than a
lighter estimate like ``maintenance plan``.
"""

from __future__ import annotations

import click

from polylogue.cli.commands.maintenance._run import _execute_and_render
from polylogue.cli.commands.maintenance._shared import _apply_scope_filter_options
from polylogue.cli.shared.types import AppEnv
from polylogue.maintenance.targets import MAINTENANCE_TARGET_NAMES, build_maintenance_target_catalog

_MAINTENANCE_TARGET_HELP = build_maintenance_target_catalog().help_text()


@click.command("run-preview")
@click.option(
    "--target",
    "targets",
    multiple=True,
    type=click.Choice(MAINTENANCE_TARGET_NAMES),
    help=_MAINTENANCE_TARGET_HELP,
)
@click.option(
    "--operation-id",
    "operation_id",
    type=str,
    default=None,
    help=(
        "Reuse a previous operation id to resume an interrupted preview; omit to mint a fresh uuid for a new operation."
    ),
)
@click.option(
    "--resume",
    "resume_cursor",
    type=str,
    default=None,
    help=(
        "Explicit resume cursor (e.g. 'target:2'). When omitted and "
        "--operation-id matches a persisted state file, the cursor is "
        "loaded automatically."
    ),
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["plain", "json"]),
    default="plain",
    show_default=True,
    help="Output format. ``json`` emits the shared MaintenanceOperationEnvelope.",
)
@_apply_scope_filter_options
@click.pass_obj
def run_preview_command(
    env: AppEnv,
    targets: tuple[str, ...],
    operation_id: str | None,
    resume_cursor: str | None,
    output_format: str,
    session_ids: tuple[str, ...],
    origin: str | None,
    source_family: str | None,
    source_root: str | None,
    since: str | None,
    until: str | None,
    failure_kind: str | None,
    parser_version: str | None,
) -> None:
    """Preview maintenance backfill operations without executing. Read-only.

    Runs the same resumable replay path as ``run`` -- including per-target
    repair simulation and checkpoint tracking -- but never mutates the
    archive. Use --operation-id together with --resume to pick up an
    interrupted preview from its last checkpoint.
    """
    _execute_and_render(
        env=env,
        targets=targets,
        dry_run=True,
        operation_id=operation_id,
        resume_cursor=resume_cursor,
        output_format=output_format,
        session_ids=session_ids,
        origin=origin,
        source_family=source_family,
        source_root=source_root,
        since=since,
        until=until,
        failure_kind=failure_kind,
        parser_version=parser_version,
    )
