"""Shared Click option decorators for the check command."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

from polylogue.maintenance_targets import MAINTENANCE_TARGET_NAMES, build_maintenance_target_catalog

_MAINTENANCE_TARGET_HELP = build_maintenance_target_catalog().help_text()

CHECK_COMMAND_OPTION_DECORATORS: tuple[Callable[[Callable[..., Any]], Callable[..., Any]], ...] = (
    click.option("--json", "json_output", is_flag=True, help="Output as JSON"),
    click.option("--verbose", "-v", is_flag=True, help="Show breakdown by provider"),
    click.option("--repair", is_flag=True, help="Run safe derived-data and database maintenance repairs"),
    click.option(
        "--cleanup", is_flag=True, help="Run destructive archive cleanup for orphaned or empty persisted data"
    ),
    click.option(
        "--target",
        "maintenance_targets",
        multiple=True,
        type=click.Choice(MAINTENANCE_TARGET_NAMES),
        help=_MAINTENANCE_TARGET_HELP,
    ),
    click.option(
        "--preview", is_flag=True, help="Preview maintenance without executing (requires --repair or --cleanup)"
    ),
    click.option(
        "--vacuum", is_flag=True, help="Reclaim unused space after maintenance (requires --repair or --cleanup)"
    ),
    click.option(
        "--deep",
        is_flag=True,
        help="Run SQLite integrity and expensive orphan scans (slow on large databases)",
    ),
    click.option("--runtime", is_flag=True, help="Run environment and runtime verification checks"),
    click.option("--blob", "check_blob", is_flag=True, help="Verify blob store integrity (missing/orphaned blobs)"),
    click.option("--schemas", "check_schemas", is_flag=True, help="Run raw-corpus schema verification (non-mutating)"),
    click.option("--proof", "check_proof", is_flag=True, help="Run durable artifact support proof"),
    click.option("--artifacts", "check_artifacts", is_flag=True, help="List durable artifact observations"),
    click.option("--cohorts", "check_cohorts", is_flag=True, help="Summarize durable artifact cohorts"),
    click.option(
        "--schema-provider",
        "schema_providers",
        multiple=True,
        help="Limit schema verification to DB provider name (repeatable)",
    ),
    click.option(
        "--artifact-provider",
        "artifact_providers",
        multiple=True,
        help="Limit artifact proof/listing/cohorting to effective provider (repeatable)",
    ),
    click.option(
        "--artifact-status",
        "artifact_statuses",
        multiple=True,
        help="Limit artifact listing/cohorting to support status (repeatable)",
    ),
    click.option(
        "--artifact-kind",
        "artifact_kinds",
        multiple=True,
        help="Limit artifact listing/cohorting to artifact kind (repeatable)",
    ),
    click.option(
        "--artifact-limit", type=int, default=None, help="Limit artifact proof/listing/cohorting to N observation rows"
    ),
    click.option(
        "--artifact-offset",
        type=int,
        default=0,
        show_default=True,
        help="Start offset for artifact proof/listing/cohorting",
    ),
    click.option(
        "--schema-samples",
        default="all",
        show_default=True,
        help="Validation samples per raw payload: positive integer or 'all'",
    ),
    click.option(
        "--schema-record-limit",
        type=int,
        default=None,
        help="Limit schema verification to N raw records (for chunked runs)",
    ),
    click.option(
        "--schema-record-offset",
        type=int,
        default=0,
        show_default=True,
        help="Start offset for chunked schema verification",
    ),
    click.option(
        "--schema-quarantine-malformed",
        is_flag=True,
        help="Mark malformed raw payloads as failed validation during schema verification (mutates DB)",
    ),
)


def apply_check_command_options(func: Callable[..., Any]) -> Callable[..., Any]:
    for decorator in reversed(CHECK_COMMAND_OPTION_DECORATORS):
        func = decorator(func)
    return func


__all__ = ["CHECK_COMMAND_OPTION_DECORATORS", "apply_check_command_options"]
