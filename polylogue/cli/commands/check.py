"""Health check command."""

from __future__ import annotations

from typing import Any

import click

from polylogue.cli.check_rendering import emit_json_output, render_plain_output
from polylogue.cli.check_support import (
    format_count_mapping as _format_count_mapping_impl,
)
from polylogue.cli.check_support import (
    format_semantic_metric_summary as _format_semantic_metric_summary_impl,
)
from polylogue.cli.check_support import (
    make_schema_progress_callback as _make_schema_progress_callback_impl,
)
from polylogue.cli.check_support import (
    parse_schema_samples as _parse_schema_samples_impl,
)
from polylogue.cli.check_support import run_vacuum as _run_vacuum_impl
from polylogue.cli.check_support import vacuum_database as _vacuum_database_impl
from polylogue.cli.check_workflow import (
    CheckCommandOptions,
    run_check_workflow,
    validate_check_options,
)
from polylogue.cli.types import AppEnv


def _format_count_mapping(counts: dict[str, int]) -> str:
    return _format_count_mapping_impl(counts)


def _format_semantic_metric_summary(metric_summary: dict[str, dict[str, int]]) -> str:
    return _format_semantic_metric_summary_impl(metric_summary)


@click.command("check")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show breakdown by provider")
@click.option("--cached", "use_cached_health", is_flag=True, help="Use the recent cached archive-health report when available")
@click.option("--repair", is_flag=True, help="Run safe derived-data and database maintenance repairs")
@click.option("--cleanup", is_flag=True, help="Run destructive archive cleanup for orphaned or empty persisted data")
@click.option("--preview", is_flag=True, help="Preview maintenance without executing (requires --repair or --cleanup)")
@click.option("--vacuum", is_flag=True, help="Reclaim unused space after maintenance (requires --repair or --cleanup)")
@click.option("--deep", is_flag=True, help="Run SQLite integrity check (slow on large databases)")
@click.option("--runtime", is_flag=True, help="Run environment and runtime verification checks")
@click.option("--schemas", "check_schemas", is_flag=True, help="Run raw-corpus schema verification (non-mutating)")
@click.option("--proof", "check_proof", is_flag=True, help="Run durable artifact support proof")
@click.option("--artifacts", "check_artifacts", is_flag=True, help="List durable artifact observations")
@click.option("--cohorts", "check_cohorts", is_flag=True, help="Summarize durable artifact cohorts")
@click.option(
    "--semantic-proof",
    "check_semantic_proof",
    is_flag=True,
    help="Run semantic preservation proof across canonical, export, query, stream, and MCP read surfaces",
)
@click.option(
    "--semantic-contracts",
    "check_semantic_contracts",
    is_flag=True,
    help="List declared semantic-proof surface contracts and aliases",
)
@click.option(
    "--roundtrip-proof",
    "check_roundtrip_proof",
    is_flag=True,
    help="Run the synthetic schema-and-evidence roundtrip proof lane in an isolated workspace",
)
@click.option("--schema-provider", "schema_providers", multiple=True, help="Limit schema verification to DB provider name (repeatable)")
@click.option(
    "--artifact-provider",
    "artifact_providers",
    multiple=True,
    help="Limit artifact proof/listing/cohorting to effective provider (repeatable)",
)
@click.option(
    "--artifact-status",
    "artifact_statuses",
    multiple=True,
    help="Limit artifact listing/cohorting to support status (repeatable)",
)
@click.option(
    "--artifact-kind",
    "artifact_kinds",
    multiple=True,
    help="Limit artifact listing/cohorting to artifact kind (repeatable)",
)
@click.option(
    "--artifact-limit",
    type=int,
    default=None,
    help="Limit artifact proof/listing/cohorting to N observation rows",
)
@click.option(
    "--artifact-offset",
    type=int,
    default=0,
    show_default=True,
    help="Start offset for artifact proof/listing/cohorting",
)
@click.option(
    "--semantic-provider",
    "semantic_providers",
    multiple=True,
    help="Limit semantic proof to conversation providers (repeatable)",
)
@click.option(
    "--semantic-surface",
    "semantic_surfaces",
    multiple=True,
    help="Limit semantic proof to canonical/export/query/stream/MCP surfaces such as canonical, export_all, query_all, stream_all, mcp_all, read_all, or all",
)
@click.option(
    "--semantic-limit",
    type=int,
    default=None,
    help="Limit semantic proof to N conversations",
)
@click.option(
    "--semantic-offset",
    type=int,
    default=0,
    show_default=True,
    help="Start offset for semantic proof",
)
@click.option(
    "--roundtrip-provider",
    "roundtrip_providers",
    multiple=True,
    help="Limit roundtrip proof to specific providers (repeatable)",
)
@click.option(
    "--roundtrip-count",
    type=int,
    default=1,
    show_default=True,
    help="Synthetic artifacts per provider for roundtrip proof",
)
@click.option(
    "--schema-samples",
    default="all",
    show_default=True,
    help="Validation samples per raw payload: positive integer or 'all'",
)
@click.option(
    "--schema-record-limit",
    type=int,
    default=None,
    help="Limit schema verification to N raw records (for chunked runs)",
)
@click.option(
    "--schema-record-offset",
    type=int,
    default=0,
    show_default=True,
    help="Start offset for chunked schema verification",
)
@click.option(
    "--schema-quarantine-malformed",
    is_flag=True,
    help="Mark malformed raw payloads as failed validation during schema verification (mutates DB)",
)
@click.pass_obj
def check_command(
    env: AppEnv,
    json_output: bool,
    verbose: bool,
    use_cached_health: bool,
    repair: bool,
    cleanup: bool,
    preview: bool,
    vacuum: bool,
    deep: bool,
    runtime: bool,
    check_schemas: bool,
    check_proof: bool,
    check_artifacts: bool,
    check_cohorts: bool,
    check_semantic_proof: bool,
    check_semantic_contracts: bool,
    check_roundtrip_proof: bool,
    schema_providers: tuple[str, ...],
    artifact_providers: tuple[str, ...],
    artifact_statuses: tuple[str, ...],
    artifact_kinds: tuple[str, ...],
    artifact_limit: int | None,
    artifact_offset: int,
    semantic_providers: tuple[str, ...],
    semantic_surfaces: tuple[str, ...],
    semantic_limit: int | None,
    semantic_offset: int,
    roundtrip_providers: tuple[str, ...],
    roundtrip_count: int,
    schema_samples: str,
    schema_record_limit: int | None,
    schema_record_offset: int,
    schema_quarantine_malformed: bool,
) -> None:
    """Health check with optional maintenance and cleanup previews."""
    options = CheckCommandOptions(
        json_output=json_output,
        verbose=verbose,
        use_cached_health=use_cached_health,
        repair=repair,
        cleanup=cleanup,
        preview=preview,
        vacuum=vacuum,
        deep=deep,
        runtime=runtime,
        check_schemas=check_schemas,
        check_proof=check_proof,
        check_artifacts=check_artifacts,
        check_cohorts=check_cohorts,
        check_semantic_proof=check_semantic_proof,
        check_semantic_contracts=check_semantic_contracts,
        check_roundtrip_proof=check_roundtrip_proof,
        schema_providers=schema_providers,
        artifact_providers=artifact_providers,
        artifact_statuses=artifact_statuses,
        artifact_kinds=artifact_kinds,
        artifact_limit=artifact_limit,
        artifact_offset=artifact_offset,
        semantic_providers=semantic_providers,
        semantic_surfaces=semantic_surfaces,
        semantic_limit=semantic_limit,
        semantic_offset=semantic_offset,
        roundtrip_providers=roundtrip_providers,
        roundtrip_count=roundtrip_count,
        schema_samples=schema_samples,
        schema_record_limit=schema_record_limit,
        schema_record_offset=schema_record_offset,
        schema_quarantine_malformed=schema_quarantine_malformed,
    )
    validate_check_options(options)
    result = run_check_workflow(env, options)
    if options.json_output:
        emit_json_output(result, options)
        return
    render_plain_output(env, result, options)


def _make_schema_progress_callback():
    return _make_schema_progress_callback_impl()


def _run_vacuum(env: AppEnv) -> None:
    _run_vacuum_impl(env)


def _vacuum_database(env: AppEnv) -> dict[str, Any]:
    return _vacuum_database_impl(env)


def _parse_schema_samples(raw: str) -> int | None:
    return _parse_schema_samples_impl(raw)


__all__ = ["check_command"]
