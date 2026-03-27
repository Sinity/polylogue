"""Health check command."""

from __future__ import annotations

from typing import Any

import click

from polylogue.cli.check_options import apply_check_command_options
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
from polylogue.cli.check_workflow import CheckCommandOptions, run_check_workflow, validate_check_options
from polylogue.cli.types import AppEnv


def _format_count_mapping(counts: dict[str, int]) -> str:
    return _format_count_mapping_impl(counts)


def _format_semantic_metric_summary(metric_summary: dict[str, dict[str, int]]) -> str:
    return _format_semantic_metric_summary_impl(metric_summary)


@click.command("check")
@apply_check_command_options
@click.pass_obj
def check_command(
    env: AppEnv,
    json_output: bool,
    verbose: bool,
    use_cached_health: bool,
    repair: bool,
    cleanup: bool,
    maintenance_targets: tuple[str, ...],
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
        maintenance_targets=maintenance_targets,
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
