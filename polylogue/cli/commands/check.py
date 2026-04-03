"""Health check command."""

from __future__ import annotations

from typing import Any

import click

from polylogue.cli.check_options import apply_check_command_options
from polylogue.cli.check_rendering_json import emit_json_output
from polylogue.cli.check_rendering_plain import render_plain_output
from polylogue.cli.check_support import (
    format_count_mapping as _format_count_mapping_impl,
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


@click.command("doctor")
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
    check_blob: bool,
    check_schemas: bool,
    check_proof: bool,
    check_artifacts: bool,
    check_cohorts: bool,
    schema_providers: tuple[str, ...],
    artifact_providers: tuple[str, ...],
    artifact_statuses: tuple[str, ...],
    artifact_kinds: tuple[str, ...],
    artifact_limit: int | None,
    artifact_offset: int,
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
        check_blob=check_blob,
        check_schemas=check_schemas,
        check_proof=check_proof,
        check_artifacts=check_artifacts,
        check_cohorts=check_cohorts,
        schema_providers=schema_providers,
        artifact_providers=artifact_providers,
        artifact_statuses=artifact_statuses,
        artifact_kinds=artifact_kinds,
        artifact_limit=artifact_limit,
        artifact_offset=artifact_offset,
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
