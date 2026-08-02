"""Health check command."""

from __future__ import annotations

import click

from polylogue.cli.shared.check_options import apply_check_command_options
from polylogue.cli.shared.check_rendering_json import emit_json_output
from polylogue.cli.shared.check_rendering_plain import render_plain_output
from polylogue.cli.shared.check_workflow import CheckCommandOptions, run_check_workflow, validate_check_options
from polylogue.cli.shared.types import AppEnv


@click.command("doctor")
@apply_check_command_options
@click.pass_obj
def check_command(
    env: AppEnv,
    output_format: str | None,
    verbose: bool,
    repair: bool,
    cleanup: bool,
    maintenance_targets: tuple[str, ...],
    preview: bool,
    vacuum: bool,
    deep: bool,
    runtime: bool,
    check_daemon: bool,
    check_blob: bool,
    blob_integrity_full: bool,
    check_schemas: bool,
    check_artifact_coverage: bool,
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
        json_output=output_format == "json",
        verbose=verbose,
        repair=repair,
        cleanup=cleanup,
        maintenance_targets=maintenance_targets,
        preview=preview,
        vacuum=vacuum,
        deep=deep,
        runtime=runtime,
        check_daemon=check_daemon,
        check_blob=check_blob,
        blob_integrity_full=blob_integrity_full,
        check_schemas=check_schemas,
        check_artifact_coverage=check_artifact_coverage,
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


__all__ = ["check_command"]
