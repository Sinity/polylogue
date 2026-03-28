"""Execution and validation workflow for the check command."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from polylogue.cli.check_maintenance import (
    build_preview_counts as _build_preview_counts,
)
from polylogue.cli.check_maintenance import (
    persist_maintenance_run,
)
from polylogue.cli.check_maintenance import (
    resolve_selected_maintenance_targets as _resolve_selected_maintenance_targets,
)
from polylogue.cli.check_validation import (
    validate_check_options as _validate_check_options,
)
from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.health_archive import get_health
from polylogue.health_runtime import run_runtime_health
from polylogue.schemas.operator_workflow import (
    list_artifact_cohorts,
    list_artifact_observations,
    run_artifact_proof,
    run_schema_verification,
)
from polylogue.schemas.verification_requests import (
    ArtifactObservationQuery,
    ArtifactProofRequest,
    SchemaVerificationRequest,
)
from polylogue.storage.repair import (
    CLEANUP_TARGETS,
    SAFE_REPAIR_TARGETS,
    run_selected_maintenance,
)

from .check_support import make_schema_progress_callback, parse_schema_samples, vacuum_database


@dataclass(frozen=True)
class CheckCommandOptions:
    json_output: bool
    verbose: bool
    use_cached_health: bool
    repair: bool
    cleanup: bool
    preview: bool
    vacuum: bool
    deep: bool
    runtime: bool
    check_schemas: bool
    check_proof: bool
    check_artifacts: bool
    check_cohorts: bool
    check_roundtrip_proof: bool
    schema_providers: tuple[str, ...]
    artifact_providers: tuple[str, ...]
    artifact_statuses: tuple[str, ...]
    artifact_kinds: tuple[str, ...]
    artifact_limit: int | None
    artifact_offset: int
    roundtrip_providers: tuple[str, ...]
    roundtrip_count: int
    schema_samples: str
    schema_record_limit: int | None
    schema_record_offset: int
    schema_quarantine_malformed: bool
    maintenance_targets: tuple[str, ...]


@dataclass
class CheckCommandResult:
    report: Any
    runtime_report: Any | None = None
    schema_report: Any | None = None
    proof_report: Any | None = None
    artifact_rows: list[Any] | None = None
    cohort_rows: list[Any] | None = None
    roundtrip_report: Any | None = None
    maintenance_results: list[Any] | None = None
    vacuum_result: dict[str, Any] | None = None

def validate_check_options(options: CheckCommandOptions) -> None:
    _validate_check_options(
        options,
        safe_repair_targets=SAFE_REPAIR_TARGETS,
        cleanup_targets=CLEANUP_TARGETS,
    )


def run_check_workflow(env: AppEnv, options: CheckCommandOptions) -> CheckCommandResult:
    config = load_effective_config(env)
    report = get_health(config, deep=options.deep, use_cached=options.use_cached_health)
    result = CheckCommandResult(report=report)

    if options.runtime:
        result.runtime_report = run_runtime_health(config)

    if options.check_schemas:
        result.schema_report = run_schema_verification(
            SchemaVerificationRequest(
                providers=list(options.schema_providers) if options.schema_providers else None,
                max_samples=parse_schema_samples(options.schema_samples),
                record_limit=options.schema_record_limit,
                record_offset=options.schema_record_offset,
                quarantine_malformed=options.schema_quarantine_malformed,
                progress_callback=make_schema_progress_callback(),
            ),
            db_path=config.db_path,
        )
        print(file=sys.stderr)

    if options.check_proof:
        result.proof_report = run_artifact_proof(
            ArtifactProofRequest(
                providers=list(options.artifact_providers) if options.artifact_providers else None,
                record_limit=options.artifact_limit,
                record_offset=options.artifact_offset,
            ),
            db_path=config.db_path,
        ).report

    if options.check_artifacts:
        result.artifact_rows = list_artifact_observations(
            ArtifactObservationQuery(
                providers=list(options.artifact_providers) if options.artifact_providers else None,
                support_statuses=list(options.artifact_statuses) if options.artifact_statuses else None,
                artifact_kinds=list(options.artifact_kinds) if options.artifact_kinds else None,
                record_limit=options.artifact_limit,
                record_offset=options.artifact_offset,
            ),
            db_path=config.db_path,
        ).rows

    if options.check_cohorts:
        result.cohort_rows = list_artifact_cohorts(
            ArtifactObservationQuery(
                providers=list(options.artifact_providers) if options.artifact_providers else None,
                support_statuses=list(options.artifact_statuses) if options.artifact_statuses else None,
                artifact_kinds=list(options.artifact_kinds) if options.artifact_kinds else None,
                record_limit=options.artifact_limit,
                record_offset=options.artifact_offset,
            ),
            db_path=config.db_path,
        ).rows

    if options.check_roundtrip_proof:
        from polylogue.schemas.roundtrip_proof import prove_schema_evidence_roundtrip_suite

        try:
            result.roundtrip_report = prove_schema_evidence_roundtrip_suite(
                providers=list(options.roundtrip_providers) if options.roundtrip_providers else None,
                count=options.roundtrip_count,
            )
        except ValueError as exc:
            fail("check", str(exc))

    if options.repair or options.cleanup:
        preview_counts = _build_preview_counts(report) if options.preview else None
        selected_targets = _resolve_selected_maintenance_targets(
            options,
            safe_repair_targets=SAFE_REPAIR_TARGETS,
            cleanup_targets=CLEANUP_TARGETS,
        )
        result.maintenance_results = run_selected_maintenance(
            config,
            repair=options.repair,
            cleanup=options.cleanup,
            dry_run=options.preview,
            preview_counts=preview_counts,
            targets=selected_targets,
        )

    if (options.repair or options.cleanup) and options.vacuum:
        if options.preview:
            result.vacuum_result = {
                "ok": True,
                "preview": True,
                "detail": "Preview mode: VACUUM skipped.",
            }
        elif options.json_output:
            result.vacuum_result = vacuum_database(env)

    if result.maintenance_results is not None:
        selected_targets = _resolve_selected_maintenance_targets(
            options,
            safe_repair_targets=SAFE_REPAIR_TARGETS,
            cleanup_targets=CLEANUP_TARGETS,
        )
        preview_counts = _build_preview_counts(report) if options.preview else None
        persist_maintenance_run(
            env,
            report=report,
            options=options,
            targets=selected_targets,
            maintenance_results=result.maintenance_results,
            vacuum_result=result.vacuum_result,
            preview_counts=preview_counts,
        )

    return result
