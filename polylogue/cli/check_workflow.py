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
from polylogue.cli.helpers import load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.health import get_health, run_runtime_health
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
from polylogue.storage.backends.connection import connection_context
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
    repair: bool
    cleanup: bool
    preview: bool
    vacuum: bool
    deep: bool
    runtime: bool
    check_blob: bool
    check_schemas: bool
    check_proof: bool
    check_artifacts: bool
    check_cohorts: bool
    schema_providers: tuple[str, ...]
    artifact_providers: tuple[str, ...]
    artifact_statuses: tuple[str, ...]
    artifact_kinds: tuple[str, ...]
    artifact_limit: int | None
    artifact_offset: int
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
    maintenance_results: list[Any] | None = None
    vacuum_result: dict[str, Any] | None = None


def validate_check_options(options: CheckCommandOptions) -> None:
    _validate_check_options(
        options,
        safe_repair_targets=SAFE_REPAIR_TARGETS,
        cleanup_targets=CLEANUP_TARGETS,
    )


def _runtime_only_requested(options: CheckCommandOptions) -> bool:
    return options.runtime and not any(
        (
            options.repair,
            options.cleanup,
            options.deep,
            options.check_blob,
            options.check_schemas,
            options.check_proof,
            options.check_artifacts,
            options.check_cohorts,
        )
    )


def run_check_workflow(env: AppEnv, options: CheckCommandOptions) -> CheckCommandResult:
    config = load_effective_config(env)
    if _runtime_only_requested(options):
        return CheckCommandResult(report=run_runtime_health(config))

    report = get_health(config, deep=options.deep)
    result = CheckCommandResult(report=report)

    if options.runtime:
        result.runtime_report = run_runtime_health(config)

    if options.check_blob:
        from polylogue.storage.blob_store import get_blob_store

        blob_store = get_blob_store()
        db_raw_ids: set[str] = set()
        with connection_context(config.db_path) as conn:
            for row in conn.execute("SELECT raw_id FROM raw_conversations"):
                db_raw_ids.add(row[0])
        disk_hashes = set(blob_store.iter_all())
        missing = db_raw_ids - disk_hashes
        orphaned = disk_hashes - db_raw_ids
        env.ui.console.print(f"Blob store: {len(disk_hashes)} blobs on disk, {len(db_raw_ids)} raw records in DB")
        if missing:
            env.ui.console.print(f"  MISSING: {len(missing)} blobs referenced in DB but not on disk")
            for h in sorted(missing)[:10]:
                env.ui.console.print(f"    {h[:16]}...")
        if orphaned:
            env.ui.console.print(f"  Orphaned: {len(orphaned)} blobs on disk not in DB")
        if not missing and not orphaned:
            env.ui.console.print("  All blobs verified.")

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
