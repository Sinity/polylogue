"""Execution and validation workflow for the check command."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from polylogue.cli.shared.check_maintenance import (
    build_preview_counts as _build_preview_counts,
)
from polylogue.cli.shared.check_maintenance import (
    persist_maintenance_run,
)
from polylogue.cli.shared.check_maintenance import (
    resolve_selected_maintenance_targets as _resolve_selected_maintenance_targets,
)
from polylogue.cli.shared.check_models import CheckCommandResult, VacuumResult
from polylogue.cli.shared.check_validation import validate_check_options as _validate_check_options
from polylogue.cli.shared.helpers import load_effective_config
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.core.json import JSONDocument, json_document
from polylogue.daemon.status import daemon_status_payload
from polylogue.protocols import ProgressCallback
from polylogue.readiness import ReadinessReport, get_readiness, run_runtime_readiness
from polylogue.schemas.operator.workflow import (
    list_artifact_cohorts,
    list_artifact_observations,
    run_artifact_proof,
    run_schema_verification,
)
from polylogue.schemas.validation.models import SchemaVerificationReport
from polylogue.schemas.validation.requests import (
    ArtifactObservationQuery,
    ArtifactProofRequest,
    SchemaVerificationRequest,
)
from polylogue.storage.repair import run_selected_maintenance
from polylogue.storage.sqlite.connection import connection_context

from .check_support import (
    make_schema_progress_callback,
    make_session_insight_progress_callback,
    parse_schema_samples,
    vacuum_database,
)


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
    check_daemon: bool
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


@dataclass(frozen=True)
class _MaintenanceRunInputs:
    selected_targets: tuple[str, ...]
    preview_counts: dict[str, int] | None


def validate_check_options(options: CheckCommandOptions) -> None:
    _validate_check_options(options)


def _runtime_only_requested(options: CheckCommandOptions) -> bool:
    return options.runtime and not any(
        (
            options.repair,
            options.cleanup,
            options.preview,
            options.vacuum,
            options.deep,
            options.check_daemon,
            options.check_blob,
            options.check_schemas,
            options.check_proof,
            options.check_artifacts,
            options.check_cohorts,
        )
    )


def _provider_filter(values: tuple[str, ...]) -> list[str] | None:
    return list(values) if values else None


def _artifact_query(options: CheckCommandOptions) -> ArtifactObservationQuery:
    return ArtifactObservationQuery(
        providers=_provider_filter(options.artifact_providers),
        support_statuses=list(options.artifact_statuses) if options.artifact_statuses else None,
        artifact_kinds=list(options.artifact_kinds) if options.artifact_kinds else None,
        record_limit=options.artifact_limit,
        record_offset=options.artifact_offset,
    )


def _run_blob_store_check(
    env: AppEnv, config: Config, *, deep: bool = False, json_output: bool = False
) -> JSONDocument | None:
    """Verify blob store integrity: missing/orphaned blobs + optional integrity check.

    Uses ``detect_orphans()`` for memory-bounded orphan counting with byte
    totals.  When *deep* is True, also runs ``verify_all()`` to re-hash every
    blob on disk — this is I/O-intensive on large archives.
    """
    from polylogue.storage.blob_store import get_blob_store

    blob_store = get_blob_store()
    db_raw_ids: set[str] = set()
    with connection_context(config.db_path) as conn:
        for row in conn.execute("SELECT raw_id FROM raw_conversations"):
            db_raw_ids.add(row[0])

    # Missing: DB references without disk blobs
    disk_hashes = set(blob_store.iter_all())
    missing = sorted(db_raw_ids - disk_hashes)

    # Orphans: disk blobs without DB references (with byte totals)
    orphan_result = blob_store.detect_orphans(db_raw_ids)

    # Deep integrity: re-hash every blob on disk
    verify_result = None
    if deep and disk_hashes:
        verify_result = blob_store.verify_all()

    if json_output:
        payload: dict[str, object] = {
            "total_blobs": len(disk_hashes),
            "total_raw_records": len(db_raw_ids),
            "missing_count": len(missing),
            "orphaned_count": orphan_result.orphan_count,
            "orphaned_bytes": orphan_result.orphan_bytes,
            "missing": missing[:10],
            "orphaned": list(orphan_result.orphan_samples),
        }
        if verify_result is not None:
            payload["integrity"] = {
                "checked": verify_result.checked,
                "checked_bytes": verify_result.checked_bytes,
                "failed_count": verify_result.failed_count,
                "truncated": verify_result.truncated,
                "failures": [{"hash": f.hash, "reason": f.reason, "detail": f.detail} for f in verify_result.failures],
            }
        return json_document(payload)

    env.ui.console.print(f"Blob store: {len(disk_hashes)} blobs on disk, {len(db_raw_ids)} raw records in DB")
    if missing:
        env.ui.console.print(f"  MISSING: {len(missing)} blobs referenced in DB but not on disk")
        for h in sorted(missing)[:10]:
            env.ui.console.print(f"    {h[:16]}...")

    if orphan_result.orphan_count:
        env.ui.console.print(
            f"  Orphaned: {orphan_result.orphan_count} blobs ({orphan_result.orphan_bytes:,} bytes) not referenced in DB"
        )
        for h in orphan_result.orphan_samples:
            env.ui.console.print(f"    {h[:16]}...")
    else:
        env.ui.console.print("  No orphaned blobs.")

    if verify_result is not None:
        if verify_result.passed:
            env.ui.console.print(
                f"  Integrity: {verify_result.checked} blobs verified ({verify_result.checked_bytes:,} bytes) — all passed"
            )
        else:
            env.ui.console.print(
                f"  Integrity: {verify_result.failed_count} failures out of {verify_result.checked} checked"
                + (" (truncated)" if verify_result.truncated else "")
            )
            for f in verify_result.failures:
                env.ui.console.print(f"    {f.hash[:16]}...  {f.reason}: {f.detail}")

    if not missing and orphan_result.orphan_count == 0 and (verify_result is None or verify_result.passed):
        env.ui.console.print("  All blobs verified.")
    return None


def _run_schema_verification(options: CheckCommandOptions, config: Config) -> SchemaVerificationReport:
    report = run_schema_verification(
        SchemaVerificationRequest(
            providers=_provider_filter(options.schema_providers),
            max_samples=parse_schema_samples(options.schema_samples),
            record_limit=options.schema_record_limit,
            record_offset=options.schema_record_offset,
            quarantine_malformed=options.schema_quarantine_malformed,
            progress_callback=make_schema_progress_callback(),
        ),
        db_path=config.db_path,
    )
    print(file=sys.stderr)
    return report


def _session_insight_progress_callback(
    options: CheckCommandOptions,
    selected_targets: tuple[str, ...],
) -> ProgressCallback | None:
    if (
        options.repair
        and not options.preview
        and not options.json_output
        and (not selected_targets or "session_insights" in selected_targets)
    ):
        return make_session_insight_progress_callback()
    return None


def _maintenance_run_inputs(options: CheckCommandOptions, report: ReadinessReport) -> _MaintenanceRunInputs:
    return _MaintenanceRunInputs(
        selected_targets=_resolve_selected_maintenance_targets(options),
        preview_counts=_build_preview_counts(report) if options.preview else None,
    )


def _run_maintenance(
    config: Config,
    result: CheckCommandResult,
    options: CheckCommandOptions,
    inputs: _MaintenanceRunInputs,
) -> None:
    result.maintenance_targets = inputs.selected_targets
    result.maintenance_results = run_selected_maintenance(
        config,
        repair=options.repair,
        cleanup=options.cleanup,
        dry_run=options.preview,
        preview_counts=inputs.preview_counts,
        targets=inputs.selected_targets,
        session_insight_progress_callback=_session_insight_progress_callback(options, inputs.selected_targets),
    )


def _persist_maintenance_run(
    env: AppEnv,
    *,
    report: ReadinessReport,
    result: CheckCommandResult,
    options: CheckCommandOptions,
    inputs: _MaintenanceRunInputs,
) -> None:
    persist_maintenance_run(
        env,
        report=report,
        options=options,
        targets=inputs.selected_targets,
        maintenance_results=result.maintenance_results or [],
        vacuum_result=result.vacuum_result,
        preview_counts=inputs.preview_counts,
    )


def run_check_workflow(env: AppEnv, options: CheckCommandOptions) -> CheckCommandResult:
    config = load_effective_config(env)
    if _runtime_only_requested(options):
        return CheckCommandResult(report=run_runtime_readiness(config))

    report = get_readiness(
        config,
        deep=options.deep,
        probe_only=not (options.deep or options.repair or options.cleanup),
    )
    result = CheckCommandResult(report=report)
    maintenance_inputs: _MaintenanceRunInputs | None = None

    if options.runtime:
        result.runtime_report = run_runtime_readiness(config)

    if options.check_daemon:
        result.daemon_report = daemon_status_payload()

    if options.check_blob:
        result.blob_report = _run_blob_store_check(env, config, deep=options.deep, json_output=options.json_output)

    if options.check_schemas:
        result.schema_report = _run_schema_verification(options, config)

    if options.check_proof:
        result.proof_report = run_artifact_proof(
            ArtifactProofRequest(
                providers=_provider_filter(options.artifact_providers),
                record_limit=options.artifact_limit,
                record_offset=options.artifact_offset,
            ),
            db_path=config.db_path,
        ).report

    if options.check_artifacts:
        result.artifact_rows = list_artifact_observations(
            _artifact_query(options),
            db_path=config.db_path,
        ).rows

    if options.check_cohorts:
        result.cohort_rows = list_artifact_cohorts(
            _artifact_query(options),
            db_path=config.db_path,
        ).rows

    if options.repair or options.cleanup:
        maintenance_inputs = _maintenance_run_inputs(options, report)
        _run_maintenance(config, result, options, maintenance_inputs)

    if (options.repair or options.cleanup) and options.vacuum:
        if options.preview:
            result.vacuum_result = VacuumResult(ok=True, preview=True, detail="Preview mode: VACUUM skipped.")
        elif options.json_output:
            result.vacuum_result = vacuum_database(env)

    if result.maintenance_results is not None and maintenance_inputs is not None:
        _persist_maintenance_run(
            env,
            report=report,
            result=result,
            options=options,
            inputs=maintenance_inputs,
        )

    return result
