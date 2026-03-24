"""Execution and validation workflow for the check command."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

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
from polylogue.storage.archive_debt import preview_counts_from_archive_debt
from polylogue.storage.backends.connection import _clear_connection_cache
from polylogue.storage.repair import (
    CLEANUP_TARGETS,
    SAFE_REPAIR_TARGETS,
    run_selected_maintenance,
)
from polylogue.storage.store import MaintenanceRunRecord
from polylogue.sync_bridge import run_coroutine_sync

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
    check_semantic_proof: bool
    check_semantic_contracts: bool
    check_roundtrip_proof: bool
    schema_providers: tuple[str, ...]
    artifact_providers: tuple[str, ...]
    artifact_statuses: tuple[str, ...]
    artifact_kinds: tuple[str, ...]
    artifact_limit: int | None
    artifact_offset: int
    semantic_providers: tuple[str, ...]
    semantic_surfaces: tuple[str, ...]
    semantic_limit: int | None
    semantic_offset: int
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
    semantic_report: Any | None = None
    semantic_contracts: list[Any] | None = None
    roundtrip_report: Any | None = None
    maintenance_results: list[Any] | None = None
    vacuum_result: dict[str, Any] | None = None


def _build_preview_counts(report: Any) -> dict[str, int]:
    """Extract maintenance preview counts from the already-computed health report."""
    return preview_counts_from_archive_debt(getattr(report, "archive_debt", {}) or {})


def _resolve_selected_maintenance_targets(options: CheckCommandOptions) -> tuple[str, ...]:
    if options.maintenance_targets:
        return tuple(options.maintenance_targets)
    targets: list[str] = []
    if options.repair:
        targets.extend(SAFE_REPAIR_TARGETS)
    if options.cleanup:
        targets.extend(CLEANUP_TARGETS)
    return tuple(targets)


def _build_maintenance_manifest(
    *,
    report: Any,
    options: CheckCommandOptions,
    targets: tuple[str, ...],
    maintenance_results: list[Any],
    vacuum_result: dict[str, Any] | None,
    preview_counts: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "report_summary": dict(report.summary),
        "report_provenance": report.provenance.to_dict(),
        "derived_models": {
            name: status.to_dict()
            for name, status in sorted((report.derived_models or {}).items())
        },
        "preview_counts": dict(preview_counts or {}),
        "targets": list(targets),
        "results": [result.to_dict() for result in maintenance_results],
        "vacuum": vacuum_result,
    }


def validate_check_options(options: CheckCommandOptions) -> None:
    if options.vacuum and not (options.repair or options.cleanup):
        fail("check", "--vacuum requires --repair or --cleanup")
    if options.preview and not (options.repair or options.cleanup):
        fail("check", "--preview requires --repair or --cleanup")
    if options.maintenance_targets and not (options.repair or options.cleanup):
        fail("check", "--target requires --repair or --cleanup")
    if options.schema_providers and not options.check_schemas:
        fail("check", "--schema-provider requires --schemas")
    if options.schema_samples != "all" and not options.check_schemas:
        fail("check", "--schema-samples requires --schemas")
    if options.schema_record_limit is not None and not options.check_schemas:
        fail("check", "--schema-record-limit requires --schemas")
    if options.schema_record_offset != 0 and not options.check_schemas:
        fail("check", "--schema-record-offset requires --schemas")
    if options.schema_quarantine_malformed and not options.check_schemas:
        fail("check", "--schema-quarantine-malformed requires --schemas")
    if options.artifact_providers and not (
        options.check_proof or options.check_artifacts or options.check_cohorts
    ):
        fail("check", "--artifact-provider requires --proof, --artifacts, or --cohorts")
    if options.artifact_statuses and not (options.check_artifacts or options.check_cohorts):
        fail("check", "--artifact-status requires --artifacts or --cohorts")
    if options.artifact_kinds and not (options.check_artifacts or options.check_cohorts):
        fail("check", "--artifact-kind requires --artifacts or --cohorts")
    if options.artifact_limit is not None and not (
        options.check_proof or options.check_artifacts or options.check_cohorts
    ):
        fail("check", "--artifact-limit requires --proof, --artifacts, or --cohorts")
    if options.artifact_offset != 0 and not (
        options.check_proof or options.check_artifacts or options.check_cohorts
    ):
        fail("check", "--artifact-offset requires --proof, --artifacts, or --cohorts")
    if options.semantic_providers and not options.check_semantic_proof:
        fail("check", "--semantic-provider requires --semantic-proof")
    if options.semantic_surfaces and not (options.check_semantic_proof or options.check_semantic_contracts):
        fail("check", "--semantic-surface requires --semantic-proof or --semantic-contracts")
    if options.semantic_limit is not None and not options.check_semantic_proof:
        fail("check", "--semantic-limit requires --semantic-proof")
    if options.semantic_offset != 0 and not options.check_semantic_proof:
        fail("check", "--semantic-offset requires --semantic-proof")
    if options.schema_record_limit is not None and options.schema_record_limit <= 0:
        fail("check", "--schema-record-limit must be a positive integer")
    if options.schema_record_offset < 0:
        fail("check", "--schema-record-offset must be >= 0")
    if options.artifact_limit is not None and options.artifact_limit <= 0:
        fail("check", "--artifact-limit must be a positive integer")
    if options.artifact_offset < 0:
        fail("check", "--artifact-offset must be >= 0")
    if options.semantic_limit is not None and options.semantic_limit <= 0:
        fail("check", "--semantic-limit must be a positive integer")
    if options.semantic_offset < 0:
        fail("check", "--semantic-offset must be >= 0")
    if options.roundtrip_providers and not options.check_roundtrip_proof:
        fail("check", "--roundtrip-provider requires --roundtrip-proof")
    if options.roundtrip_count <= 0:
        fail("check", "--roundtrip-count must be a positive integer")
    if options.maintenance_targets:
        if options.repair and not options.cleanup and not any(name in SAFE_REPAIR_TARGETS for name in options.maintenance_targets):
            fail("check", "--target only selected cleanup targets while running --repair")
        if options.cleanup and not options.repair and not any(name in CLEANUP_TARGETS for name in options.maintenance_targets):
            fail("check", "--target only selected repair targets while running --cleanup")


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

    if options.check_semantic_proof:
        from polylogue.rendering.semantic_proof import prove_semantic_surface_suite

        try:
            result.semantic_report = prove_semantic_surface_suite(
                providers=list(options.semantic_providers) if options.semantic_providers else None,
                surfaces=list(options.semantic_surfaces) if options.semantic_surfaces else None,
                record_limit=options.semantic_limit,
                record_offset=options.semantic_offset,
            )
        except ValueError as exc:
            fail("check", str(exc))

    if options.check_semantic_contracts:
        from polylogue.rendering.semantic_surface_registry import (
            list_semantic_surface_specs,
            resolve_semantic_surfaces,
            semantic_surface_spec,
        )

        try:
            if options.semantic_surfaces:
                resolved = resolve_semantic_surfaces(list(options.semantic_surfaces))
                result.semantic_contracts = [semantic_surface_spec(name) for name in resolved]
            else:
                result.semantic_contracts = list(list_semantic_surface_specs())
        except ValueError as exc:
            fail("check", str(exc))

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
        selected_targets = _resolve_selected_maintenance_targets(options)
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
        selected_targets = _resolve_selected_maintenance_targets(options)
        preview_counts = _build_preview_counts(report) if options.preview else None
        vacuum_ok = result.vacuum_result is None or bool(result.vacuum_result.get("ok", False))
        _clear_connection_cache()
        record = MaintenanceRunRecord(
            maintenance_run_id=f"maint-{uuid4().hex[:16]}",
            executed_at=datetime.now(timezone.utc).isoformat(),
            mode="preview" if options.preview else "apply",
            preview=options.preview,
            repair_selected=options.repair,
            cleanup_selected=options.cleanup,
            vacuum_requested=options.vacuum,
            target_names=selected_targets,
            success=all(result_item.success for result_item in result.maintenance_results) and vacuum_ok,
            manifest=_build_maintenance_manifest(
                report=report,
                options=options,
                targets=selected_targets,
                maintenance_results=result.maintenance_results,
                vacuum_result=result.vacuum_result,
                preview_counts=preview_counts,
            ),
        )
        run_coroutine_sync(env.backend.record_maintenance_run(record))

    return result
