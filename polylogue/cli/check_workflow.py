"""Execution and validation workflow for the check command."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

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
from polylogue.storage.repair import run_all_repairs

from .check_support import make_schema_progress_callback, parse_schema_samples, vacuum_database


@dataclass(frozen=True)
class CheckCommandOptions:
    json_output: bool
    verbose: bool
    repair: bool
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
    repair_results: list[Any] | None = None
    vacuum_result: dict[str, Any] | None = None


def validate_check_options(options: CheckCommandOptions) -> None:
    if options.vacuum and not options.repair:
        fail("check", "--vacuum requires --repair")
    if options.preview and not options.repair:
        fail("check", "--preview requires --repair")
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


def run_check_workflow(env: AppEnv, options: CheckCommandOptions) -> CheckCommandResult:
    config = load_effective_config(env)
    report = get_health(config, deep=options.deep)
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

    if options.repair:
        result.repair_results = run_all_repairs(config, dry_run=options.preview)

    if options.repair and options.vacuum:
        if options.preview:
            result.vacuum_result = {
                "ok": True,
                "preview": True,
                "detail": "Preview mode: VACUUM skipped.",
            }
        elif options.json_output:
            result.vacuum_result = vacuum_database(env)

    return result
