"""Option validation for the check command workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.cli.shared.helpers import fail
from polylogue.maintenance.targets import MaintenanceTargetMode, build_maintenance_target_catalog

if TYPE_CHECKING:
    from polylogue.cli.shared.check_workflow import CheckCommandOptions


def validate_check_options(options: CheckCommandOptions) -> None:
    catalog = build_maintenance_target_catalog()
    if options.vacuum and not (options.repair or options.cleanup):
        fail("doctor", "--vacuum requires --repair or --cleanup")
    if options.preview and not (options.repair or options.cleanup):
        fail("doctor", "--preview requires --repair or --cleanup")
    if options.maintenance_targets and not (options.repair or options.cleanup):
        fail("doctor", "--target requires --repair or --cleanup")
    if options.schema_providers and not options.check_schemas:
        fail("doctor", "--schema-provider requires --schemas")
    if options.schema_samples != "all" and not options.check_schemas:
        fail("doctor", "--schema-samples requires --schemas")
    if options.schema_record_limit is not None and not options.check_schemas:
        fail("doctor", "--schema-record-limit requires --schemas")
    if options.schema_record_offset != 0 and not options.check_schemas:
        fail("doctor", "--schema-record-offset requires --schemas")
    if options.schema_quarantine_malformed and not options.check_schemas:
        fail("doctor", "--schema-quarantine-malformed requires --schemas")
    if options.artifact_providers and not (options.check_proof or options.check_artifacts or options.check_cohorts):
        fail("doctor", "--artifact-provider requires --proof, --artifacts, or --cohorts")
    if options.artifact_statuses and not (options.check_artifacts or options.check_cohorts):
        fail("doctor", "--artifact-status requires --artifacts or --cohorts")
    if options.artifact_kinds and not (options.check_artifacts or options.check_cohorts):
        fail("doctor", "--artifact-kind requires --artifacts or --cohorts")
    if options.artifact_limit is not None and not (
        options.check_proof or options.check_artifacts or options.check_cohorts
    ):
        fail("doctor", "--artifact-limit requires --proof, --artifacts, or --cohorts")
    if options.artifact_offset != 0 and not (options.check_proof or options.check_artifacts or options.check_cohorts):
        fail("doctor", "--artifact-offset requires --proof, --artifacts, or --cohorts")
    if options.schema_record_limit is not None and options.schema_record_limit <= 0:
        fail("doctor", "--schema-record-limit must be a positive integer")
    if options.schema_record_offset < 0:
        fail("doctor", "--schema-record-offset must be >= 0")
    if options.artifact_limit is not None and options.artifact_limit <= 0:
        fail("doctor", "--artifact-limit must be a positive integer")
    if options.artifact_offset < 0:
        fail("doctor", "--artifact-offset must be >= 0")
    if options.maintenance_targets:
        selected = catalog.resolve(tuple(options.maintenance_targets))
        if (
            options.repair
            and not options.cleanup
            and not any(spec.mode is MaintenanceTargetMode.REPAIR for spec in selected)
        ):
            fail("doctor", "--target only selected cleanup targets while running --repair")
        if (
            options.cleanup
            and not options.repair
            and not any(spec.mode is MaintenanceTargetMode.CLEANUP for spec in selected)
        ):
            fail("doctor", "--target only selected repair targets while running --cleanup")


__all__ = ["validate_check_options"]
