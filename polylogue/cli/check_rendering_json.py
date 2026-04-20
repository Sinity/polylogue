"""JSON rendering helpers for the check command."""

from __future__ import annotations

from polylogue.cli.machine_errors import emit_success

from .check_models import CheckCommandResult
from .check_workflow import CheckCommandOptions


def emit_json_output(result: CheckCommandResult, options: CheckCommandOptions) -> None:
    out = result.report.to_dict()
    if result.runtime_report is not None:
        out["runtime"] = result.runtime_report.to_dict()
    if result.schema_report is not None:
        out["schema_verification"] = result.schema_report.to_dict()
    if result.proof_report is not None:
        out["artifact_proof"] = result.proof_report.to_dict()
    if result.artifact_rows is not None:
        out["artifact_observations"] = {
            "record_limit": options.artifact_limit if options.artifact_limit is not None else "all",
            "record_offset": max(0, options.artifact_offset),
            "count": len(result.artifact_rows),
            "items": [row.model_dump(mode="json") for row in result.artifact_rows],
        }
    if result.cohort_rows is not None:
        out["artifact_cohorts"] = {
            "record_limit": options.artifact_limit if options.artifact_limit is not None else "all",
            "record_offset": max(0, options.artifact_offset),
            "count": len(result.cohort_rows),
            "items": [row.model_dump(mode="json") for row in result.cohort_rows],
        }
    if result.maintenance_results is not None:
        out["maintenance"] = {
            "targets": list(result.maintenance_targets),
            "items": [repair.to_dict() for repair in result.maintenance_results],
        }
    if result.vacuum_result is not None:
        out["vacuum"] = result.vacuum_result.to_dict()
    emit_success(out)
