"""JSON rendering helpers for the check command."""

from __future__ import annotations

from polylogue.cli.machine_errors import emit_success

from .check_workflow import CheckCommandOptions, CheckCommandResult


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
    if result.semantic_report is not None:
        out["semantic_proof"] = result.semantic_report.to_dict()
    if result.semantic_contracts is not None:
        out["semantic_contracts"] = {
            "count": len(result.semantic_contracts),
            "items": [
                {
                    "surface": spec.name,
                    "category": spec.category,
                    "aliases": list(spec.aliases),
                    "export_format": spec.export_format,
                    "stream_format": spec.stream_format,
                    "contract_count": len(spec.contracts),
                    "contracts": [contract.to_dict() for contract in spec.contracts],
                }
                for spec in result.semantic_contracts
            ],
        }
    if result.roundtrip_report is not None:
        out["roundtrip_proof"] = result.roundtrip_report.to_dict()
    if result.maintenance_results is not None:
        out["maintenance"] = {
            "targets": list(options.maintenance_targets),
            "items": [repair.to_dict() for repair in result.maintenance_results],
        }
    if result.vacuum_result is not None:
        out["vacuum"] = result.vacuum_result
    emit_success(out)
