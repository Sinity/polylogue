"""Artifact/schema sections for plain check output."""

from __future__ import annotations

from polylogue.cli.check_support import format_count_mapping
from polylogue.cli.check_workflow import CheckCommandResult


def append_schema_lines(lines: list[str], result: CheckCommandResult) -> None:
    if result.schema_report is None:
        return
    lines.extend(
        [
            "",
            f"Schema verification: {result.schema_report.total_records:,} raw records "
            f"(samples={result.schema_report.max_samples if result.schema_report.max_samples is not None else 'all'}, "
            f"records={result.schema_report.record_limit if result.schema_report.record_limit is not None else 'all'}, "
            f"offset={result.schema_report.record_offset})",
        ]
    )
    for provider, stats in sorted(result.schema_report.providers.items()):
        lines.append(
            f"  {provider}: valid={stats.valid_records:,} invalid={stats.invalid_records:,} "
            f"drift={stats.drift_records:,} skipped={stats.skipped_no_schema:,} "
            f"decode_errors={stats.decode_errors:,} quarantined={stats.quarantined_records:,}"
        )


def append_artifact_proof_lines(lines: list[str], result: CheckCommandResult) -> None:
    if result.proof_report is None:
        return
    proof_summary = result.proof_report.to_dict()["summary"]
    lines.extend(
        [
            "",
            f"Artifact proof: {result.proof_report.total_records:,} artifact observations "
            f"(contract_backed={proof_summary['contract_backed_records']:,}, "
            f"unsupported={proof_summary['unsupported_parseable_records']:,}, "
            f"non_parseable={proof_summary['recognized_non_parseable_records']:,}, "
            f"unknown={proof_summary['unknown_records']:,}, "
            f"decode_errors={proof_summary['decode_errors']:,})",
        ]
    )
    if proof_summary["subagent_streams"]:
        lines.append(
            f"  Claude subagents: linked_sidecars={proof_summary['linked_sidecars']:,} "
            f"orphan_sidecars={proof_summary['orphan_sidecars']:,} "
            f"streams={proof_summary['subagent_streams']:,}"
        )
    if proof_summary["package_versions"]:
        lines.append(f"  Resolved packages: {format_count_mapping(proof_summary['package_versions'])}")
    if proof_summary["element_kinds"]:
        lines.append(f"  Resolved elements: {format_count_mapping(proof_summary['element_kinds'])}")
    if proof_summary["resolution_reasons"]:
        lines.append(f"  Resolution reasons: {format_count_mapping(proof_summary['resolution_reasons'])}")
    for provider, stats in sorted(result.proof_report.providers.items()):
        lines.append(
            f"  {provider}: contract_backed={stats.contract_backed_records:,} "
            f"unsupported={stats.unsupported_parseable_records:,} "
            f"non_parseable={stats.recognized_non_parseable_records:,} "
            f"unknown={stats.unknown_records:,} "
            f"decode_errors={stats.decode_errors:,}"
        )
        if stats.package_versions:
            lines.append(f"    packages: {format_count_mapping(stats.package_versions)}")
        if stats.element_kinds:
            lines.append(f"    elements: {format_count_mapping(stats.element_kinds)}")
        if stats.resolution_reasons:
            lines.append(f"    reasons: {format_count_mapping(stats.resolution_reasons)}")


def append_artifact_observation_lines(lines: list[str], result: CheckCommandResult) -> None:
    if result.artifact_rows is not None:
        lines.extend(["", f"Artifact observations: {len(result.artifact_rows):,} rows"])
        for row in result.artifact_rows:
            resolved = ""
            if row.resolved_package_version and row.resolved_element_kind:
                resolved = (
                    f" -> {row.resolved_package_version}/{row.resolved_element_kind}"
                    f" [{row.resolution_reason}]"
                )
            lines.append(
                f"  {row.support_status} {row.payload_provider or row.provider_name} "
                f"{row.artifact_kind} {row.source_path}{resolved}"
            )

    if result.cohort_rows is not None:
        lines.extend(["", f"Artifact cohorts: {len(result.cohort_rows):,} cohorts"])
        for row in result.cohort_rows:
            lines.append(
                f"  {row.provider_name} {row.artifact_kind} {row.support_status} "
                f"count={row.observation_count:,} cohort={row.cohort_id or '-'} "
                f"version={row.resolved_package_version or '-'} "
                f"element={row.resolved_element_kind or '-'}"
            )


__all__ = [
    "append_artifact_observation_lines",
    "append_artifact_proof_lines",
    "append_schema_lines",
]
