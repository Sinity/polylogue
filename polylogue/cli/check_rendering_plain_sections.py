"""Section builders for plain check output."""

from __future__ import annotations

from polylogue.cli.check_rendering_plain_support import status_icon
from polylogue.cli.check_support import format_count_mapping, format_semantic_metric_summary
from polylogue.cli.check_workflow import CheckCommandOptions, CheckCommandResult
from polylogue.cli.types import AppEnv
from polylogue.health_models import VerifyStatus


def build_report_lines(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> list[str]:
    """Build the full plain-mode report body."""
    lines = _build_health_lines(env, result, options)
    _append_derived_model_lines(lines, result)
    _append_schema_lines(lines, result)
    _append_artifact_proof_lines(lines, result)
    _append_artifact_observation_lines(lines, result)
    _append_semantic_lines(lines, result)
    _append_runtime_lines(lines, result, plain=env.ui.plain)
    _append_roundtrip_lines(lines, result)
    return lines


def _build_health_lines(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> list[str]:
    lines: list[str] = []
    for check in result.report.checks:
        lines.append(f"{status_icon(check.status, plain=env.ui.plain)} {check.name}: {check.detail}")
        if check.breakdown and (options.verbose or check.status in (VerifyStatus.WARNING, VerifyStatus.ERROR)):
            for provider, count in sorted(check.breakdown.items(), key=lambda item: -item[1]):
                lines.append(f"    {provider}: {count:,}")

    summary = result.report.summary
    provenance = result.report.provenance
    lines.extend(
        [
            "",
            (
                f"Summary: {summary.get('ok', 0)} ok, {summary.get('warning', 0)} warnings, "
                f"{summary.get('error', 0)} errors (source={provenance.source.value}"
                + (
                    f", age={provenance.cache_age_seconds}s, ttl={provenance.cache_ttl_seconds}s"
                    if provenance.cache_age_seconds is not None
                    else ""
                )
                + ")"
            ),
        ]
    )
    return lines


def _append_derived_model_lines(lines: list[str], result: CheckCommandResult) -> None:
    if not result.report.derived_models:
        return
    lines.extend(["", "Derived Models:"])
    for name, status in sorted(result.report.derived_models.items()):
        state = "ready" if status.ready else "pending"
        lines.append(
            f"  {name}: {state}; docs {status.materialized_documents:,}/{status.source_documents:,}; "
            f"rows {status.materialized_rows:,}/{status.source_rows:,}; "
            f"pending_docs={status.pending_documents:,}; pending_rows={status.pending_rows:,}; "
            f"stale={status.stale_rows:,}; orphan={status.orphan_rows:,}; "
            f"missing_provenance={status.missing_provenance_rows:,}"
        )


def _append_schema_lines(lines: list[str], result: CheckCommandResult) -> None:
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


def _append_artifact_proof_lines(lines: list[str], result: CheckCommandResult) -> None:
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


def _append_artifact_observation_lines(lines: list[str], result: CheckCommandResult) -> None:
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


def _append_semantic_lines(lines: list[str], result: CheckCommandResult) -> None:
    if result.semantic_report is not None:
        semantic_summary = result.semantic_report.to_dict()["summary"]
        lines.extend(
            [
                "",
                f"Semantic proof: {semantic_summary['surface_count']:,} surfaces "
                f"(clean={semantic_summary['clean_surfaces']:,}, "
                f"critical={semantic_summary['critical_surfaces']:,}, "
                f"total_conversations={semantic_summary['total_conversations']:,}, "
                f"preserved_checks={semantic_summary['preserved_checks']:,}, "
                f"declared_loss_checks={semantic_summary['declared_loss_checks']:,}, "
                f"critical_loss_checks={semantic_summary['critical_loss_checks']:,})",
            ]
        )
        if semantic_summary["metric_summary"]:
            lines.append(
                f"  Metrics: {format_semantic_metric_summary(semantic_summary['metric_summary'])}"
            )
        for surface, surface_report in sorted(result.semantic_report.surfaces.items()):
            surface_summary = surface_report.to_dict()["summary"]
            lines.append(
                f"  {surface}: conversations={surface_summary['total_conversations']:,} "
                f"clean={surface_summary['clean_conversations']:,} "
                f"critical={surface_summary['critical_conversations']:,} "
                f"preserved_checks={surface_summary['preserved_checks']:,} "
                f"declared_loss_checks={surface_summary['declared_loss_checks']:,} "
                f"critical_loss_checks={surface_summary['critical_loss_checks']:,}"
            )
            for provider, stats in sorted(surface_report.providers.items()):
                lines.append(
                    f"    {provider}: conversations={stats.total_conversations:,} "
                    f"clean={stats.clean_conversations:,} "
                    f"critical={stats.critical_conversations:,} "
                    f"preserved_checks={stats.preserved_checks:,} "
                    f"declared_loss_checks={stats.declared_loss_checks:,} "
                    f"critical_loss_checks={stats.critical_loss_checks:,}"
                )

    if result.semantic_contracts is not None:
        lines.extend(["", f"Semantic contracts: {len(result.semantic_contracts):,} surfaces"])
        for spec in result.semantic_contracts:
            details: list[str] = [f"category={spec.category}"]
            if spec.aliases:
                details.append(f"aliases={','.join(spec.aliases)}")
            if spec.export_format:
                details.append(f"export_format={spec.export_format}")
            if spec.stream_format:
                details.append(f"stream_format={spec.stream_format}")
            details.append(f"contracts={len(spec.contracts)}")
            lines.append(f"  {spec.name}: {'; '.join(details)}")
            metric_bits = [f"{contract.metric}:{contract.mode}" for contract in spec.contracts]
            lines.append(f"    metrics={', '.join(metric_bits)}")


def _append_runtime_lines(lines: list[str], result: CheckCommandResult, *, plain: bool) -> None:
    if result.runtime_report is None:
        return
    lines.extend(["", "Runtime Environment:"])
    for check in result.runtime_report.checks:
        lines.append(
            f"  {status_icon(check.status, plain=plain)} {check.name}: {check.detail}"
        )
    rt_summary = result.runtime_report.summary
    lines.append(
        f"  Runtime: {rt_summary.get('ok', 0)} ok, {rt_summary.get('warning', 0)} warnings, "
        f"{rt_summary.get('error', 0)} errors"
    )


def _append_roundtrip_lines(lines: list[str], result: CheckCommandResult) -> None:
    if result.roundtrip_report is None:
        return
    roundtrip_summary = result.roundtrip_report.summary
    lines.extend(
        [
            "",
            f"Roundtrip proof: {roundtrip_summary['provider_count']:,} providers "
            f"(clean={roundtrip_summary['clean_providers']:,}, "
            f"failed={roundtrip_summary['failed_providers']:,}, "
            f"artifacts={roundtrip_summary['artifact_count']:,}, "
            f"parsed_conversations={roundtrip_summary['parsed_conversations']:,}, "
            f"persisted_conversations={roundtrip_summary['persisted_conversations']:,})",
        ]
    )
    for provider, provider_report in sorted(result.roundtrip_report.provider_reports.items()):
        summary = provider_report.summary
        status = "clean" if provider_report.is_clean else "failed"
        lines.append(
            f"  {provider}: {status}, package={provider_report.package_version}, "
            f"element={provider_report.element_kind or '-'}, "
            f"failed_stages={','.join(summary['failed_stages']) or '-'}"
        )


__all__ = ["build_report_lines"]
