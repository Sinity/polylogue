"""Rendering helpers for the check command."""

from __future__ import annotations

import click

from polylogue.cli.machine_errors import emit_success
from polylogue.cli.types import AppEnv
from polylogue.health_models import VerifyStatus

from .check_support import format_count_mapping, format_semantic_metric_summary, run_vacuum
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
        out["maintenance"] = [repair.to_dict() for repair in result.maintenance_results]
    if result.vacuum_result is not None:
        out["vacuum"] = result.vacuum_result
    emit_success(out)


def render_plain_output(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> None:
    lines: list[str] = []
    for check in result.report.checks:
        status_icon = {
            VerifyStatus.OK: "[green]✓[/green]",
            VerifyStatus.WARNING: "[yellow]![/yellow]",
            VerifyStatus.ERROR: "[red]✗[/red]",
        }.get(check.status, "?")
        if env.ui.plain:
            status_icon = {
                VerifyStatus.OK: "OK",
                VerifyStatus.WARNING: "WARN",
                VerifyStatus.ERROR: "ERR",
            }.get(check.status, "?")
        lines.append(f"{status_icon} {check.name}: {check.detail}")

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
    if result.report.derived_models:
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

    if result.schema_report is not None:
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

    if result.proof_report is not None:
        summary = result.proof_report.to_dict()["summary"]
        lines.extend(
            [
                "",
                f"Artifact proof: {result.proof_report.total_records:,} artifact observations "
                f"(contract_backed={summary['contract_backed_records']:,}, "
                f"unsupported={summary['unsupported_parseable_records']:,}, "
                f"non_parseable={summary['recognized_non_parseable_records']:,}, "
                f"unknown={summary['unknown_records']:,}, "
                f"decode_errors={summary['decode_errors']:,})",
            ]
        )
        if summary["subagent_streams"]:
            lines.append(
                f"  Claude subagents: linked_sidecars={summary['linked_sidecars']:,} "
                f"orphan_sidecars={summary['orphan_sidecars']:,} "
                f"streams={summary['subagent_streams']:,}"
            )
        if summary["package_versions"]:
            lines.append(f"  Resolved packages: {format_count_mapping(summary['package_versions'])}")
        if summary["element_kinds"]:
            lines.append(f"  Resolved elements: {format_count_mapping(summary['element_kinds'])}")
        if summary["resolution_reasons"]:
            lines.append(f"  Resolution reasons: {format_count_mapping(summary['resolution_reasons'])}")
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
            lines.append(f"  Metrics: {format_semantic_metric_summary(semantic_summary['metric_summary'])}")
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

    if result.runtime_report is not None:
        lines.extend(["", "Runtime Environment:"])
        for check in result.runtime_report.checks:
            status_icon = {
                VerifyStatus.OK: "[green]✓[/green]",
                VerifyStatus.WARNING: "[yellow]![/yellow]",
                VerifyStatus.ERROR: "[red]✗[/red]",
            }.get(check.status, "?")
            if env.ui.plain:
                status_icon = {
                    VerifyStatus.OK: "OK",
                    VerifyStatus.WARNING: "WARN",
                    VerifyStatus.ERROR: "ERR",
                }.get(check.status, "?")
            lines.append(f"  {status_icon} {check.name}: {check.detail}")
        rt_summary = result.runtime_report.summary
        lines.append(
            f"  Runtime: {rt_summary.get('ok', 0)} ok, {rt_summary.get('warning', 0)} warnings, "
            f"{rt_summary.get('error', 0)} errors"
        )

    if result.roundtrip_report is not None:
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

    env.ui.summary("Health Check", lines)

    if result.maintenance_results is not None:
        click.echo("")
        mode_label = "Preview of maintenance" if options.preview else "Running maintenance"
        click.echo(f"{mode_label}...")
        total_repaired = 0
        for repair in result.maintenance_results:
            if repair.repaired_count > 0 or not repair.success:
                status = "[green]✓[/green]" if repair.success else "[red]✗[/red]"
                if env.ui.plain:
                    status = "OK" if repair.success else "FAIL"
                mode = f"{repair.category.value}{' destructive' if repair.destructive else ''}"
                env.ui.console.print(f"  {status} {repair.name} [{mode}]: {repair.detail}")
                total_repaired += repair.repaired_count

        if total_repaired > 0:
            action = "Would change" if options.preview else "Changed"
            click.echo(f"\n{action} {total_repaired} issue(s)")
        else:
            click.echo("  No selected maintenance work was needed.")
    elif options.repair or options.cleanup:
        env.ui.console.print("No maintenance operations were selected.")

    if (options.repair or options.cleanup) and options.vacuum and options.preview:
        env.ui.console.print("")
        env.ui.console.print("Preview mode: VACUUM skipped.")
    elif (options.repair or options.cleanup) and options.vacuum:
        run_vacuum(env)
