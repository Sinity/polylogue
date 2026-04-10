"""Plain/rich rendering helpers for the check command."""

from __future__ import annotations

import click

from polylogue.cli.check_support import format_count_mapping, run_vacuum
from polylogue.cli.check_workflow import CheckCommandOptions, CheckCommandResult
from polylogue.cli.types import AppEnv
from polylogue.health import VerifyStatus

# ---------------------------------------------------------------------------
# Support helpers
# ---------------------------------------------------------------------------


def status_icon(status: VerifyStatus, *, plain: bool) -> str:
    if plain:
        return {
            VerifyStatus.OK: "OK",
            VerifyStatus.WARNING: "WARN",
            VerifyStatus.ERROR: "ERR",
        }.get(status, "?")
    return {
        VerifyStatus.OK: "[green]✓[/green]",
        VerifyStatus.WARNING: "[yellow]![/yellow]",
        VerifyStatus.ERROR: "[red]✗[/red]",
    }.get(status, "?")


# ---------------------------------------------------------------------------
# Health / derived-model / runtime sections
# ---------------------------------------------------------------------------


def build_health_lines(
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
    source_val = getattr(provenance.source, "value", provenance.source) if hasattr(provenance, "source") else "live"
    lines.extend(
        [
            "",
            (
                f"Summary: {summary.get('ok', 0)} ok, {summary.get('warning', 0)} warnings, "
                f"{summary.get('error', 0)} errors (source={source_val})"
            ),
        ]
    )
    return lines


def append_derived_model_lines(lines: list[str], result: CheckCommandResult) -> None:
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


def append_runtime_lines(lines: list[str], result: CheckCommandResult, *, plain: bool) -> None:
    if result.runtime_report is None:
        return
    lines.extend(["", "Runtime Environment:"])
    for check in result.runtime_report.checks:
        lines.append(f"  {status_icon(check.status, plain=plain)} {check.name}: {check.detail}")
    rt_summary = result.runtime_report.summary
    lines.append(
        f"  Runtime: {rt_summary.get('ok', 0)} ok, {rt_summary.get('warning', 0)} warnings, "
        f"{rt_summary.get('error', 0)} errors"
    )


# ---------------------------------------------------------------------------
# Artifact / schema sections
# ---------------------------------------------------------------------------


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
                resolved = f" -> {row.resolved_package_version}/{row.resolved_element_kind} [{row.resolution_reason}]"
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


# ---------------------------------------------------------------------------
# Section assembly
# ---------------------------------------------------------------------------


def build_report_lines(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> list[str]:
    """Build the full plain-mode report body."""
    lines = build_health_lines(env, result, options)
    append_derived_model_lines(lines, result)
    append_schema_lines(lines, result)
    append_artifact_proof_lines(lines, result)
    append_artifact_observation_lines(lines, result)
    append_runtime_lines(lines, result, plain=env.ui.plain)
    return lines


# ---------------------------------------------------------------------------
# Maintenance output
# ---------------------------------------------------------------------------


def emit_maintenance_output(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> None:
    """Render maintenance/correction output after the health report."""
    if result.maintenance_results is not None:
        click.echo("")
        mode_label = "Preview of maintenance" if options.preview else "Running maintenance"
        click.echo(f"{mode_label}...")
        if options.maintenance_targets:
            click.echo(f"  Targets: {', '.join(options.maintenance_targets)}")
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_plain_output(
    env: AppEnv,
    result: CheckCommandResult,
    options: CheckCommandOptions,
) -> None:
    env.ui.summary("Health Check", build_report_lines(env, result, options))
    emit_maintenance_output(env, result, options)


__all__ = [
    "emit_maintenance_output",
    "render_plain_output",
    "status_icon",
]
