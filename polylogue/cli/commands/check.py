"""Health check command."""

from __future__ import annotations

import sys
import time
from typing import Any

import click

from polylogue.cli.helpers import fail, load_effective_config
from polylogue.cli.types import AppEnv
from polylogue.health import VerifyStatus, get_health, run_all_repairs


@click.command("check")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show breakdown by provider")
@click.option("--repair", is_flag=True, help="Attempt to repair detected issues")
@click.option("--preview", is_flag=True, help="Preview repairs without executing (requires --repair)")
@click.option("--vacuum", is_flag=True, help="Reclaim unused space after repair (requires --repair)")
@click.option("--deep", is_flag=True, help="Run SQLite integrity check (slow on large databases)")
@click.option("--schemas", "check_schemas", is_flag=True, help="Run raw-corpus schema verification (non-mutating)")
@click.option("--proof", "check_proof", is_flag=True, help="Run durable artifact support proof")
@click.option("--artifacts", "check_artifacts", is_flag=True, help="List durable artifact observations")
@click.option("--cohorts", "check_cohorts", is_flag=True, help="Summarize durable artifact cohorts")
@click.option("--schema-provider", "schema_providers", multiple=True, help="Limit schema verification to DB provider name (repeatable)")
@click.option(
    "--artifact-provider",
    "artifact_providers",
    multiple=True,
    help="Limit artifact proof/listing/cohorting to effective provider (repeatable)",
)
@click.option(
    "--artifact-status",
    "artifact_statuses",
    multiple=True,
    help="Limit artifact listing/cohorting to support status (repeatable)",
)
@click.option(
    "--artifact-kind",
    "artifact_kinds",
    multiple=True,
    help="Limit artifact listing/cohorting to artifact kind (repeatable)",
)
@click.option(
    "--artifact-limit",
    type=int,
    default=None,
    help="Limit artifact proof/listing/cohorting to N observation rows",
)
@click.option(
    "--artifact-offset",
    type=int,
    default=0,
    show_default=True,
    help="Start offset for artifact proof/listing/cohorting",
)
@click.option(
    "--schema-samples",
    default="all",
    show_default=True,
    help="Validation samples per raw payload: positive integer or 'all'",
)
@click.option(
    "--schema-record-limit",
    type=int,
    default=None,
    help="Limit schema verification to N raw records (for chunked runs)",
)
@click.option(
    "--schema-record-offset",
    type=int,
    default=0,
    show_default=True,
    help="Start offset for chunked schema verification",
)
@click.option(
    "--schema-quarantine-malformed",
    is_flag=True,
    help="Mark malformed raw payloads as failed validation during schema verification (mutates DB)",
)
@click.pass_obj
def check_command(
    env: AppEnv,
    json_output: bool,
    verbose: bool,
    repair: bool,
    preview: bool,
    vacuum: bool,
    deep: bool,
    check_schemas: bool,
    check_proof: bool,
    check_artifacts: bool,
    check_cohorts: bool,
    schema_providers: tuple[str, ...],
    artifact_providers: tuple[str, ...],
    artifact_statuses: tuple[str, ...],
    artifact_kinds: tuple[str, ...],
    artifact_limit: int | None,
    artifact_offset: int,
    schema_samples: str,
    schema_record_limit: int | None,
    schema_record_offset: int,
    schema_quarantine_malformed: bool,
) -> None:
    """Health check with optional repair."""
    if vacuum and not repair:
        fail("check", "--vacuum requires --repair")
    if preview and not repair:
        fail("check", "--preview requires --repair")
    if schema_providers and not check_schemas:
        fail("check", "--schema-provider requires --schemas")
    if schema_samples != "all" and not check_schemas:
        fail("check", "--schema-samples requires --schemas")
    if schema_record_limit is not None and not check_schemas:
        fail("check", "--schema-record-limit requires --schemas")
    if schema_record_offset != 0 and not check_schemas:
        fail("check", "--schema-record-offset requires --schemas")
    if schema_quarantine_malformed and not check_schemas:
        fail("check", "--schema-quarantine-malformed requires --schemas")
    if artifact_providers and not (check_proof or check_artifacts or check_cohorts):
        fail("check", "--artifact-provider requires --proof, --artifacts, or --cohorts")
    if artifact_statuses and not (check_artifacts or check_cohorts):
        fail("check", "--artifact-status requires --artifacts or --cohorts")
    if artifact_kinds and not (check_artifacts or check_cohorts):
        fail("check", "--artifact-kind requires --artifacts or --cohorts")
    if artifact_limit is not None and not (check_proof or check_artifacts or check_cohorts):
        fail("check", "--artifact-limit requires --proof, --artifacts, or --cohorts")
    if artifact_offset != 0 and not (check_proof or check_artifacts or check_cohorts):
        fail("check", "--artifact-offset requires --proof, --artifacts, or --cohorts")
    if schema_record_limit is not None and schema_record_limit <= 0:
        fail("check", "--schema-record-limit must be a positive integer")
    if schema_record_offset < 0:
        fail("check", "--schema-record-offset must be >= 0")
    if artifact_limit is not None and artifact_limit <= 0:
        fail("check", "--artifact-limit must be a positive integer")
    if artifact_offset < 0:
        fail("check", "--artifact-offset must be >= 0")

    config = load_effective_config(env)
    report = get_health(config, deep=deep)
    schema_report = None
    proof_report = None
    artifact_rows = None
    cohort_rows = None
    if check_schemas:
        from polylogue.schemas.verification import verify_raw_corpus

        parsed_max_samples = _parse_schema_samples(schema_samples)
        schema_report = verify_raw_corpus(
            providers=list(schema_providers) if schema_providers else None,
            max_samples=parsed_max_samples,
            record_limit=schema_record_limit,
            record_offset=schema_record_offset,
            quarantine_malformed=schema_quarantine_malformed,
        )
        print(file=sys.stderr)  # End the \r progress line
    if check_proof:
        from polylogue.schemas.verification import prove_raw_artifact_coverage

        proof_report = prove_raw_artifact_coverage(
            providers=list(artifact_providers) if artifact_providers else None,
            record_limit=artifact_limit,
            record_offset=artifact_offset,
        )
    if check_artifacts:
        from polylogue.schemas.verification import list_artifact_observation_rows

        artifact_rows = list_artifact_observation_rows(
            providers=list(artifact_providers) if artifact_providers else None,
            support_statuses=list(artifact_statuses) if artifact_statuses else None,
            artifact_kinds=list(artifact_kinds) if artifact_kinds else None,
            record_limit=artifact_limit,
            record_offset=artifact_offset,
        )
    if check_cohorts:
        from polylogue.schemas.verification import list_artifact_cohort_rows

        cohort_rows = list_artifact_cohort_rows(
            providers=list(artifact_providers) if artifact_providers else None,
            support_statuses=list(artifact_statuses) if artifact_statuses else None,
            artifact_kinds=list(artifact_kinds) if artifact_kinds else None,
            record_limit=artifact_limit,
            record_offset=artifact_offset,
        )

    # Run repairs before output so JSON mode includes repair results
    repair_results: list | None = None
    if repair:
        repair_results = run_all_repairs(config, dry_run=preview)

    vacuum_result: dict[str, Any] | None = None
    if repair and vacuum:
        if preview:
            vacuum_result = {
                "ok": True,
                "preview": True,
                "detail": "Preview mode: VACUUM skipped.",
            }
        elif json_output:
            vacuum_result = _vacuum_database(env)

    if json_output:
        out = report.to_dict()
        if schema_report is not None:
            out["schema_verification"] = schema_report.to_dict()
        if proof_report is not None:
            out["artifact_proof"] = proof_report.to_dict()
        if artifact_rows is not None:
            out["artifact_observations"] = {
                "record_limit": artifact_limit if artifact_limit is not None else "all",
                "record_offset": max(0, artifact_offset),
                "count": len(artifact_rows),
                "items": [row.model_dump(mode="json") for row in artifact_rows],
            }
        if cohort_rows is not None:
            out["artifact_cohorts"] = {
                "record_limit": artifact_limit if artifact_limit is not None else "all",
                "record_offset": max(0, artifact_offset),
                "count": len(cohort_rows),
                "items": [row.model_dump(mode="json") for row in cohort_rows],
            }
        if repair_results is not None:
            out["repairs"] = [r.to_dict() for r in repair_results]
        if vacuum_result is not None:
            out["vacuum"] = vacuum_result
        click.echo(json.dumps(out, indent=2))
        return

    lines = []
    for check in report.checks:
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
        line = f"{status_icon} {check.name}: {check.detail}"
        lines.append(line)

        # Show breakdown for warnings/errors or if verbose
        if check.breakdown and (verbose or check.status in (VerifyStatus.WARNING, VerifyStatus.ERROR)):
            for provider, count in sorted(check.breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"    {provider}: {count:,}")

    summary = report.summary
    summary_line = (
        f"Summary: {summary.get('ok', 0)} ok, {summary.get('warning', 0)} warnings, {summary.get('error', 0)} errors"
    )
    lines.append("")
    lines.append(summary_line)

    if schema_report is not None:
        lines.append("")
        lines.append(
            f"Schema verification: {schema_report.total_records:,} raw records "
            f"(samples={schema_report.max_samples if schema_report.max_samples is not None else 'all'}, "
            f"records={schema_report.record_limit if schema_report.record_limit is not None else 'all'}, "
            f"offset={schema_report.record_offset})"
        )
        for provider, stats in sorted(schema_report.providers.items()):
                lines.append(
                    f"  {provider}: valid={stats.valid_records:,} invalid={stats.invalid_records:,} "
                    f"drift={stats.drift_records:,} skipped={stats.skipped_no_schema:,} "
                    f"decode_errors={stats.decode_errors:,} quarantined={stats.quarantined_records:,}"
                )

    if proof_report is not None:
        summary = proof_report.to_dict()["summary"]
        lines.append("")
        lines.append(
            f"Artifact proof: {proof_report.total_records:,} artifact observations "
            f"(contract_backed={summary['contract_backed_records']:,}, "
            f"unsupported={summary['unsupported_parseable_records']:,}, "
            f"non_parseable={summary['recognized_non_parseable_records']:,}, "
            f"unknown={summary['unknown_records']:,}, "
            f"decode_errors={summary['decode_errors']:,})"
        )
        if summary["subagent_streams"]:
            lines.append(
                f"  Claude subagents: linked_sidecars={summary['linked_sidecars']:,} "
                f"orphan_sidecars={summary['orphan_sidecars']:,} "
                f"streams={summary['subagent_streams']:,}"
            )
        for provider, stats in sorted(proof_report.providers.items()):
            lines.append(
                f"  {provider}: contract_backed={stats.contract_backed_records:,} "
                f"unsupported={stats.unsupported_parseable_records:,} "
                f"non_parseable={stats.recognized_non_parseable_records:,} "
                f"unknown={stats.unknown_records:,} "
                f"decode_errors={stats.decode_errors:,}"
            )

    if artifact_rows is not None:
        lines.append("")
        lines.append(f"Artifact observations: {len(artifact_rows):,} rows")
        for row in artifact_rows:
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

    if cohort_rows is not None:
        lines.append("")
        lines.append(f"Artifact cohorts: {len(cohort_rows):,} cohorts")
        for row in cohort_rows:
            lines.append(
                f"  {row.provider_name} {row.artifact_kind} {row.support_status} "
                f"count={row.observation_count:,} cohort={row.cohort_id or '-'} "
                f"version={row.resolved_package_version or '-'} "
                f"element={row.resolved_element_kind or '-'}"
            )

    if runtime_report is not None:
        lines.append("")
        lines.append("Runtime Environment:")
        for check in runtime_report.checks:
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
        rt_summary = runtime_report.summary
        lines.append(
            f"  Runtime: {rt_summary.get('ok', 0)} ok, {rt_summary.get('warning', 0)} warnings, "
            f"{rt_summary.get('error', 0)} errors"
        )

    env.ui.summary("Health Check", lines)

    # Show repair results in plain text mode
    if repair_results is not None:
        click.echo("")
        mode_label = "Preview of repairs" if preview else "Running repairs"
        click.echo(f"{mode_label}...")
        total_repaired = 0
        for result in repair_results:
            if result.repaired_count > 0 or not result.success:
                status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                if env.ui.plain:
                    status = "OK" if result.success else "FAIL"
                env.ui.console.print(f"  {status} {result.name}: {result.detail}")
                total_repaired += result.repaired_count

        if total_repaired > 0:
            action = "Would repair" if preview else "Repaired"
            click.echo(f"\n{action} {total_repaired} issue(s)")
        else:
            click.echo("  No issues found that could be automatically repaired.")
    elif repair:
        env.ui.console.print("No issues to repair.")

    if repair and vacuum and preview:
        env.ui.console.print("")
        env.ui.console.print("Preview mode: VACUUM skipped.")
    elif repair and vacuum:
        _run_vacuum(env)


def _run_vacuum(env: AppEnv) -> None:
    """Run VACUUM to reclaim unused space."""
    result = _vacuum_database(env)
    env.ui.console.print("")
    env.ui.console.print(result["detail"])


def _vacuum_database(env: AppEnv) -> dict[str, Any]:
    """Run VACUUM and return a machine-readable result."""
    from polylogue.storage.backends.connection import open_connection

    try:
        with open_connection(env.config.db_path) as conn:
            conn.execute("VACUUM")
        return {"ok": True, "detail": "Running VACUUM to reclaim space...\n  VACUUM complete."}
    except Exception as exc:
        return {"ok": False, "detail": f"Running VACUUM to reclaim space...\n  VACUUM failed: {exc}"}


def _parse_schema_samples(raw: str) -> int | None:
    value = raw.strip().lower()
    if value == "all":
        return None
    try:
        parsed = int(value)
    except ValueError:
        fail("check", "--schema-samples must be a positive integer or 'all'")
    if parsed <= 0:
        fail("check", "--schema-samples must be a positive integer or 'all'")
    return parsed


__all__ = ["check_command"]
