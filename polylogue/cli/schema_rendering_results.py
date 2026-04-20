"""Non-explain rendering helpers for schema CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.cli.machine_errors import emit_success
from polylogue.scenarios import CorpusScenario, CorpusSpec
from polylogue.schemas.audit_models import AuditReport
from polylogue.schemas.operator_models import (
    SchemaCompareResult,
    SchemaInferResult,
    SchemaListResult,
    SchemaPromoteResult,
)


def _render_corpus_spec_preview(*, corpus_specs: tuple[CorpusSpec, ...], header: str) -> None:
    if not corpus_specs:
        return
    click.echo(header)
    for spec in corpus_specs[:3]:
        target = spec.profile.scope_token(element_kind=spec.element_kind)
        click.echo(
            f"    - {spec.provider}:{spec.package_version}:{target} "
            f"x{spec.count} messages={spec.messages_min}-{spec.messages_max}"
        )
    if len(corpus_specs) > 3:
        click.echo(f"    … {len(corpus_specs) - 3} more")


def _render_corpus_scenario_preview(*, corpus_scenarios: tuple[CorpusScenario, ...], header: str) -> None:
    if not corpus_scenarios:
        return
    click.echo(header)
    for scenario in corpus_scenarios[:3]:
        click.echo(
            f"    - {scenario.provider}:{scenario.package_version} "
            f"variants={len(scenario.corpus_specs)} "
            f"targets={', '.join(scenario.target_labels)}"
        )
    if len(corpus_scenarios) > 3:
        click.echo(f"    … {len(corpus_scenarios) - 3} more")


def _corpus_scenario_payloads(corpus_scenarios: tuple[CorpusScenario, ...]) -> list[dict[str, object]]:
    return [
        {
            "provider": scenario.provider,
            "package_version": scenario.package_version,
            "corpus_specs": [spec.to_payload() for spec in scenario.corpus_specs],
        }
        for scenario in corpus_scenarios
    ]


def render_schema_generate_result(
    *,
    provider: str,
    result: SchemaInferResult,
    json_output: bool,
    report: bool,
) -> None:
    """Render schema generation output and optional redaction report."""
    generation = result.generation
    if report and generation.redaction_report is not None:
        click.echo(generation.redaction_report.format_summary(), err=True)
        if not json_output:
            report_path = Path(f"{provider}-redaction-report.md")
            report_path.write_text(generation.redaction_report.format_markdown(), encoding="utf-8")
            click.echo(f"  Redaction report: {report_path}", err=True)

    if json_output:
        payload: dict[str, object] = {
            "provider": provider,
            "generation": {
                "success": generation.success,
                "sample_count": generation.sample_count,
                "cluster_count": generation.cluster_count,
                "package_count": generation.package_count,
                "versions": generation.versions,
                "default_version": generation.default_version,
                "artifact_counts": generation.artifact_counts,
                "schema": generation.schema,
            },
        }
        if result.manifest is not None:
            payload["manifest"] = result.manifest.to_dict()
            payload["manifest_path"] = str(result.manifest_path) if result.manifest_path else None
        if result.corpus_specs:
            payload["corpus_specs"] = [spec.to_payload() for spec in result.corpus_specs]
        if result.corpus_scenarios:
            payload["corpus_scenarios"] = _corpus_scenario_payloads(result.corpus_scenarios)
        emit_success(payload)
        return

    click.echo(
        f"Generated schema package set for {provider}: "
        f"{generation.sample_count:,} samples, "
        f"{generation.package_count} package(s), "
        f"{generation.cluster_count} evidence cluster(s)"
    )
    if generation.versions:
        click.echo(f"  Versions: {', '.join(generation.versions)}")
    if generation.default_version:
        click.echo(f"  Default package: {generation.default_version}")
    if result.manifest_path is not None:
        click.echo(f"  Evidence manifest: {result.manifest_path}")
    _render_corpus_scenario_preview(
        corpus_scenarios=result.corpus_scenarios,
        header="  Suggested synthetic scenarios:",
    )
    _render_corpus_spec_preview(corpus_specs=result.corpus_specs, header="  Suggested synthetic corpus specs:")


def render_schema_list_result(
    *,
    provider: str | None,
    result: SchemaListResult,
    json_output: bool,
) -> None:
    """Render schema list output in JSON or text mode."""
    if provider:
        selected = result.selected
        if json_output:
            emit_success(selected.to_dict() if selected is not None else {"provider": provider, "versions": []})
            return
        if selected is None or (not selected.versions and selected.catalog is None):
            click.echo(f"No schemas found for provider: {provider}")
            return

        click.echo(f"Provider: {selected.provider}")
        click.echo(f"Versions: {', '.join(selected.versions)}")
        if selected.catalog is not None:
            click.echo(
                f"Default={selected.catalog.default_version}, latest={selected.catalog.latest_version}, "
                f"recommended={selected.catalog.recommended_version}"
            )
            if selected.catalog.orphan_adjunct_counts:
                counts = ", ".join(
                    f"{kind}={count}" for kind, count in sorted(selected.catalog.orphan_adjunct_counts.items())
                )
                click.echo(f"Orphan adjunct evidence: {counts}")
            click.echo()
            for package in selected.catalog.packages:
                click.echo(
                    f"  {package.version}: anchor={package.anchor_kind}, "
                    f"default={package.default_element_kind}, "
                    f"anchor-family={package.anchor_profile_family_id or 'n/a'}, "
                    f"profiles={len(package.profile_family_ids)}, scopes={package.bundle_scope_count}, "
                    f"window={package.first_seen} -> {package.last_seen}"
                )
                for element in package.elements:
                    click.echo(
                        f"    - {element.element_kind}: "
                        f"{element.sample_count} samples / {element.artifact_count} artifacts, "
                        f"profiles={len(element.profile_family_ids)}, scopes={element.bundle_scope_count}, "
                        f"window={element.first_seen or 'unknown'} -> {element.last_seen or 'unknown'}"
                    )
        if selected.manifest is not None:
            click.echo(f"\nEvidence manifest ({len(selected.manifest.clusters)} clusters):")
            for cluster in selected.manifest.clusters:
                status = (
                    f" [promoted package: {cluster.promoted_package_version}]"
                    if cluster.promoted_package_version
                    else ""
                )
                click.echo(
                    f"  {cluster.cluster_id}: {cluster.sample_count:,} samples, confidence={cluster.confidence}{status}"
                )
        if selected.corpus_specs:
            click.echo()
            _render_corpus_spec_preview(
                corpus_specs=selected.corpus_specs,
                header="Suggested synthetic corpus specs:",
            )
        if selected.corpus_scenarios:
            click.echo()
            _render_corpus_scenario_preview(
                corpus_scenarios=selected.corpus_scenarios,
                header="Suggested synthetic scenarios:",
            )
        return

    if json_output:
        emit_success({"providers": result.to_dict()})
        return
    if not result.providers:
        click.echo("No schemas found.")
        return
    click.echo(f"Found {len(result.providers)} provider(s):\n")
    for snapshot in result.providers:
        latest = (
            snapshot.catalog.latest_version
            if snapshot.catalog is not None
            else (snapshot.versions[-1] if snapshot.versions else None)
        ) or "none"
        age_str = f" ({snapshot.latest_age_days}d old)" if snapshot.latest_age_days is not None else ""
        package_str = f", packages={len(snapshot.catalog.packages)}" if snapshot.catalog else ""
        corpus_spec_str = f", corpus-specs={len(snapshot.corpus_specs)}" if snapshot.corpus_specs else ""
        corpus_scenario_str = (
            f", corpus-scenarios={len(snapshot.corpus_scenarios)}" if snapshot.corpus_scenarios else ""
        )
        click.echo(
            f"  {snapshot.provider}: {len(snapshot.versions)} version(s){package_str}{corpus_spec_str}{corpus_scenario_str}, "
            f"latest={latest}{age_str}"
        )


def render_schema_compare_result(*, result: SchemaCompareResult, json_output: bool, md_output: bool) -> None:
    """Render schema comparison output."""
    if json_output:
        emit_success(result.to_dict())
    elif md_output:
        click.echo(result.diff.to_markdown())
    else:
        click.echo(result.diff.to_text())


def render_schema_promote_result(*, result: SchemaPromoteResult, json_output: bool) -> None:
    """Render schema cluster promotion output."""
    if json_output:
        emit_success(result.to_dict())
        return
    click.echo(f"Promoted cluster {result.cluster_id} -> package {result.package_version}")
    click.echo(f"Schema package registered for {result.provider} as {result.package_version}")
    click.echo(f"Available versions: {', '.join(result.versions)}")


def render_schema_audit_result(*, report: AuditReport, json_output: bool) -> None:
    """Render schema audit output."""
    if json_output:
        emit_success(report.to_json())
    else:
        click.echo(report.format_text())


__all__ = [
    "render_schema_audit_result",
    "render_schema_compare_result",
    "render_schema_generate_result",
    "render_schema_list_result",
    "render_schema_promote_result",
]
