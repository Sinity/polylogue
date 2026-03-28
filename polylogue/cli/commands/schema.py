"""Schema generation, package inspection, comparison, and audit commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from polylogue.cli.helpers import fail
from polylogue.cli.types import AppEnv
from polylogue.schemas.operator_models import (
    SchemaAuditRequest,
    SchemaCompareRequest,
    SchemaExplainRequest,
    SchemaInferRequest,
    SchemaListRequest,
    SchemaPromoteRequest,
)
from polylogue.schemas.operator_workflow import (
    audit_schemas,
    compare_schema_versions,
    explain_schema,
    infer_schema,
    list_schemas,
    promote_schema_cluster,
)


@click.group("schema")
@click.pass_context
def schema_command(ctx: click.Context) -> None:
    """Schema generation, package versioning, and evidence inspection."""
    del ctx


@schema_command.command("generate")
@click.option("--provider", required=True, help="Provider to generate schema for")
@click.option("--cluster", is_flag=True, help="Also cluster observed samples by structural fingerprint")
@click.option("--max-samples", type=int, default=None, help="Limit samples for generation")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option(
    "--privacy",
    type=click.Choice(["strict", "standard", "permissive"], case_sensitive=False),
    default=None,
    help="Privacy preset level (default: standard)",
)
@click.option(
    "--privacy-config",
    "privacy_config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to TOML privacy config overrides",
)
@click.option("--report", is_flag=True, help="Write a redaction report alongside the schema")
@click.pass_obj
def schema_generate(
    env: AppEnv,
    provider: str,
    cluster: bool,
    max_samples: int | None,
    json_output: bool,
    privacy: str | None,
    privacy_config_path: Path | None,
    report: bool,
) -> None:
    """Generate provider schema packages and optional evidence clusters."""
    from polylogue.schemas.privacy_config import PrivacyConfig, load_privacy_config

    cli_overrides: dict[str, Any] = {}
    if privacy:
        cli_overrides["level"] = privacy
    if privacy_config_path:
        privacy_config = load_privacy_config(
            cli_overrides=cli_overrides,
            project_path=privacy_config_path.parent,
        )
    elif cli_overrides:
        privacy_config = PrivacyConfig(**cli_overrides)
    else:
        privacy_config = None

    result = infer_schema(
        SchemaInferRequest(
            provider=provider,
            db_path=env.config.db_path,
            max_samples=max_samples,
            privacy_config=privacy_config,
            cluster=cluster,
        )
    )
    generation = result.generation
    if not generation.success:
        fail("schema generate", generation.error or "Schema generation failed")
    if cluster and result.manifest is None:
        fail("schema generate", "No samples found for clustering")

    if report and generation.redaction_report is not None:
        click.echo(generation.redaction_report.format_summary(), err=True)
        if not json_output:
            report_path = Path(f"{provider}-redaction-report.md")
            report_path.write_text(generation.redaction_report.format_markdown())
            click.echo(f"  Redaction report: {report_path}", err=True)

    if json_output:
        payload: dict[str, Any] = {
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
        click.echo(json.dumps(payload, indent=2))
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


@schema_command.command("list")
@click.option("--provider", default=None, help="Filter to specific provider")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_list(env: AppEnv, provider: str | None, json_output: bool) -> None:
    """List available schema packages, versions, and evidence manifests."""
    del env
    result = list_schemas(SchemaListRequest(provider=provider))
    if provider:
        selected = result.selected
        if json_output:
            click.echo(json.dumps(result.to_dict(), indent=2))
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
                    f"{kind}={count}"
                    for kind, count in sorted(selected.catalog.orphan_adjunct_counts.items())
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
                    f"  {cluster.cluster_id}: {cluster.sample_count:,} samples, "
                    f"confidence={cluster.confidence}{status}"
                )
        return

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
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
        click.echo(
            f"  {snapshot.provider}: {len(snapshot.versions)} version(s){package_str}, "
            f"latest={latest}{age_str}"
        )


@schema_command.command("compare")
@click.option("--provider", required=True, help="Provider name")
@click.option("--from", "from_version", required=True, help="Source version (e.g., v1)")
@click.option("--to", "to_version", required=True, help="Target version (e.g., v2)")
@click.option("--element", "element_kind", default=None, help="Element kind inside the package")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--markdown", "md_output", is_flag=True, help="Output as Markdown")
@click.pass_obj
def schema_compare(
    env: AppEnv,
    provider: str,
    from_version: str,
    to_version: str,
    element_kind: str | None,
    json_output: bool,
    md_output: bool,
) -> None:
    """Compare two schema package versions for a provider."""
    del env
    try:
        result = compare_schema_versions(
            SchemaCompareRequest(
                provider=provider,
                from_version=from_version,
                to_version=to_version,
                element_kind=element_kind,
            )
        )
    except ValueError as exc:
        fail("schema compare", str(exc))

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
    elif md_output:
        click.echo(result.diff.to_markdown())
    else:
        click.echo(result.diff.to_text())


@schema_command.command("promote")
@click.option("--provider", required=True, help="Provider name")
@click.option("--cluster", "cluster_id", required=True, help="Evidence cluster ID to promote")
@click.option("--with-samples", is_flag=True, help="Re-load samples for full schema generation")
@click.option("--max-samples", type=int, default=500, help="Max samples when using --with-samples")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_promote(
    env: AppEnv,
    provider: str,
    cluster_id: str,
    with_samples: bool,
    max_samples: int,
    json_output: bool,
) -> None:
    """Promote an evidence cluster to a new registered package version."""
    try:
        result = promote_schema_cluster(
            SchemaPromoteRequest(
                provider=provider,
                cluster_id=cluster_id,
                db_path=env.config.db_path,
                with_samples=with_samples,
                max_samples=max_samples,
            )
        )
    except ValueError as exc:
        fail("schema promote", str(exc))

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    click.echo(f"Promoted cluster {result.cluster_id} -> package {result.package_version}")
    click.echo(f"Schema package registered for {result.provider} as {result.package_version}")
    click.echo(f"Available versions: {', '.join(result.versions)}")


def _render_explain_verbose(result) -> None:
    roles = result.annotations.roles
    if roles:
        click.echo("  Semantic Roles:")
        for role in roles:
            evidence_str = ", ".join(f"{key}={value}" for key, value in role.evidence.items())
            click.echo(f"    {role.role} -> {role.path} (confidence={role.confidence:.3f})")
            if evidence_str:
                click.echo(f"      evidence: {evidence_str}")

    coverage = result.annotations.coverage
    if coverage.total_fields:
        click.echo(f"\n  Annotation Coverage ({coverage.total_fields} fields):")
        click.echo(
            f"    Format:        {coverage.with_format}/{coverage.total_fields} "
            f"({coverage.with_format / coverage.total_fields * 100:.0f}%)"
        )
        click.echo(
            f"    Enum values:   {coverage.with_values}/{coverage.total_fields} "
            f"({coverage.with_values / coverage.total_fields * 100:.0f}%)"
        )
        click.echo(
            f"    Semantic role: {coverage.with_role}/{coverage.total_fields} "
            f"({coverage.with_role / coverage.total_fields * 100:.0f}%)"
        )


@schema_command.command("explain")
@click.option("--provider", required=True, help="Provider name")
@click.option("--version", default="latest", help="Schema version (default: latest)")
@click.option("--element", "element_kind", default=None, help="Element kind inside the package")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show semantic roles and coverage")
@click.pass_obj
def schema_explain(
    env: AppEnv,
    provider: str,
    version: str,
    element_kind: str | None,
    json_output: bool,
    verbose: bool,
) -> None:
    """Explain a package element schema with evidence and annotations."""
    del env
    try:
        result = explain_schema(
            SchemaExplainRequest(
                provider=provider,
                version=version,
                element_kind=element_kind,
            )
        )
    except ValueError as exc:
        fail("schema explain", str(exc))

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    props = result.schema.get("properties", {})
    resolved_element = result.element_kind or (
        result.package.default_element_kind
        if result.package is not None
        else result.schema.get("x-polylogue-element-kind", "?")
    )
    click.echo(f"Schema: {result.provider} {result.version} [{resolved_element}]")
    if result.package is not None:
        click.echo(
            f"  Package anchor={result.package.anchor_kind}, "
            f"anchor-family={result.package.anchor_profile_family_id or 'n/a'}, "
            f"profiles={len(result.package.profile_family_ids)}, scopes={result.package.bundle_scope_count}, "
            f"window={result.package.first_seen} -> {result.package.last_seen}"
        )
    click.echo(
        f"  {len(props)} properties, {result.schema.get('x-polylogue-sample-count', '?')} samples, "
        f"{result.annotations.semantic_count} semantic roles, {result.annotations.format_count} format annotations"
    )
    click.echo(
        f"  Privacy: standard ({result.annotations.values_count} fields with enums, "
        f"{result.annotations.total_enum_values} values included)"
    )
    click.echo()
    click.echo(f"  $id: {result.schema.get('$id', 'N/A')}")
    click.echo(f"  Title: {result.schema.get('title', 'N/A')}")
    click.echo(f"  Description: {result.schema.get('description', 'N/A')}")
    click.echo()

    click.echo("  Metadata:")
    for key, label in [
        ("x-polylogue-version", "Version"),
        ("x-polylogue-generated-at", "Generated"),
        ("x-polylogue-registered-at", "Registered"),
        ("x-polylogue-promoted-at", "Promoted"),
        ("x-polylogue-sample-count", "Samples"),
        ("x-polylogue-sample-granularity", "Granularity"),
        ("x-polylogue-anchor-profile-family-id", "Anchor profile family"),
        ("x-polylogue-profile-family-ids", "Element profile families"),
        ("x-polylogue-package-profile-family-ids", "Package profile families"),
        ("x-polylogue-observed-artifact-count", "Observed artifacts"),
        ("x-polylogue-evidence-confidence", "Evidence confidence"),
    ]:
        value = result.schema.get(key)
        if value is not None:
            click.echo(f"    {label}: {value}")

    click.echo(f"\n  Properties ({len(props)}):")
    for name, prop_schema in sorted(props.items()):
        type_str = prop_schema.get("type", "?")
        annotations: list[str] = []
        if prop_schema.get("x-polylogue-semantic-role"):
            annotations.append(f"role={prop_schema['x-polylogue-semantic-role']}")
        if prop_schema.get("x-polylogue-format"):
            annotations.append(f"fmt={prop_schema['x-polylogue-format']}")
        if prop_schema.get("x-polylogue-frequency") is not None:
            annotations.append(f"freq={prop_schema['x-polylogue-frequency']}")
        if prop_schema.get("x-polylogue-values"):
            values = prop_schema["x-polylogue-values"]
            preview = ", ".join(str(value) for value in values[:5])
            if len(values) > 5:
                preview += f" (+{len(values) - 5} more)"
            annotations.append(f"values=[{preview}]")
        ann_str = f" ({', '.join(annotations)})" if annotations else ""
        click.echo(f"    {name}: {type_str}{ann_str}")

    for key in ("x-polylogue-foreign-keys", "x-polylogue-time-deltas", "x-polylogue-mutually-exclusive"):
        value = result.schema.get(key)
        if value:
            click.echo(f"\n  {key}:")
            for entry in value:
                click.echo(f"    {json.dumps(entry)}")

    if verbose:
        click.echo()
        _render_explain_verbose(result)


@schema_command.command("audit")
@click.option("--provider", default=None, help="Audit a specific provider (default: all)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_audit(
    env: AppEnv,
    provider: str | None,
    json_output: bool,
) -> None:
    """Run automated quality checks on committed schema packages."""
    del env
    report = audit_schemas(SchemaAuditRequest(provider=provider))
    if json_output:
        click.echo(json.dumps(report.to_json(), indent=2))
    else:
        click.echo(report.format_text())
    if not report.all_passed:
        raise SystemExit(1)


__all__ = ["schema_command"]
