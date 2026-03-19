"""Schema inference, clustering, versioning, and promotion CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from polylogue.cli.helpers import fail
from polylogue.cli.types import AppEnv


@click.group("schema")
@click.pass_context
def schema_command(ctx: click.Context) -> None:
    """Schema inference, versioning, and promotion."""
    pass


@schema_command.command("infer")
@click.option("--provider", required=True, help="Provider to infer schema for")
@click.option("--cluster", is_flag=True, help="Cluster samples by structural fingerprint")
@click.option("--max-samples", type=int, default=None, help="Limit samples for inference")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option(
    "--privacy",
    type=click.Choice(["strict", "standard", "permissive"], case_sensitive=False),
    default=None,
    help="Privacy preset level (default: standard)",
)
@click.option("--privacy-config", "privacy_config_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to TOML privacy config overrides")
@click.option("--report", is_flag=True, help="Write a redaction report alongside the schema")
@click.pass_obj
def schema_infer(
    env: AppEnv,
    provider: str,
    cluster: bool,
    max_samples: int | None,
    json_output: bool,
    privacy: str | None,
    privacy_config_path: Path | None,
    report: bool,
) -> None:
    """Infer schema from provider data, optionally clustering by structure."""
    from polylogue.schemas.privacy_config import PrivacyConfig, load_privacy_config
    from polylogue.schemas.registry import SchemaRegistry
    from polylogue.schemas.schema_generation import generate_provider_schema

    # Build privacy config from cascade
    cli_overrides: dict[str, Any] = {}
    if privacy:
        cli_overrides["level"] = privacy
    if privacy_config_path:
        p_config = load_privacy_config(
            cli_overrides=cli_overrides,
            project_path=privacy_config_path.parent,
        )
    elif cli_overrides:
        p_config = PrivacyConfig(**cli_overrides)
    else:
        p_config = None

    result = generate_provider_schema(
        provider,
        db_path=env.config.db_path,
        max_samples=max_samples,
        privacy_config=p_config,
    )

    if not result.success:
        fail("schema infer", result.error or "Schema generation failed")

    # Redaction report (when --report is set)
    if report and result.redaction_report:
        import sys

        click.echo(result.redaction_report.format_summary(), err=True)
        if not json_output:
            report_md = result.redaction_report.format_markdown()
            report_path = Path(f"{provider}-redaction-report.md")
            report_path.write_text(report_md)
            click.echo(f"  Redaction report: {report_path}", err=True)

    registry = SchemaRegistry()

    if cluster:
        # Load samples for clustering
        from polylogue.schemas.sampling import PROVIDERS, load_samples_from_db

        config = PROVIDERS.get(provider)
        if config is None:
            fail("schema infer", f"Unknown provider: {provider}")

        if config.db_provider_name:
            samples = load_samples_from_db(
                config.db_provider_name,
                db_path=env.config.db_path,
                max_samples=max_samples or 500,
            )
        else:
            samples = []

        if not samples:
            fail("schema infer", "No samples found for clustering")

        manifest = registry.cluster_samples(provider, samples)
        manifest_path = registry.save_cluster_manifest(manifest)

        if json_output:
            click.echo(json.dumps(manifest.to_dict(), indent=2))
        else:
            click.echo(f"Clustered {result.sample_count:,} samples into {len(manifest.clusters)} cluster(s)")
            click.echo(f"Manifest saved to: {manifest_path}")
            click.echo()
            for c in manifest.clusters:
                pct = (c.sample_count / result.sample_count * 100) if result.sample_count else 0
                click.echo(f"  [{c.cluster_id}] {c.sample_count:,} samples ({pct:.1f}%)")
                click.echo(f"    confidence: {c.confidence}")
                click.echo(f"    keys: {', '.join(c.dominant_keys[:10])}")
                if c.schema_version:
                    click.echo(f"    version: {c.schema_version}")
    else:
        if json_output:
            click.echo(json.dumps(result.schema, indent=2))
        else:
            click.echo(f"Inferred schema for {provider}: {result.sample_count:,} samples")
            props = result.schema.get("properties", {}) if result.schema else {}
            click.echo(f"  Properties: {len(props)}")
            if props:
                click.echo(f"  Top-level keys: {', '.join(sorted(props.keys())[:15])}")

            version = result.schema.get("x-polylogue-version") if result.schema else None
            if version:
                click.echo(f"  Schema version: v{version}")


@schema_command.command("list")
@click.option("--provider", default=None, help="Filter to specific provider")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_list(env: AppEnv, provider: str | None, json_output: bool) -> None:
    """List available schemas and versions."""
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()

    if provider:
        catalog = registry.load_package_catalog(provider)
        versions = registry.list_versions(provider)
        manifest = registry.load_cluster_manifest(provider)

        if json_output:
            out: dict[str, Any] = {
                "provider": provider,
                "versions": versions,
            }
            if catalog:
                out["catalog"] = catalog.to_dict()
            if manifest:
                out["manifest"] = manifest.to_dict()
            click.echo(json.dumps(out, indent=2))
        else:
            if not versions and catalog is None:
                click.echo(f"No schemas found for provider: {provider}")
                return

            click.echo(f"Provider: {provider}")
            click.echo(f"Versions: {', '.join(versions)}")
            if catalog:
                click.echo(
                    f"Default={catalog.default_version}, latest={catalog.latest_version}, "
                    f"recommended={catalog.recommended_version}"
                )
                if catalog.orphan_adjunct_counts:
                    counts = ", ".join(f"{kind}={count}" for kind, count in sorted(catalog.orphan_adjunct_counts.items()))
                    click.echo(f"Orphan adjunct evidence: {counts}")
                click.echo()
                for package in catalog.packages:
                    click.echo(
                        f"  {package.version}: anchor={package.anchor_kind}, "
                        f"default={package.default_element_kind}, scopes={package.bundle_scope_count}, "
                        f"window={package.first_seen} -> {package.last_seen}"
                    )
                    for element in package.elements:
                        click.echo(
                            f"    - {element.element_kind}: "
                            f"{element.sample_count} samples / {element.artifact_count} artifacts"
                        )
            else:
                latest = versions[-1] if versions else None
                if latest:
                    schema = registry.get_schema(provider, version=latest)
                    if schema:
                        age = registry.get_schema_age_days(provider)
                        props = schema.get("properties", {})
                        click.echo(f"Latest: {latest} ({len(props)} properties)")
                        if age is not None:
                            click.echo(f"Age: {age} days")
                        sample_count = schema.get("x-polylogue-sample-count")
                        if sample_count:
                            click.echo(f"Sample count: {sample_count:,}")

            if manifest:
                click.echo(f"\nCluster manifest ({len(manifest.clusters)} clusters):")
                for c in manifest.clusters:
                    status = f" [version: {c.schema_version}]" if c.schema_version else ""
                    click.echo(f"  {c.cluster_id}: {c.sample_count:,} samples, confidence={c.confidence}{status}")
    else:
        providers = registry.list_providers()

        if json_output:
            result = []
            for p in providers:
                versions = registry.list_versions(p)
                entry: dict[str, Any] = {"provider": p, "versions": versions}
                catalog = registry.load_package_catalog(p)
                if catalog:
                    entry["package_count"] = len(catalog.packages)
                    entry["default_version"] = catalog.default_version
                    entry["latest_version"] = catalog.latest_version
                manifest = registry.load_cluster_manifest(p)
                if manifest:
                    entry["cluster_count"] = len(manifest.clusters)
                result.append(entry)
            click.echo(json.dumps(result, indent=2))
        else:
            if not providers:
                click.echo("No schemas found.")
                return

            click.echo(f"Found {len(providers)} provider(s):\n")
            for p in providers:
                versions = registry.list_versions(p)
                catalog = registry.load_package_catalog(p)
                latest = (catalog.latest_version if catalog else (versions[-1] if versions else None)) or "none"
                age = registry.get_schema_age_days(p)
                age_str = f" ({age}d old)" if age is not None else ""
                package_str = f", packages={len(catalog.packages)}" if catalog else ""
                click.echo(f"  {p}: {len(versions)} version(s){package_str}, latest={latest}{age_str}")


@schema_command.command("compare")
@click.option("--provider", required=True, help="Provider name")
@click.option("--from", "from_version", required=True, help="Source version (e.g., v1)")
@click.option("--to", "to_version", required=True, help="Target version (e.g., v2)")
@click.option("--element", "element_kind", default=None, help="Element kind inside the package (defaults to package default)")
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
    """Compare two schema versions for a provider."""
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()

    try:
        diff = registry.compare_versions(provider, from_version, to_version, element_kind=element_kind)
    except ValueError as exc:
        fail("schema compare", str(exc))

    if json_output:
        click.echo(json.dumps(diff.to_dict(), indent=2))
    elif md_output:
        click.echo(diff.to_markdown())
    else:
        click.echo(diff.to_text())


@schema_command.command("promote")
@click.option("--provider", required=True, help="Provider name")
@click.option("--cluster", "cluster_id", required=True, help="Cluster ID to promote")
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
    """Promote a cluster's schema to a new registered version."""
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()

    samples: list[dict[str, Any]] | None = None
    if with_samples:
        from polylogue.schemas.schema_generation import _structure_fingerprint
        from polylogue.schemas.registry import _fingerprint_hash
        from polylogue.schemas.sampling import PROVIDERS, load_samples_from_db

        config = PROVIDERS.get(provider)
        if config is None:
            fail("schema promote", f"Unknown provider: {provider}")

        if config.db_provider_name:
            all_samples = load_samples_from_db(
                config.db_provider_name,
                db_path=env.config.db_path,
                max_samples=max_samples,
            )
        else:
            all_samples = []

        # Filter to samples matching the cluster fingerprint
        samples = [
            s for s in all_samples
            if _fingerprint_hash(_structure_fingerprint(s)) == cluster_id
        ]

        if not samples:
            fail("schema promote", f"No samples match cluster {cluster_id}")

    try:
        new_version = registry.promote_cluster(provider, cluster_id, samples=samples)
    except ValueError as exc:
        fail("schema promote", str(exc))

    if json_output:
        package = registry.get_package(provider, version=new_version)
        schema = registry.get_element_schema(provider, version=new_version)
        click.echo(json.dumps({
            "provider": provider,
            "cluster_id": cluster_id,
            "package_version": new_version,
            "package": package.to_dict() if package else None,
            "schema": schema,
        }, indent=2))
    else:
        click.echo(f"Promoted cluster {cluster_id} -> package {new_version}")
        click.echo(f"Schema package registered for {provider} as {new_version}")

        # Show what's now available
        versions = registry.list_versions(provider)
        click.echo(f"Available versions: {', '.join(versions)}")


@schema_command.command("explain")
@click.option("--provider", required=True, help="Provider name")
@click.option("--version", "version", default="latest", help="Schema version (default: latest)")
@click.option("--element", "element_kind", default=None, help="Element kind inside the package (defaults to package default)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show semantic roles, privacy summary, annotation coverage")
@click.pass_obj
def schema_explain(
    env: AppEnv,
    provider: str,
    version: str,
    element_kind: str | None,
    json_output: bool,
    verbose: bool,
) -> None:
    """Explain a schema version with annotations and metadata."""
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()
    package = registry.get_package(provider, version=version)
    schema = registry.get_element_schema(provider, version=version, element_kind=element_kind)

    if schema is None:
        fail(
            "schema explain",
            f"No schema found for {provider} version={version}"
            + (f" element={element_kind}" if element_kind else ""),
        )

    if json_output:
        payload: dict[str, Any] = {"schema": schema}
        if package is not None:
            payload["package"] = package.to_dict()
        click.echo(json.dumps(payload, indent=2))
        return

    # Count annotations for summary line
    props = schema.get("properties", {})
    n_semantic = 0
    n_format = 0
    n_values = 0
    total_enum_values = 0

    def _count_annotations(s: dict[str, Any]) -> None:
        nonlocal n_semantic, n_format, n_values, total_enum_values
        if not isinstance(s, dict):
            return
        if "x-polylogue-semantic-role" in s:
            n_semantic += 1
        if "x-polylogue-format" in s:
            n_format += 1
        if "x-polylogue-values" in s:
            n_values += 1
            total_enum_values += len(s["x-polylogue-values"])
        for sub in s.get("properties", {}).values():
            _count_annotations(sub)
        if isinstance(s.get("items"), dict):
            _count_annotations(s["items"])
        if isinstance(s.get("additionalProperties"), dict):
            _count_annotations(s["additionalProperties"])
        for kw in ("anyOf", "oneOf", "allOf"):
            for sub in s.get(kw, []):
                _count_annotations(sub)

    _count_annotations(schema)

    # Summary header
    sample_count = schema.get("x-polylogue-sample-count", "?")
    resolved_element = element_kind or (
        package.default_element_kind if package else schema.get("x-polylogue-element-kind", "?")
    )
    click.echo(f"Schema: {provider} {version} [{resolved_element}]")
    if package is not None:
        click.echo(
            f"  Package anchor={package.anchor_kind}, scopes={package.bundle_scope_count}, "
            f"window={package.first_seen} -> {package.last_seen}"
        )
    click.echo(
        f"  {len(props)} properties, {sample_count} samples, "
        f"{n_semantic} semantic roles, {n_format} format annotations"
    )
    click.echo(
        f"  Privacy: standard ({n_values} fields with enums, "
        f"{total_enum_values} values included)"
    )
    click.echo()

    # Basic metadata
    click.echo(f"  $id: {schema.get('$id', 'N/A')}")
    click.echo(f"  Title: {schema.get('title', 'N/A')}")
    click.echo(f"  Description: {schema.get('description', 'N/A')}")
    click.echo()

    # Version metadata
    meta_keys = [
        ("x-polylogue-version", "Version"),
        ("x-polylogue-generated-at", "Generated"),
        ("x-polylogue-registered-at", "Registered"),
        ("x-polylogue-promoted-at", "Promoted"),
        ("x-polylogue-sample-count", "Samples"),
        ("x-polylogue-sample-granularity", "Granularity"),
        ("x-polylogue-cluster-id", "Cluster ID"),
        ("x-polylogue-cluster-sample-count", "Cluster samples"),
        ("x-polylogue-cluster-confidence", "Cluster confidence"),
    ]
    click.echo("  Metadata:")
    for key, label in meta_keys:
        val = schema.get(key)
        if val is not None:
            click.echo(f"    {label}: {val}")

    # Properties
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
            vals = prop_schema["x-polylogue-values"]
            preview = ", ".join(str(v) for v in vals[:5])
            if len(vals) > 5:
                preview += f" (+{len(vals) - 5} more)"
            annotations.append(f"values=[{preview}]")

        ann_str = f"  ({', '.join(annotations)})" if annotations else ""
        click.echo(f"    {name}: {type_str}{ann_str}")

    # Schema-level relational annotations
    for key in ("x-polylogue-foreign-keys", "x-polylogue-time-deltas", "x-polylogue-mutually-exclusive"):
        val = schema.get(key)
        if val:
            click.echo(f"\n  {key}:")
            for entry in val:
                click.echo(f"    {json.dumps(entry)}")

    # Verbose mode: additional detail
    if verbose:
        click.echo()
        _explain_verbose(schema)


def _explain_verbose(schema: dict[str, Any]) -> None:
    """Print verbose explain sections: semantic roles, annotation coverage."""
    # Semantic role assignments
    roles: list[tuple[str, str, float, dict]] = []

    def _collect_roles(s: dict[str, Any], path: str = "$") -> None:
        if not isinstance(s, dict):
            return
        role = s.get("x-polylogue-semantic-role")
        if role:
            confidence = s.get("x-polylogue-confidence", 0.0)
            evidence = s.get("x-polylogue-evidence", {})
            roles.append((path, role, confidence, evidence))
        for name, prop in s.get("properties", {}).items():
            _collect_roles(prop, f"{path}.{name}")
        if isinstance(s.get("items"), dict):
            _collect_roles(s["items"], f"{path}[*]")
        if isinstance(s.get("additionalProperties"), dict):
            _collect_roles(s["additionalProperties"], f"{path}.*")
        for kw in ("anyOf", "oneOf", "allOf"):
            for sub in s.get(kw, []):
                _collect_roles(sub, path)

    _collect_roles(schema)

    if roles:
        click.echo("  Semantic Roles:")
        for path, role, confidence, evidence in sorted(roles, key=lambda r: -r[2]):
            evidence_str = ", ".join(f"{k}={v}" for k, v in evidence.items())
            click.echo(f"    {role} → {path} (confidence={confidence:.3f})")
            if evidence_str:
                click.echo(f"      evidence: {evidence_str}")

    # Annotation coverage
    total = 0
    with_format = 0
    with_values = 0
    with_role = 0

    def _count_coverage(s: dict[str, Any]) -> None:
        nonlocal total, with_format, with_values, with_role
        if not isinstance(s, dict):
            return
        for prop in s.get("properties", {}).values():
            if isinstance(prop, dict):
                total += 1
                if "x-polylogue-format" in prop:
                    with_format += 1
                if "x-polylogue-values" in prop:
                    with_values += 1
                if "x-polylogue-semantic-role" in prop:
                    with_role += 1
                _count_coverage(prop)
        if isinstance(s.get("items"), dict):
            _count_coverage(s["items"])
        if isinstance(s.get("additionalProperties"), dict):
            _count_coverage(s["additionalProperties"])
        for kw in ("anyOf", "oneOf", "allOf"):
            for sub in s.get(kw, []):
                _count_coverage(sub)

    _count_coverage(schema)

    if total:
        click.echo(f"\n  Annotation Coverage ({total} fields):")
        click.echo(f"    Format:        {with_format}/{total} ({with_format/total*100:.0f}%)")
        click.echo(f"    Enum values:   {with_values}/{total} ({with_values/total*100:.0f}%)")
        click.echo(f"    Semantic role:  {with_role}/{total} ({with_role/total*100:.0f}%)")


@schema_command.command("audit")
@click.option("--provider", default=None, help="Audit a specific provider (default: all)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_audit(
    env: AppEnv,
    provider: str | None,
    json_output: bool,
) -> None:
    """Run automated quality checks on committed schemas."""
    from polylogue.schemas.audit import audit_all_providers, audit_provider

    if provider:
        report = audit_provider(provider)
    else:
        report = audit_all_providers()

    if json_output:
        click.echo(json.dumps(report.to_json(), indent=2))
    else:
        click.echo(report.format_text())

    if not report.all_passed:
        raise SystemExit(1)


__all__ = ["schema_command"]
