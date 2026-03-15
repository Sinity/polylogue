"""Schema inference, clustering, versioning, and promotion CLI commands."""

from __future__ import annotations

import json
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
@click.pass_obj
def schema_infer(
    env: AppEnv,
    provider: str,
    cluster: bool,
    max_samples: int | None,
    json_output: bool,
) -> None:
    """Infer schema from provider data, optionally clustering by structure."""
    from polylogue.schemas.registry import SchemaRegistry
    from polylogue.schemas.schema_generation import generate_provider_schema

    result = generate_provider_schema(
        provider,
        db_path=env.config.db_path,
        max_samples=max_samples,
    )

    if not result.success:
        fail("schema infer", result.error or "Schema generation failed")

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
                if c.promoted_version:
                    click.echo(f"    promoted: {c.promoted_version}")
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
        versions = registry.list_versions(provider)
        manifest = registry.load_cluster_manifest(provider)

        if json_output:
            out: dict[str, Any] = {
                "provider": provider,
                "versions": versions,
            }
            if manifest:
                out["manifest"] = manifest.to_dict()
            click.echo(json.dumps(out, indent=2))
        else:
            if not versions:
                click.echo(f"No schemas found for provider: {provider}")
                return

            click.echo(f"Provider: {provider}")
            click.echo(f"Versions: {', '.join(versions)}")
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
                    status = f" [promoted: {c.promoted_version}]" if c.promoted_version else ""
                    click.echo(f"  {c.cluster_id}: {c.sample_count:,} samples, confidence={c.confidence}{status}")
    else:
        providers = registry.list_providers()

        if json_output:
            result = []
            for p in providers:
                versions = registry.list_versions(p)
                entry: dict[str, Any] = {"provider": p, "versions": versions}
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
                latest = versions[-1] if versions else "none"
                age = registry.get_schema_age_days(p)
                age_str = f" ({age}d old)" if age is not None else ""
                click.echo(f"  {p}: {len(versions)} version(s), latest={latest}{age_str}")


@schema_command.command("compare")
@click.option("--provider", required=True, help="Provider name")
@click.option("--from", "from_version", required=True, help="Source version (e.g., v1)")
@click.option("--to", "to_version", required=True, help="Target version (e.g., v2)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--markdown", "md_output", is_flag=True, help="Output as Markdown")
@click.pass_obj
def schema_compare(
    env: AppEnv,
    provider: str,
    from_version: str,
    to_version: str,
    json_output: bool,
    md_output: bool,
) -> None:
    """Compare two schema versions for a provider."""
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()

    try:
        diff = registry.compare_versions(provider, from_version, to_version)
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
        schema = registry.get_schema(provider, version=new_version)
        click.echo(json.dumps({
            "provider": provider,
            "cluster_id": cluster_id,
            "promoted_version": new_version,
            "schema": schema,
        }, indent=2))
    else:
        click.echo(f"Promoted cluster {cluster_id} -> {new_version}")
        click.echo(f"Schema registered for {provider} as {new_version}")

        # Show what's now available
        versions = registry.list_versions(provider)
        click.echo(f"Available versions: {', '.join(versions)}")


@schema_command.command("explain")
@click.option("--provider", required=True, help="Provider name")
@click.option("--version", "version", default="latest", help="Schema version (default: latest)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_obj
def schema_explain(
    env: AppEnv,
    provider: str,
    version: str,
    json_output: bool,
) -> None:
    """Explain a schema version with annotations and metadata."""
    from polylogue.schemas.registry import SchemaRegistry

    registry = SchemaRegistry()
    schema = registry.get_schema(provider, version=version)

    if schema is None:
        fail("schema explain", f"No schema found for {provider} version={version}")

    if json_output:
        click.echo(json.dumps(schema, indent=2))
        return

    # Human-readable explanation
    click.echo(f"Schema: {provider} {version}")
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
    props = schema.get("properties", {})
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


__all__ = ["schema_command"]
