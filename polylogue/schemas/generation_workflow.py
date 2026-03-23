"""Workflow orchestration for schema generation and package emission."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.paths import db_path as default_db_path
from polylogue.schemas.field_stats import _collect_field_stats
from polylogue.schemas.generation_analysis import (
    _artifact_priority,
    _build_package_candidates,
    _cluster_profile_tokens,
    _cluster_reservoir_size,
    _cluster_sort_key,
    _collect_cluster_accumulators,
    _element_profile_tokens,
    _membership_observed_window,
    _merge_representative_paths,
)
from polylogue.schemas.generation_models import GenerationResult, _ProviderBundle, _UnitMembership
from polylogue.schemas.generation_support import (
    GENSON_AVAILABLE,
    SchemaBuilder,
    _annotate_schema,
    _annotate_semantic_and_relational,
    _build_redaction_report,
    _remove_nested_required,
    collapse_dynamic_keys,
)
from polylogue.schemas.observation import PROVIDERS, ProviderConfig, resolve_provider_config
from polylogue.schemas.packages import (
    SchemaElementManifest,
    SchemaPackageCatalog,
    SchemaVersionPackage,
)
from polylogue.schemas.redaction_report import SchemaReport
from polylogue.schemas.registry import ClusterManifest, SchemaCluster, SchemaRegistry
from polylogue.schemas.shape_fingerprint import _structure_fingerprint
from polylogue.types import Provider

_STRUCTURE_EXEMPLARS_PER_FINGERPRINT = 8


def _generate_cluster_schema(
    provider: str,
    config: ProviderConfig,
    samples: list[dict[str, Any]],
    conv_ids: list[str | None],
    *,
    privacy_config: Any | None,
) -> tuple[dict[str, Any], SchemaReport | None]:
    """Generate one schema version from the bounded cluster reservoir."""
    if not samples:
        return {"type": "object", "description": "No samples available"}, None

    builder = SchemaBuilder()
    fingerprint_counts: dict[Any, int] = {}
    for sample in samples:
        fingerprint = _structure_fingerprint(sample)
        seen = fingerprint_counts.get(fingerprint, 0)
        if seen < _STRUCTURE_EXEMPLARS_PER_FINGERPRINT:
            builder.add_object(sample)
            fingerprint_counts[fingerprint] = seen + 1

    schema = builder.to_schema()
    schema = collapse_dynamic_keys(schema)
    schema = _remove_nested_required(schema)
    if config.sample_granularity == "record":
        schema.pop("required", None)

    conv_ids_for_stats: list[str | None] | None = conv_ids if any(conv_id is not None for conv_id in conv_ids) else None
    field_stats = _collect_field_stats(samples, conversation_ids=conv_ids_for_stats)
    schema = _annotate_schema(
        schema,
        field_stats,
        min_conversation_count=3,
        privacy_config=privacy_config,
    )
    schema = _annotate_semantic_and_relational(schema, field_stats)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

    redaction_report = _build_redaction_report(
        provider,
        field_stats,
        schema,
        privacy_config=privacy_config,
        privacy_level=getattr(privacy_config, "level", "standard") if privacy_config else "standard",
    )
    return schema, redaction_report


def _apply_schema_metadata(
    schema: dict[str, Any],
    *,
    provider: str,
    config: ProviderConfig,
    schema_sample_count: int,
    anchor_profile_family_id: str,
    artifact_kind: str,
    observed_artifact_count: int,
) -> None:
    schema["title"] = f"{provider} export format ({artifact_kind})"
    schema["description"] = config.description
    schema["x-polylogue-generated-at"] = datetime.now(tz=timezone.utc).isoformat()
    schema["x-polylogue-sample-count"] = schema_sample_count
    schema["x-polylogue-generator"] = "polylogue.schemas.schema_inference"
    schema["x-polylogue-sample-granularity"] = config.sample_granularity
    schema["x-polylogue-anchor-profile-family-id"] = anchor_profile_family_id
    schema["x-polylogue-observed-artifact-count"] = observed_artifact_count
    schema["x-polylogue-artifact-kind"] = artifact_kind


def generate_schema_from_samples(
    samples: list[dict[str, Any]],
    *,
    annotate: bool = True,
    max_stats_samples: int = 500,
    max_genson_samples: int | None = None,
) -> dict[str, Any]:
    """Generate JSON schema from samples using genson, with optional annotations."""
    if not GENSON_AVAILABLE:
        raise ImportError("genson is required for schema generation. Install with: pip install genson")

    if not samples:
        return {"type": "object", "description": "No samples available"}

    genson_samples = samples
    if max_genson_samples and len(samples) > max_genson_samples:
        import random

        rng = random.Random(0)
        genson_samples = rng.sample(samples, max_genson_samples)

    builder = SchemaBuilder()
    for sample in genson_samples:
        builder.add_object(sample)

    schema = builder.to_schema()
    schema = collapse_dynamic_keys(schema)
    schema = _remove_nested_required(schema)

    if annotate:
        stats_samples = samples
        if max_stats_samples and len(samples) > max_stats_samples:
            import random

            rng = random.Random(42)
            stats_samples = rng.sample(samples, max_stats_samples)

        field_stats = _collect_field_stats(stats_samples)
        schema = _annotate_schema(schema, field_stats)
        schema = _annotate_semantic_and_relational(schema, field_stats)

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def generate_provider_schema(
    provider: str,
    db_path: Path | None = None,
    max_samples: int | None = None,
    privacy_config: Any | None = None,
) -> GenerationResult:
    """Generate the default inferred schema for a provider."""
    return _build_provider_bundle(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        privacy_config=privacy_config,
    ).result


def _build_provider_bundle(
    provider: str,
    *,
    db_path: Path | None,
    max_samples: int | None,
    privacy_config: Any | None,
) -> _ProviderBundle:
    """Generate all inferred schema versions plus the default result for a provider."""
    provider_token = Provider.from_string(provider)
    if provider_token not in PROVIDERS:
        return _ProviderBundle(
            result=GenerationResult(
                provider=str(provider_token),
                schema=None,
                sample_count=0,
                error=f"Unknown provider: {provider}. Known: {[str(item) for item in PROVIDERS]}",
            ),
        )
    if db_path is None:
        db_path = default_db_path()
    if not GENSON_AVAILABLE:
        return _ProviderBundle(
            result=GenerationResult(
                provider=str(provider_token),
                schema=None,
                sample_count=0,
                error="genson not installed",
            ),
        )

    config = resolve_provider_config(provider_token)

    try:
        clusters, memberships, sample_count, artifact_counts = _collect_cluster_accumulators(
            provider,
            db_path=db_path,
            max_samples=max_samples,
            reservoir_size=_cluster_reservoir_size(config, max_samples),
        )
        if not clusters:
            return _ProviderBundle(
                result=GenerationResult(
                    provider=str(provider_token),
                    schema=None,
                    sample_count=0,
                    error="No samples found",
                ),
            )
        packages, orphan_adjunct_counts = _build_package_candidates(
            str(provider_token),
            memberships=memberships,
            clusters=clusters,
        )
        if not packages:
            return _ProviderBundle(
                result=GenerationResult(
                    provider=str(provider_token),
                    schema=None,
                    sample_count=sample_count,
                    error="No anchor-backed schema packages found",
                    cluster_count=len(clusters),
                    artifact_counts=artifact_counts,
                ),
                manifest=ClusterManifest(
                    provider=provider_token,
                    clusters=[
                        SchemaCluster(
                            cluster_id=cluster_id,
                            provider=provider_token,
                            sample_count=acc.sample_count,
                            first_seen=acc.first_seen or "",
                            last_seen=acc.last_seen or "",
                            representative_paths=acc.representative_paths,
                            dominant_keys=acc.dominant_keys,
                            confidence=1.0,
                            artifact_kind=acc.artifact_kind,
                            profile_tokens=list(_cluster_profile_tokens(acc)),
                            exact_structure_ids=sorted(acc.exact_structure_ids),
                            bundle_scope_count=len(acc.bundle_scopes),
                        )
                        for cluster_id, acc in sorted(clusters.items(), key=_cluster_sort_key, reverse=True)
                    ],
                    artifact_counts=artifact_counts,
                ),
            )

        total_units = max(sum(acc.sample_count for acc in clusters.values()), 1)
        package_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        package_reports: dict[str, dict[str, SchemaReport | None]] = {}
        catalog_packages: list[SchemaVersionPackage] = []
        cluster_to_package_version: dict[str, str] = {}

        for index, package_acc in enumerate(packages, start=1):
            version = f"v{index}"
            package_schemas[version] = {}
            package_reports[version] = {}

            element_memberships: dict[str, list[_UnitMembership]] = {}
            for membership in package_acc.memberships:
                element_memberships.setdefault(membership.unit.artifact_kind, []).append(membership)

            elements: list[SchemaElementManifest] = []
            total_package_samples = 0
            for element_kind, kind_memberships in sorted(
                element_memberships.items(),
                key=lambda item: (_artifact_priority(item[0]), item[0]),
                reverse=True,
            ):
                schema_samples: list[dict[str, Any]] = []
                conv_ids: list[str | None] = []
                representative_paths: list[str] = []
                exact_structure_ids = sorted({membership.unit.exact_structure_id for membership in kind_memberships})
                profile_family_ids = sorted({membership.profile_family_id for membership in kind_memberships})
                element_bundle_scopes = sorted(
                    {membership.unit.bundle_scope for membership in kind_memberships if membership.unit.bundle_scope}
                )
                element_first_seen, element_last_seen = _membership_observed_window(kind_memberships)
                for membership in kind_memberships:
                    schema_samples.extend(membership.unit.schema_samples)
                    conv_ids.extend([membership.unit.conversation_id] * len(membership.unit.schema_samples))
                    if membership.unit.source_path:
                        _merge_representative_paths(representative_paths, [membership.unit.source_path])

                total_package_samples += len(schema_samples)
                schema, redaction_report = _generate_cluster_schema(
                    provider,
                    config,
                    schema_samples,
                    conv_ids,
                    privacy_config=privacy_config,
                )
                _apply_schema_metadata(
                    schema,
                    provider=str(provider_token),
                    config=config,
                    schema_sample_count=len(schema_samples),
                    anchor_profile_family_id=package_acc.anchor_family_id,
                    artifact_kind=element_kind,
                    observed_artifact_count=len(kind_memberships),
                )
                schema["x-polylogue-package-version"] = version
                schema["x-polylogue-profile-family-ids"] = profile_family_ids
                schema["x-polylogue-exact-structure-ids"] = exact_structure_ids
                if element_first_seen:
                    schema["x-polylogue-element-first-seen"] = element_first_seen
                if element_last_seen:
                    schema["x-polylogue-element-last-seen"] = element_last_seen
                schema["x-polylogue-element-bundle-scope-count"] = len(element_bundle_scopes)
                schema["x-polylogue-anchor-profile-family-id"] = package_acc.anchor_family_id
                schema["x-polylogue-package-profile-family-ids"] = sorted(package_acc.profile_family_ids)
                package_schemas[version][element_kind] = schema
                package_reports[version][element_kind] = redaction_report
                elements.append(
                    SchemaElementManifest(
                        element_kind=element_kind,
                        schema_file=f"{element_kind}.schema.json.gz",
                        sample_count=len(schema_samples),
                        artifact_count=len(kind_memberships),
                        first_seen=element_first_seen or "",
                        last_seen=element_last_seen or "",
                        bundle_scope_count=len(element_bundle_scopes),
                        bundle_scopes=element_bundle_scopes,
                        exact_structure_ids=exact_structure_ids,
                        profile_family_ids=profile_family_ids,
                        profile_tokens=_element_profile_tokens(kind_memberships),
                        representative_paths=representative_paths,
                        observed_artifact_count=len(kind_memberships),
                    )
                )

            package = SchemaVersionPackage(
                provider=provider_token,
                version=version,
                anchor_kind=package_acc.anchor_kind,
                default_element_kind=package_acc.anchor_kind,
                first_seen=package_acc.first_seen or datetime.now(tz=timezone.utc).isoformat(),
                last_seen=package_acc.last_seen or package_acc.first_seen or datetime.now(tz=timezone.utc).isoformat(),
                bundle_scope_count=len(package_acc.bundle_scopes),
                sample_count=total_package_samples,
                anchor_profile_family_id=package_acc.anchor_family_id,
                bundle_scopes=sorted(package_acc.bundle_scopes),
                profile_family_ids=sorted(package_acc.profile_family_ids),
                representative_paths=package_acc.representative_paths,
                elements=elements,
            )
            catalog_packages.append(package)
            for cluster_id in package.profile_family_ids:
                cluster_to_package_version[cluster_id] = version

        latest_version = catalog_packages[-1].version if catalog_packages else None
        catalog = SchemaPackageCatalog(
            provider=provider_token,
            packages=catalog_packages,
            latest_version=latest_version,
            default_version=latest_version,
            recommended_version=latest_version,
            orphan_adjunct_counts=orphan_adjunct_counts,
        )
        manifest_clusters: list[SchemaCluster] = []
        for cluster_id, acc in sorted(clusters.items(), key=_cluster_sort_key, reverse=True):
            manifest_clusters.append(
                SchemaCluster(
                    cluster_id=cluster_id,
                    provider=provider_token,
                    sample_count=acc.sample_count,
                    first_seen=acc.first_seen or "",
                    last_seen=acc.last_seen or "",
                    representative_paths=acc.representative_paths,
                    dominant_keys=acc.dominant_keys,
                    confidence=round(min(1.0, acc.sample_count / max(total_units * 0.1, 1)), 3),
                    artifact_kind=acc.artifact_kind,
                    profile_tokens=list(_cluster_profile_tokens(acc)),
                    exact_structure_ids=sorted(acc.exact_structure_ids),
                    bundle_scope_count=len(acc.bundle_scopes),
                    promoted_package_version=cluster_to_package_version.get(cluster_id),
                )
            )
        manifest = ClusterManifest(
            provider=provider_token,
            clusters=manifest_clusters,
            artifact_counts=artifact_counts,
            default_version=catalog.default_version,
        )
        default_package = catalog.package(catalog.default_version) if catalog.default_version else None
        default_schema = (
            package_schemas[default_package.version][default_package.default_element_kind]
            if default_package is not None
            else None
        )
        default_redaction_report = (
            package_reports[default_package.version][default_package.default_element_kind]
            if default_package is not None
            else None
        )
        return _ProviderBundle(
            result=GenerationResult(
                provider=str(provider_token),
                schema=default_schema,
                sample_count=sample_count,
                redaction_report=default_redaction_report,
                versions=[package.version for package in catalog.packages],
                default_version=catalog.default_version,
                cluster_count=len(clusters),
                package_count=len(catalog.packages),
                artifact_counts=artifact_counts,
            ),
            catalog=catalog,
            package_schemas=package_schemas,
            manifest=manifest,
        )
    except Exception as e:
        return _ProviderBundle(
            result=GenerationResult(
                provider=str(provider_token),
                schema=None,
                sample_count=0,
                error=str(e),
            ),
        )


def generate_all_schemas(
    output_dir: Path,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
    privacy_config: Any | None = None,
) -> list[GenerationResult]:
    """Generate versioned schemas for all providers."""
    if db_path is None:
        db_path = default_db_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_list = providers or list(PROVIDERS.keys())
    results: list[GenerationResult] = []
    for provider in provider_list:
        bundle = _build_provider_bundle(
            provider,
            db_path=db_path,
            max_samples=max_samples,
            privacy_config=privacy_config,
        )
        result = bundle.result
        results.append(result)

        if result.success and bundle.manifest is not None and bundle.catalog is not None:
            registry = SchemaRegistry(storage_root=output_dir)
            registry.replace_provider_packages(provider, bundle.catalog, bundle.package_schemas)
            registry.save_cluster_manifest(bundle.manifest)

            for legacy_name in (f"{provider}.schema.json.gz", f"{provider}.schema.json"):
                legacy_path = output_dir / legacy_name
                if legacy_path.exists():
                    legacy_path.unlink()

    return results


__all__ = [
    "_apply_schema_metadata",
    "_build_provider_bundle",
    "_generate_cluster_schema",
    "generate_all_schemas",
    "generate_provider_schema",
    "generate_schema_from_samples",
]
