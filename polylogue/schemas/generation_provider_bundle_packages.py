"""Catalog and manifest assembly for provider-bundle generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from polylogue.schemas.generation_cluster_support import (
    _artifact_priority,
    _cluster_profile_tokens,
    _cluster_sort_key,
)
from polylogue.schemas.generation_models import (
    GenerationResult,
    _ClusterAccumulator,
    _PackageAccumulator,
    _ProviderBundle,
    _UnitMembership,
)
from polylogue.schemas.generation_packages import (
    _element_profile_tokens,
    _membership_observed_window,
    _merge_representative_paths,
)
from polylogue.schemas.generation_schema_builder import (
    _apply_schema_metadata,
    _generate_cluster_schema,
)
from polylogue.schemas.observation import ProviderConfig
from polylogue.schemas.packages import (
    SchemaElementManifest,
    SchemaPackageCatalog,
    SchemaVersionPackage,
)
from polylogue.schemas.redaction_report import SchemaReport
from polylogue.schemas.registry import ClusterManifest, SchemaCluster
from polylogue.types import Provider


@dataclass(frozen=True)
class ProviderCatalogArtifacts:
    """Fully assembled provider package artifacts."""

    catalog: SchemaPackageCatalog
    package_schemas: dict[str, dict[str, dict[str, Any]]]
    package_reports: dict[str, dict[str, SchemaReport | None]]
    manifest: ClusterManifest


def build_provider_catalog_artifacts(
    *,
    provider_token: Provider,
    config: ProviderConfig,
    provider: str,
    clusters: dict[str, _ClusterAccumulator],
    memberships: list[_UnitMembership],
    packages: list[_PackageAccumulator],
    sample_count: int,
    artifact_counts: dict[str, int],
    orphan_adjunct_counts: dict[str, int],
    privacy_config: Any | None,
) -> ProviderCatalogArtifacts:
    """Build package schemas, catalog metadata, and manifest for a provider."""
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
                artifact_kind=element_kind,
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
    return ProviderCatalogArtifacts(
        catalog=catalog,
        package_schemas=package_schemas,
        package_reports=package_reports,
        manifest=manifest,
    )


def build_success_provider_bundle(
    *,
    provider_token: Provider,
    sample_count: int,
    clusters: dict[str, _ClusterAccumulator],
    artifact_counts: dict[str, int],
    catalog_artifacts: ProviderCatalogArtifacts,
) -> _ProviderBundle:
    """Build the final successful provider bundle from assembled artifacts."""
    default_package = (
        catalog_artifacts.catalog.package(catalog_artifacts.catalog.default_version)
        if catalog_artifacts.catalog.default_version
        else None
    )
    default_schema = (
        catalog_artifacts.package_schemas[default_package.version][default_package.default_element_kind]
        if default_package is not None
        else None
    )
    default_redaction_report = (
        catalog_artifacts.package_reports[default_package.version][default_package.default_element_kind]
        if default_package is not None
        else None
    )
    return _ProviderBundle(
        result=GenerationResult(
            provider=str(provider_token),
            schema=default_schema,
            sample_count=sample_count,
            redaction_report=default_redaction_report,
            versions=[package.version for package in catalog_artifacts.catalog.packages],
            default_version=catalog_artifacts.catalog.default_version,
            cluster_count=len(clusters),
            package_count=len(catalog_artifacts.catalog.packages),
            artifact_counts=artifact_counts,
        ),
        catalog=catalog_artifacts.catalog,
        package_schemas=catalog_artifacts.package_schemas,
        manifest=catalog_artifacts.manifest,
    )


__all__ = [
    "ProviderCatalogArtifacts",
    "build_provider_catalog_artifacts",
    "build_success_provider_bundle",
]
