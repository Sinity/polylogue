"""Catalog and manifest assembly for provider-bundle generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.schemas.generation.cluster_support import (
    _artifact_priority,
    _cluster_profile_tokens,
    _cluster_sort_key,
)
from polylogue.schemas.generation.models import (
    GenerationResult,
    _ClusterAccumulator,
    _PackageAccumulator,
    _ProviderBundle,
    _UnitMembership,
)
from polylogue.schemas.generation.observation_journal import JournalMemberships, ObservationJournal
from polylogue.schemas.generation.packages import (
    _element_profile_tokens,
    _membership_observed_window,
    _merge_representative_paths,
    _package_bundle_scope_count,
)
from polylogue.schemas.generation.replay import (
    MembershipSamples,
    MembershipSessionIds,
    metadata_memberships,
    select_artifact_memberships,
)
from polylogue.schemas.generation.schema_builder import (
    _apply_schema_metadata,
    _generate_cluster_schema,
)
from polylogue.schemas.generation.workload_profiles import build_package_workload_profile
from polylogue.schemas.observation import ProviderConfig
from polylogue.schemas.packages import (
    SchemaElementManifest,
    SchemaPackageCatalog,
    SchemaVersionPackage,
)
from polylogue.schemas.privacy_config import SchemaPrivacyConfig
from polylogue.schemas.redaction_report import SchemaReport
from polylogue.schemas.registry import ClusterManifest, SchemaCluster


@dataclass(frozen=True)
class ProviderCatalogArtifacts:
    """Fully assembled provider package artifacts."""

    catalog: SchemaPackageCatalog
    package_schemas: dict[str, dict[str, JSONDocument]]
    package_reports: dict[str, dict[str, SchemaReport | None]]
    package_workload_profiles: dict[str, JSONDocument]
    manifest: ClusterManifest


def _coverage_rank(package: SchemaVersionPackage) -> tuple[int, int, str, str]:
    """Rank fallback fitness without letting one long transcript dominate."""
    return (
        package.bundle_scope_count,
        package.sample_count,
        package.last_seen,
        package.version,
    )


def _select_catalog_versions(
    packages: Sequence[SchemaVersionPackage],
) -> tuple[str | None, str | None, str | None, JSONDocument]:
    """Keep temporal and coverage semantics distinct and explain the choice."""
    if not packages:
        return None, None, None, {}

    latest = packages[-1]
    recommended = max(packages, key=_coverage_rank)
    rationale = json_document(
        {
            "latest": {
                "version": latest.version,
                "strategy": "latest_first_observed_family",
                "first_seen": latest.first_seen,
            },
            "recommended": {
                "version": recommended.version,
                "strategy": "coverage_first",
                "rank_fields": ["bundle_scope_count", "sample_count", "last_seen", "version"],
                "bundle_scope_count": recommended.bundle_scope_count,
                "sample_count": recommended.sample_count,
            },
            "default": {
                "version": recommended.version,
                "strategy": "recommended_fallback",
            },
        }
    )
    return latest.version, recommended.version, recommended.version, rationale


def _json_text_values(values: Iterable[str]) -> list[JSONValue]:
    return list(values)


def _membership_exact_structure_ids(memberships: Sequence[_UnitMembership]) -> list[str]:
    if isinstance(memberships, JournalMemberships):
        return list(memberships.iter_distinct_values("exact_structure_id"))
    return sorted({membership.unit.exact_structure_id for membership in memberships})


def _membership_profile_family_ids(memberships: Sequence[_UnitMembership]) -> list[str]:
    if isinstance(memberships, JournalMemberships):
        return list(memberships.iter_distinct_values("profile_family_id"))
    return sorted({membership.profile_family_id for membership in memberships})


def _membership_bundle_scopes(memberships: Sequence[_UnitMembership]) -> list[str]:
    if isinstance(memberships, JournalMemberships):
        return list(memberships.iter_distinct_values("bundle_scope"))
    return sorted({membership.unit.bundle_scope for membership in memberships if membership.unit.bundle_scope})


def _package_scope_keys(package: _PackageAccumulator) -> list[str]:
    if isinstance(package.memberships, JournalMemberships):
        return list(package.memberships.iter_scope_keys())
    return sorted(package.bundle_scopes)


def build_provider_catalog_artifacts(
    *,
    provider_token: Provider,
    config: ProviderConfig,
    provider: str,
    clusters: dict[str, _ClusterAccumulator],
    memberships: Sequence[_UnitMembership],
    packages: list[_PackageAccumulator],
    sample_count: int,
    artifact_counts: dict[str, int],
    orphan_adjunct_counts: dict[str, int],
    privacy_config: SchemaPrivacyConfig | None,
    observation_outcomes: JSONDocument,
    journal: ObservationJournal | None = None,
) -> ProviderCatalogArtifacts:
    """Build package schemas, catalog metadata, and manifest for a provider."""
    total_units = max(sum(acc.sample_count for acc in clusters.values()), 1)
    package_schemas: dict[str, dict[str, JSONDocument]] = {}
    package_reports: dict[str, dict[str, SchemaReport | None]] = {}
    package_workload_profiles: dict[str, JSONDocument] = {}
    catalog_packages: list[SchemaVersionPackage] = []
    cluster_to_package_version: dict[str, str] = {}

    for index, package_acc in enumerate(packages, start=1):
        version = f"v{index}"
        package_schemas[version] = {}
        package_reports[version] = {}

        package_metadata = metadata_memberships(package_acc.memberships)
        element_kinds = {membership.unit.artifact_kind for membership in package_metadata}
        package_profile_family_ids = _membership_profile_family_ids(package_metadata)
        package_profile_family_ids_json = _json_text_values(package_profile_family_ids)

        elements: list[SchemaElementManifest] = []
        total_package_samples = 0
        for element_kind in sorted(
            element_kinds,
            key=lambda item: (_artifact_priority(item), item),
            reverse=True,
        ):
            kind_memberships = select_artifact_memberships(package_acc.memberships, element_kind)
            kind_metadata = select_artifact_memberships(package_metadata, element_kind)
            schema_samples = MembershipSamples(kind_memberships)
            conv_ids = MembershipSessionIds(kind_memberships)
            representative_paths: list[str] = []
            exact_structure_ids = _membership_exact_structure_ids(kind_metadata)
            profile_family_ids = _membership_profile_family_ids(kind_metadata)
            element_bundle_scopes = _membership_bundle_scopes(kind_metadata)
            element_first_seen, element_last_seen = _membership_observed_window(kind_metadata)
            for membership in kind_metadata:
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
                observed_artifact_count=len(kind_metadata),
            )
            profile_family_ids_json = _json_text_values(profile_family_ids)
            exact_structure_ids_json = _json_text_values(exact_structure_ids)
            schema["x-polylogue-package-version"] = version
            schema["x-polylogue-profile-family-ids"] = profile_family_ids_json
            schema["x-polylogue-exact-structure-ids"] = exact_structure_ids_json
            if element_first_seen:
                schema["x-polylogue-element-first-seen"] = element_first_seen
            if element_last_seen:
                schema["x-polylogue-element-last-seen"] = element_last_seen
            schema["x-polylogue-element-bundle-scope-count"] = len(element_bundle_scopes)
            schema["x-polylogue-anchor-profile-family-id"] = package_acc.anchor_family_id
            schema["x-polylogue-package-profile-family-ids"] = package_profile_family_ids_json
            package_schemas[version][element_kind] = schema
            package_reports[version][element_kind] = redaction_report
            elements.append(
                SchemaElementManifest(
                    element_kind=element_kind,
                    schema_file=f"{element_kind}.schema.json.gz",
                    sample_count=len(schema_samples),
                    artifact_count=len(kind_metadata),
                    first_seen=element_first_seen or "",
                    last_seen=element_last_seen or "",
                    bundle_scope_count=len(element_bundle_scopes),
                    bundle_scopes=element_bundle_scopes,
                    exact_structure_ids=exact_structure_ids,
                    profile_family_ids=profile_family_ids,
                    profile_tokens=_element_profile_tokens(kind_metadata),
                    representative_paths=representative_paths,
                    observed_artifact_count=len(kind_metadata),
                )
            )

        package = SchemaVersionPackage(
            provider=provider_token,
            version=version,
            anchor_kind=package_acc.anchor_kind,
            default_element_kind=package_acc.anchor_kind,
            first_seen=package_acc.first_seen or datetime.now(tz=timezone.utc).isoformat(),
            last_seen=package_acc.last_seen or package_acc.first_seen or datetime.now(tz=timezone.utc).isoformat(),
            bundle_scope_count=_package_bundle_scope_count(package_acc),
            sample_count=total_package_samples,
            anchor_profile_family_id=package_acc.anchor_family_id,
            bundle_scopes=_package_scope_keys(package_acc),
            profile_family_ids=package_profile_family_ids,
            representative_paths=package_acc.representative_paths,
            elements=elements,
            workload_profile_file="workload-profile.json.gz",
        )
        package_workload_profiles[version] = build_package_workload_profile(
            provider=str(provider_token),
            version=version,
            package=package_acc,
            element_schemas=package_schemas[version],
            privacy_policy=privacy_config.level if privacy_config is not None else "standard",
            observation_outcomes=observation_outcomes,
        )
        catalog_packages.append(package)
        for cluster_id in package.profile_family_ids:
            cluster_to_package_version[cluster_id] = version

    latest_version, default_version, recommended_version, selection_rationale = _select_catalog_versions(
        catalog_packages
    )
    catalog = SchemaPackageCatalog(
        provider=provider_token,
        packages=catalog_packages,
        latest_version=latest_version,
        default_version=default_version,
        recommended_version=recommended_version,
        orphan_adjunct_counts=orphan_adjunct_counts,
        selection_rationale=selection_rationale,
        observation_outcomes=observation_outcomes,
    )
    manifest_clusters: list[SchemaCluster] = []
    for cluster_id, acc in sorted(clusters.items(), key=_cluster_sort_key, reverse=True):
        cluster_memberships = (
            journal.memberships(profile_family_id=cluster_id, include_samples=False) if journal is not None else None
        )
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
                exact_structure_ids=(
                    list(cluster_memberships.iter_distinct_values("exact_structure_id"))
                    if cluster_memberships is not None
                    else sorted(acc.exact_structure_ids)
                ),
                bundle_scope_count=(
                    cluster_memberships.distinct_count("bundle_scope")
                    if cluster_memberships is not None
                    else len(acc.bundle_scopes)
                ),
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
        package_workload_profiles=package_workload_profiles,
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
        package_workload_profiles=catalog_artifacts.package_workload_profiles,
        manifest=catalog_artifacts.manifest,
    )


__all__ = [
    "ProviderCatalogArtifacts",
    "build_provider_catalog_artifacts",
    "build_success_provider_bundle",
]
