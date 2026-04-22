"""Provider-bundle assembly for schema generation."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import db_path as archive_db_path
from polylogue.schemas.generation_cluster_collection import (
    _collect_cluster_accumulators,
)
from polylogue.schemas.generation_cluster_support import (
    _cluster_profile_tokens,
    _cluster_reservoir_size,
    _cluster_sort_key,
)
from polylogue.schemas.generation_models import GenerationResult, _ProviderBundle
from polylogue.schemas.generation_packages import (
    _build_package_candidates,
)
from polylogue.schemas.generation_provider_bundle_packages import (
    build_provider_catalog_artifacts,
    build_success_provider_bundle,
)
from polylogue.schemas.generation_support import GENSON_AVAILABLE
from polylogue.schemas.observation import PROVIDERS, ProviderConfig, resolve_provider_config
from polylogue.schemas.privacy_config import SchemaPrivacyConfig
from polylogue.schemas.registry import ClusterManifest, SchemaCluster
from polylogue.types import Provider


def build_provider_error_bundle(
    provider: str,
    *,
    error: str,
    sample_count: int = 0,
    cluster_count: int = 0,
    artifact_counts: dict[str, int] | None = None,
    manifest: ClusterManifest | None = None,
) -> _ProviderBundle:
    """Build a provider bundle carrying only an error result."""
    return _ProviderBundle(
        result=GenerationResult(
            provider=provider,
            schema=None,
            sample_count=sample_count,
            error=error,
            cluster_count=cluster_count,
            artifact_counts=artifact_counts or {},
        ),
        manifest=manifest,
    )


def _build_provider_bundle(
    provider: str,
    *,
    db_path: Path | None,
    max_samples: int | None,
    privacy_config: SchemaPrivacyConfig | None,
    full_corpus: bool = False,
) -> _ProviderBundle:
    """Generate all inferred schema versions plus the default result for a provider."""
    provider_token = Provider.from_string(provider)
    if provider_token not in PROVIDERS:
        return build_provider_error_bundle(
            str(provider_token),
            error=f"Unknown provider: {provider}. Known: {[str(item) for item in PROVIDERS]}",
        )
    if db_path is None:
        db_path = archive_db_path()
    if not GENSON_AVAILABLE:
        return build_provider_error_bundle(
            str(provider_token),
            error="genson not installed",
        )

    config: ProviderConfig = resolve_provider_config(provider_token)

    try:
        clusters, memberships, sample_count, artifact_counts = _collect_cluster_accumulators(
            provider,
            db_path=db_path,
            max_samples=max_samples,
            reservoir_size=_cluster_reservoir_size(config, max_samples),
            full_corpus=full_corpus,
        )
        if not clusters:
            return build_provider_error_bundle(
                str(provider_token),
                error="No samples found",
            )
        packages, orphan_adjunct_counts = _build_package_candidates(
            str(provider_token),
            memberships=memberships,
            clusters=clusters,
        )
        if not packages:
            return build_provider_error_bundle(
                str(provider_token),
                error="No anchor-backed schema packages found",
                sample_count=sample_count,
                cluster_count=len(clusters),
                artifact_counts=artifact_counts,
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
        catalog_artifacts = build_provider_catalog_artifacts(
            provider_token=provider_token,
            config=config,
            provider=provider,
            clusters=clusters,
            memberships=memberships,
            packages=packages,
            sample_count=sample_count,
            artifact_counts=artifact_counts,
            orphan_adjunct_counts=orphan_adjunct_counts,
            privacy_config=privacy_config,
        )
        return build_success_provider_bundle(
            provider_token=provider_token,
            sample_count=sample_count,
            clusters=clusters,
            artifact_counts=artifact_counts,
            catalog_artifacts=catalog_artifacts,
        )
    except Exception as e:
        return build_provider_error_bundle(
            str(provider_token),
            error=str(e),
        )


__all__ = ["_build_provider_bundle"]
