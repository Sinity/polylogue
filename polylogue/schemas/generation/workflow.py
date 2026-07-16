"""Workflow orchestration for schema generation and package emission."""

from __future__ import annotations

from pathlib import Path

from polylogue.core.json import JSONDocument
from polylogue.paths import db_path as index_db_path
from polylogue.schemas.generation.archive_workload_profile import (
    build_archive_workload_profile,
    write_archive_workload_profile,
)
from polylogue.schemas.generation.models import GenerationResult, _ProviderBundle
from polylogue.schemas.generation.provider_bundle import _build_provider_bundle
from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
from polylogue.schemas.observation import PROVIDERS
from polylogue.schemas.privacy_config import SchemaPrivacyConfig
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.runtime_registry import ElementSchemaMap


def _package_schemas(bundle: _ProviderBundle) -> dict[str, ElementSchemaMap]:
    return {
        version: {element_kind: dict(schema) for element_kind, schema in element_schemas.items()}
        for version, element_schemas in bundle.package_schemas.items()
    }


def _package_workload_profiles(bundle: _ProviderBundle) -> dict[str, JSONDocument]:
    profiles = getattr(bundle, "package_workload_profiles", {})
    return {version: dict(profile) for version, profile in profiles.items()}


def persist_generated_provider_bundle(output_dir: Path, provider: str, bundle: _ProviderBundle) -> None:
    """Persist a generated provider bundle into the registry storage."""
    result = bundle.result
    if not result.success or bundle.manifest is None or bundle.catalog is None:
        return

    registry = SchemaRegistry(storage_root=output_dir)
    registry.replace_provider_packages(
        provider,
        bundle.catalog,
        _package_schemas(bundle),
        package_workload_profiles=_package_workload_profiles(bundle),
    )
    registry.save_cluster_manifest(bundle.manifest)

    for old_name in (f"{provider}.schema.json.gz", f"{provider}.schema.json"):
        old_path = output_dir / old_name
        if old_path.exists():
            old_path.unlink()


def generate_provider_schema(
    provider: str,
    db_path: Path | None = None,
    max_samples: int | None = None,
    privacy_config: SchemaPrivacyConfig | None = None,
    full_corpus: bool = False,
) -> GenerationResult:
    """Generate the default inferred schema for a provider."""
    return _build_provider_bundle(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        privacy_config=privacy_config,
        full_corpus=full_corpus,
    ).result


def generate_all_schemas(
    output_dir: Path,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
    privacy_config: SchemaPrivacyConfig | None = None,
    include_archive_workload_profile: bool = False,
) -> list[GenerationResult]:
    """Generate versioned schemas for all providers."""
    if db_path is None:
        db_path = index_db_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_list = providers or list(PROVIDERS.keys())
    results = []
    package_mix: dict[str, dict[str, int]] = {}
    for provider in provider_list:
        bundle = _build_provider_bundle(
            provider,
            db_path=db_path,
            max_samples=max_samples,
            privacy_config=privacy_config,
        )
        results.append(bundle.result)
        persist_generated_provider_bundle(output_dir, provider, bundle)
        if bundle.catalog is not None:
            package_mix[provider] = {package.version: package.bundle_scope_count for package in bundle.catalog.packages}

    if include_archive_workload_profile:
        archive_profile = build_archive_workload_profile(
            db_path,
            package_mix=package_mix,
            privacy_policy=privacy_config.level if privacy_config is not None else "standard",
        )
        if archive_profile is not None:
            write_archive_workload_profile(output_dir, archive_profile)

    return results


__all__ = [
    "_build_provider_bundle",
    "generate_all_schemas",
    "generate_provider_schema",
    "generate_schema_from_samples",
]
