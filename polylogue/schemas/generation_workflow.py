"""Workflow orchestration for schema generation and package emission."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import db_path as archive_db_path
from polylogue.schemas.generation_models import GenerationResult, _ProviderBundle
from polylogue.schemas.generation_provider_bundle import _build_provider_bundle
from polylogue.schemas.generation_schema_builder import generate_schema_from_samples
from polylogue.schemas.observation import PROVIDERS
from polylogue.schemas.privacy_config import SchemaPrivacyConfig
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.runtime_registry import ElementSchemaMap


def _package_schemas(bundle: _ProviderBundle) -> dict[str, ElementSchemaMap]:
    return {
        version: {element_kind: dict(schema) for element_kind, schema in element_schemas.items()}
        for version, element_schemas in bundle.package_schemas.items()
    }


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
    )
    registry.save_cluster_manifest(bundle.manifest)

    for legacy_name in (f"{provider}.schema.json.gz", f"{provider}.schema.json"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()


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
) -> list[GenerationResult]:
    """Generate versioned schemas for all providers."""
    if db_path is None:
        db_path = archive_db_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_list = providers or list(PROVIDERS.keys())
    results = []
    for provider in provider_list:
        bundle = _build_provider_bundle(
            provider,
            db_path=db_path,
            max_samples=max_samples,
            privacy_config=privacy_config,
        )
        results.append(bundle.result)
        persist_generated_provider_bundle(output_dir, provider, bundle)

    return results


__all__ = [
    "_build_provider_bundle",
    "generate_all_schemas",
    "generate_provider_schema",
    "generate_schema_from_samples",
]
