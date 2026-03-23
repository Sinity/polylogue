"""Public facade for staged schema generation and package assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polylogue.paths import db_path as default_db_path
from polylogue.schemas import generation_analysis as _generation_analysis
from polylogue.schemas.field_stats import UUID_PATTERN, is_dynamic_key
from polylogue.schemas.generation_models import GenerationResult
from polylogue.schemas.generation_support import (
    _annotate_schema,
    _annotate_semantic_and_relational,
    _merge_schemas,
    _remove_nested_required,
    collapse_dynamic_keys,
)
from polylogue.schemas.generation_workflow import (
    _build_provider_bundle as _workflow_build_provider_bundle,
)
from polylogue.schemas.generation_workflow import (
    generate_provider_schema as _workflow_generate_provider_schema,
)
from polylogue.schemas.generation_workflow import (
    generate_schema_from_samples as _workflow_generate_schema_from_samples,
)
from polylogue.schemas.observation import PROVIDERS
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.sampling import iter_schema_units
from polylogue.schemas.shape_fingerprint import _structure_fingerprint


def _sync_generation_patch_surfaces() -> None:
    _generation_analysis.iter_schema_units = iter_schema_units


def _collect_cluster_accumulators(
    provider: str,
    *,
    db_path: Path,
    max_samples: int | None,
    reservoir_size: int,
):
    _sync_generation_patch_surfaces()
    return _generation_analysis._collect_cluster_accumulators(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        reservoir_size=reservoir_size,
    )


def _cluster_reservoir_size(config, max_samples: int | None) -> int:
    return _generation_analysis._cluster_reservoir_size(config, max_samples)


def _build_package_candidates(provider: str, *, memberships, clusters):
    return _generation_analysis._build_package_candidates(
        provider,
        memberships=memberships,
        clusters=clusters,
    )


def generate_schema_from_samples(
    samples: list[dict[str, Any]],
    *,
    annotate: bool = True,
    max_stats_samples: int = 500,
    max_genson_samples: int | None = None,
) -> dict[str, Any]:
    return _workflow_generate_schema_from_samples(
        samples,
        annotate=annotate,
        max_stats_samples=max_stats_samples,
        max_genson_samples=max_genson_samples,
    )


def _build_provider_bundle(
    provider: str,
    *,
    db_path: Path | None,
    max_samples: int | None,
    privacy_config: Any | None,
):
    _sync_generation_patch_surfaces()
    return _workflow_build_provider_bundle(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        privacy_config=privacy_config,
    )


def generate_provider_schema(
    provider: str,
    db_path: Path | None = None,
    max_samples: int | None = None,
    privacy_config: Any | None = None,
) -> GenerationResult:
    _sync_generation_patch_surfaces()
    return _workflow_generate_provider_schema(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        privacy_config=privacy_config,
    )


def generate_all_schemas(
    output_dir: Path,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
    privacy_config: Any | None = None,
) -> list[GenerationResult]:
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
    "GenerationResult",
    "_annotate_schema",
    "_annotate_semantic_and_relational",
    "_build_package_candidates",
    "_build_provider_bundle",
    "_cluster_reservoir_size",
    "_collect_cluster_accumulators",
    "_merge_schemas",
    "_remove_nested_required",
    "_structure_fingerprint",
    "UUID_PATTERN",
    "collapse_dynamic_keys",
    "generate_all_schemas",
    "generate_provider_schema",
    "generate_schema_from_samples",
    "is_dynamic_key",
]
