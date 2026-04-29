"""Inference/listing/promotion helpers for schema operator workflows."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.lib.json import JSONDocument
from polylogue.scenarios import CorpusScenario, CorpusSpec, build_corpus_scenarios, build_inferred_corpus_specs
from polylogue.schemas.audit.models import AuditReport
from polylogue.schemas.operator.models import (
    SchemaAuditRequest,
    SchemaCompareRequest,
    SchemaCompareResult,
    SchemaInferRequest,
    SchemaInferResult,
    SchemaListRequest,
    SchemaListResult,
    SchemaPromoteRequest,
    SchemaPromoteResult,
    SchemaProviderSnapshot,
    operator_json_document,
)
from polylogue.schemas.operator.registry import SchemaRegistryLike, schema_registry
from polylogue.schemas.packages import SchemaPackageCatalog
from polylogue.schemas.privacy_config import PrivacyConfig, PrivacyLevel
from polylogue.schemas.tooling_models import ClusterManifest
from polylogue.types import Provider


def _typed_registry() -> SchemaRegistryLike:
    return schema_registry()


def _registry_catalog(registry: SchemaRegistryLike, provider: str) -> SchemaPackageCatalog | None:
    return registry.load_package_catalog(provider)


def _build_inferred_outputs(
    *,
    provider: str,
    package_version: str,
    manifest: ClusterManifest | None = None,
    sample_count: int = 0,
    catalog: SchemaPackageCatalog | None = None,
) -> tuple[tuple[CorpusSpec, ...], tuple[CorpusScenario, ...]]:
    corpus_specs = build_inferred_corpus_specs(
        provider=provider,
        package_version=package_version,
        manifest=manifest,
        sample_count=sample_count,
        catalog=catalog,
    )
    corpus_scenarios = build_corpus_scenarios(
        corpus_specs,
        origin="compiled.inferred-corpus-scenario",
        tags=("inferred", "schema", "synthetic", "scenario"),
    )
    return corpus_specs, corpus_scenarios


def _privacy_level(value: object) -> PrivacyLevel:
    if value == "strict":
        return "strict"
    if value == "permissive":
        return "permissive"
    return "standard"


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str) and isinstance(item, str)}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _privacy_config(payload: Mapping[str, object] | None) -> PrivacyConfig | None:
    if payload is None:
        return None
    field_overrides = payload.get("field_overrides")
    allow_patterns = payload.get("allow_value_patterns")
    deny_patterns = payload.get("deny_value_patterns")
    level = payload.get("level")
    safe_enum_max_length = payload.get("safe_enum_max_length", 50)
    high_entropy_min_length = payload.get("high_entropy_min_length", 10)
    cross_conv_min_count = payload.get("cross_conv_min_count", 3)
    return PrivacyConfig(
        level=_privacy_level(level),
        safe_enum_max_length=safe_enum_max_length if isinstance(safe_enum_max_length, int) else 50,
        high_entropy_min_length=high_entropy_min_length if isinstance(high_entropy_min_length, int) else 10,
        cross_conv_min_count=cross_conv_min_count if isinstance(cross_conv_min_count, int) else 3,
        cross_conv_proportional=bool(payload.get("cross_conv_proportional", False)),
        field_overrides=_string_mapping(field_overrides),
        allow_value_patterns=_string_list(allow_patterns),
        deny_value_patterns=_string_list(deny_patterns),
    )


def infer_schema(request: SchemaInferRequest) -> SchemaInferResult:
    from polylogue.schemas.generation.workflow import generate_provider_schema
    from polylogue.schemas.observation import PROVIDERS
    from polylogue.schemas.sampling import load_samples_from_db

    result = generate_provider_schema(
        request.provider,
        db_path=request.db_path,
        max_samples=request.max_samples,
        privacy_config=_privacy_config(request.privacy_config),
        full_corpus=request.full_corpus,
    )
    package_version = result.default_version or "default"
    registry = _typed_registry()
    if not request.cluster or not result.success:
        corpus_specs, corpus_scenarios = _build_inferred_outputs(
            provider=request.provider,
            package_version=package_version,
            sample_count=result.sample_count,
            catalog=_registry_catalog(registry, request.provider),
        )
        return SchemaInferResult(
            generation=result,
            corpus_specs=corpus_specs,
            corpus_scenarios=corpus_scenarios if result.success else (),
        )

    provider_token = Provider.from_string(request.provider)
    config = PROVIDERS.get(provider_token)
    if config is None or not config.db_provider_name:
        corpus_specs, corpus_scenarios = _build_inferred_outputs(
            provider=request.provider,
            package_version=package_version,
            sample_count=result.sample_count,
            catalog=_registry_catalog(registry, request.provider),
        )
        return SchemaInferResult(
            generation=result,
            corpus_specs=corpus_specs,
            corpus_scenarios=corpus_scenarios,
        )

    samples = load_samples_from_db(
        config.db_provider_name,
        db_path=request.db_path,
        max_samples=request.max_samples or request.cluster_sample_limit,
    )
    if not samples:
        corpus_specs, corpus_scenarios = _build_inferred_outputs(
            provider=request.provider,
            package_version=package_version,
            sample_count=result.sample_count,
            catalog=_registry_catalog(registry, request.provider),
        )
        return SchemaInferResult(
            generation=result,
            corpus_specs=corpus_specs,
            corpus_scenarios=corpus_scenarios,
        )

    manifest = registry.cluster_samples(request.provider, samples)
    manifest_path = registry.save_cluster_manifest(manifest)
    corpus_specs, corpus_scenarios = _build_inferred_outputs(
        provider=request.provider,
        package_version=package_version,
        manifest=manifest,
        sample_count=result.sample_count,
        catalog=_registry_catalog(registry, request.provider),
    )
    return SchemaInferResult(
        generation=result,
        manifest=manifest,
        manifest_path=manifest_path,
        corpus_specs=corpus_specs,
        corpus_scenarios=corpus_scenarios,
    )


def list_inferred_corpus_specs(
    *,
    provider: str | None = None,
    registry: SchemaRegistryLike | None = None,
) -> tuple[CorpusSpec, ...]:
    registry = registry or _typed_registry()
    providers = (provider,) if provider is not None else tuple(registry.list_providers())
    specs: list[CorpusSpec] = []
    for provider_name in providers:
        catalog = registry.load_package_catalog(provider_name)
        manifest = registry.load_cluster_manifest(provider_name)
        package_version = "default"
        sample_count = 0
        if catalog is not None:
            package_version = (
                catalog.default_version or catalog.latest_version or catalog.recommended_version or package_version
            )
            package = catalog.package(package_version)
            if package is not None:
                sample_count = package.sample_count
        elif manifest is not None and manifest.default_version:
            package_version = manifest.default_version
        specs.extend(
            build_inferred_corpus_specs(
                provider=provider_name,
                package_version=package_version,
                manifest=manifest,
                sample_count=sample_count,
                catalog=catalog,
            )
        )
    return tuple(specs)


def list_inferred_corpus_scenarios(
    *,
    provider: str | None = None,
    registry: SchemaRegistryLike | None = None,
) -> tuple[CorpusScenario, ...]:
    return build_corpus_scenarios(
        list_inferred_corpus_specs(provider=provider, registry=registry),
        origin="compiled.inferred-corpus-scenario",
        tags=("inferred", "schema", "synthetic", "scenario"),
    )


def list_schemas(request: SchemaListRequest) -> SchemaListResult:
    registry = _typed_registry()
    if request.provider is not None:
        provider = request.provider
        return SchemaListResult(
            provider=provider,
            selected=SchemaProviderSnapshot(
                provider=provider,
                versions=registry.list_versions(provider),
                catalog=registry.load_package_catalog(provider),
                manifest=registry.load_cluster_manifest(provider),
                latest_age_days=registry.get_schema_age_days(provider),
                corpus_specs=list_inferred_corpus_specs(provider=provider, registry=registry),
                corpus_scenarios=list_inferred_corpus_scenarios(provider=provider, registry=registry),
            ),
        )

    all_corpus_specs = list_inferred_corpus_specs(registry=registry)
    corpus_specs_by_provider: dict[str, list[CorpusSpec]] = {}
    for spec in all_corpus_specs:
        corpus_specs_by_provider.setdefault(spec.provider, []).append(spec)
    snapshots = [
        SchemaProviderSnapshot(
            provider=provider,
            versions=registry.list_versions(provider),
            catalog=registry.load_package_catalog(provider),
            manifest=registry.load_cluster_manifest(provider),
            latest_age_days=registry.get_schema_age_days(provider),
            corpus_specs=tuple(corpus_specs_by_provider.get(provider, ())),
            corpus_scenarios=list_inferred_corpus_scenarios(provider=provider, registry=registry),
        )
        for provider in registry.list_providers()
    ]
    return SchemaListResult(provider=None, providers=snapshots)


def compare_schema_versions(request: SchemaCompareRequest) -> SchemaCompareResult:
    registry = _typed_registry()
    return SchemaCompareResult(
        diff=registry.compare_versions(
            request.provider,
            request.from_version,
            request.to_version,
            element_kind=request.element_kind,
        )
    )


def promote_schema_cluster(request: SchemaPromoteRequest) -> SchemaPromoteResult:
    from polylogue.schemas.observation import PROVIDERS, fingerprint_hash
    from polylogue.schemas.sampling import load_samples_from_db
    from polylogue.schemas.shape_fingerprint import _structure_fingerprint

    registry = _typed_registry()
    samples: list[JSONDocument] | None = None
    if request.with_samples:
        provider_token = Provider.from_string(request.provider)
        config = PROVIDERS.get(provider_token)
        if config is None:
            raise ValueError(f"Unknown provider: {request.provider}")
        all_samples = (
            load_samples_from_db(
                config.db_provider_name,
                db_path=request.db_path,
                max_samples=request.max_samples,
            )
            if config.db_provider_name
            else []
        )
        samples = [
            sample for sample in all_samples if fingerprint_hash(_structure_fingerprint(sample)) == request.cluster_id
        ]
        if not samples:
            raise ValueError(f"No samples match cluster {request.cluster_id}")

    new_version = registry.promote_cluster(
        request.provider,
        request.cluster_id,
        samples=samples,
    )
    return SchemaPromoteResult(
        provider=request.provider,
        cluster_id=request.cluster_id,
        package_version=new_version,
        package=registry.get_package(request.provider, version=new_version),
        schema=(
            operator_json_document(schema)
            if (schema := registry.get_element_schema(request.provider, version=new_version)) is not None
            else None
        ),
        versions=registry.list_versions(request.provider),
    )


def audit_schemas(request: SchemaAuditRequest) -> AuditReport:
    from polylogue.schemas.audit.workflow import audit_all_providers, audit_provider

    return audit_provider(request.provider) if request.provider else audit_all_providers()


__all__ = [
    "audit_schemas",
    "compare_schema_versions",
    "infer_schema",
    "list_inferred_corpus_scenarios",
    "list_inferred_corpus_specs",
    "list_schemas",
    "promote_schema_cluster",
]
