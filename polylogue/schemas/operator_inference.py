"""Inference/listing/promotion helpers for schema operator workflows."""

from __future__ import annotations

from typing import Any

from polylogue.scenarios import CorpusScenario, CorpusSpec, build_corpus_scenarios, build_inferred_corpus_specs
from polylogue.schemas.audit_models import AuditReport
from polylogue.schemas.operator_models import (
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
)
from polylogue.schemas.operator_registry import schema_registry


def _registry_catalog(registry, provider: str):
    load_catalog = getattr(registry, "load_package_catalog", None)
    if load_catalog is None:
        return None
    return load_catalog(provider)


def _build_inferred_outputs(
    *,
    provider: str,
    package_version: str,
    manifest=None,
    sample_count: int = 0,
    catalog=None,
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


def infer_schema(request: SchemaInferRequest) -> SchemaInferResult:
    from polylogue.schemas.generation_workflow import generate_provider_schema
    from polylogue.schemas.observation import PROVIDERS
    from polylogue.schemas.sampling import load_samples_from_db

    result = generate_provider_schema(
        request.provider,
        db_path=request.db_path,
        max_samples=request.max_samples,
        privacy_config=request.privacy_config,
        full_corpus=request.full_corpus,
    )
    package_version = result.default_version or "default"
    registry = schema_registry()
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

    config = PROVIDERS.get(request.provider)
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
    registry: Any | None = None,
) -> tuple[CorpusSpec, ...]:
    registry = registry or schema_registry()
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
    registry: Any | None = None,
) -> tuple[CorpusScenario, ...]:
    return build_corpus_scenarios(
        list_inferred_corpus_specs(provider=provider, registry=registry),
        origin="compiled.inferred-corpus-scenario",
        tags=("inferred", "schema", "synthetic", "scenario"),
    )


def list_schemas(request: SchemaListRequest) -> SchemaListResult:
    registry = schema_registry()
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
    registry = schema_registry()
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

    registry = schema_registry()
    samples: list[dict[str, Any]] | None = None
    if request.with_samples:
        config = PROVIDERS.get(request.provider)
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
        schema=registry.get_element_schema(request.provider, version=new_version),
        versions=registry.list_versions(request.provider),
    )


def audit_schemas(request: SchemaAuditRequest) -> AuditReport:
    from polylogue.schemas.audit_workflow import audit_all_providers, audit_provider

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
