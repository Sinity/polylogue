"""Workflow orchestration for schema generation and package emission."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDocument
from polylogue.paths import cache_home
from polylogue.paths import db_path as index_db_path
from polylogue.schemas.generation.archive_workload_profile import (
    build_archive_workload_profile,
    write_archive_workload_profile,
)
from polylogue.schemas.generation.cluster_support import _artifact_priority
from polylogue.schemas.generation.dynamic_keys import canonicalize_structure_schema, retain_exact_structure_witnesses
from polylogue.schemas.generation.evidence import SchemaEvidence, merge_evidence
from polylogue.schemas.generation.models import GenerationProgressCallback, GenerationResult, _ProviderBundle
from polylogue.schemas.generation.provider_bundle import _build_provider_bundle
from polylogue.schemas.generation.provider_bundle_packages import allocate_package_versions
from polylogue.schemas.generation.schema_builder import emit_schema_from_evidence, generate_schema_from_samples
from polylogue.schemas.observation import PROVIDERS, resolve_provider_config
from polylogue.schemas.packages import SchemaElementManifest, SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.privacy_config import SchemaPrivacyConfig
from polylogue.schemas.registry import ClusterManifest, SchemaRegistry
from polylogue.schemas.runtime_registry import ElementSchemaMap, canonical_schema_provider
from polylogue.schemas.source_inference import SchemaSourceInput, infer_sources


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
        cluster_manifest=bundle.manifest.to_dict(),
        redact_observed_numeric_values=True,
    )

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
    progress_callback: GenerationProgressCallback | None = None,
) -> GenerationResult:
    """Generate the default inferred schema for a provider."""
    return _build_provider_bundle(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        privacy_config=privacy_config,
        full_corpus=full_corpus,
        progress_callback=progress_callback,
    ).result


def generate_provider_schema_from_sources(
    provider: str,
    *,
    source_inputs: tuple[SchemaSourceInput, ...],
    cache_path: Path | None,
    max_workers: int,
    privacy_config: SchemaPrivacyConfig | None,
    progress_callback: GenerationProgressCallback | None = None,
) -> GenerationResult:
    """Preview the same package bundle used by source commit."""
    return build_provider_bundle_from_sources(
        provider,
        source_inputs=source_inputs,
        cache_path=cache_path,
        max_workers=max_workers,
        privacy_config=privacy_config,
        prior_catalog=None,
        progress_callback=progress_callback,
    ).result


def build_provider_bundle_from_sources(
    provider: str,
    *,
    source_inputs: tuple[SchemaSourceInput, ...],
    cache_path: Path | None,
    max_workers: int,
    privacy_config: SchemaPrivacyConfig | None,
    prior_catalog: SchemaPackageCatalog | None,
    progress_callback: GenerationProgressCallback | None = None,
) -> _ProviderBundle:
    """Build a multi-element package with identity independent of statistics."""
    provider_token = str(canonical_schema_provider(provider))
    inputs = tuple(item for item in source_inputs if str(canonical_schema_provider(item.provider)) == provider_token)
    if not inputs:
        return _ProviderBundle(
            GenerationResult(provider=provider_token, schema=None, sample_count=0, error="No declared source inputs")
        )
    provider = provider_token
    if progress_callback is not None:
        progress_callback("source_inventory", {"state": "started"})
    source = infer_sources(
        inputs,
        cache_path=cache_path or cache_home() / "schema-source-evidence.sqlite3",
        max_workers=max_workers,
        progress=progress_callback,
    )
    if progress_callback is not None:
        progress_callback("source_evidence", {"state": "completed", **source.provenance()})
    evidence_by_kind = {
        kind: merge_evidence(SchemaEvidence.from_json(row) for row in rows)
        for kind, rows in source.evidence_by_element.items()
    }
    if not evidence_by_kind:
        return _ProviderBundle(
            GenerationResult(
                provider=provider,
                schema=None,
                sample_count=0,
                error="No source evidence",
                phase_receipt={"source": source.provenance()},
            )
        )
    config = resolve_provider_config(provider)
    emitted: dict[str, JSONDocument] = {}
    reports = {}
    for kind, evidence in evidence_by_kind.items():
        element_config = (
            replace(config, sample_granularity="record", record_type_key="type")
            if kind == "session_record_stream"
            else config
        )
        emitted[kind], reports[kind] = emit_schema_from_evidence(
            provider, element_config, evidence, privacy_config=privacy_config, artifact_kind=kind
        )
        emitted[kind]["x-polylogue-sample-granularity"] = element_config.sample_granularity
        emitted[kind]["x-polylogue-exact-structure-ids"] = list(evidence.shape_hashes)
    preferred_anchor = "session_record_stream" if config.sample_granularity == "record" else "session_document"
    if provider == "claude-code" and "coordinator_session_stream" in emitted:
        preferred_anchor = "coordinator_session_stream"
    anchor = (
        preferred_anchor
        if preferred_anchor in emitted
        else max(emitted, key=lambda kind: (_artifact_priority(kind), kind))
    )
    anchor_structure = evidence_by_kind[anchor].structure
    canonical_family = hash_payload({"anchor": anchor, "structure": canonicalize_structure_schema(anchor_structure)})
    legacy_family = hash_payload({"anchor": anchor, "structure": anchor_structure})
    version = allocate_package_versions(
        prior_catalog,
        [(anchor, canonical_family)],
        legacy_family_ids={(anchor, canonical_family): legacy_family},
    )[0]
    now = datetime.now(tz=timezone.utc).isoformat()
    prior = next((item for item in prior_catalog.packages if item.version == version), None) if prior_catalog else None
    family = prior.anchor_profile_family_id if prior is not None else canonical_family
    first_seen = prior.first_seen if prior is not None else now
    retained_witnesses = {}
    for kind, evidence in evidence_by_kind.items():
        prior_element = prior.element(kind) if prior is not None else None
        retained = retain_exact_structure_witnesses(
            prior_element.exact_structure_ids if prior_element is not None else (), evidence.shape_hashes
        )
        retained_witnesses[kind] = retained
        emitted[kind]["x-polylogue-exact-structure-ids"] = list(retained.exact_structure_ids)
        emitted[kind]["x-polylogue-publication-omitted-structure-witness-count"] = (
            retained.omitted_current_witness_count
        )
        emitted[kind]["x-polylogue-source-evidence-unretained-shape-observation-lower-bound"] = (
            evidence.unretained_shape_observation_lower_bound
        )
    counts = {kind: evidence.current_record_count for kind, evidence in evidence_by_kind.items()}
    elements = [
        SchemaElementManifest(
            element_kind=kind,
            schema_file=f"{kind}.schema.json.gz",
            sample_count=evidence.current_record_count,
            artifact_count=evidence.current_source_count,
            bundle_scope_count=evidence.current_source_count,
            observed_artifact_count=evidence.current_source_count,
            first_seen=first_seen,
            last_seen=now,
            exact_structure_ids=list(retained_witnesses[kind].exact_structure_ids),
            publication_omitted_structure_witness_count=retained_witnesses[kind].omitted_current_witness_count,
            source_evidence_unretained_shape_observation_lower_bound=(
                evidence.unretained_shape_observation_lower_bound
            ),
        )
        for kind, evidence in sorted(evidence_by_kind.items())
    ]
    package = SchemaVersionPackage(
        provider=provider,
        version=version,
        anchor_kind=anchor,
        default_element_kind=anchor,
        first_seen=first_seen,
        last_seen=now,
        bundle_scope_count=evidence_by_kind[anchor].current_source_count,
        sample_count=sum(counts.values()),
        anchor_profile_family_id=family,
        canonical_anchor_profile_family_id=canonical_family if family != canonical_family else None,
        elements=elements,
    )
    catalog = SchemaPackageCatalog(
        provider=provider,
        packages=[package],
        latest_version=version,
        default_version=version,
        recommended_version=version,
        observation_outcomes=source.provenance(),
    )
    manifest = ClusterManifest(provider=provider, artifact_counts=counts, default_version=version)
    result = GenerationResult(
        provider=provider,
        schema=emitted[anchor],
        sample_count=sum(counts.values()),
        redaction_report=reports[anchor],
        versions=[version],
        default_version=version,
        package_count=1,
        artifact_counts=counts,
        phase_receipt={"source": source.provenance()},
    )
    return _ProviderBundle(result, catalog=catalog, package_schemas={version: emitted}, manifest=manifest)


def generate_all_schemas(
    output_dir: Path,
    db_path: Path | None = None,
    providers: list[str] | None = None,
    max_samples: int | None = None,
    privacy_config: SchemaPrivacyConfig | None = None,
    include_archive_workload_profile: bool = False,
    full_corpus: bool = False,
) -> list[GenerationResult]:
    """Generate versioned schemas for all providers."""
    if db_path is None:
        db_path = index_db_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_list = providers or list(PROVIDERS.keys())
    results = []
    package_bundle_scope_counts: dict[str, dict[str, int]] = {}
    for provider in provider_list:
        bundle = _build_provider_bundle(
            provider,
            db_path=db_path,
            max_samples=max_samples,
            privacy_config=privacy_config,
            full_corpus=full_corpus,
        )
        results.append(bundle.result)
        persist_generated_provider_bundle(output_dir, provider, bundle)
        if bundle.catalog is not None:
            package_bundle_scope_counts[provider] = {
                package.version: package.bundle_scope_count for package in bundle.catalog.packages
            }

    if include_archive_workload_profile:
        archive_profile = build_archive_workload_profile(
            db_path,
            package_bundle_scope_counts=package_bundle_scope_counts,
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
