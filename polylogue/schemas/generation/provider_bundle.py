"""Provider-bundle assembly for schema generation."""

from __future__ import annotations

import time
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.paths import db_path as index_db_path
from polylogue.schemas.generation.cluster_collection import (
    _collect_cluster_accumulators,
)
from polylogue.schemas.generation.cluster_support import (
    _cluster_profile_tokens,
    _cluster_sort_key,
)
from polylogue.schemas.generation.models import GenerationProgressCallback, GenerationResult, _ProviderBundle
from polylogue.schemas.generation.observation_journal import ObservationJournal, recover_stale_journals
from polylogue.schemas.generation.packages import (
    _build_package_candidates,
)
from polylogue.schemas.generation.provider_bundle_packages import (
    build_provider_catalog_artifacts,
    build_success_provider_bundle,
)
from polylogue.schemas.generation.support import GENSON_AVAILABLE
from polylogue.schemas.observation import PROVIDERS, ProviderConfig, resolve_provider_config
from polylogue.schemas.privacy_config import SchemaPrivacyConfig
from polylogue.schemas.registry import ClusterManifest, SchemaCluster


def _journal_storage_bytes(journal: ObservationJournal) -> dict[str, int]:
    """Return aggregate private-spool sizes without exposing observation data."""
    path = journal.path
    wal_path = path.with_name(f"{path.name}-wal")
    shm_path = path.with_name(f"{path.name}-shm")
    return {
        "db_bytes": path.stat().st_size if path.exists() else 0,
        "wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
        "shm_bytes": shm_path.stat().st_size if shm_path.exists() else 0,
    }


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
    progress_callback: GenerationProgressCallback | None = None,
) -> _ProviderBundle:
    """Generate all inferred schema versions plus the default result for a provider."""
    provider_token = Provider.from_string(provider)
    if provider_token not in PROVIDERS:
        return build_provider_error_bundle(
            str(provider_token),
            error=f"Unknown provider: {provider}. Known: {[str(item) for item in PROVIDERS]}",
        )
    if db_path is None:
        db_path = index_db_path()
    if not GENSON_AVAILABLE:
        return build_provider_error_bundle(
            str(provider_token),
            error="genson not installed",
        )

    config: ProviderConfig = resolve_provider_config(provider_token)

    phase_receipt: JSONDocument = {"version": 1, "status": "running", "phases": []}

    def report(phase: str, state: str, **details: JSONValue) -> None:
        payload = json_document({"phase": phase, "state": state, **details})
        if progress_callback is not None:
            progress_callback(phase, payload)

    def complete_phase(phase: str, started_ns: int, journal: ObservationJournal, **details: JSONValue) -> None:
        phase_payload = json_document(
            {
                "name": phase,
                "elapsed_ms": round((time.monotonic_ns() - started_ns) / 1_000_000, 3),
                "journal": _journal_storage_bytes(journal),
                **details,
            }
        )
        phases = phase_receipt["phases"]
        assert isinstance(phases, list)
        phases.append(phase_payload)
        report(phase, "completed", **phase_payload)

    try:
        recover_stale_journals(minimum_age_s=0)
        with ObservationJournal.create(forbidden_roots=(db_path.parent,)) as journal:
            observe_started_ns = time.monotonic_ns()
            report("observe_and_cluster", "started")

            def observe_progress(details: JSONDocument) -> None:
                elapsed_ms = round((time.monotonic_ns() - observe_started_ns) / 1_000_000, 3)
                unit_count = details.get("unit_count")
                units_per_s = (
                    round(int(unit_count) / max(elapsed_ms / 1_000, 0.001), 3) if isinstance(unit_count, int) else None
                )
                report(
                    "observe_and_cluster",
                    "progress",
                    elapsed_ms=elapsed_ms,
                    units_per_s=units_per_s,
                    estimated_total_units=None,
                    estimate_status="source_total_unavailable",
                    **details,
                )

            clusters, memberships, sample_count, artifact_counts = _collect_cluster_accumulators(
                provider,
                db_path=db_path,
                max_samples=max_samples,
                full_corpus=full_corpus,
                journal=journal,
                progress_callback=observe_progress,
            )
            complete_phase(
                "observe_and_cluster",
                observe_started_ns,
                journal,
                cluster_count=len(clusters),
                unit_count=journal.unit_count,
                sample_count=sample_count,
            )
            if not clusters:
                bundle = build_provider_error_bundle(
                    str(provider_token),
                    error="No samples found",
                )
                bundle.result.phase_receipt = {**phase_receipt, "status": "empty"}
                return bundle
            package_started_ns = time.monotonic_ns()
            report("assemble_packages", "started")
            packages, orphan_adjunct_counts = _build_package_candidates(
                str(provider_token),
                memberships=memberships,
                clusters=clusters,
                journal=journal,
            )
            complete_phase(
                "assemble_packages",
                package_started_ns,
                journal,
                package_count=len(packages),
            )
            if not packages:
                bundle = build_provider_error_bundle(
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
                                exact_structure_ids=list(
                                    journal.memberships(
                                        profile_family_id=cluster_id,
                                        include_samples=False,
                                    ).iter_distinct_values("exact_structure_id")
                                ),
                                bundle_scope_count=journal.memberships(
                                    profile_family_id=cluster_id,
                                    include_samples=False,
                                ).distinct_count("bundle_scope"),
                            )
                            for cluster_id, acc in sorted(clusters.items(), key=_cluster_sort_key, reverse=True)
                        ],
                        artifact_counts=artifact_counts,
                    ),
                )
                bundle.result.phase_receipt = {**phase_receipt, "status": "empty"}
                return bundle
            observation_outcomes = journal.terminal_summary()
            if observation_outcomes.get("total") == 0:
                observation_outcomes = {}
            catalog_started_ns = time.monotonic_ns()
            report("build_catalog", "started")
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
                observation_outcomes=observation_outcomes,
                journal=journal,
            )
            complete_phase(
                "build_catalog",
                catalog_started_ns,
                journal,
                package_count=len(catalog_artifacts.catalog.packages),
            )
            bundle = build_success_provider_bundle(
                provider_token=provider_token,
                sample_count=sample_count,
                clusters=clusters,
                artifact_counts=artifact_counts,
                catalog_artifacts=catalog_artifacts,
            )
            bundle.result.phase_receipt = {**phase_receipt, "status": "succeeded"}
            return bundle
    except Exception as e:
        bundle = build_provider_error_bundle(
            str(provider_token),
            error=str(e),
        )
        bundle.result.phase_receipt = {**phase_receipt, "status": "failed", "error": str(e)}
        return bundle


__all__ = ["_build_provider_bundle"]
