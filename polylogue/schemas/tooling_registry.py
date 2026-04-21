"""Schema tooling layer: evidence manifests, diffs, clustering, and promotion."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from polylogue.lib.json import json_document, json_document_list, require_json_value
from polylogue.schemas.observation import schema_cluster_id
from polylogue.schemas.observation_models import SchemaClusterPayload
from polylogue.schemas.packages import SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.runtime_registry import (
    ElementSchemaMap,
    PublicSchemaDocument,
    SchemaInputDocument,
    canonical_schema_provider,
)
from polylogue.schemas.tooling_diff import diff_schemas
from polylogue.schemas.tooling_models import ClusterManifest, PropertyChange, SchemaCluster, SchemaDiff
from polylogue.types import Provider

SchemaPayload: TypeAlias = SchemaInputDocument
ObservedSchemaSample: TypeAlias = object


def _dominant_keys(sample: object) -> list[str]:
    document = json_document(sample)
    if document:
        return sorted(document)
    documents = json_document_list(sample)
    if documents:
        return sorted(documents[0])
    return []


def _cluster_payload(sample: object) -> SchemaClusterPayload:
    document = json_document(sample)
    if document:
        return document
    documents = json_document_list(sample)
    if documents:
        return require_json_value(documents, context="schema cluster payload")
    if sample is None or isinstance(sample, (str, int, float, bool)):
        return sample
    return None


class SchemaRegistryToolingMixin:
    """Tooling mixin layered on top of the runtime registry base."""

    storage_root: Path

    if TYPE_CHECKING:

        def _single_element_package(
            self,
            provider: str,
            *,
            version: str,
            schema: SchemaInputDocument,
            element_kind: str = "conversation_document",
            first_seen: str | None = None,
            last_seen: str | None = None,
        ) -> tuple[SchemaVersionPackage, ElementSchemaMap]: ...

        def replace_provider_packages(
            self,
            provider: str,
            catalog: SchemaPackageCatalog,
            package_schemas: Mapping[str, ElementSchemaMap],
        ) -> None: ...

        def _catalog_path(self, provider: str) -> Path: ...

        def get_element_schema(
            self,
            provider: str,
            *,
            version: str = "default",
            element_kind: str | None = None,
        ) -> PublicSchemaDocument | None: ...

        def register_schema(self, provider: str, schema: SchemaInputDocument) -> str: ...

    def replace_provider_schemas(
        self,
        provider: str | Provider,
        versioned_schemas: Sequence[tuple[str, SchemaPayload]],
        *,
        manifest: ClusterManifest | None = None,
    ) -> Path:
        provider_token = canonical_schema_provider(provider)
        packages: list[SchemaVersionPackage] = []
        package_schemas: dict[str, ElementSchemaMap] = {}
        for version, schema in versioned_schemas:
            package, schemas = self._single_element_package(provider_token, version=version, schema=schema)
            packages.append(package)
            package_schemas[version] = schemas
        latest_version = packages[-1].version if packages else None
        recommended_version = manifest.default_version if manifest is not None else latest_version
        catalog = SchemaPackageCatalog(
            provider=provider_token,
            packages=sorted(packages, key=lambda item: int(item.version[1:])),
            latest_version=latest_version,
            default_version=recommended_version,
            recommended_version=recommended_version,
        )
        self.replace_provider_packages(provider_token, catalog, package_schemas)
        if manifest is not None:
            self.save_cluster_manifest(manifest)
        return self._catalog_path(provider_token)

    def compare_versions(
        self,
        provider: str | Provider,
        v1: str,
        v2: str,
        *,
        element_kind: str | None = None,
    ) -> SchemaDiff:
        provider_token = canonical_schema_provider(provider)
        schema_a = self.get_element_schema(provider_token, version=v1, element_kind=element_kind)
        schema_b = self.get_element_schema(provider_token, version=v2, element_kind=element_kind)
        if schema_a is None or schema_b is None:
            raise ValueError(
                f"Schema not found for {provider_token}: {v1}, {v2}"
                + (f", element={element_kind}" if element_kind else "")
            )
        return diff_schemas(provider_token, v1, v2, schema_a, schema_b)

    def cluster_samples(
        self,
        provider: str | Provider,
        samples: Sequence[ObservedSchemaSample],
        *,
        source_paths: list[str] | None = None,
        artifact_kinds: list[str] | None = None,
    ) -> ClusterManifest:
        provider_token = canonical_schema_provider(provider)
        now = datetime.now(tz=timezone.utc).isoformat()
        groups: dict[str, list[int]] = {}
        artifact_by_cluster: dict[str, str] = {}
        for index, sample in enumerate(samples):
            artifact_kind = (
                artifact_kinds[index] if artifact_kinds is not None and index < len(artifact_kinds) else "unspecified"
            )
            cluster_id = schema_cluster_id(_cluster_payload(sample), artifact_kind)
            groups.setdefault(cluster_id, []).append(index)
            artifact_by_cluster[cluster_id] = artifact_kind

        clusters: list[SchemaCluster] = []
        for cluster_id, indices in sorted(groups.items(), key=lambda item: -len(item[1])):
            rep_paths: list[str] = []
            if source_paths:
                seen: set[str] = set()
                for index in indices[:5]:
                    if index < len(source_paths) and source_paths[index] not in seen:
                        rep_paths.append(source_paths[index])
                        seen.add(source_paths[index])
            clusters.append(
                SchemaCluster(
                    cluster_id=cluster_id,
                    provider=provider_token,
                    sample_count=len(indices),
                    first_seen=now,
                    last_seen=now,
                    representative_paths=rep_paths,
                    dominant_keys=_dominant_keys(samples[indices[0]])[:20],
                    confidence=round(min(1.0, len(indices) / max(len(samples) * 0.1, 1)), 3),
                    artifact_kind=artifact_by_cluster.get(cluster_id, "unspecified"),
                )
            )

        artifact_counts: dict[str, int] = {}
        for cluster in clusters:
            artifact_counts[cluster.artifact_kind] = (
                artifact_counts.get(cluster.artifact_kind, 0) + cluster.sample_count
            )
        return ClusterManifest(
            provider=provider_token,
            clusters=clusters,
            generated_at=now,
            artifact_counts=artifact_counts,
        )

    def save_cluster_manifest(self, manifest: ClusterManifest) -> Path:
        provider_dir = self.storage_root / str(canonical_schema_provider(manifest.provider))
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = provider_dir / "manifest.json"
        path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_cluster_manifest(
        self,
        provider: str | Provider,
    ) -> ClusterManifest | None:
        provider_token = canonical_schema_provider(provider)
        path = self.storage_root / str(provider_token) / "manifest.json"
        if not path.exists():
            return None
        return ClusterManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def promote_cluster(
        self,
        provider: str | Provider,
        cluster_id: str,
        *,
        samples: Sequence[SchemaPayload] | None = None,
    ) -> str:
        provider_token = canonical_schema_provider(provider)
        manifest = self.load_cluster_manifest(provider_token)
        if manifest is None:
            raise ValueError(f"No cluster manifest found for provider: {provider_token}")
        target_cluster = next((cluster for cluster in manifest.clusters if cluster.cluster_id == cluster_id), None)
        if target_cluster is None:
            raise ValueError(f"Cluster {cluster_id} not found for {provider_token}")
        if target_cluster.promoted_package_version is not None:
            raise ValueError(f"Cluster {cluster_id} already promoted as {target_cluster.promoted_package_version}")

        if samples:
            from polylogue.schemas.generation_workflow import generate_schema_from_samples

            schema = json_document(generate_schema_from_samples(samples))
        else:
            schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "title": f"{provider_token} export format ({target_cluster.artifact_kind})",
                "properties": {key: {} for key in target_cluster.dominant_keys},
            }

        schema["title"] = schema.get("title") or f"{provider_token} export format ({target_cluster.artifact_kind})"
        schema["x-polylogue-anchor-profile-family-id"] = cluster_id
        schema["x-polylogue-profile-family-ids"] = [cluster_id]
        schema["x-polylogue-package-profile-family-ids"] = [cluster_id]
        schema["x-polylogue-observed-artifact-count"] = target_cluster.sample_count
        schema["x-polylogue-evidence-confidence"] = target_cluster.confidence
        schema["x-polylogue-artifact-kind"] = target_cluster.artifact_kind
        schema["x-polylogue-promoted-at"] = datetime.now(tz=timezone.utc).isoformat()

        new_version = self.register_schema(provider_token, schema)
        target_cluster.promoted_package_version = new_version
        if manifest.default_version is None:
            manifest.default_version = new_version
        self.save_cluster_manifest(manifest)
        return new_version


__all__ = [
    "ClusterManifest",
    "PropertyChange",
    "SchemaCluster",
    "SchemaDiff",
    "SchemaRegistryToolingMixin",
]
