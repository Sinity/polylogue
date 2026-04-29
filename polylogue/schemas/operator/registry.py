"""Registry access helpers for schema operator workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from polylogue.lib.json import JSONDocument
from polylogue.schemas.packages import SchemaPackageCatalog, SchemaResolution, SchemaVersionPackage
from polylogue.schemas.tooling_models import ClusterManifest, SchemaDiff


class RuntimeSchemaRegistryLike(Protocol):
    def load_package_catalog(self, provider: str) -> SchemaPackageCatalog | None: ...

    def get_package(self, provider: str, version: str = "default") -> SchemaVersionPackage | None: ...

    def get_element_schema(
        self,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> JSONDocument | None: ...

    def list_versions(self, provider: str) -> list[str]: ...

    def list_providers(self) -> list[str]: ...

    def get_schema_age_days(self, provider: str) -> int | None: ...

    def resolve_payload(
        self,
        provider: str,
        payload: Mapping[str, object],
        *,
        source_path: str | None = None,
    ) -> SchemaResolution | None: ...


class SchemaRegistryLike(RuntimeSchemaRegistryLike, Protocol):
    def compare_versions(
        self,
        provider: str,
        v1: str,
        v2: str,
        *,
        element_kind: str | None = None,
    ) -> SchemaDiff: ...

    def cluster_samples(self, provider: str, samples: Sequence[Mapping[str, object]]) -> ClusterManifest: ...

    def save_cluster_manifest(self, manifest: ClusterManifest) -> Path: ...

    def load_cluster_manifest(self, provider: str) -> ClusterManifest | None: ...

    def promote_cluster(
        self,
        provider: str,
        cluster_id: str,
        *,
        samples: Sequence[Mapping[str, object]] | None = None,
    ) -> str: ...


def schema_registry() -> SchemaRegistryLike:
    from polylogue.schemas.registry import SchemaRegistry

    return SchemaRegistry()


def runtime_schema_registry() -> RuntimeSchemaRegistryLike:
    from polylogue.schemas.runtime_registry import SchemaRegistry

    return SchemaRegistry()


__all__ = [
    "RuntimeSchemaRegistryLike",
    "SchemaRegistryLike",
    "runtime_schema_registry",
    "schema_registry",
]
