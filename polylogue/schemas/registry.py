"""Public schema registry facade combining runtime authority with tooling helpers."""

from __future__ import annotations

from polylogue.schemas.runtime_registry import SCHEMA_DIR, canonical_schema_provider
from polylogue.schemas.runtime_registry import SchemaRegistry as RuntimeSchemaRegistry
from polylogue.schemas.sampling import fingerprint_hash as _fingerprint_hash
from polylogue.schemas.tooling_registry import (
    ClusterManifest,
    PropertyChange,
    SchemaCluster,
    SchemaDiff,
    SchemaRegistryToolingMixin,
)


class SchemaRegistry(SchemaRegistryToolingMixin, RuntimeSchemaRegistry):
    """Full schema registry surface for tooling-heavy callers."""


__all__ = [
    "SCHEMA_DIR",
    "ClusterManifest",
    "PropertyChange",
    "SchemaCluster",
    "SchemaDiff",
    "SchemaRegistry",
    "_fingerprint_hash",
    "canonical_schema_provider",
]
