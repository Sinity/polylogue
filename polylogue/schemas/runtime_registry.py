"""Runtime schema authority for package catalogs, schemas, and payload resolution."""

from __future__ import annotations

from pathlib import Path

from polylogue.schemas.runtime_registry_resolution import SchemaRegistryResolutionMixin
from polylogue.schemas.runtime_registry_support import SCHEMA_DIR, SchemaProvider, canonical_schema_provider


class SchemaRegistry(SchemaRegistryResolutionMixin):
    """Runtime package/catalog authority for schema resolution."""

    def __init__(self, storage_root: Path | None = None):
        self._storage_root = storage_root


__all__ = ["SCHEMA_DIR", "SchemaProvider", "SchemaRegistry", "canonical_schema_provider"]
