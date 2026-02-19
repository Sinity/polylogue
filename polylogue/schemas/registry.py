"""Versioned schema registry for provider export formats.

The registry manages two tiers of schemas:
  - **Baseline schemas**: Shipped in-package under ``providers/*.schema.json.gz``
  - **Versioned schemas**: Generated at runtime, stored under
    ``DATA_HOME/schemas/{provider}/v{N}.schema.json.gz``

Usage::

    registry = SchemaRegistry()
    schema = registry.get_schema("chatgpt")             # latest
    schema = registry.get_schema("chatgpt", version="v1")  # specific version
    diff = registry.compare_versions("chatgpt", "v1", "v2")
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.paths import data_home

# In-package baseline schemas (canonical definition — imported by validator, synthetic)
SCHEMA_DIR = Path(__file__).parent / "providers"


@dataclass
class SchemaDiff:
    """Difference between two schema versions."""

    provider: str
    version_a: str
    version_b: str
    added_properties: list[str] = field(default_factory=list)
    removed_properties: list[str] = field(default_factory=list)
    changed_properties: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added_properties or self.removed_properties or self.changed_properties)

    def summary(self) -> str:
        parts = []
        if self.added_properties:
            parts.append(f"+{len(self.added_properties)} properties")
        if self.removed_properties:
            parts.append(f"-{len(self.removed_properties)} properties")
        if self.changed_properties:
            parts.append(f"~{len(self.changed_properties)} changed")
        return ", ".join(parts) if parts else "no changes"


class SchemaRegistry:
    """Registry of versioned provider schemas."""

    def __init__(self, storage_root: Path | None = None):
        """Initialize the registry.

        Args:
            storage_root: Root directory for versioned schemas.
                          Defaults to ``data_home() / "schemas"``.
        """
        self._storage_root = storage_root

    @property
    def storage_root(self) -> Path:
        if self._storage_root is not None:
            return self._storage_root
        return data_home() / "schemas"

    # --- Read operations ---

    def get_schema(self, provider: str, version: str = "latest") -> dict[str, Any] | None:
        """Get a schema by provider and version.

        Args:
            provider: Provider name (e.g., "chatgpt", "claude-ai")
            version: Version string ("v1", "v2", ...) or "latest"

        Returns:
            Schema dict, or None if not found.
        """
        if version == "latest":
            versions = self.list_versions(provider)
            if versions:
                version = versions[-1]  # sorted, last is latest
            else:
                # Fall back to baseline
                return self._load_baseline(provider)

        # Try versioned storage first
        schema = self._load_versioned(provider, version)
        if schema is not None:
            return schema

        # Fall back to baseline for v1 (or if only baseline exists)
        if version == "v1":
            return self._load_baseline(provider)

        return None

    def list_versions(self, provider: str) -> list[str]:
        """List all versions for a provider, sorted by version number.

        Returns:
            Sorted list of version strings like ["v1", "v2", "v3"].
        """
        provider_dir = self.storage_root / provider
        if not provider_dir.exists():
            # Check if baseline exists
            if self._baseline_path(provider).exists():
                return ["v1"]
            return []

        versions: list[str] = []
        for p in provider_dir.glob("v*.schema.json.gz"):
            v = p.name.split(".")[0]  # "v2.schema.json.gz" -> "v2"
            versions.append(v)

        # Always include v1 if baseline exists
        if "v1" not in versions and self._baseline_path(provider).exists():
            versions.append("v1")

        return sorted(versions, key=lambda v: int(v[1:]))

    def list_providers(self) -> list[str]:
        """List all providers with at least one schema (baseline or versioned)."""
        providers: set[str] = set()

        # Baseline providers
        for p in SCHEMA_DIR.glob("*.schema.json.gz"):
            providers.add(p.name.replace(".schema.json.gz", ""))

        # Versioned providers
        if self.storage_root.exists():
            for d in self.storage_root.iterdir():
                if d.is_dir() and any(d.glob("v*.schema.json.gz")):
                    providers.add(d.name)

        return sorted(providers)

    # --- Write operations ---

    def register_schema(self, provider: str, schema: dict[str, Any]) -> str:
        """Register a new schema version for a provider.

        Auto-increments the version number based on existing versions.
        Adds/updates ``x-polylogue-version`` and ``$id`` metadata.

        Args:
            provider: Provider name
            schema: Schema dict to register

        Returns:
            Version string assigned (e.g., "v3")
        """
        versions = self.list_versions(provider)
        if versions:
            last_num = int(versions[-1][1:])
            new_version = f"v{last_num + 1}"
        else:
            new_version = "v1"

        # Inject metadata
        schema["$id"] = f"polylogue://schemas/{provider}/{new_version}"
        schema["x-polylogue-version"] = int(new_version[1:])
        schema["x-polylogue-registered-at"] = datetime.now(tz=timezone.utc).isoformat()

        # Write to storage
        provider_dir = self.storage_root / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = provider_dir / f"{new_version}.schema.json.gz"
        path.write_bytes(gzip.compress(json.dumps(schema, indent=2).encode("utf-8")))

        return new_version

    # --- Comparison ---

    def compare_versions(self, provider: str, v1: str, v2: str) -> SchemaDiff:
        """Compare two schema versions for a provider.

        Args:
            provider: Provider name
            v1: First version (e.g., "v1")
            v2: Second version (e.g., "v2")

        Returns:
            SchemaDiff with added/removed/changed properties.

        Raises:
            ValueError: If either version doesn't exist.
        """
        schema_a = self.get_schema(provider, version=v1)
        schema_b = self.get_schema(provider, version=v2)

        if schema_a is None:
            raise ValueError(f"Schema not found: {provider} {v1}")
        if schema_b is None:
            raise ValueError(f"Schema not found: {provider} {v2}")

        return self._diff_schemas(provider, v1, v2, schema_a, schema_b)

    # --- Schema metadata ---

    def get_schema_age_days(self, provider: str) -> int | None:
        """Get the age in days of the latest schema for a provider.

        Returns:
            Age in days, or None if no schema or no timestamp metadata.
        """
        schema = self.get_schema(provider, version="latest")
        if schema is None:
            return None

        generated_at = schema.get("x-polylogue-generated-at")
        if not generated_at:
            return None

        try:
            ts = datetime.fromisoformat(generated_at)
            delta = datetime.now(tz=timezone.utc) - ts
            return delta.days
        except (ValueError, TypeError):
            return None

    # --- Internal helpers ---

    def _baseline_path(self, provider: str) -> Path:
        return SCHEMA_DIR / f"{provider}.schema.json.gz"

    def _load_baseline(self, provider: str) -> dict[str, Any] | None:
        path = self._baseline_path(provider)
        if not path.exists():
            return None
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))

    def _load_versioned(self, provider: str, version: str) -> dict[str, Any] | None:
        path = self.storage_root / provider / f"{version}.schema.json.gz"
        if not path.exists():
            return None
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))

    def _diff_schemas(
        self,
        provider: str,
        v1: str,
        v2: str,
        schema_a: dict[str, Any],
        schema_b: dict[str, Any],
    ) -> SchemaDiff:
        """Compare top-level and nested properties between two schemas."""
        props_a = set(schema_a.get("properties", {}).keys())
        props_b = set(schema_b.get("properties", {}).keys())

        added = sorted(props_b - props_a)
        removed = sorted(props_a - props_b)

        # Check for type changes in common properties
        changed = []
        common = props_a & props_b
        for prop in sorted(common):
            type_a = schema_a["properties"][prop].get("type")
            type_b = schema_b["properties"][prop].get("type")
            if type_a != type_b:
                changed.append(prop)

        return SchemaDiff(
            provider=provider,
            version_a=v1,
            version_b=v2,
            added_properties=added,
            removed_properties=removed,
            changed_properties=changed,
        )
