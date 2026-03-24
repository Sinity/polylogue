"""Catalog/schema read helpers for runtime schema packages."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from typing import Any

from polylogue.schemas.packages import SchemaPackageCatalog, SchemaVersionPackage
from polylogue.schemas.runtime_registry_paths import SchemaRegistryPathsMixin
from polylogue.schemas.runtime_registry_support import canonical_schema_provider


class SchemaRegistryCatalogMixin(SchemaRegistryPathsMixin):
    """Catalog and element-schema read support."""

    def load_package_catalog(self, provider: str) -> SchemaPackageCatalog | None:
        provider_dir = self._provider_dir_for_catalog(provider)
        if provider_dir is None:
            return None
        return SchemaPackageCatalog.from_dict(json.loads((provider_dir / "catalog.json").read_text(encoding="utf-8")))

    def save_package_catalog(self, catalog: SchemaPackageCatalog):
        provider_dir = self._provider_dir(catalog.provider)
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = self._catalog_path(catalog.provider)
        path.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
        return path

    def _resolve_catalog_version(self, catalog: SchemaPackageCatalog, version: str) -> str | None:
        if version == "default":
            return catalog.default_version or catalog.latest_version or catalog.recommended_version
        if version == "latest":
            return catalog.latest_version or catalog.default_version or catalog.recommended_version
        if version == "recommended":
            return catalog.recommended_version or catalog.default_version or catalog.latest_version
        return version

    def get_package(self, provider: str, version: str = "default") -> SchemaVersionPackage | None:
        provider_token = str(canonical_schema_provider(provider))
        catalog = self.load_package_catalog(provider_token)
        if catalog is None:
            return None
        resolved_version = self._resolve_catalog_version(catalog, version)
        return catalog.package(resolved_version) if resolved_version is not None else None

    def get_element_schema(
        self,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> dict[str, Any] | None:
        provider_token = str(canonical_schema_provider(provider))
        package = self.get_package(provider_token, version=version)
        if package is None:
            return None
        element = package.element(element_kind)
        if element is None or element.schema_file is None:
            return None
        provider_dir = self._provider_dir_for_package(provider_token, package.version)
        if provider_dir is None:
            return None
        path = provider_dir / "versions" / package.version / "elements" / element.schema_file
        if not path.exists():
            return None
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))

    def get_schema(self, provider: str, version: str = "default") -> dict[str, Any] | None:
        return self.get_element_schema(provider, version=version)

    def list_versions(self, provider: str) -> list[str]:
        provider_token = str(canonical_schema_provider(provider))
        catalog = self.load_package_catalog(provider_token)
        if catalog is None:
            return []
        return sorted((package.version for package in catalog.packages), key=lambda value: int(value[1:]))

    def list_providers(self) -> list[str]:
        providers: set[str] = set()
        scanned_roots: set = set()
        from polylogue.schemas.runtime_registry_support import SCHEMA_DIR

        for root in (self.storage_root, SCHEMA_DIR):
            if root in scanned_roots or not root.exists():
                continue
            scanned_roots.add(root)
            for path in root.iterdir():
                if path.is_dir() and (path / "catalog.json").exists():
                    providers.add(path.name)
        return sorted(providers)

    def get_schema_age_days(self, provider: str) -> int | None:
        package = self.get_package(provider, version="latest")
        if package is None:
            return None
        try:
            delta = datetime.now(tz=timezone.utc) - datetime.fromisoformat(package.last_seen)
        except (ValueError, TypeError):
            return None
        return delta.days


__all__ = ["SchemaRegistryCatalogMixin"]
