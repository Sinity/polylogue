"""Filesystem/path helpers for runtime schema packages."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import data_home
from polylogue.schemas.runtime_registry_support import SCHEMA_DIR, canonical_schema_provider


class SchemaRegistryPathsMixin:
    """Path resolution helpers for runtime schema storage."""

    _storage_root: Path | None

    @property
    def storage_root(self) -> Path:
        return self._storage_root if self._storage_root is not None else data_home() / "schemas"

    def _provider_dir(self, provider: str) -> Path:
        return self.storage_root / str(canonical_schema_provider(provider))

    def _bundled_provider_dir(self, provider: str) -> Path:
        return SCHEMA_DIR / str(canonical_schema_provider(provider))

    def _provider_search_roots(self, provider: str) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()
        for candidate in (self._provider_dir(provider), self._bundled_provider_dir(provider)):
            if candidate in seen:
                continue
            seen.add(candidate)
            roots.append(candidate)
        return roots

    def _catalog_path(self, provider: str) -> Path:
        return self._provider_dir(provider) / "catalog.json"

    def _package_dir(self, provider: str, version: str) -> Path:
        return self._provider_dir(provider) / "versions" / version

    def _package_manifest_path(self, provider: str, version: str) -> Path:
        return self._package_dir(provider, version) / "package.json"

    def _catalog_path_in(self, provider_dir: Path) -> Path:
        return provider_dir / "catalog.json"

    def _provider_dir_for_catalog(self, provider: str) -> Path | None:
        for provider_dir in self._provider_search_roots(provider):
            if self._catalog_path_in(provider_dir).exists():
                return provider_dir
        return None

    def _provider_dir_for_package(self, provider: str, version: str) -> Path | None:
        for provider_dir in self._provider_search_roots(provider):
            manifest_path = provider_dir / "versions" / version / "package.json"
            if manifest_path.exists():
                return provider_dir
        return None


__all__ = ["SchemaRegistryPathsMixin"]
