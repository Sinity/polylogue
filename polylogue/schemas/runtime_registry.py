"""Runtime schema authority for package catalogs, schemas, and payload resolution."""

from __future__ import annotations

import copy
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.lib.provider_identity import canonical_schema_provider as _canonical_schema_provider
from polylogue.lib.provider_identity import normalize_provider_token
from polylogue.paths import data_home
from polylogue.schemas.observation import (
    derive_bundle_scope,
    extract_schema_units_from_payload,
    profile_similarity,
    resolve_provider_config,
)
from polylogue.schemas.packages import (
    SchemaElementManifest,
    SchemaPackageCatalog,
    SchemaResolution,
    SchemaVersionPackage,
)
from polylogue.types import Provider

SCHEMA_DIR = Path(__file__).parent / "providers"
SchemaProvider = Provider | str


def canonical_schema_provider(provider: str | Provider) -> SchemaProvider:
    normalized = normalize_provider_token(str(provider))
    if not normalized:
        return Provider.UNKNOWN

    canonical = _canonical_schema_provider(normalized, default="")
    if canonical:
        provider_token = Provider.from_string(canonical)
        if provider_token is not Provider.UNKNOWN:
            return provider_token
    return normalized


class SchemaRegistry:
    """Runtime package/catalog authority for schema resolution."""

    def __init__(self, storage_root: Path | None = None):
        self._storage_root = storage_root
        self._catalog_cache: dict[str, SchemaPackageCatalog | None] = {}
        self._schema_cache: dict[tuple[str, str, str | None], dict[str, Any] | None] = {}

    def clear_cache(self) -> None:
        """Clear internal caches. Call after modifying schema packages."""
        self._catalog_cache.clear()
        self._schema_cache.clear()

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

    def load_package_catalog(self, provider: str) -> SchemaPackageCatalog | None:
        if provider in self._catalog_cache:
            return self._catalog_cache[provider]
        provider_dir = self._provider_dir_for_catalog(provider)
        if provider_dir is None:
            self._catalog_cache[provider] = None
            return None
        catalog = SchemaPackageCatalog.from_dict(
            json.loads((provider_dir / "catalog.json").read_text(encoding="utf-8"))
        )
        self._catalog_cache[provider] = catalog
        return catalog

    def save_package_catalog(self, catalog: SchemaPackageCatalog):
        provider_dir = self._provider_dir(catalog.provider)
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = self._catalog_path(catalog.provider)
        path.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
        self.clear_cache()
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
        cache_key = (provider, version, element_kind)
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]
        provider_token = str(canonical_schema_provider(provider))
        package = self.get_package(provider_token, version=version)
        if package is None:
            self._schema_cache[cache_key] = None
            return None
        element = package.element(element_kind)
        if element is None or element.schema_file is None:
            self._schema_cache[cache_key] = None
            return None
        provider_dir = self._provider_dir_for_package(provider_token, package.version)
        if provider_dir is None:
            self._schema_cache[cache_key] = None
            return None
        path = provider_dir / "versions" / package.version / "elements" / element.schema_file
        if not path.exists():
            self._schema_cache[cache_key] = None
            return None
        schema = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        self._schema_cache[cache_key] = schema
        return schema

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
        scanned_roots: set[Path] = set()
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

    def write_package(
        self,
        package: SchemaVersionPackage,
        *,
        element_schemas: dict[str, dict[str, Any]],
    ):
        provider_token = str(canonical_schema_provider(package.provider))
        package_dir = self._package_dir(provider_token, package.version)
        elements_dir = package_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)

        for element in package.elements:
            if element.schema_file is None:
                continue
            schema = copy.deepcopy(element_schemas[element.element_kind])
            schema["$id"] = f"polylogue://schemas/{provider_token}/{package.version}/{element.element_kind}"
            schema["x-polylogue-version"] = int(package.version[1:]) if package.version.startswith("v") else package.version
            schema["x-polylogue-package-version"] = package.version
            schema["x-polylogue-element-kind"] = element.element_kind
            schema["x-polylogue-registered-at"] = datetime.now(tz=timezone.utc).isoformat()
            schema_path = elements_dir / element.schema_file
            schema_path.write_bytes(gzip.compress(json.dumps(schema, indent=2).encode("utf-8")))

        manifest_path = self._package_manifest_path(provider_token, package.version)
        manifest_path.write_text(json.dumps(package.to_dict(), indent=2), encoding="utf-8")
        self.clear_cache()
        return manifest_path

    def replace_provider_packages(
        self,
        provider: str,
        catalog: SchemaPackageCatalog,
        package_schemas: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        provider_token = str(canonical_schema_provider(provider))
        provider_dir = self._provider_dir(provider_token)
        provider_dir.mkdir(parents=True, exist_ok=True)
        versions_dir = provider_dir / "versions"
        if versions_dir.exists():
            for path in versions_dir.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(versions_dir.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
        versions_dir.mkdir(parents=True, exist_ok=True)

        for package in catalog.packages:
            self.write_package(package, element_schemas=package_schemas[package.version])
        self.save_package_catalog(catalog)

    def _single_element_package(
        self,
        provider: str,
        *,
        version: str,
        schema: dict[str, Any],
        element_kind: str = "conversation_document",
        first_seen: str | None = None,
        last_seen: str | None = None,
    ) -> tuple[SchemaVersionPackage, dict[str, dict[str, Any]]]:
        provider_token = str(canonical_schema_provider(provider))
        now = datetime.now(tz=timezone.utc).isoformat()
        observed_at = schema.get("x-polylogue-generated-at")
        profile_family_ids = [
            str(item)
            for item in schema.get(
                "x-polylogue-package-profile-family-ids",
                schema.get("x-polylogue-profile-family-ids", []),
            )
            if isinstance(item, str)
        ]
        if not profile_family_ids:
            profile_family_ids = [
                str(item)
                for item in schema.get("x-polylogue-profile-family-ids", [])
                if isinstance(item, str)
            ]
        element_profile_family_ids = [
            str(item)
            for item in schema.get("x-polylogue-profile-family-ids", profile_family_ids)
            if isinstance(item, str)
        ]
        anchor_profile_family_id = str(schema.get("x-polylogue-anchor-profile-family-id", "")).strip() or next(
            (item for item in profile_family_ids if item),
            "",
        )
        observed_artifact_count = int(schema.get("x-polylogue-observed-artifact-count", 0) or 0)
        package = SchemaVersionPackage(
            provider=provider_token,
            version=version,
            anchor_kind=element_kind,
            default_element_kind=element_kind,
            first_seen=first_seen or observed_at or now,
            last_seen=last_seen or first_seen or observed_at or now,
            bundle_scope_count=0,
            sample_count=int(schema.get("x-polylogue-sample-count", 0) or 0),
            anchor_profile_family_id=anchor_profile_family_id,
            profile_family_ids=profile_family_ids,
            elements=[
                SchemaElementManifest(
                    element_kind=element_kind,
                    schema_file=f"{element_kind}.schema.json.gz",
                    sample_count=int(schema.get("x-polylogue-sample-count", 0) or 0),
                    artifact_count=observed_artifact_count,
                    first_seen=str(schema.get("x-polylogue-element-first-seen", observed_at or "")),
                    last_seen=str(schema.get("x-polylogue-element-last-seen", observed_at or "")),
                    bundle_scope_count=int(schema.get("x-polylogue-element-bundle-scope-count", 0) or 0),
                    exact_structure_ids=[str(item) for item in schema.get("x-polylogue-exact-structure-ids", [])],
                    profile_family_ids=element_profile_family_ids,
                    profile_tokens=[
                        str(item)
                        for item in schema.get("x-polylogue-profile-tokens", [])
                        if isinstance(item, str)
                    ],
                    representative_paths=[
                        str(item)
                        for item in schema.get("x-polylogue-representative-paths", [])
                        if isinstance(item, str)
                    ],
                    observed_artifact_count=observed_artifact_count,
                )
            ],
        )
        return package, {element_kind: schema}

    def register_schema(self, provider: str, schema: dict[str, Any]) -> str:
        provider_token = str(canonical_schema_provider(provider))
        versions = self.list_versions(provider_token)
        new_version = f"v{int(versions[-1][1:]) + 1}" if versions else "v1"
        self.write_schema_version(provider_token, new_version, schema)
        return new_version

    def write_schema_version(self, provider: str, version: str, schema: dict[str, Any]):
        provider_token = str(canonical_schema_provider(provider))
        package, schemas = self._single_element_package(provider_token, version=version, schema=copy.deepcopy(schema))
        catalog = self.load_package_catalog(provider_token) or SchemaPackageCatalog(provider=provider_token)
        existing_packages = [item for item in catalog.packages if item.version != version]
        existing_packages.append(package)
        existing_packages.sort(key=lambda item: int(item.version[1:]))
        catalog.packages = existing_packages
        catalog.latest_version = existing_packages[-1].version if existing_packages else version
        catalog.default_version = catalog.latest_version
        catalog.recommended_version = catalog.latest_version
        self.write_package(package, element_schemas=schemas)
        self.save_package_catalog(catalog)
        return self._package_dir(provider_token, version) / "elements" / f"{package.default_element_kind}.schema.json.gz"

    def resolve_payload(
        self,
        provider: str,
        payload: Any,
        *,
        source_path: str | None = None,
    ) -> SchemaResolution | None:
        provider_token = str(canonical_schema_provider(provider))
        catalog = self.load_package_catalog(provider_token)
        if catalog is None or not catalog.packages:
            return None

        config = resolve_provider_config(provider_token)
        units = extract_schema_units_from_payload(
            payload,
            provider_name=provider_token,
            source_path=source_path,
            raw_id=None,
            observed_at=None,
            config=config,
            max_samples=64,
        )
        if not units:
            default_package = self.get_package(provider_token, version="default")
            if default_package is None:
                return None
            return SchemaResolution(
                provider=provider_token,
                package_version=default_package.version,
                element_kind=default_package.default_element_kind,
                exact_structure_id=None,
                bundle_scope=derive_bundle_scope(provider_token, source_path),
                reason="package_default",
            )

        unit = units[0]
        bundle_scope = unit.bundle_scope or derive_bundle_scope(provider_token, source_path)

        bundle_resolution = self._resolve_by_bundle_scope(
            catalog.packages,
            unit.artifact_kind,
            bundle_scope,
            provider_token,
            unit.exact_structure_id,
        )
        if bundle_resolution is not None:
            return bundle_resolution

        exact_resolution = self._resolve_by_exact_structure(
            catalog.packages,
            unit.artifact_kind,
            unit.exact_structure_id,
            bundle_scope,
            provider_token,
        )
        if exact_resolution is not None:
            return exact_resolution

        profile_resolution = self._resolve_by_profile_similarity(
            catalog.packages,
            unit.artifact_kind,
            unit.profile_tokens,
            unit.exact_structure_id,
            bundle_scope,
            provider_token,
        )
        if profile_resolution is not None:
            return profile_resolution

        default_package = self.get_package(provider_token, version="default")
        if default_package is None:
            return None
        return SchemaResolution(
            provider=provider_token,
            package_version=default_package.version,
            element_kind=default_package.default_element_kind,
            exact_structure_id=unit.exact_structure_id,
            bundle_scope=bundle_scope,
            reason="package_default",
        )

    def _resolve_by_bundle_scope(
        self,
        packages: list[SchemaVersionPackage],
        artifact_kind: str,
        bundle_scope: str | None,
        provider: str,
        exact_structure_id: str | None,
    ) -> SchemaResolution | None:
        if bundle_scope is None:
            return None
        for package in packages:
            if bundle_scope not in package.bundle_scopes:
                continue
            element = package.element(artifact_kind)
            if element is None:
                continue
            return SchemaResolution(
                provider=provider,
                package_version=package.version,
                element_kind=element.element_kind,
                exact_structure_id=exact_structure_id,
                bundle_scope=bundle_scope,
                reason="bundle_scope",
            )
        return None

    def _resolve_by_exact_structure(
        self,
        packages: list[SchemaVersionPackage],
        artifact_kind: str,
        exact_structure_id: str | None,
        bundle_scope: str | None,
        provider: str,
    ) -> SchemaResolution | None:
        for package in packages:
            element = package.element(artifact_kind)
            if element is None:
                continue
            if exact_structure_id in element.exact_structure_ids:
                return SchemaResolution(
                    provider=provider,
                    package_version=package.version,
                    element_kind=element.element_kind,
                    exact_structure_id=exact_structure_id,
                    bundle_scope=bundle_scope,
                    reason="exact_structure",
                )
        return None

    def _resolve_by_profile_similarity(
        self,
        packages: list[SchemaVersionPackage],
        artifact_kind: str,
        profile_tokens: list[str],
        exact_structure_id: str | None,
        bundle_scope: str | None,
        provider: str,
    ) -> SchemaResolution | None:
        best_match: tuple[float, SchemaVersionPackage, SchemaElementManifest] | None = None
        for package in packages:
            element = package.element(artifact_kind)
            if element is None or not element.profile_tokens:
                continue
            score = profile_similarity(set(element.profile_tokens), set(profile_tokens))
            if best_match is None or score > best_match[0]:
                best_match = (score, package, element)
        if best_match is None or best_match[0] <= 0.0:
            return None
        _score, package, element = best_match
        return SchemaResolution(
            provider=provider,
            package_version=package.version,
            element_kind=element.element_kind,
            exact_structure_id=exact_structure_id,
            bundle_scope=bundle_scope,
            reason="profile_family",
        )

    def match_payload_version(
        self,
        provider: str,
        payload: Any,
        *,
        source_path: str | None = None,
    ) -> str | None:
        resolution = self.resolve_payload(provider, payload, source_path=source_path)
        return resolution.package_version if resolution is not None else None


__all__ = ["SCHEMA_DIR", "SchemaProvider", "SchemaRegistry", "canonical_schema_provider"]
