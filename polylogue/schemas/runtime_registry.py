"""Runtime schema authority for package catalogs, schemas, and payload resolution."""

from __future__ import annotations

import copy
import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from polylogue.archive.raw_payload.decode import JSONRecord
from polylogue.core.json import json_document
from polylogue.core.provider_identity import canonical_schema_provider as _canonical_schema_provider
from polylogue.core.provider_identity import normalize_provider_token
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
    SchemaResolutionReason,
    SchemaVersionPackage,
)
from polylogue.types import Provider

SCHEMA_DIR = Path(__file__).parent / "providers"
SchemaProvider = Provider | str
SchemaCacheKey = tuple[str, str, str | None]
SchemaInputDocument = Mapping[str, object]
PublicSchemaDocument = JSONRecord
ElementSchemaMap = dict[str, PublicSchemaDocument]

_PROFILE_SAMPLE_LIMIT = 64
_RESOLUTION_PRIORITY: dict[SchemaResolutionReason, int] = {
    "exact_structure": 3,
    "bundle_scope": 2,
    "profile_family": 1,
    "package_default": 0,
}


def _provider_token(provider: str | Provider) -> str:
    return str(canonical_schema_provider(provider))


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _nonblank_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _read_json_dict(path: Path) -> PublicSchemaDocument:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return {str(key): value for key, value in loaded.items()}


def _read_gzip_json_dict(path: Path) -> PublicSchemaDocument:
    loaded = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected gzipped JSON object in {path}")
    return {str(key): value for key, value in loaded.items()}


def _catalog_path(provider_dir: Path) -> Path:
    return provider_dir / "catalog.json"


def _version_sort_key(version: str) -> tuple[int, str]:
    if version.startswith("v") and version[1:].isdigit():
        return (int(version[1:]), version)
    return (-1, version)


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


@dataclass(frozen=True)
class _SchemaEvidence:
    observed_at: str | None
    sample_count: int
    observed_artifact_count: int
    package_profile_family_ids: list[str]
    element_profile_family_ids: list[str]
    anchor_profile_family_id: str
    element_first_seen: str
    element_last_seen: str
    element_bundle_scope_count: int
    exact_structure_ids: list[str]
    profile_tokens: list[str]
    representative_paths: list[str]


@dataclass(frozen=True)
class _ObservedPayload:
    artifact_kind: str
    bundle_scope: str | None
    exact_structure_id: str | None
    profile_tokens: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedElement:
    package: SchemaVersionPackage
    element: SchemaElementManifest


@dataclass(frozen=True)
class _ResolutionCandidate:
    reason: SchemaResolutionReason
    resolved: _ResolvedElement
    exact_structure_id: str | None
    bundle_scope: str | None
    observation_index: int
    profile_score: float | None = None


def _schema_evidence(schema: SchemaInputDocument) -> _SchemaEvidence:
    observed_at = _string_value(schema.get("x-polylogue-generated-at"))
    package_profile_family_ids = _string_list(
        schema.get(
            "x-polylogue-package-profile-family-ids",
            schema.get("x-polylogue-profile-family-ids", []),
        )
    )
    if not package_profile_family_ids:
        package_profile_family_ids = _string_list(schema.get("x-polylogue-profile-family-ids", []))

    element_profile_family_ids = _string_list(
        schema.get("x-polylogue-profile-family-ids", package_profile_family_ids)
    ) or list(package_profile_family_ids)
    anchor_profile_family_id = _nonblank_string(schema.get("x-polylogue-anchor-profile-family-id")) or next(
        (item for item in package_profile_family_ids if item),
        "",
    )
    element_first_seen = _string_value(schema.get("x-polylogue-element-first-seen")) or (observed_at or "")
    element_last_seen = _string_value(schema.get("x-polylogue-element-last-seen")) or (observed_at or "")

    return _SchemaEvidence(
        observed_at=observed_at,
        sample_count=_int_value(schema.get("x-polylogue-sample-count", 0)),
        observed_artifact_count=_int_value(schema.get("x-polylogue-observed-artifact-count", 0)),
        package_profile_family_ids=package_profile_family_ids,
        element_profile_family_ids=element_profile_family_ids,
        anchor_profile_family_id=anchor_profile_family_id,
        element_first_seen=element_first_seen,
        element_last_seen=element_last_seen,
        element_bundle_scope_count=_int_value(schema.get("x-polylogue-element-bundle-scope-count", 0)),
        exact_structure_ids=_string_list(schema.get("x-polylogue-exact-structure-ids", [])),
        profile_tokens=_string_list(schema.get("x-polylogue-profile-tokens", [])),
        representative_paths=_string_list(schema.get("x-polylogue-representative-paths", [])),
    )


def _resolved_package_version(catalog: SchemaPackageCatalog, version: str) -> str | None:
    if version == "default":
        return catalog.default_version or catalog.latest_version or catalog.recommended_version
    if version == "latest":
        return catalog.latest_version or catalog.default_version or catalog.recommended_version
    if version == "recommended":
        return catalog.recommended_version or catalog.default_version or catalog.latest_version
    return version


def _package_rank_key(
    catalog: SchemaPackageCatalog,
    package: SchemaVersionPackage,
) -> tuple[bool, bool, bool, int, str]:
    version_score, version_text = _version_sort_key(package.version)
    return (
        package.version != (catalog.recommended_version or ""),
        package.version != (catalog.default_version or ""),
        package.version != (catalog.latest_version or ""),
        -version_score,
        version_text,
    )


def _candidate_sort_key(
    candidate: _ResolutionCandidate,
    *,
    package_rank: Mapping[str, int],
) -> tuple[int, int, float, int, str, str]:
    return (
        -_RESOLUTION_PRIORITY[candidate.reason],
        package_rank[candidate.resolved.package.version],
        -(candidate.profile_score or 0.0),
        candidate.observation_index,
        candidate.resolved.element.element_kind,
        candidate.resolved.package.version,
    )


class SchemaRegistry:
    """Runtime package/catalog authority for schema resolution."""

    def __init__(self, storage_root: Path | None = None) -> None:
        self._storage_root = storage_root
        self._catalog_cache: dict[str, SchemaPackageCatalog | None] = {}
        self._schema_cache: dict[SchemaCacheKey, PublicSchemaDocument | None] = {}

    def clear_cache(self) -> None:
        """Clear internal caches. Call after modifying schema packages."""
        self._catalog_cache.clear()
        self._schema_cache.clear()

    @property
    def storage_root(self) -> Path:
        return self._storage_root if self._storage_root is not None else data_home() / "schemas"

    def _provider_dir(self, provider: str) -> Path:
        return self.storage_root / _provider_token(provider)

    def _bundled_provider_dir(self, provider: str) -> Path:
        return SCHEMA_DIR / _provider_token(provider)

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

    def _provider_dir_for_catalog(self, provider: str) -> Path | None:
        for provider_dir in self._provider_search_roots(provider):
            if _catalog_path(provider_dir).exists():
                return provider_dir
        return None

    def _provider_dir_for_package(self, provider: str, version: str) -> Path | None:
        for provider_dir in self._provider_search_roots(provider):
            manifest_path = provider_dir / "versions" / version / "package.json"
            if manifest_path.exists():
                return provider_dir
        return None

    def load_package_catalog(self, provider: str) -> SchemaPackageCatalog | None:
        provider_token = _provider_token(provider)
        if provider_token in self._catalog_cache:
            return self._catalog_cache[provider_token]
        provider_dir = self._provider_dir_for_catalog(provider_token)
        if provider_dir is None:
            self._catalog_cache[provider_token] = None
            return None
        catalog = SchemaPackageCatalog.from_dict(_read_json_dict(_catalog_path(provider_dir)))
        self._catalog_cache[provider_token] = catalog
        return catalog

    def save_package_catalog(self, catalog: SchemaPackageCatalog) -> Path:
        provider_token = _provider_token(catalog.provider)
        provider_dir = self._provider_dir(provider_token)
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = self._catalog_path(provider_token)
        path.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
        self.clear_cache()
        return path

    def get_package(self, provider: str, version: str = "default") -> SchemaVersionPackage | None:
        provider_token = _provider_token(provider)
        catalog = self.load_package_catalog(provider_token)
        if catalog is None:
            return None
        resolved_version = _resolved_package_version(catalog, version)
        return catalog.package(resolved_version) if resolved_version is not None else None

    def get_element_schema(
        self,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> PublicSchemaDocument | None:
        provider_token = _provider_token(provider)
        cache_key: SchemaCacheKey = (provider_token, version, element_kind)
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

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

        schema = _read_gzip_json_dict(path)
        self._schema_cache[cache_key] = schema
        return schema

    def get_schema(self, provider: str, version: str = "default") -> PublicSchemaDocument | None:
        return self.get_element_schema(provider, version=version)

    def list_versions(self, provider: str) -> list[str]:
        provider_token = _provider_token(provider)
        catalog = self.load_package_catalog(provider_token)
        if catalog is None:
            return []
        return sorted((package.version for package in catalog.packages), key=_version_sort_key)

    def list_providers(self) -> list[str]:
        providers: set[str] = set()
        scanned_roots: set[Path] = set()
        for root in (self.storage_root, SCHEMA_DIR):
            if root in scanned_roots or not root.exists():
                continue
            scanned_roots.add(root)
            for path in root.iterdir():
                if path.is_dir() and _catalog_path(path).exists():
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
        element_schemas: ElementSchemaMap,
    ) -> Path:
        provider_token = _provider_token(package.provider)
        package_dir = self._package_dir(provider_token, package.version)
        elements_dir = package_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)

        for element in package.elements:
            if element.schema_file is None:
                continue
            schema = copy.deepcopy(element_schemas[element.element_kind])
            schema["$id"] = f"polylogue://schemas/{provider_token}/{package.version}/{element.element_kind}"
            schema["x-polylogue-version"] = (
                int(package.version[1:]) if package.version.startswith("v") else package.version
            )
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
        package_schemas: Mapping[str, ElementSchemaMap],
    ) -> None:
        provider_token = _provider_token(provider)
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
        schema: SchemaInputDocument,
        element_kind: str = "conversation_document",
        first_seen: str | None = None,
        last_seen: str | None = None,
    ) -> tuple[SchemaVersionPackage, ElementSchemaMap]:
        provider_token = _provider_token(provider)
        now = datetime.now(tz=timezone.utc).isoformat()
        evidence = _schema_evidence(schema)
        package = SchemaVersionPackage(
            provider=provider_token,
            version=version,
            anchor_kind=element_kind,
            default_element_kind=element_kind,
            first_seen=first_seen or evidence.observed_at or now,
            last_seen=last_seen or first_seen or evidence.observed_at or now,
            bundle_scope_count=0,
            sample_count=evidence.sample_count,
            anchor_profile_family_id=evidence.anchor_profile_family_id,
            profile_family_ids=evidence.package_profile_family_ids,
            elements=[
                SchemaElementManifest(
                    element_kind=element_kind,
                    schema_file=f"{element_kind}.schema.json.gz",
                    sample_count=evidence.sample_count,
                    artifact_count=evidence.observed_artifact_count,
                    first_seen=evidence.element_first_seen,
                    last_seen=evidence.element_last_seen,
                    bundle_scope_count=evidence.element_bundle_scope_count,
                    exact_structure_ids=evidence.exact_structure_ids,
                    profile_family_ids=evidence.element_profile_family_ids,
                    profile_tokens=evidence.profile_tokens,
                    representative_paths=evidence.representative_paths,
                    observed_artifact_count=evidence.observed_artifact_count,
                )
            ],
        )
        return package, {element_kind: json_document(copy.deepcopy(dict(schema)))}

    def register_schema(self, provider: str, schema: SchemaInputDocument) -> str:
        provider_token = _provider_token(provider)
        versions = self.list_versions(provider_token)
        new_version = f"v{int(versions[-1][1:]) + 1}" if versions else "v1"
        self.write_schema_version(provider_token, new_version, schema)
        return new_version

    def write_schema_version(self, provider: str, version: str, schema: SchemaInputDocument) -> Path:
        provider_token = _provider_token(provider)
        package, schemas = self._single_element_package(
            provider_token,
            version=version,
            schema=copy.deepcopy(dict(schema)),
        )
        catalog = self.load_package_catalog(provider_token) or SchemaPackageCatalog(provider=provider_token)
        existing_packages = [item for item in catalog.packages if item.version != version]
        existing_packages.append(package)
        existing_packages.sort(key=lambda item: _version_sort_key(item.version))
        catalog.packages = existing_packages
        catalog.latest_version = existing_packages[-1].version if existing_packages else version
        catalog.default_version = catalog.latest_version
        catalog.recommended_version = catalog.latest_version
        self.write_package(package, element_schemas=schemas)
        self.save_package_catalog(catalog)
        return (
            self._package_dir(provider_token, version) / "elements" / f"{package.default_element_kind}.schema.json.gz"
        )

    def _package_rank(self, catalog: SchemaPackageCatalog) -> dict[str, int]:
        return {package.version: index for index, package in enumerate(self._ranked_packages(catalog))}

    @staticmethod
    def _package_rank_from_sorted(packages: Sequence[SchemaVersionPackage]) -> dict[str, int]:
        return {package.version: index for index, package in enumerate(packages)}

    def _ranked_packages(self, catalog: SchemaPackageCatalog) -> list[SchemaVersionPackage]:
        return sorted(catalog.packages, key=lambda package: _package_rank_key(catalog, package))

    def _observed_payloads(
        self,
        provider: str,
        payload: object,
        *,
        source_path: str | None,
    ) -> list[_ObservedPayload]:
        provider_token = _provider_token(provider)
        config = resolve_provider_config(provider_token)
        fallback_bundle_scope = derive_bundle_scope(provider_token, source_path)
        units = extract_schema_units_from_payload(
            payload,
            provider_name=Provider.from_string(provider_token),
            source_path=source_path,
            raw_id=None,
            observed_at=None,
            config=config,
            max_samples=_PROFILE_SAMPLE_LIMIT,
        )
        return [
            _ObservedPayload(
                artifact_kind=unit.artifact_kind,
                bundle_scope=unit.bundle_scope or fallback_bundle_scope,
                exact_structure_id=unit.exact_structure_id or None,
                profile_tokens=unit.profile_tokens,
            )
            for unit in units
        ]

    def _resolve_observation(
        self,
        packages: Sequence[SchemaVersionPackage],
        observation: _ObservedPayload,
        *,
        package_rank: Mapping[str, int],
        observation_index: int,
    ) -> _ResolutionCandidate | None:
        candidates: list[_ResolutionCandidate] = []
        observed_profile_tokens = set(observation.profile_tokens)
        for package in packages:
            element = package.element(observation.artifact_kind)
            if element is None:
                continue
            resolved = _ResolvedElement(package=package, element=element)

            if observation.exact_structure_id and observation.exact_structure_id in element.exact_structure_ids:
                candidates.append(
                    _ResolutionCandidate(
                        reason="exact_structure",
                        resolved=resolved,
                        exact_structure_id=observation.exact_structure_id,
                        bundle_scope=observation.bundle_scope,
                        observation_index=observation_index,
                    )
                )
            if observation.bundle_scope and observation.bundle_scope in package.element_bundle_scopes(
                element.element_kind
            ):
                candidates.append(
                    _ResolutionCandidate(
                        reason="bundle_scope",
                        resolved=resolved,
                        exact_structure_id=observation.exact_structure_id,
                        bundle_scope=observation.bundle_scope,
                        observation_index=observation_index,
                    )
                )
            if observed_profile_tokens and element.profile_tokens:
                score = profile_similarity(set(element.profile_tokens), observed_profile_tokens)
                if score > 0.0:
                    candidates.append(
                        _ResolutionCandidate(
                            reason="profile_family",
                            resolved=resolved,
                            exact_structure_id=observation.exact_structure_id,
                            bundle_scope=observation.bundle_scope,
                            observation_index=observation_index,
                            profile_score=score,
                        )
                    )

        if not candidates:
            return None

        winner = min(
            candidates,
            key=lambda candidate: _candidate_sort_key(candidate, package_rank=package_rank),
        )
        return winner

    def _default_resolution(
        self,
        provider: str,
        *,
        observations: Sequence[_ObservedPayload],
        source_path: str | None,
    ) -> SchemaResolution | None:
        default_package = self.get_package(provider, version="default")
        if default_package is None:
            return None

        matching_observation = next(
            (
                observation
                for observation in observations
                if default_package.element(observation.artifact_kind) is not None
            ),
            None,
        )
        if matching_observation is not None:
            element = default_package.element(matching_observation.artifact_kind)
            if element is not None:
                return SchemaResolution(
                    provider=provider,
                    package_version=default_package.version,
                    element_kind=element.element_kind,
                    exact_structure_id=matching_observation.exact_structure_id,
                    bundle_scope=matching_observation.bundle_scope,
                    reason="package_default",
                )

        return SchemaResolution(
            provider=provider,
            package_version=default_package.version,
            element_kind=default_package.default_element_kind,
            exact_structure_id=None,
            bundle_scope=derive_bundle_scope(provider, source_path),
            reason="package_default",
        )

    def resolve_payload(
        self,
        provider: str,
        payload: object,
        *,
        source_path: str | None = None,
    ) -> SchemaResolution | None:
        provider_token = _provider_token(provider)
        catalog = self.load_package_catalog(provider_token)
        if catalog is None or not catalog.packages:
            return None

        observations = self._observed_payloads(provider_token, payload, source_path=source_path)
        ranked_packages = self._ranked_packages(catalog)
        package_rank = self._package_rank_from_sorted(ranked_packages)
        best_candidate: _ResolutionCandidate | None = None
        for index, observation in enumerate(observations):
            candidate = self._resolve_observation(
                ranked_packages,
                observation,
                package_rank=package_rank,
                observation_index=index,
            )
            if candidate is None:
                continue
            if best_candidate is None:
                best_candidate = candidate
                continue
            if _candidate_sort_key(candidate, package_rank=package_rank) < _candidate_sort_key(
                best_candidate,
                package_rank=package_rank,
            ):
                best_candidate = candidate

        if best_candidate is not None:
            return SchemaResolution(
                provider=provider_token,
                package_version=best_candidate.resolved.package.version,
                element_kind=best_candidate.resolved.element.element_kind,
                exact_structure_id=best_candidate.exact_structure_id,
                bundle_scope=best_candidate.bundle_scope,
                reason=best_candidate.reason,
                profile_score=best_candidate.profile_score,
            )
        return self._default_resolution(
            provider_token,
            observations=observations,
            source_path=source_path,
        )

    def match_payload_version(
        self,
        provider: str,
        payload: object,
        *,
        source_path: str | None = None,
    ) -> str | None:
        resolution = self.resolve_payload(provider, payload, source_path=source_path)
        return resolution.package_version if resolution is not None else None


__all__ = ["SCHEMA_DIR", "SchemaProvider", "SchemaRegistry", "canonical_schema_provider"]
