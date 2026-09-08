"""Runtime schema authority for package catalogs, schemas, and payload resolution."""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
import shutil
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from polylogue.archive.artifact_taxonomy import classify_artifact
from polylogue.archive.raw_payload.decode import JSONRecord
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue, json_document
from polylogue.core.provider_identity import canonical_schema_provider as _canonical_schema_provider
from polylogue.core.provider_identity import normalize_provider_token
from polylogue.core.schema_subjects import SCHEMA_PACKAGE_DIRECTORIES, SCHEMA_SUBJECTS
from polylogue.paths import data_home
from polylogue.schemas.generation.dynamic_keys import (
    is_source_structure_witness,
    legacy_structure_schema_digest,
    observed_structure_schema,
    retain_exact_structure_witnesses,
    structure_schema_digest,
)
from polylogue.schemas.observation import (
    derive_bundle_scope,
    extract_schema_units_from_payload,
    profile_similarity,
    resolve_provider_config,
)
from polylogue.schemas.package_publication import publish_provider_tree, read_provider_snapshot
from polylogue.schemas.packages import (
    SchemaElementManifest,
    SchemaPackageCatalog,
    SchemaResolution,
    SchemaResolutionReason,
    SchemaVersionPackage,
)

SCHEMA_DIR = Path(__file__).parent / "providers"
SchemaProvider = Provider | str
SchemaCacheKey = tuple[str, str, str | None]
WorkloadProfileCacheKey = tuple[str, str]
SchemaInputDocument = Mapping[str, object]
PublicSchemaDocument = JSONRecord
ElementSchemaMap = dict[str, PublicSchemaDocument]

_STATISTICAL_ANNOTATIONS = frozenset(
    {
        "x-polylogue-observed-distribution",
        "x-polylogue-frequency",
        "x-polylogue-range",
        "x-polylogue-array-lengths",
        "x-polylogue-multiline",
        "x-polylogue-values",
    }
)
_PROFILE_SAMPLE_LIMIT = 64
_RESOLUTION_PRIORITY: dict[SchemaResolutionReason, int] = {
    "exact_structure": 3,
    "bundle_scope": 2,
    "profile_family": 1,
    "package_catalog": 0,
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


def _publication_omission_bound(schema: SchemaInputDocument) -> int:
    return _int_value(
        schema.get(
            "x-polylogue-publication-omitted-structure-witness-count",
            schema.get("x-polylogue-omitted-current-structure-witness-count", 0),
        )
    )


def _structure_witnesses(samples: Sequence[object]) -> tuple[tuple[str, ...], ...]:
    """Return canonical and shipped-order aliases for every sampled record."""

    aliases: list[tuple[str, ...]] = []
    for sample in samples:
        schema = observed_structure_schema(sample)
        canonical = structure_schema_digest(schema)
        legacy = legacy_structure_schema_digest(schema)
        aliases.append((canonical,) if canonical == legacy else (canonical, legacy))
    return tuple(aliases)


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


def schema_subject_diagnostics(root: Path | None = None) -> tuple[str, ...]:
    """Return declaration/package drift diagnostics for the committed tree."""
    package_root = root or SCHEMA_DIR
    actual = (
        tuple(
            sorted(path.name for path in package_root.iterdir() if path.is_dir() and (path / "catalog.json").exists())
        )
        if package_root.exists()
        else ()
    )
    declared = tuple(sorted(SCHEMA_PACKAGE_DIRECTORIES))
    diagnostics: list[str] = []
    for name in sorted(set(declared) - set(actual)):
        diagnostics.append(f"declared schema package is missing: {name}")
    for name in sorted(set(actual) - set(declared)):
        diagnostics.append(f"undeclared schema package directory: {name}")
    for subject in SCHEMA_SUBJECTS:
        if subject.requires_package and canonical_schema_provider(subject.token) != subject.token:
            diagnostics.append(f"schema subject is not canonically reachable: {subject.token}")
    return tuple(diagnostics)


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


@dataclass(frozen=True)
class _ObservedPayload:
    artifact_kind: str
    bundle_scope: str | None
    exact_structure_id: str | None
    profile_tokens: tuple[str, ...]
    schema_samples: Sequence[object] = ()


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
        self._workload_profile_cache: dict[WorkloadProfileCacheKey, PublicSchemaDocument | None] = {}
        self._snapshots: dict[Path, dict[str, bytes]] = {}
        self._cache_lock = threading.RLock()

    def _snapshot(self, provider_dir: Path) -> dict[str, bytes]:
        with self._cache_lock:
            if provider_dir not in self._snapshots:
                self._snapshots[provider_dir] = read_provider_snapshot(provider_dir)
            return self._snapshots[provider_dir]

    def _snapshot_json(self, provider_dir: Path, relative: str) -> PublicSchemaDocument | None:
        payload = self._snapshot(provider_dir).get(relative)
        if payload is None:
            return None
        decoded = gzip.decompress(payload) if relative.endswith(".gz") else payload
        value = json.loads(decoded)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {relative}")
        return {str(key): item for key, item in value.items()}

    def clear_cache(self) -> None:
        """Clear internal caches. Call after modifying schema packages."""
        with self._cache_lock:
            self._snapshots.clear()
            self._catalog_cache.clear()
            self._schema_cache.clear()
            self._workload_profile_cache.clear()

    @property
    def storage_root(self) -> Path:
        return self._storage_root if self._storage_root is not None else data_home() / "schemas"

    def _provider_dir(self, provider: str) -> Path:
        return self.storage_root / _provider_token(provider)

    def _committed_provider_dir(self, provider: str) -> Path:
        """Return the literal provider directory in this registry's tree.

        Committed-bundle audits must inspect the directory they discovered.
        Canonical provider aliases are appropriate for runtime resolution, but
        they must not redirect an inventory audit to another directory.
        """
        return self.storage_root / provider

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
        return next(
            (root for root in self._provider_search_roots(provider) if "catalog.json" in self._snapshot(root)), None
        )

    def _provider_dir_for_package(self, provider: str, version: str) -> Path | None:
        root = self._provider_dir_for_catalog(provider)
        return root if root is not None and f"versions/{version}/package.json" in self._snapshot(root) else None

    def load_package_catalog(self, provider: str) -> SchemaPackageCatalog | None:
        provider_token = _provider_token(provider)
        with self._cache_lock:
            if provider_token not in self._catalog_cache:
                root = self._provider_dir_for_catalog(provider_token)
                payload = self._snapshot_json(root, "catalog.json") if root is not None else None
                self._catalog_cache[provider_token] = (
                    SchemaPackageCatalog.from_dict(payload) if payload is not None else None
                )
            return self._catalog_cache[provider_token]

    def _load_local_catalog(self, provider: str) -> SchemaPackageCatalog | None:
        payload = self._snapshot_json(self._provider_dir(provider), "catalog.json")
        return SchemaPackageCatalog.from_dict(payload) if payload is not None else None

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
        with self._cache_lock:
            if cache_key in self._schema_cache:
                return self._schema_cache[cache_key]

            schema = self._load_element_schema(provider_token, version=version, element_kind=element_kind)
            self._schema_cache[cache_key] = schema
            return schema

    def _load_element_schema(
        self,
        provider_token: str,
        *,
        version: str,
        element_kind: str | None,
    ) -> PublicSchemaDocument | None:
        package = self.get_package(provider_token, version=version)
        if package is None:
            return None

        element = package.element(element_kind)
        if element is None or element.schema_file is None:
            return None

        provider_dir = self._provider_dir_for_package(provider_token, package.version)
        if provider_dir is None:
            return None

        return self._snapshot_json(provider_dir, f"versions/{package.version}/elements/{element.schema_file}")

    def get_schema(self, provider: str, version: str = "default") -> PublicSchemaDocument | None:
        return self.get_element_schema(provider, version=version)

    def get_workload_profile(
        self,
        provider: str,
        version: str = "default",
    ) -> PublicSchemaDocument | None:
        """Load the privacy-safe workload profile carried by a package."""
        provider_token = _provider_token(provider)
        cache_key = (provider_token, version)
        with self._cache_lock:
            if cache_key in self._workload_profile_cache:
                return self._workload_profile_cache[cache_key]
            profile = self._load_workload_profile(provider_token, version=version)
            self._workload_profile_cache[cache_key] = profile
            return profile

    def _load_workload_profile(self, provider_token: str, *, version: str) -> PublicSchemaDocument | None:
        package = self.get_package(provider_token, version=version)
        if package is None or package.workload_profile_file is None:
            return None
        provider_dir = self._provider_dir_for_package(provider_token, package.version)
        if provider_dir is None:
            return None
        return self._snapshot_json(provider_dir, f"versions/{package.version}/{package.workload_profile_file}")

    def list_versions(self, provider: str) -> list[str]:
        provider_token = _provider_token(provider)
        catalog = self.load_package_catalog(provider_token)
        if catalog is None:
            return []
        return sorted((package.version for package in catalog.packages), key=_version_sort_key)

    def list_committed_versions(self, provider: str) -> list[str]:
        snapshot = self._snapshot(self._committed_provider_dir(provider))
        return sorted(
            {path.split("/")[1] for path in snapshot if path.startswith("versions/") and path.count("/") >= 2},
            key=_version_sort_key,
        )

    def list_committed_providers(self) -> list[str]:
        """List provider directories in this registry's own committed tree."""
        root = self.storage_root
        if not root.is_dir():
            return []
        return sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and (_catalog_path(path).is_file() or bool(self.list_committed_versions(path.name)))
        )

    def read_committed_file(self, provider: str, relative_path: str) -> bytes | None:
        """Read artifact bytes from the same snapshot as committed catalog queries."""
        return self._snapshot(self._committed_provider_dir(provider)).get(relative_path)

    def load_committed_catalog(self, provider: str) -> SchemaPackageCatalog | None:
        payload = self._snapshot_json(self._committed_provider_dir(provider), "catalog.json")
        return SchemaPackageCatalog.from_dict(payload) if payload is not None else None

    def load_committed_package(self, provider: str, version: str) -> SchemaVersionPackage | None:
        payload = self._snapshot_json(self._committed_provider_dir(provider), f"versions/{version}/package.json")
        return SchemaVersionPackage.from_dict(payload) if payload is not None else None

    def list_committed_schema_files(self, provider: str, version: str) -> list[str]:
        prefix = f"versions/{version}/elements/"
        return sorted(
            path[len(prefix) :]
            for path in self._snapshot(self._committed_provider_dir(provider))
            if path.startswith(prefix) and path.endswith(".schema.json.gz")
        )

    def load_committed_schema_file(self, provider: str, version: str, schema_file: str) -> PublicSchemaDocument | None:
        return self._snapshot_json(self._committed_provider_dir(provider), f"versions/{version}/elements/{schema_file}")

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
        workload_profile: Mapping[str, object] | None = None,
    ) -> Path:
        provider_token = _provider_token(package.provider)
        self._preflight_package_write(
            package,
            element_schemas=element_schemas,
            workload_profile=workload_profile,
        )
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
            schema.pop("x-polylogue-registered-at", None)
            schema_path = elements_dir / element.schema_file
            schema_path.write_bytes(
                gzip.compress(json.dumps(schema, indent=2, sort_keys=True).encode("utf-8"), mtime=0)
            )

        if package.workload_profile_file is not None:
            if workload_profile is None:
                raise ValueError(
                    f"Package {provider_token}/{package.version} declares "
                    f"{package.workload_profile_file} but no workload profile was supplied"
                )
            profile_path = package_dir / package.workload_profile_file
            profile_path.write_bytes(
                gzip.compress(
                    json.dumps(dict(workload_profile), indent=2, sort_keys=True).encode("utf-8"),
                    mtime=0,
                )
            )

        manifest_path = self._package_manifest_path(provider_token, package.version)
        manifest_path.write_text(json.dumps(package.to_dict(), indent=2), encoding="utf-8")
        self.clear_cache()
        return manifest_path

    @staticmethod
    def _preflight_package_write(
        package: SchemaVersionPackage,
        *,
        element_schemas: ElementSchemaMap,
        workload_profile: Mapping[str, object] | None,
    ) -> None:
        missing_elements = sorted(
            element.element_kind
            for element in package.elements
            if element.schema_file is not None and element.element_kind not in element_schemas
        )
        if missing_elements:
            raise ValueError(
                f"Package {package.provider}/{package.version} is missing schemas for: {', '.join(missing_elements)}"
            )
        if package.workload_profile_file is not None and workload_profile is None:
            raise ValueError(
                f"Package {package.provider}/{package.version} declares "
                f"{package.workload_profile_file} but no workload profile was supplied"
            )
        if workload_profile is not None:
            try:
                json.dumps(dict(workload_profile), sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Package {package.provider}/{package.version} has a non-JSON workload profile"
                ) from exc

    def _load_local_element_schema(
        self,
        provider_token: str,
        *,
        version: str,
        element_kind: str | None,
    ) -> PublicSchemaDocument | None:
        """Load an element schema strictly from this registry's own storage root.

        Mirrors ``_load_local_catalog``'s no-bundled-fallback guarantee: a
        caller merging against "the committed prior schema"
        (``replace_provider_packages``) must not silently pull in an
        unrelated version from the bundled ``SCHEMA_DIR`` tree when
        ``storage_root`` is isolated (e.g. a test's ``tmp_path``, or a
        ``devtools schema-generate`` run pointed at a scratch output dir).
        """
        catalog = self._load_local_catalog(provider_token)
        if catalog is None:
            return None
        resolved_version = _resolved_package_version(catalog, version)
        if resolved_version is None:
            return None
        package = catalog.package(resolved_version)
        if package is None:
            return None
        element = package.element(element_kind)
        if element is None or element.schema_file is None:
            return None
        return self._read_local_element_schema_file(provider_token, package.version, element.schema_file)

    def _read_local_element_schema_file(
        self, provider_token: str, package_version: str, schema_file: str
    ) -> PublicSchemaDocument | None:
        return self._snapshot_json(
            self._provider_dir(provider_token), f"versions/{package_version}/elements/{schema_file}"
        )

    @staticmethod
    def _annotate_merged_schema_node(
        merged: PublicSchemaDocument,
        *,
        existing: PublicSchemaDocument | None,
        candidate: PublicSchemaDocument | None,
    ) -> PublicSchemaDocument:
        """Preserve qualitative annotations and distinguish current from historical statistics."""
        result: PublicSchemaDocument = dict(merged)
        candidate_present = candidate is not None
        existing = existing or {}
        candidate = candidate or {}
        fresh_statistics = "x-polylogue-observed-distribution" in candidate
        for key, value in existing.items():
            if key.startswith("x-polylogue-") and key not in candidate:
                if fresh_statistics and (
                    key in _STATISTICAL_ANNOTATIONS
                    or key in {"x-polylogue-statistics-status", "x-polylogue-observation-status"}
                ):
                    continue
                result[key] = value
        for key, value in candidate.items():
            if key.startswith("x-polylogue-"):
                result[key] = value
        if not fresh_statistics and any(key in existing for key in _STATISTICAL_ANNOTATIONS):
            result["x-polylogue-statistics-status"] = "historical"
        if not candidate_present:
            result["x-polylogue-observation-status"] = "historical"
            result["x-polylogue-frequency"] = 0.0

        merged_properties = json_document(merged.get("properties"))
        if merged_properties:
            existing_properties = json_document(existing.get("properties"))
            candidate_properties = json_document(candidate.get("properties"))
            result["properties"] = {
                name: SchemaRegistry._annotate_merged_schema_node(
                    json_document(child),
                    existing=json_document(existing_properties.get(name)) or None,
                    candidate=json_document(candidate_properties.get(name)) or None,
                )
                for name, child in merged_properties.items()
            }

        merged_items = json_document(merged.get("items"))
        if merged_items:
            result["items"] = SchemaRegistry._annotate_merged_schema_node(
                merged_items,
                existing=json_document(existing.get("items")) or None,
                candidate=json_document(candidate.get("items")) or None,
            )

        merged_additional = json_document(merged.get("additionalProperties"))
        if merged_additional:
            result["additionalProperties"] = SchemaRegistry._annotate_merged_schema_node(
                merged_additional,
                existing=json_document(existing.get("additionalProperties")) or None,
                candidate=json_document(candidate.get("additionalProperties")) or None,
            )
        return result

    @staticmethod
    def _merge_element_schema_with_existing(
        existing: PublicSchemaDocument | None,
        candidate: PublicSchemaDocument,
    ) -> PublicSchemaDocument:
        """Preserve previously observed structure within one schema family."""
        if existing is None:
            return candidate
        from polylogue.schemas.generation.dynamic_keys import merge_observed_structure_schemas

        merged = json_document(merge_observed_structure_schemas([json_document(existing), candidate]))
        merged = SchemaRegistry._annotate_merged_schema_node(merged, existing=existing, candidate=candidate)
        for key, value in candidate.items():
            if key in ("$schema", "title"):
                merged[key] = value
        return merged

    @staticmethod
    def _retain_element_structure_witnesses(
        package: SchemaVersionPackage,
        *,
        schemas: ElementSchemaMap,
        merged: ElementSchemaMap,
        prior: SchemaVersionPackage | None,
        prior_schemas: ElementSchemaMap,
    ) -> tuple[SchemaVersionPackage, ElementSchemaMap]:
        """Keep package manifests and schema extensions on one witness set."""

        prior_elements = {element.element_kind: element for element in prior.elements} if prior is not None else {}
        elements: list[SchemaElementManifest] = []
        for element in package.elements:
            kind = element.element_kind
            incoming_schema = schemas.get(kind, {})
            previous_schema = prior_schemas.get(kind, {})
            previous_element = prior_elements.get(kind)
            current_ids = [
                *element.exact_structure_ids,
                *_string_list(incoming_schema.get("x-polylogue-exact-structure-ids", [])),
            ]
            incoming_omitted_count = max(
                element.publication_omitted_structure_witness_count,
                _publication_omission_bound(incoming_schema),
            )
            prior_ids = [
                *(previous_element.exact_structure_ids if previous_element is not None else ()),
                *_string_list(previous_schema.get("x-polylogue-exact-structure-ids", [])),
            ]
            retained = retain_exact_structure_witnesses(prior_ids, current_ids)
            omitted_count = max(
                incoming_omitted_count,
                retained.omitted_current_witness_count,
                previous_element.publication_omitted_structure_witness_count if previous_element else 0,
                _publication_omission_bound(previous_schema),
            )
            evidence_loss = max(
                element.source_evidence_unretained_shape_observation_lower_bound,
                previous_element.source_evidence_unretained_shape_observation_lower_bound if previous_element else 0,
                _int_value(
                    previous_schema.get("x-polylogue-source-evidence-unretained-shape-observation-lower-bound", 0)
                ),
                _int_value(
                    incoming_schema.get("x-polylogue-source-evidence-unretained-shape-observation-lower-bound", 0)
                ),
            )
            has_witness_metadata = bool(
                current_ids
                or prior_ids
                or omitted_count
                or evidence_loss
                or "x-polylogue-omitted-current-structure-witness-count" in incoming_schema
                or "x-polylogue-omitted-current-structure-witness-count" in previous_schema
            )
            elements.append(
                dataclasses.replace(
                    element,
                    exact_structure_ids=list(retained.exact_structure_ids),
                    publication_omitted_structure_witness_count=omitted_count,
                    source_evidence_unretained_shape_observation_lower_bound=evidence_loss,
                )
            )
            if has_witness_metadata and kind in merged:
                synchronized = dict(merged[kind])
                synchronized.pop("x-polylogue-omitted-current-structure-witness-count", None)
                synchronized["x-polylogue-exact-structure-ids"] = list(retained.exact_structure_ids)
                synchronized["x-polylogue-publication-omitted-structure-witness-count"] = omitted_count
                synchronized["x-polylogue-source-evidence-unretained-shape-observation-lower-bound"] = evidence_loss
                merged[kind] = synchronized
        return dataclasses.replace(package, elements=elements), merged

    def replace_provider_packages(
        self,
        provider: str,
        catalog: SchemaPackageCatalog,
        package_schemas: Mapping[str, ElementSchemaMap],
        *,
        package_workload_profiles: Mapping[str, Mapping[str, object]] | None = None,
        cluster_manifest: Mapping[str, object] | None = None,
        redact_observed_numeric_values: bool = False,
    ) -> None:
        """Replace a complete package set while preserving observed family structure."""
        with self._cache_lock:
            provider_token = _provider_token(provider)
            self.clear_cache()
            existing_catalog = self._load_local_catalog(provider_token)
            prior_packages = (
                {package.version: package for package in existing_catalog.packages} if existing_catalog else {}
            )
            prepared: list[tuple[SchemaVersionPackage, ElementSchemaMap, Mapping[str, object] | None]] = []
            for package in catalog.packages:
                schemas = package_schemas.get(package.version)
                if schemas is None:
                    raise ValueError(f"Package {provider_token}/{package.version} has no schema mapping")
                prior = prior_packages.get(package.version)
                if (
                    prior is not None
                    and prior.anchor_profile_family_id
                    and package.anchor_profile_family_id
                    and (prior.anchor_kind, prior.anchor_profile_family_id)
                    != (package.anchor_kind, package.anchor_profile_family_id)
                ):
                    raise ValueError(f"Schema version {package.version} already belongs to another structural family")
                if prior is not None and prior.canonical_anchor_profile_family_id is not None:
                    if package.canonical_anchor_profile_family_id not in (
                        None,
                        prior.canonical_anchor_profile_family_id,
                    ):
                        raise ValueError(f"Schema version {package.version} already has another canonical family")
                    package = dataclasses.replace(
                        package, canonical_anchor_profile_family_id=prior.canonical_anchor_profile_family_id
                    )
                prior_schemas: ElementSchemaMap = {}
                if prior is not None:
                    for element in prior.elements:
                        if element.schema_file is not None:
                            value = self._read_local_element_schema_file(
                                provider_token, prior.version, element.schema_file
                            )
                            if value is None:
                                raise ValueError(f"Existing package {provider_token}/{prior.version} is incomplete")
                            prior_schemas[element.element_kind] = value
                merged = {
                    kind: self._merge_element_schema_with_existing(prior_schemas.get(kind), value)
                    for kind, value in schemas.items()
                }
                elements = list(package.elements)
                if prior is not None:
                    for element in prior.elements:
                        if element.element_kind not in schemas:
                            if element.element_kind in prior_schemas:
                                merged[element.element_kind] = prior_schemas[element.element_kind]
                            elements.append(dataclasses.replace(element, observation_status="historical"))
                package, merged = self._retain_element_structure_witnesses(
                    dataclasses.replace(package, elements=elements),
                    schemas=schemas,
                    merged=merged,
                    prior=prior,
                    prior_schemas=prior_schemas,
                )
                profile = (
                    package_workload_profiles.get(package.version) if package_workload_profiles is not None else None
                )
                if redact_observed_numeric_values:
                    from polylogue.schemas.numeric_privacy import (
                        redact_observed_numeric_schema,
                        redact_observed_numeric_workload,
                    )

                    merged = {kind: redact_observed_numeric_schema(value) for kind, value in merged.items()}
                    if profile is not None:
                        profile = redact_observed_numeric_workload(profile)
                self._preflight_package_write(package, element_schemas=merged, workload_profile=profile)
                prepared.append((package, merged, profile))

            incoming_versions = {package.version for package, _, _ in prepared}
            historical_packages = [
                dataclasses.replace(package, observation_status="historical")
                for version, package in prior_packages.items()
                if version not in incoming_versions
            ]
            final_catalog = dataclasses.replace(
                catalog,
                packages=sorted(
                    [package for package, _, _ in prepared] + historical_packages,
                    key=lambda package: _version_sort_key(package.version),
                ),
            )
            final_catalog.family_versions()
            provider_dir = self._provider_dir(provider_token)
            self.storage_root.mkdir(parents=True, exist_ok=True)
            baseline = dict(self._snapshot(provider_dir))
            with tempfile.TemporaryDirectory(prefix=f".{provider_token}.staging-", dir=self.storage_root) as temporary:
                staging_root = Path(temporary)
                staged_provider = staging_root / provider_token
                if provider_dir.exists():
                    shutil.copytree(provider_dir, staged_provider)
                else:
                    staged_provider.mkdir()
                staged_registry = type(self)(storage_root=staging_root)
                for package, schemas, profile in prepared:
                    staged_registry.write_package(package, element_schemas=schemas, workload_profile=profile)
                for package in historical_packages:
                    manifest_path = staged_registry._package_manifest_path(provider_token, package.version)
                    manifest_path.write_text(json.dumps(package.to_dict(), indent=2), encoding="utf-8")
                staged_registry.save_package_catalog(final_catalog)
                if cluster_manifest is not None:
                    (staged_provider / "manifest.json").write_text(
                        json.dumps(dict(cluster_manifest), indent=2, sort_keys=True), encoding="utf-8"
                    )
                publish_provider_tree(staged_provider, provider_dir, expected_snapshot=baseline)
            self.clear_cache()

    def _single_element_package(
        self,
        provider: str,
        *,
        version: str,
        schema: SchemaInputDocument,
        element_kind: str = "session_document",
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
                    observed_artifact_count=evidence.observed_artifact_count,
                )
            ],
        )
        return package, {element_kind: json_document(copy.deepcopy(dict(schema)))}

    def register_schema(
        self,
        provider: str,
        schema: SchemaInputDocument,
        *,
        element_kind: str = "session_document",
    ) -> str:
        with self._cache_lock:
            self.clear_cache()
            provider_token = _provider_token(provider)
            versions = self.list_versions(provider_token)
            new_version = f"v{int(versions[-1][1:]) + 1}" if versions else "v1"
            self.write_schema_version(provider_token, new_version, schema, element_kind=element_kind)
            return new_version

    def write_schema_version(
        self,
        provider: str,
        version: str,
        schema: SchemaInputDocument,
        *,
        element_kind: str = "session_document",
    ) -> Path:
        with self._cache_lock:
            provider_token = _provider_token(provider)
            package, schemas = self._single_element_package(
                provider_token,
                version=version,
                schema=copy.deepcopy(dict(schema)),
                element_kind=element_kind,
            )
            self.clear_cache()
            prior_catalog = self._load_local_catalog(provider_token)
            versions = [item.version for item in prior_catalog.packages] if prior_catalog else []
            latest_version = max([*versions, version], key=_version_sort_key)
            catalog = SchemaPackageCatalog(
                provider=provider_token,
                packages=[package],
                latest_version=latest_version,
                default_version=latest_version,
                recommended_version=latest_version,
            )
            self.replace_provider_packages(provider_token, catalog, {version: schemas})
            return (
                self._package_dir(provider_token, version)
                / "elements"
                / f"{package.default_element_kind}.schema.json.gz"
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
        admitted_artifact_kind = None
        if provider_token == "gemini-cli" and isinstance(payload, list):
            artifact = classify_artifact(
                cast(JSONValue, payload), provider=Provider.GEMINI_CLI, source_path=source_path
            )
            if artifact.schema_eligible and artifact.cohort == "session_record_stream":
                config = dataclasses.replace(config, sample_granularity="record", record_type_key="type")
                admitted_artifact_kind = artifact.cohort
        units = extract_schema_units_from_payload(
            payload,
            source_name=Provider.from_string(provider_token),
            source_path=source_path,
            raw_id=None,
            observed_at=None,
            config=config,
            max_samples=_PROFILE_SAMPLE_LIMIT,
            admitted_artifact_kind=admitted_artifact_kind,
        )
        return [
            _ObservedPayload(
                artifact_kind=unit.artifact_kind,
                bundle_scope=unit.bundle_scope or fallback_bundle_scope,
                exact_structure_id=unit.exact_structure_id or None,
                profile_tokens=unit.profile_tokens,
                schema_samples=unit.schema_samples,
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
        source_witnesses: tuple[tuple[str, ...], ...] = ()
        if any(
            is_source_structure_witness(structure_id)
            for package in packages
            if (element := package.element(observation.artifact_kind)) is not None
            for structure_id in element.exact_structure_ids
        ):
            source_witnesses = _structure_witnesses(observation.schema_samples)
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
            source_matches = tuple(
                next((witness for witness in aliases if witness in element.exact_structure_ids), None)
                for aliases in source_witnesses
            )
            if source_witnesses and all(source_matches):
                candidates.append(
                    _ResolutionCandidate(
                        reason="exact_structure",
                        resolved=resolved,
                        exact_structure_id=source_matches[0],
                        bundle_scope=observation.bundle_scope,
                        observation_index=observation_index,
                    )
                )
            if observation.bundle_scope and package.matches_bundle_scope(
                observation.bundle_scope, element.element_kind
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


__all__ = ["SCHEMA_DIR", "SchemaProvider", "SchemaRegistry", "canonical_schema_provider", "schema_subject_diagnostics"]
