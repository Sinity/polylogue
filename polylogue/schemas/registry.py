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

from polylogue.lib.provider_identity import (
    canonical_schema_provider as _canonical_schema_provider,
)
from polylogue.lib.provider_identity import (
    normalize_provider_token,
)
from polylogue.paths import data_home
from polylogue.schemas.packages import (
    SchemaElementManifest,
    SchemaPackageCatalog,
    SchemaResolution,
    SchemaVersionPackage,
)
from polylogue.schemas.sampling import (
    _resolve_provider_config,
    derive_bundle_scope,
    extract_schema_units_from_payload,
    profile_similarity,
    schema_cluster_id,
)
from polylogue.schemas.sampling import (
    fingerprint_hash as _stable_fingerprint_hash,
)
from polylogue.types import Provider

# In-package baseline schemas (canonical definition — imported by validator, synthetic)
SCHEMA_DIR = Path(__file__).parent / "providers"

SchemaProvider = Provider | str

def canonical_schema_provider(provider: str) -> str:
    """Normalize provider names to canonical schema identifiers."""
    return _canonical_schema_provider(provider, preserve_unknown=True, default=provider)


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": str(self.provider),
            "version_a": self.version_a,
            "version_b": self.version_b,
            "summary": self.summary(),
            "has_changes": self.has_changes,
            "added_properties": self.added_properties,
            "removed_properties": self.removed_properties,
            "changed_properties": self.changed_properties,
            "classified_changes": [change.to_dict() for change in self.classified_changes],
        }

    def to_text(self) -> str:
        lines = [
            f"Schema diff: {self.provider} {self.version_a} -> {self.version_b}",
            f"Summary: {self.summary()}",
            "",
        ]
        if self.classified_changes:
            by_kind: dict[str, list[PropertyChange]] = {}
            for change in self.classified_changes:
                by_kind.setdefault(change.kind, []).append(change)

            kind_labels = {
                "added": "Additive (new properties)",
                "removed": "Subtractive (removed properties)",
                "type_mutation": "Type mutations",
                "requiredness": "Requiredness changes",
                "semantic_role": "Semantic annotation changes",
                "relational": "Relational annotation changes",
            }
            for kind, label in kind_labels.items():
                changes = by_kind.get(kind, [])
                if changes:
                    lines.append(f"  {label}:")
                    for change in changes:
                        lines.append(f"    {change.path}: {change.detail}")
                    lines.append("")
        elif self.has_changes:
            if self.added_properties:
                lines.append("  Added:")
                for prop in self.added_properties:
                    lines.append(f"    + {prop}")
            if self.removed_properties:
                lines.append("  Removed:")
                for prop in self.removed_properties:
                    lines.append(f"    - {prop}")
            if self.changed_properties:
                lines.append("  Changed:")
                for prop in self.changed_properties:
                    lines.append(f"    ~ {prop}")
        else:
            lines.append("  No changes detected.")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            f"# Schema Diff: {self.provider}",
            f"**{self.version_a}** -> **{self.version_b}**",
            "",
            f"**Summary:** {self.summary()}",
            "",
        ]
        if self.classified_changes:
            by_kind: dict[str, list[PropertyChange]] = {}
            for change in self.classified_changes:
                by_kind.setdefault(change.kind, []).append(change)

            kind_labels = {
                "added": "Additive Changes (new properties)",
                "removed": "Subtractive Changes (removed properties)",
                "type_mutation": "Type Mutations",
                "requiredness": "Requiredness Changes",
                "semantic_role": "Semantic Annotation Changes",
                "relational": "Relational Annotation Changes",
            }
            for kind, label in kind_labels.items():
                changes = by_kind.get(kind, [])
                if changes:
                    lines.append(f"## {label}")
                    lines.append("")
                    lines.append("| Path | Detail |")
                    lines.append("|------|--------|")
                    for change in changes:
                        lines.append(f"| `{change.path}` | {change.detail} |")
                    lines.append("")
        elif self.has_changes:
            if self.added_properties:
                lines.append("## Added Properties")
                lines.append("")
                for prop in self.added_properties:
                    lines.append(f"- `{prop}`")
                lines.append("")
            if self.removed_properties:
                lines.append("## Removed Properties")
                lines.append("")
                for prop in self.removed_properties:
                    lines.append(f"- `{prop}`")
                lines.append("")
            if self.changed_properties:
                lines.append("## Changed Properties")
                lines.append("")
                for prop in self.changed_properties:
                    lines.append(f"- `{prop}`")
                lines.append("")
        else:
            lines.append("No changes detected.")
        return "\n".join(lines)


@dataclass
class SchemaCluster:
    cluster_id: str
    provider: SchemaProvider
    sample_count: int
    first_seen: str
    last_seen: str
    representative_paths: list[str] = field(default_factory=list)
    dominant_keys: list[str] = field(default_factory=list)
    confidence: float = 1.0
    artifact_kind: str = "unspecified"
    profile_tokens: list[str] = field(default_factory=list)
    exact_structure_ids: list[str] = field(default_factory=list)
    bundle_scope_count: int = 0
    promoted_package_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provider"] = str(self.provider)
        return data


@dataclass
class ClusterManifest:
    provider: SchemaProvider
    clusters: list[SchemaCluster] = field(default_factory=list)
    generated_at: str = ""
    artifact_counts: dict[str, int] = field(default_factory=dict)
    default_version: str | None = None

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": str(self.provider),
            "generated_at": self.generated_at,
            "cluster_count": len(self.clusters),
            "artifact_counts": self.artifact_counts,
            "default_version": self.default_version,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterManifest:
        return cls(
            provider=canonical_schema_provider(data["provider"]),
            clusters=[
                SchemaCluster(
                    cluster_id=cluster["cluster_id"],
                    provider=canonical_schema_provider(cluster["provider"]),
                    sample_count=int(cluster["sample_count"]),
                    first_seen=cluster["first_seen"],
                    last_seen=cluster["last_seen"],
                    representative_paths=list(cluster.get("representative_paths", [])),
                    dominant_keys=list(cluster.get("dominant_keys", [])),
                    confidence=float(cluster.get("confidence", 1.0)),
                    artifact_kind=str(cluster.get("artifact_kind", "unspecified")),
                    profile_tokens=list(cluster.get("profile_tokens", [])),
                    exact_structure_ids=list(cluster.get("exact_structure_ids", [])),
                    bundle_scope_count=int(cluster.get("bundle_scope_count", 0)),
                    promoted_package_version=cluster.get("promoted_package_version"),
                )
                for cluster in data.get("clusters", [])
            ],
            generated_at=data.get("generated_at", ""),
            artifact_counts={str(key): int(value) for key, value in data.get("artifact_counts", {}).items()},
            default_version=data.get("default_version"),
        )


def _fingerprint_hash(fingerprint: Any) -> str:
    return _stable_fingerprint_hash(fingerprint)


def _dominant_keys(sample: Any) -> list[str]:
    if isinstance(sample, dict):
        return sorted(sample.keys())
    if isinstance(sample, list):
        first_dict = next((item for item in sample if isinstance(item, dict)), None)
        if first_dict is not None:
            return sorted(first_dict.keys())
    return []


def _type_label(schema: dict[str, Any]) -> str:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " | ".join(str(item) for item in schema_type)
    return str(schema_type)


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

    def _bundled_provider_dir(self, provider: str | Provider) -> Path:
        return SCHEMA_DIR / str(canonical_schema_provider(provider))

    def _provider_search_roots(self, provider: str | Provider) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()
        for candidate in (
            self._provider_dir(provider),
            self._bundled_provider_dir(provider),
        ):
            if candidate in seen:
                continue
            seen.add(candidate)
            roots.append(candidate)
        return roots

    def _catalog_path(self, provider: str | Provider) -> Path:
        return self._provider_dir(provider) / "catalog.json"

        Args:
            provider: Provider name (e.g., "chatgpt", "claude-ai")
            version: Version string ("v1", "v2", ...) or "latest"

    def _package_manifest_path(self, provider: str | Provider, version: str) -> Path:
        return self._package_dir(provider, version) / "package.json"

    def _element_schema_path(self, provider: str | Provider, version: str, element_kind: str) -> Path:
        return self._package_dir(provider, version) / "elements" / f"{element_kind}.schema.json.gz"

    def _catalog_path_in(self, provider_dir: Path) -> Path:
        return provider_dir / "catalog.json"

    def _provider_dir_for_catalog(self, provider: str | Provider) -> Path | None:
        for provider_dir in self._provider_search_roots(provider):
            if self._catalog_path_in(provider_dir).exists():
                return provider_dir
        return None

    def _provider_dir_for_package(self, provider: str | Provider, version: str) -> Path | None:
        for provider_dir in self._provider_search_roots(provider):
            manifest_path = provider_dir / "versions" / version / "package.json"
            if manifest_path.exists():
                return provider_dir
        return None

    def load_package_catalog(self, provider: str | Provider) -> SchemaPackageCatalog | None:
        provider_dir = self._provider_dir_for_catalog(provider)
        if provider_dir is None:
            return None
        path = self._catalog_path_in(provider_dir)
        return SchemaPackageCatalog.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_package_catalog(self, catalog: SchemaPackageCatalog) -> Path:
        provider_dir = self._provider_dir(catalog.provider)
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = self._catalog_path(catalog.provider)
        path.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
        return path

    def _resolve_catalog_version(self, catalog: SchemaPackageCatalog, version: str) -> str | None:
        if version == "default":
            return catalog.default_version or catalog.latest_version or catalog.recommended_version
        if version == "latest":
            versions = self.list_versions(canonical_provider)
            if versions:
                version = versions[-1]  # sorted, last is latest
            else:
                # Fall back to baseline
                return self._load_baseline(canonical_provider)

    def write_package(
        self,
        package: SchemaVersionPackage,
        *,
        element_schemas: dict[str, dict[str, Any]],
    ) -> Path:
        provider_token = canonical_schema_provider(package.provider)
        package_dir = self._package_dir(provider_token, package.version)
        elements_dir = package_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)

        for element in package.elements:
            if element.schema_file is None:
                continue
            schema = copy.deepcopy(element_schemas[element.element_kind])
            schema["$id"] = (
                f"polylogue://schemas/{provider_token}/{package.version}/{element.element_kind}"
            )
            schema["x-polylogue-version"] = int(package.version[1:]) if package.version.startswith("v") else package.version
            schema["x-polylogue-package-version"] = package.version
            schema["x-polylogue-element-kind"] = element.element_kind
            schema["x-polylogue-registered-at"] = datetime.now(tz=timezone.utc).isoformat()
            schema_path = elements_dir / element.schema_file
            schema_path.write_bytes(gzip.compress(json.dumps(schema, indent=2).encode("utf-8")))

        manifest_path = self._package_manifest_path(provider_token, package.version)
        manifest_path.write_text(json.dumps(package.to_dict(), indent=2), encoding="utf-8")
        return manifest_path

    def replace_provider_packages(
        self,
        provider: str | Provider,
        catalog: SchemaPackageCatalog,
        package_schemas: dict[str, dict[str, dict[str, Any]]],
        *,
        manifest: ClusterManifest | None = None,
    ) -> None:
        provider_token = canonical_schema_provider(provider)
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
        if manifest is not None:
            self.save_cluster_manifest(manifest)

    def get_package(self, provider: str | Provider, version: str = "default") -> SchemaVersionPackage | None:
        provider_token = canonical_schema_provider(provider)
        catalog = self.load_package_catalog(provider_token)
        if catalog is not None:
            resolved_version = self._resolve_catalog_version(catalog, version)
            if resolved_version:
                package = catalog.package(resolved_version)
                if package is not None:
                    return package
        return None

    def get_element_schema(
        self,
        provider: str | Provider,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> dict[str, Any] | None:
        provider_token = canonical_schema_provider(provider)
        package = self.get_package(provider_token, version=version)
        if package is not None:
            element = package.element(element_kind)
            if element is None or element.schema_file is None:
                return None
            provider_dir = self._provider_dir_for_package(provider_token, package.version)
            if provider_dir is None:
                return None
            path = provider_dir / "versions" / package.version / "elements" / element.schema_file
            if path.exists():
                return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        return None

    def list_versions(self, provider: str) -> list[str]:
        """List all versions for a provider, sorted by version number.

    def list_versions(self, provider: str | Provider) -> list[str]:
        provider_token = canonical_schema_provider(provider)
        catalog = self.load_package_catalog(provider_token)
        if catalog is not None:
            return sorted((package.version for package in catalog.packages), key=lambda value: int(value[1:]))
        return []

    def list_providers(self) -> list[str]:
        """List all providers with at least one schema (baseline or versioned)."""
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

    def _single_element_package(
        self,
        provider: str | Provider,
        *,
        version: str,
        schema: dict[str, Any],
        element_kind: str = "conversation_document",
        first_seen: str | None = None,
        last_seen: str | None = None,
    ) -> tuple[SchemaVersionPackage, dict[str, dict[str, Any]]]:
        provider_token = canonical_schema_provider(provider)
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

        return new_version

    # --- Comparison ---

    def compare_versions(self, provider: str, v1: str, v2: str) -> SchemaDiff:
        """Compare two schema versions for a provider.

        Args:
            provider: Provider name
            v1: First version (e.g., "v1")
            v2: Second version (e.g., "v2")

    def get_schema_age_days(self, provider: str | Provider) -> int | None:
        package = self.get_package(provider, version="latest")
        if package is not None:
            try:
                delta = datetime.now(tz=timezone.utc) - datetime.fromisoformat(package.last_seen)
            except (ValueError, TypeError):
                return None
            return delta.days
        return None

    # --- Internal helpers ---

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
            artifact_counts[cluster.artifact_kind] = artifact_counts.get(cluster.artifact_kind, 0) + cluster.sample_count
        return ClusterManifest(provider=provider_token, clusters=clusters, generated_at=now, artifact_counts=artifact_counts)

    def save_cluster_manifest(self, manifest: ClusterManifest) -> Path:
        provider_dir = self.storage_root / str(canonical_schema_provider(manifest.provider))
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = provider_dir / "manifest.json"
        path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_cluster_manifest(self, provider: str | Provider) -> ClusterManifest | None:
        provider_token = canonical_schema_provider(provider)
        path = self.storage_root / str(provider_token) / "manifest.json"
        if not path.exists():
            return None
        return ClusterManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def resolve_payload(
        self,
        provider: str | Provider,
        payload: Any,
        *,
        source_path: str | None = None,
    ) -> SchemaResolution | None:
        provider_token = canonical_schema_provider(provider)
        catalog = self.load_package_catalog(provider_token)
        if catalog is None or not catalog.packages:
            return None

        config = _resolve_provider_config(provider_token)
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
                provider=str(provider_token),
                package_version=default_package.version,
                element_kind=default_package.default_element_kind,
                exact_structure_id=None,
                bundle_scope=derive_bundle_scope(provider_token, source_path),
                reason="package_default",
            )

        unit = units[0]
        bundle_scope = unit.bundle_scope or derive_bundle_scope(provider_token, source_path)

        if bundle_scope is not None:
            for package in catalog.packages:
                if bundle_scope not in package.bundle_scopes:
                    continue
                element = package.element(unit.artifact_kind)
                if element is None:
                    continue
                return SchemaResolution(
                    provider=str(provider_token),
                    package_version=package.version,
                    element_kind=element.element_kind,
                    exact_structure_id=unit.exact_structure_id,
                    bundle_scope=bundle_scope,
                    reason="bundle_scope",
                )

        for package in catalog.packages:
            element = package.element(unit.artifact_kind)
            if element is None:
                continue
            if unit.exact_structure_id in element.exact_structure_ids:
                return SchemaResolution(
                    provider=str(provider_token),
                    package_version=package.version,
                    element_kind=element.element_kind,
                    exact_structure_id=unit.exact_structure_id,
                    bundle_scope=bundle_scope,
                    reason="exact_structure",
                )

        best_match: tuple[float, SchemaVersionPackage, SchemaElementManifest] | None = None
        for package in catalog.packages:
            element = package.element(unit.artifact_kind)
            if element is None or not element.profile_tokens:
                continue
            score = profile_similarity(set(element.profile_tokens), set(unit.profile_tokens))
            if best_match is None or score > best_match[0]:
                best_match = (score, package, element)
        if best_match is not None and best_match[0] > 0.0:
            _score, package, element = best_match
            return SchemaResolution(
                provider=str(provider_token),
                package_version=package.version,
                element_kind=element.element_kind,
                exact_structure_id=unit.exact_structure_id,
                bundle_scope=bundle_scope,
                reason="profile_family",
            )

        default_package = self.get_package(provider_token, version="default")
        if default_package is None:
            return None
        return SchemaResolution(
            provider=str(provider_token),
            package_version=default_package.version,
            element_kind=default_package.default_element_kind,
            exact_structure_id=unit.exact_structure_id,
            bundle_scope=bundle_scope,
            reason="package_default",
        )

    def match_payload_version(
        self,
        provider: str | Provider,
        payload: Any,
        *,
        source_path: str | None = None,
    ) -> str | None:
        resolution = self.resolve_payload(provider, payload, source_path=source_path)
        return resolution.package_version if resolution is not None else None

    def promote_cluster(
        self,
        provider: str | Provider,
        cluster_id: str,
        *,
        samples: list[dict[str, Any]] | None = None,
    ) -> str:
        provider_token = canonical_schema_provider(provider)
        manifest = self.load_cluster_manifest(provider_token)
        if manifest is None:
            raise ValueError(f"No cluster manifest found for provider: {provider_token}")
        target_cluster = next((cluster for cluster in manifest.clusters if cluster.cluster_id == cluster_id), None)
        if target_cluster is None:
            raise ValueError(f"Cluster {cluster_id} not found for {provider_token}")
        if target_cluster.promoted_package_version is not None:
            raise ValueError(
                f"Cluster {cluster_id} already promoted as {target_cluster.promoted_package_version}"
            )

        if samples:
            from polylogue.schemas.schema_generation import generate_schema_from_samples
            schema = generate_schema_from_samples(samples)
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
