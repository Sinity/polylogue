"""Schema tooling layer: evidence manifests, diffs, clustering, and promotion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from polylogue.schemas.observation import schema_cluster_id
from polylogue.schemas.packages import SchemaPackageCatalog
from polylogue.schemas.runtime_registry import SchemaProvider, canonical_schema_provider
from polylogue.types import Provider


@dataclass
class PropertyChange:
    path: str
    kind: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "detail": self.detail}


@dataclass
class SchemaDiff:
    provider: SchemaProvider
    version_a: str
    version_b: str
    added_properties: list[str] = field(default_factory=list)
    removed_properties: list[str] = field(default_factory=list)
    changed_properties: list[str] = field(default_factory=list)
    classified_changes: list[PropertyChange] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added_properties or self.removed_properties or self.changed_properties)

    def summary(self) -> str:
        parts: list[str] = []
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


class SchemaRegistryToolingMixin:
    """Tooling mixin layered on top of the runtime registry base."""

    def replace_provider_schemas(
        self,
        provider: str | Provider,
        versioned_schemas: list[tuple[str, dict[str, Any]]],
        *,
        manifest: ClusterManifest | None = None,
    ):
        provider_token = canonical_schema_provider(provider)
        packages = []
        package_schemas: dict[str, dict[str, dict[str, Any]]] = {}
        for version, schema in versioned_schemas:
            package, schemas = self._single_element_package(provider_token, version=version, schema=schema.copy())
            packages.append(package)
            package_schemas[version] = schemas
        catalog = SchemaPackageCatalog(
            provider=provider_token,
            packages=sorted(packages, key=lambda item: int(item.version[1:])),
            latest_version=packages[-1].version if packages else None,
            default_version=(manifest.default_version if manifest is not None else (packages[-1].version if packages else None)),
            recommended_version=(manifest.default_version if manifest is not None else (packages[-1].version if packages else None)),
        )
        self.replace_provider_packages(provider_token, catalog, package_schemas)
        if manifest is not None:
            self.save_cluster_manifest(manifest)
        return self._catalog_path(provider_token)

    def compare_versions(
        self,
        provider: str | Provider,
        v1: str,
        v2: str,
        *,
        element_kind: str | None = None,
    ) -> SchemaDiff:
        provider_token = canonical_schema_provider(provider)
        schema_a = self.get_element_schema(provider_token, version=v1, element_kind=element_kind)
        schema_b = self.get_element_schema(provider_token, version=v2, element_kind=element_kind)
        if schema_a is None or schema_b is None:
            raise ValueError(
                f"Schema not found for {provider_token}: {v1}, {v2}"
                + (f", element={element_kind}" if element_kind else "")
            )
        return self._diff_schemas(provider_token, v1, v2, schema_a, schema_b)

    def cluster_samples(
        self,
        provider: str | Provider,
        samples: list[Any],
        *,
        source_paths: list[str] | None = None,
        artifact_kinds: list[str] | None = None,
    ) -> ClusterManifest:
        provider_token = canonical_schema_provider(provider)
        now = datetime.now(tz=timezone.utc).isoformat()
        groups: dict[str, list[int]] = {}
        artifact_by_cluster: dict[str, str] = {}
        for index, sample in enumerate(samples):
            artifact_kind = artifact_kinds[index] if artifact_kinds is not None and index < len(artifact_kinds) else "unspecified"
            cluster_id = schema_cluster_id(sample, artifact_kind)
            groups.setdefault(cluster_id, []).append(index)
            artifact_by_cluster[cluster_id] = artifact_kind

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

    def save_cluster_manifest(self, manifest: ClusterManifest):
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
            raise ValueError(f"Cluster {cluster_id} already promoted as {target_cluster.promoted_package_version}")

        if samples:
            from polylogue.schemas.generation_workflow import generate_schema_from_samples

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
        provider: SchemaProvider,
        v1: str,
        v2: str,
        schema_a: dict[str, Any],
        schema_b: dict[str, Any],
    ) -> SchemaDiff:
        props_a = set(schema_a.get("properties", {}).keys())
        props_b = set(schema_b.get("properties", {}).keys())
        added = sorted(props_b - props_a)
        removed = sorted(props_a - props_b)
        changed: list[str] = []
        classified: list[PropertyChange] = []

        for prop in added:
            classified.append(PropertyChange(path=prop, kind="added", detail=f"new property (type: {_type_label(schema_b['properties'][prop])})"))
        for prop in removed:
            classified.append(PropertyChange(path=prop, kind="removed", detail=f"removed property (was type: {_type_label(schema_a['properties'][prop])})"))

        req_a = set(schema_a.get("required", []))
        req_b = set(schema_b.get("required", []))
        for prop in sorted(props_a & props_b):
            prop_a = schema_a["properties"][prop]
            prop_b = schema_b["properties"][prop]
            if prop_a.get("type") != prop_b.get("type"):
                changed.append(prop)
                classified.append(PropertyChange(path=prop, kind="type_mutation", detail=f"type changed: {prop_a.get('type')} -> {prop_b.get('type')}"))
            if (prop in req_a) != (prop in req_b):
                classified.append(PropertyChange(
                    path=prop,
                    kind="requiredness",
                    detail=f"{'required' if prop in req_b else 'optional'} (was {'required' if prop in req_a else 'optional'})",
                ))
            if prop_a.get("x-polylogue-semantic-role") != prop_b.get("x-polylogue-semantic-role"):
                classified.append(PropertyChange(
                    path=prop,
                    kind="semantic_role",
                    detail=f"semantic role changed: {prop_a.get('x-polylogue-semantic-role')!r} -> {prop_b.get('x-polylogue-semantic-role')!r}",
                ))
            if prop_a.get("x-polylogue-ref") != prop_b.get("x-polylogue-ref"):
                classified.append(PropertyChange(
                    path=prop,
                    kind="relational",
                    detail=f"reference changed: {prop_a.get('x-polylogue-ref')!r} -> {prop_b.get('x-polylogue-ref')!r}",
                ))

        for annotation_key in (
            "x-polylogue-foreign-keys",
            "x-polylogue-time-deltas",
            "x-polylogue-mutually-exclusive",
        ):
            val_a = schema_a.get(annotation_key)
            val_b = schema_b.get(annotation_key)
            if val_a != val_b:
                if val_a is None:
                    detail = f"{annotation_key} added"
                elif val_b is None:
                    detail = f"{annotation_key} removed"
                else:
                    detail = f"{annotation_key} changed"
                classified.append(PropertyChange(
                    path="$",
                    kind="relational",
                    detail=detail,
                ))

        return SchemaDiff(
            provider=provider,
            version_a=v1,
            version_b=v2,
            added_properties=added,
            removed_properties=removed,
            changed_properties=sorted(set(changed)),
            classified_changes=classified,
        )


__all__ = [
    "ClusterManifest",
    "PropertyChange",
    "SchemaCluster",
    "SchemaDiff",
    "SchemaRegistryToolingMixin",
]
