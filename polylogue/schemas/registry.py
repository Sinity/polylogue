"""Versioned schema registry for provider export formats."""

from __future__ import annotations

import copy
import gzip
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.lib.provider_identity import (
    canonical_schema_provider as _canonical_schema_provider,
    normalize_provider_token,
)
from polylogue.paths import data_home
from polylogue.schemas.sampling import (
    _resolve_provider_config,
    extract_schema_units_from_payload,
    fingerprint_hash as _stable_fingerprint_hash,
    profile_similarity,
    schema_cluster_id,
)
from polylogue.types import Provider

SCHEMA_DIR = Path(__file__).parent / "providers"

type SchemaProvider = Provider | str


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
    schema_version: str | None = None

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
    def from_dict(cls, data: dict[str, Any]) -> "ClusterManifest":
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
                    schema_version=cluster.get("schema_version"),
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
    def __init__(self, storage_root: Path | None = None):
        self._storage_root = storage_root

    @property
    def storage_root(self) -> Path:
        return self._storage_root if self._storage_root is not None else data_home() / "schemas"

    def get_schema(self, provider: str | Provider, version: str = "latest") -> dict[str, Any] | None:
        provider_token = canonical_schema_provider(provider)
        if version == "latest":
            manifest = self.load_cluster_manifest(provider_token)
            if manifest is not None and manifest.default_version is not None:
                version = manifest.default_version
            else:
                versions = self.list_versions(provider_token)
                if versions:
                    version = versions[-1]
                else:
                    return self._load_baseline(provider_token)
        schema = self._load_versioned(provider_token, version)
        if schema is not None:
            return schema
        if version == "v1":
            return self._load_baseline(provider_token)
        return None

    def list_versions(self, provider: str | Provider) -> list[str]:
        provider_token = canonical_schema_provider(provider)
        provider_dir = self.storage_root / str(provider_token)
        if not provider_dir.exists():
            return ["v1"] if self._baseline_exists(provider_token) else []
        versions = [path.name.split(".")[0] for path in provider_dir.glob("v*.schema.json.gz")]
        if "v1" not in versions and self._baseline_exists(provider_token):
            versions.append("v1")
        return sorted(versions, key=lambda value: int(value[1:]))

    def list_providers(self) -> list[str]:
        providers: set[str] = set()
        for pattern in ("*.schema.json.gz", "*.schema.json"):
            for path in SCHEMA_DIR.glob(pattern):
                providers.add(path.name.replace(".schema.json.gz", "").replace(".schema.json", ""))
        if self.storage_root.exists():
            for path in self.storage_root.iterdir():
                if path.is_dir() and any(path.glob("v*.schema.json.gz")):
                    providers.add(path.name)
        return sorted(providers)

    def register_schema(self, provider: str | Provider, schema: dict[str, Any]) -> str:
        provider_token = canonical_schema_provider(provider)
        versions = self.list_versions(provider_token)
        new_version = f"v{int(versions[-1][1:]) + 1}" if versions else "v1"
        self.write_schema_version(provider_token, new_version, schema)
        return new_version

    def write_schema_version(self, provider: str | Provider, version: str, schema: dict[str, Any]) -> Path:
        provider_token = canonical_schema_provider(provider)
        schema_copy = copy.deepcopy(schema)
        schema_copy["$id"] = f"polylogue://schemas/{provider_token}/{version}"
        schema_copy["x-polylogue-version"] = int(version[1:])
        schema_copy["x-polylogue-registered-at"] = datetime.now(tz=timezone.utc).isoformat()
        provider_dir = self.storage_root / str(provider_token)
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = provider_dir / f"{version}.schema.json.gz"
        path.write_bytes(gzip.compress(json.dumps(schema_copy, indent=2).encode("utf-8")))
        return path

    def replace_provider_schemas(
        self,
        provider: str | Provider,
        versioned_schemas: list[tuple[str, dict[str, Any]]],
        *,
        manifest: ClusterManifest | None = None,
    ) -> Path | None:
        provider_token = canonical_schema_provider(provider)
        provider_dir = self.storage_root / str(provider_token)
        provider_dir.mkdir(parents=True, exist_ok=True)
        for path in provider_dir.glob("v*.schema.json.gz"):
            path.unlink()
        manifest_path = provider_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        for version, schema in versioned_schemas:
            self.write_schema_version(provider_token, version, schema)
        if manifest is not None:
            return self.save_cluster_manifest(manifest)
        return None

    def compare_versions(self, provider: str | Provider, v1: str, v2: str) -> SchemaDiff:
        provider_token = canonical_schema_provider(provider)
        schema_a = self.get_schema(provider_token, version=v1)
        schema_b = self.get_schema(provider_token, version=v2)
        if schema_a is None or schema_b is None:
            raise ValueError(f"Schema not found for {provider_token}: {v1}, {v2}")
        return self._diff_schemas(provider_token, v1, v2, schema_a, schema_b)

    def get_schema_age_days(self, provider: str | Provider) -> int | None:
        schema = self.get_schema(provider, version="latest")
        if schema is None:
            return None
        generated_at = schema.get("x-polylogue-generated-at")
        if not generated_at:
            return None
        try:
            delta = datetime.now(tz=timezone.utc) - datetime.fromisoformat(generated_at)
        except (ValueError, TypeError):
            return None
        return delta.days

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

    def match_payload_version(
        self,
        provider: str | Provider,
        payload: Any,
        *,
        source_path: str | None = None,
    ) -> str | None:
        provider_token = canonical_schema_provider(provider)
        manifest = self.load_cluster_manifest(provider_token)
        if manifest is None:
            versions = self.list_versions(provider_token)
            return versions[-1] if versions else None

        config = _resolve_provider_config(provider_token)
        units = extract_schema_units_from_payload(
            payload,
            provider_name=provider_token,
            source_path=source_path,
            raw_id=None,
            config=config,
            max_samples=64,
        )
        if not units:
            return manifest.default_version

        scores: dict[str, float] = {}
        promoted_clusters = [cluster for cluster in manifest.clusters if cluster.schema_version is not None]
        exact_clusters = {cluster.cluster_id: cluster for cluster in promoted_clusters if not cluster.profile_tokens}
        for unit in units:
            exact_cluster = exact_clusters.get(schema_cluster_id(unit.cluster_payload, unit.artifact_kind))
            if exact_cluster is not None:
                scores[exact_cluster.cluster_id] = scores.get(exact_cluster.cluster_id, 0.0) + 1.0
                continue
            best_cluster: SchemaCluster | None = None
            best_score = 0.0
            for cluster in promoted_clusters:
                if cluster.artifact_kind != unit.artifact_kind or not cluster.profile_tokens:
                    continue
                score = profile_similarity(set(cluster.profile_tokens), set(unit.profile_tokens))
                if score > best_score:
                    best_cluster = cluster
                    best_score = score
            if best_cluster is not None:
                scores[best_cluster.cluster_id] = scores.get(best_cluster.cluster_id, 0.0) + best_score

        if not scores:
            return manifest.default_version

        best_cluster_id = max(
            scores.items(),
            key=lambda item: (
                item[1],
                next((cluster.sample_count for cluster in manifest.clusters if cluster.cluster_id == item[0]), 0),
            ),
        )[0]
        return next(
            (cluster.schema_version for cluster in manifest.clusters if cluster.cluster_id == best_cluster_id),
            manifest.default_version,
        )

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
        if target_cluster.schema_version is not None:
            raise ValueError(f"Cluster {cluster_id} already promoted as {target_cluster.schema_version}")

        if samples:
            from polylogue.schemas.schema_generation import generate_schema_from_samples
            schema = generate_schema_from_samples(samples)
        else:
            schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "title": f"{provider_token} export format (cluster {cluster_id})",
                "properties": {key: {} for key in target_cluster.dominant_keys},
            }

        schema["x-polylogue-cluster-id"] = cluster_id
        schema["x-polylogue-cluster-sample-count"] = target_cluster.sample_count
        schema["x-polylogue-cluster-confidence"] = target_cluster.confidence
        schema["x-polylogue-promoted-at"] = datetime.now(tz=timezone.utc).isoformat()

        new_version = self.register_schema(provider_token, schema)
        target_cluster.schema_version = new_version
        if manifest.default_version is None:
            manifest.default_version = new_version
        self.save_cluster_manifest(manifest)
        return new_version

    def _baseline_path(self, provider: SchemaProvider) -> Path:
        return SCHEMA_DIR / f"{provider}.schema.json.gz"

    def _baseline_path_plain(self, provider: SchemaProvider) -> Path:
        return SCHEMA_DIR / f"{provider}.schema.json"

    def _baseline_exists(self, provider: SchemaProvider) -> bool:
        return self._baseline_path(provider).exists() or self._baseline_path_plain(provider).exists()

    def _load_baseline(self, provider: SchemaProvider) -> dict[str, Any] | None:
        gz_path = self._baseline_path(provider)
        if gz_path.exists():
            return json.loads(gzip.decompress(gz_path.read_bytes()).decode("utf-8"))
        plain_path = self._baseline_path_plain(provider)
        if plain_path.exists():
            return json.loads(plain_path.read_text(encoding="utf-8"))
        return None

    def _load_versioned(self, provider: SchemaProvider, version: str) -> dict[str, Any] | None:
        path = self.storage_root / str(provider) / f"{version}.schema.json.gz"
        if not path.exists():
            return None
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))

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
                if val_a is None and val_b is not None:
                    detail = f"{annotation_key} added"
                elif val_a is not None and val_b is None:
                    detail = f"{annotation_key} removed"
                else:
                    detail = f"{annotation_key} changed"
                classified.append(PropertyChange(path="$", kind="relational", detail=detail))

        return SchemaDiff(
            provider=provider,
            version_a=v1,
            version_b=v2,
            added_properties=added,
            removed_properties=removed,
            changed_properties=sorted(changed),
            classified_changes=classified,
        )
