"""Versioned schema registry for provider export formats.

The registry manages two tiers of schemas:
  - **Baseline schemas**: Shipped in-package under ``providers/*.schema.json.gz``
  - **Versioned schemas**: Generated at runtime, stored under
    ``DATA_HOME/schemas/{provider}/v{N}.schema.json.gz``

Extended with schema clustering and promotion:
  - **Cluster manifests**: Group samples by structural fingerprint
  - **Promotion**: Elevate a cluster's representative schema to a new version

Usage::

    registry = SchemaRegistry()
    schema = registry.get_schema("chatgpt")             # latest
    schema = registry.get_schema("chatgpt", version="v1")  # specific version
    diff = registry.compare_versions("chatgpt", "v1", "v2")

    # Clustering
    manifest = registry.cluster_samples("chatgpt", samples)
    registry.save_cluster_manifest(manifest)
    version = registry.promote_cluster("chatgpt", cluster_id)
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.lib.provider_identity import (
    canonical_schema_provider as _canonical_schema_provider,
)
from polylogue.paths import data_home

# In-package baseline schemas (canonical definition -- imported by validator, synthetic)
SCHEMA_DIR = Path(__file__).parent / "providers"


def canonical_schema_provider(provider: str) -> str:
    """Normalize provider names to canonical schema identifiers."""
    return _canonical_schema_provider(provider, preserve_unknown=True, default=provider)


# =============================================================================
# Change Classification
# =============================================================================


@dataclass
class PropertyChange:
    """A single classified change between two schema versions."""

    path: str
    kind: str  # "added" | "removed" | "type_mutation" | "requiredness" | "semantic_role" | "relational"
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "detail": self.detail}


@dataclass
class SchemaDiff:
    """Difference between two schema versions with classified changes."""

    provider: str
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
        parts = []
        if self.added_properties:
            parts.append(f"+{len(self.added_properties)} properties")
        if self.removed_properties:
            parts.append(f"-{len(self.removed_properties)} properties")
        if self.changed_properties:
            parts.append(f"~{len(self.changed_properties)} changed")
        return ", ".join(parts) if parts else "no changes"

    # --- Output methods ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "provider": self.provider,
            "version_a": self.version_a,
            "version_b": self.version_b,
            "summary": self.summary(),
            "has_changes": self.has_changes,
            "added_properties": self.added_properties,
            "removed_properties": self.removed_properties,
            "changed_properties": self.changed_properties,
            "classified_changes": [c.to_dict() for c in self.classified_changes],
        }

    def to_text(self) -> str:
        """Render as plain text report."""
        lines: list[str] = []
        lines.append(f"Schema diff: {self.provider} {self.version_a} -> {self.version_b}")
        lines.append(f"Summary: {self.summary()}")
        lines.append("")

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
                    for c in changes:
                        lines.append(f"    {c.path}: {c.detail}")
                    lines.append("")
        elif self.has_changes:
            if self.added_properties:
                lines.append("  Added:")
                for p in self.added_properties:
                    lines.append(f"    + {p}")
            if self.removed_properties:
                lines.append("  Removed:")
                for p in self.removed_properties:
                    lines.append(f"    - {p}")
            if self.changed_properties:
                lines.append("  Changed:")
                for p in self.changed_properties:
                    lines.append(f"    ~ {p}")
        else:
            lines.append("  No changes detected.")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render as Markdown report."""
        lines: list[str] = []
        lines.append(f"# Schema Diff: {self.provider}")
        lines.append(f"**{self.version_a}** -> **{self.version_b}**")
        lines.append("")
        lines.append(f"**Summary:** {self.summary()}")
        lines.append("")

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
                    for c in changes:
                        lines.append(f"| `{c.path}` | {c.detail} |")
                    lines.append("")
        elif self.has_changes:
            if self.added_properties:
                lines.append("## Added Properties")
                lines.append("")
                for p in self.added_properties:
                    lines.append(f"- `{p}`")
                lines.append("")
            if self.removed_properties:
                lines.append("## Removed Properties")
                lines.append("")
                for p in self.removed_properties:
                    lines.append(f"- `{p}`")
                lines.append("")
            if self.changed_properties:
                lines.append("## Changed Properties")
                lines.append("")
                for p in self.changed_properties:
                    lines.append(f"- `{p}`")
                lines.append("")
        else:
            lines.append("No changes detected.")

        return "\n".join(lines)


# =============================================================================
# Schema Clustering
# =============================================================================


@dataclass
class SchemaCluster:
    """A group of samples that share the same structural fingerprint."""

    cluster_id: str  # content hash of the structural fingerprint
    provider: str
    sample_count: int
    first_seen: str  # ISO timestamp
    last_seen: str
    representative_paths: list[str] = field(default_factory=list)  # example source file paths
    dominant_keys: list[str] = field(default_factory=list)  # most common top-level keys
    confidence: float = 1.0
    promoted_version: str | None = None  # "v2" if promoted

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterManifest:
    """Per-provider manifest of schema clusters."""

    provider: str
    clusters: list[SchemaCluster] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "generated_at": self.generated_at,
            "cluster_count": len(self.clusters),
            "clusters": [c.to_dict() for c in self.clusters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterManifest:
        """Deserialize from a JSON-compatible dict."""
        clusters = [SchemaCluster(**c) for c in data.get("clusters", [])]
        return cls(
            provider=data["provider"],
            clusters=clusters,
            generated_at=data.get("generated_at", ""),
        )


def _fingerprint_hash(fingerprint: Any) -> str:
    """Compute a stable content hash of a structural fingerprint."""
    raw = repr(fingerprint).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class SchemaRegistry:
    """Registry of versioned provider schemas with clustering and promotion."""

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
        canonical_provider = canonical_schema_provider(provider)
        if version == "latest":
            versions = self.list_versions(canonical_provider)
            if versions:
                version = versions[-1]  # sorted, last is latest
            else:
                # Fall back to baseline
                return self._load_baseline(canonical_provider)

        # Try versioned storage first
        schema = self._load_versioned(canonical_provider, version)
        if schema is not None:
            return schema

        # Fall back to baseline for v1 (or if only baseline exists)
        if version == "v1":
            return self._load_baseline(canonical_provider)

        return None

    def list_versions(self, provider: str) -> list[str]:
        """List all versions for a provider, sorted by version number.

        Returns:
            Sorted list of version strings like ["v1", "v2", "v3"].
        """
        canonical_provider = canonical_schema_provider(provider)
        provider_dir = self.storage_root / canonical_provider
        if not provider_dir.exists():
            # Check if baseline exists
            if self._baseline_exists(canonical_provider):
                return ["v1"]
            return []

        versions: list[str] = []
        for p in provider_dir.glob("v*.schema.json.gz"):
            v = p.name.split(".")[0]  # "v2.schema.json.gz" -> "v2"
            versions.append(v)

        # Always include v1 if baseline exists
        if "v1" not in versions and self._baseline_exists(canonical_provider):
            versions.append("v1")

        return sorted(versions, key=lambda v: int(v[1:]))

    def list_providers(self) -> list[str]:
        """List all providers with at least one schema (baseline or versioned)."""
        providers: set[str] = set()

        # Baseline providers
        for pattern in ("*.schema.json.gz", "*.schema.json"):
            for p in SCHEMA_DIR.glob(pattern):
                providers.add(p.name.replace(".schema.json.gz", "").replace(".schema.json", ""))

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
        canonical_provider = canonical_schema_provider(provider)
        versions = self.list_versions(canonical_provider)
        if versions:
            last_num = int(versions[-1][1:])
            new_version = f"v{last_num + 1}"
        else:
            new_version = "v1"

        # Inject metadata
        schema["$id"] = f"polylogue://schemas/{canonical_provider}/{new_version}"
        schema["x-polylogue-version"] = int(new_version[1:])
        schema["x-polylogue-registered-at"] = datetime.now(tz=timezone.utc).isoformat()

        # Write to storage
        provider_dir = self.storage_root / canonical_provider
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
            SchemaDiff with added/removed/changed properties and classified changes.

        Raises:
            ValueError: If either version doesn't exist.
        """
        canonical_provider = canonical_schema_provider(provider)
        schema_a = self.get_schema(canonical_provider, version=v1)
        schema_b = self.get_schema(canonical_provider, version=v2)

        if schema_a is None:
            raise ValueError(f"Schema not found: {canonical_provider} {v1}")
        if schema_b is None:
            raise ValueError(f"Schema not found: {canonical_provider} {v2}")

        return self._diff_schemas(canonical_provider, v1, v2, schema_a, schema_b)

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

    # --- Clustering ---

    def cluster_samples(
        self,
        provider: str,
        samples: list[dict[str, Any]],
        *,
        source_paths: list[str] | None = None,
    ) -> ClusterManifest:
        """Group samples by structural fingerprint into clusters.

        Args:
            provider: Provider name
            samples: Raw data samples to cluster
            source_paths: Optional parallel list of source file paths for each sample

        Returns:
            ClusterManifest with one SchemaCluster per unique fingerprint.
        """
        from polylogue.schemas.schema_generation import _structure_fingerprint

        canonical_provider = canonical_schema_provider(provider)
        now = datetime.now(tz=timezone.utc).isoformat()

        # Group samples by fingerprint
        groups: dict[str, list[int]] = {}  # fingerprint_hash -> sample indices
        fingerprint_map: dict[str, Any] = {}  # hash -> raw fingerprint (for debugging)

        for i, sample in enumerate(samples):
            fp = _structure_fingerprint(sample)
            fp_hash = _fingerprint_hash(fp)
            groups.setdefault(fp_hash, []).append(i)
            fingerprint_map[fp_hash] = fp

        # Build clusters
        clusters: list[SchemaCluster] = []
        for fp_hash, indices in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            # Collect top-level keys from the first sample in the group
            representative_sample = samples[indices[0]]
            dominant_keys = sorted(representative_sample.keys()) if isinstance(representative_sample, dict) else []

            # Collect representative source paths
            rep_paths: list[str] = []
            if source_paths:
                seen: set[str] = set()
                for idx in indices[:5]:
                    if idx < len(source_paths) and source_paths[idx] not in seen:
                        rep_paths.append(source_paths[idx])
                        seen.add(source_paths[idx])

            # Confidence: larger clusters are more confident
            confidence = min(1.0, len(indices) / max(len(samples) * 0.1, 1))

            clusters.append(SchemaCluster(
                cluster_id=fp_hash,
                provider=canonical_provider,
                sample_count=len(indices),
                first_seen=now,
                last_seen=now,
                representative_paths=rep_paths,
                dominant_keys=dominant_keys[:20],
                confidence=round(confidence, 3),
            ))

        return ClusterManifest(
            provider=canonical_provider,
            clusters=clusters,
            generated_at=now,
        )

    def save_cluster_manifest(self, manifest: ClusterManifest) -> Path:
        """Write a cluster manifest to schemas/<provider>/manifest.json.

        Args:
            manifest: The ClusterManifest to persist.

        Returns:
            Path to the written manifest file.
        """
        canonical_provider = canonical_schema_provider(manifest.provider)
        provider_dir = self.storage_root / canonical_provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        path = provider_dir / "manifest.json"
        path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return path

    def load_cluster_manifest(self, provider: str) -> ClusterManifest | None:
        """Load a previously saved cluster manifest.

        Args:
            provider: Provider name

        Returns:
            ClusterManifest, or None if no manifest exists.
        """
        canonical_provider = canonical_schema_provider(provider)
        path = self.storage_root / canonical_provider / "manifest.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ClusterManifest.from_dict(data)

    def promote_cluster(
        self,
        provider: str,
        cluster_id: str,
        *,
        samples: list[dict[str, Any]] | None = None,
    ) -> str:
        """Promote a cluster's schema to a new version.

        Generates a schema from the cluster's samples (if provided) or from
        an existing manifest, then registers it as the next version.

        Args:
            provider: Provider name
            cluster_id: The cluster_id to promote
            samples: Optional samples belonging to this cluster. If not provided,
                     the promotion uses the manifest metadata to create a minimal
                     schema stub.

        Returns:
            Version string assigned (e.g., "v3")

        Raises:
            ValueError: If the cluster is not found in the manifest.
        """
        canonical_provider = canonical_schema_provider(provider)
        manifest = self.load_cluster_manifest(canonical_provider)
        if manifest is None:
            raise ValueError(f"No cluster manifest found for provider: {canonical_provider}")

        target_cluster: SchemaCluster | None = None
        for cluster in manifest.clusters:
            if cluster.cluster_id == cluster_id:
                target_cluster = cluster
                break

        if target_cluster is None:
            raise ValueError(
                f"Cluster {cluster_id} not found in manifest for {canonical_provider}. "
                f"Available: {[c.cluster_id for c in manifest.clusters]}"
            )

        if target_cluster.promoted_version is not None:
            raise ValueError(
                f"Cluster {cluster_id} already promoted as {target_cluster.promoted_version}"
            )

        # Generate schema from samples if provided
        if samples:
            from polylogue.schemas.schema_generation import generate_schema_from_samples

            schema = generate_schema_from_samples(samples)
        else:
            # Create a minimal schema from cluster metadata
            schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "title": f"{canonical_provider} export format (cluster {cluster_id})",
                "properties": {key: {} for key in target_cluster.dominant_keys},
            }

        # Add cluster provenance
        schema["x-polylogue-cluster-id"] = cluster_id
        schema["x-polylogue-cluster-sample-count"] = target_cluster.sample_count
        schema["x-polylogue-cluster-confidence"] = target_cluster.confidence
        schema["x-polylogue-promoted-at"] = datetime.now(tz=timezone.utc).isoformat()

        # Register as next version
        new_version = self.register_schema(canonical_provider, schema)

        # Update manifest to record promotion
        target_cluster.promoted_version = new_version
        self.save_cluster_manifest(manifest)

        return new_version

    # --- Internal helpers ---

    def _baseline_path(self, provider: str) -> Path:
        return SCHEMA_DIR / f"{provider}.schema.json.gz"

    def _baseline_path_plain(self, provider: str) -> Path:
        return SCHEMA_DIR / f"{provider}.schema.json"

    def _baseline_exists(self, provider: str) -> bool:
        return self._baseline_path(provider).exists() or self._baseline_path_plain(provider).exists()

    def _load_baseline(self, provider: str) -> dict[str, Any] | None:
        gz_path = self._baseline_path(provider)
        if gz_path.exists():
            return json.loads(gzip.decompress(gz_path.read_bytes()).decode("utf-8"))

        plain_path = self._baseline_path_plain(provider)
        if plain_path.exists():
            return json.loads(plain_path.read_text(encoding="utf-8"))

        return None

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
        """Compare top-level and nested properties between two schemas.

        Produces both the legacy added/removed/changed lists and the new
        classified_changes list with fine-grained change types.
        """
        props_a = set(schema_a.get("properties", {}).keys())
        props_b = set(schema_b.get("properties", {}).keys())

        added = sorted(props_b - props_a)
        removed = sorted(props_a - props_b)

        # Check for type changes in common properties
        changed: list[str] = []
        classified: list[PropertyChange] = []
        common = props_a & props_b

        # Classify additive changes
        for prop in added:
            classified.append(PropertyChange(
                path=prop,
                kind="added",
                detail=f"new property (type: {_type_label(schema_b['properties'][prop])})",
            ))

        # Classify subtractive changes
        for prop in removed:
            classified.append(PropertyChange(
                path=prop,
                kind="removed",
                detail=f"removed property (was type: {_type_label(schema_a['properties'][prop])})",
            ))

        # Classify changes in common properties
        req_a = set(schema_a.get("required", []))
        req_b = set(schema_b.get("required", []))

        for prop in sorted(common):
            prop_a = schema_a["properties"][prop]
            prop_b = schema_b["properties"][prop]
            type_a = prop_a.get("type")
            type_b = prop_b.get("type")

            if type_a != type_b:
                changed.append(prop)
                classified.append(PropertyChange(
                    path=prop,
                    kind="type_mutation",
                    detail=f"type changed: {type_a} -> {type_b}",
                ))

            # Requiredness changes
            was_required = prop in req_a
            is_required = prop in req_b
            if was_required != is_required:
                classified.append(PropertyChange(
                    path=prop,
                    kind="requiredness",
                    detail=f"{'required' if is_required else 'optional'} (was {'required' if was_required else 'optional'})",
                ))

            # Semantic annotation changes
            role_a = prop_a.get("x-polylogue-semantic-role")
            role_b = prop_b.get("x-polylogue-semantic-role")
            if role_a != role_b:
                classified.append(PropertyChange(
                    path=prop,
                    kind="semantic_role",
                    detail=f"semantic role changed: {role_a!r} -> {role_b!r}",
                ))

            # Relational annotation changes (x-polylogue-ref)
            ref_a = prop_a.get("x-polylogue-ref")
            ref_b = prop_b.get("x-polylogue-ref")
            if ref_a != ref_b:
                classified.append(PropertyChange(
                    path=prop,
                    kind="relational",
                    detail=f"reference changed: {ref_a!r} -> {ref_b!r}",
                ))

        # Schema-level relational annotation changes
        for annotation_key in (
            "x-polylogue-foreign-keys",
            "x-polylogue-time-deltas",
            "x-polylogue-mutually-exclusive",
        ):
            val_a = schema_a.get(annotation_key)
            val_b = schema_b.get(annotation_key)
            if val_a != val_b:
                if val_a is None and val_b is not None:
                    classified.append(PropertyChange(
                        path="$",
                        kind="relational",
                        detail=f"{annotation_key} added",
                    ))
                elif val_a is not None and val_b is None:
                    classified.append(PropertyChange(
                        path="$",
                        kind="relational",
                        detail=f"{annotation_key} removed",
                    ))
                else:
                    classified.append(PropertyChange(
                        path="$",
                        kind="relational",
                        detail=f"{annotation_key} changed",
                    ))

        return SchemaDiff(
            provider=provider,
            version_a=v1,
            version_b=v2,
            added_properties=added,
            removed_properties=removed,
            changed_properties=changed,
            classified_changes=classified,
        )


def _type_label(prop_schema: dict[str, Any]) -> str:
    """Extract a human-readable type label from a property schema."""
    t = prop_schema.get("type")
    if t:
        return str(t)
    if "anyOf" in prop_schema:
        types = [s.get("type", "?") for s in prop_schema["anyOf"]]
        return " | ".join(str(t) for t in types)
    if "oneOf" in prop_schema:
        types = [s.get("type", "?") for s in prop_schema["oneOf"]]
        return " | ".join(str(t) for t in types)
    return "unknown"
