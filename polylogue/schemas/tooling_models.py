"""Typed schema-tooling models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from polylogue.schemas.json_types import JSONDocument, json_document, json_document_list
from polylogue.schemas.runtime_registry import SchemaProvider, canonical_schema_provider


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_dict(value: object) -> dict[str, int]:
    return {str(key): int(item) for key, item in json_document(value).items() if isinstance(item, (str, int, float))}


def _int_scalar(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, (str, int, float)) and not isinstance(value, bool) else default


def _float_scalar(value: object, default: float = 1.0) -> float:
    return float(value) if isinstance(value, (str, int, float)) and not isinstance(value, bool) else default


@dataclass
class PropertyChange:
    path: str
    kind: str
    detail: str

    def to_dict(self) -> JSONDocument:
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

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
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
        )

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

    def to_dict(self) -> JSONDocument:
        data = asdict(self)
        data["provider"] = str(self.provider)
        return json_document(data)


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

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "provider": str(self.provider),
                "generated_at": self.generated_at,
                "cluster_count": len(self.clusters),
                "artifact_counts": self.artifact_counts,
                "default_version": self.default_version,
                "clusters": [cluster.to_dict() for cluster in self.clusters],
            }
        )

    @classmethod
    def from_dict(cls, data: JSONDocument) -> ClusterManifest:
        cluster_payloads = json_document_list(data.get("clusters"))
        return cls(
            provider=canonical_schema_provider(str(data["provider"])),
            clusters=[
                SchemaCluster(
                    cluster_id=str(cluster["cluster_id"]),
                    provider=canonical_schema_provider(str(cluster["provider"])),
                    sample_count=_int_scalar(cluster.get("sample_count")),
                    first_seen=str(cluster["first_seen"]),
                    last_seen=str(cluster["last_seen"]),
                    representative_paths=_string_list(cluster.get("representative_paths")),
                    dominant_keys=_string_list(cluster.get("dominant_keys")),
                    confidence=_float_scalar(cluster.get("confidence")),
                    artifact_kind=str(cluster.get("artifact_kind", "unspecified")),
                    profile_tokens=_string_list(cluster.get("profile_tokens")),
                    exact_structure_ids=_string_list(cluster.get("exact_structure_ids")),
                    bundle_scope_count=_int_scalar(cluster.get("bundle_scope_count")),
                    promoted_package_version=(
                        str(cluster["promoted_package_version"])
                        if isinstance(cluster.get("promoted_package_version"), str)
                        else None
                    ),
                )
                for cluster in cluster_payloads
            ],
            generated_at=str(data.get("generated_at", "")),
            artifact_counts=_int_dict(data.get("artifact_counts")),
            default_version=str(data["default_version"]) if isinstance(data.get("default_version"), str) else None,
        )


__all__ = [
    "ClusterManifest",
    "PropertyChange",
    "SchemaCluster",
    "SchemaDiff",
]
