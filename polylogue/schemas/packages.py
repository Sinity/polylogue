"""Package-aware schema manifest models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from polylogue.types import Provider


def _provider_value(provider: str | Provider) -> str:
    provider_token = Provider.from_string(provider)
    if provider_token is Provider.UNKNOWN and isinstance(provider, str):
        return provider
    return str(provider_token)


@dataclass
class SchemaElementManifest:
    element_kind: str
    schema_file: str | None
    sample_count: int
    artifact_count: int
    supported: bool = True
    exact_structure_ids: list[str] = field(default_factory=list)
    profile_family_ids: list[str] = field(default_factory=list)
    profile_tokens: list[str] = field(default_factory=list)
    representative_paths: list[str] = field(default_factory=list)
    observed_artifact_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaElementManifest":
        return cls(
            element_kind=str(data["element_kind"]),
            schema_file=data.get("schema_file"),
            sample_count=int(data.get("sample_count", 0)),
            artifact_count=int(data.get("artifact_count", 0)),
            supported=bool(data.get("supported", True)),
            exact_structure_ids=[str(item) for item in data.get("exact_structure_ids", [])],
            profile_family_ids=[str(item) for item in data.get("profile_family_ids", [])],
            profile_tokens=[str(item) for item in data.get("profile_tokens", [])],
            representative_paths=[str(item) for item in data.get("representative_paths", [])],
            observed_artifact_count=int(data.get("observed_artifact_count", 0)),
        )


@dataclass
class SchemaVersionPackage:
    provider: str | Provider
    version: str
    anchor_kind: str
    default_element_kind: str
    first_seen: str
    last_seen: str
    bundle_scope_count: int
    sample_count: int
    bundle_scopes: list[str] = field(default_factory=list)
    source_cluster_ids: list[str] = field(default_factory=list)
    representative_paths: list[str] = field(default_factory=list)
    elements: list[SchemaElementManifest] = field(default_factory=list)
    orphan_adjunct_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": _provider_value(self.provider),
            "version": self.version,
            "anchor_kind": self.anchor_kind,
            "default_element_kind": self.default_element_kind,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "bundle_scope_count": self.bundle_scope_count,
            "sample_count": self.sample_count,
            "bundle_scopes": self.bundle_scopes,
            "source_cluster_ids": self.source_cluster_ids,
            "representative_paths": self.representative_paths,
            "elements": [element.to_dict() for element in self.elements],
            "orphan_adjunct_counts": self.orphan_adjunct_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaVersionPackage":
        return cls(
            provider=_provider_value(data["provider"]),
            version=str(data["version"]),
            anchor_kind=str(data["anchor_kind"]),
            default_element_kind=str(data.get("default_element_kind", data["anchor_kind"])),
            first_seen=str(data["first_seen"]),
            last_seen=str(data["last_seen"]),
            bundle_scope_count=int(data.get("bundle_scope_count", 0)),
            sample_count=int(data.get("sample_count", 0)),
            bundle_scopes=[str(item) for item in data.get("bundle_scopes", [])],
            source_cluster_ids=[str(item) for item in data.get("source_cluster_ids", [])],
            representative_paths=[str(item) for item in data.get("representative_paths", [])],
            elements=[SchemaElementManifest.from_dict(item) for item in data.get("elements", [])],
            orphan_adjunct_counts={
                str(key): int(value) for key, value in data.get("orphan_adjunct_counts", {}).items()
            },
        )

    def element(self, element_kind: str | None = None) -> SchemaElementManifest | None:
        target = element_kind or self.default_element_kind
        return next((item for item in self.elements if item.element_kind == target), None)


@dataclass
class SchemaPackageCatalog:
    provider: str | Provider
    packages: list[SchemaVersionPackage] = field(default_factory=list)
    generated_at: str = ""
    latest_version: str | None = None
    default_version: str | None = None
    recommended_version: str | None = None
    orphan_adjunct_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": _provider_value(self.provider),
            "generated_at": self.generated_at,
            "latest_version": self.latest_version,
            "default_version": self.default_version,
            "recommended_version": self.recommended_version,
            "orphan_adjunct_counts": self.orphan_adjunct_counts,
            "packages": [package.to_dict() for package in self.packages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaPackageCatalog":
        return cls(
            provider=_provider_value(data["provider"]),
            generated_at=str(data.get("generated_at", "")),
            latest_version=data.get("latest_version"),
            default_version=data.get("default_version"),
            recommended_version=data.get("recommended_version"),
            orphan_adjunct_counts={
                str(key): int(value) for key, value in data.get("orphan_adjunct_counts", {}).items()
            },
            packages=[SchemaVersionPackage.from_dict(item) for item in data.get("packages", [])],
        )

    def package(self, version: str) -> SchemaVersionPackage | None:
        return next((item for item in self.packages if item.version == version), None)


@dataclass(frozen=True)
class SchemaResolution:
    provider: str
    package_version: str
    element_kind: str
    exact_structure_id: str | None
    bundle_scope: str | None
    reason: str


__all__ = [
    "SchemaElementManifest",
    "SchemaPackageCatalog",
    "SchemaResolution",
    "SchemaVersionPackage",
]
