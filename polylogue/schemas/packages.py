"""Package-aware schema manifest and resolution models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, TypeAlias

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, json_document, json_document_list


def _provider_value(provider: str | Provider) -> str:
    provider_token = Provider.from_string(provider)
    if provider_token is Provider.UNKNOWN and isinstance(provider, str):
        return provider
    return str(provider_token)


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_value(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, (str, int, float)) else default


def _string_int_dict(value: object) -> dict[str, int]:
    return {str(key): _int_value(item) for key, item in json_document(value).items()}


SchemaResolutionReason: TypeAlias = Literal[
    "bundle_scope",
    "exact_structure",
    "package_default",
    "profile_family",
]


@dataclass
class SchemaElementManifest:
    element_kind: str
    schema_file: str | None
    sample_count: int
    artifact_count: int
    supported: bool = True
    first_seen: str = ""
    last_seen: str = ""
    bundle_scope_count: int = 0
    bundle_scope_identities: list[str] = field(default_factory=list)
    exact_structure_ids: list[str] = field(default_factory=list)
    profile_family_ids: list[str] = field(default_factory=list)
    profile_tokens: list[str] = field(default_factory=list)
    observed_artifact_count: int = 0

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "element_kind": self.element_kind,
                "schema_file": self.schema_file,
                "sample_count": self.sample_count,
                "artifact_count": self.artifact_count,
                "supported": self.supported,
                "first_seen": self.first_seen,
                "last_seen": self.last_seen,
                "bundle_scope_count": self.bundle_scope_count,
                "bundle_scope_identities": self.bundle_scope_identities,
                "exact_structure_ids": self.exact_structure_ids,
                "profile_family_ids": self.profile_family_ids,
                "profile_tokens": self.profile_tokens,
                "observed_artifact_count": self.observed_artifact_count,
            }
        )

    @classmethod
    def from_dict(cls, data: JSONDocument) -> SchemaElementManifest:
        return cls(
            element_kind=str(data["element_kind"]),
            schema_file=_string_or_none(data.get("schema_file")),
            sample_count=_int_value(data.get("sample_count")),
            artifact_count=_int_value(data.get("artifact_count")),
            supported=bool(data.get("supported", True)),
            first_seen=str(data.get("first_seen", "")),
            last_seen=str(data.get("last_seen", "")),
            bundle_scope_count=_int_value(data.get("bundle_scope_count")),
            bundle_scope_identities=_string_list(data.get("bundle_scope_identities")),
            exact_structure_ids=_string_list(data.get("exact_structure_ids")),
            profile_family_ids=_string_list(data.get("profile_family_ids")),
            profile_tokens=_string_list(data.get("profile_tokens")),
            observed_artifact_count=_int_value(data.get("observed_artifact_count")),
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
    anchor_profile_family_id: str = ""
    bundle_scope_identities: list[str] = field(default_factory=list)
    profile_family_ids: list[str] = field(default_factory=list)
    elements: list[SchemaElementManifest] = field(default_factory=list)
    orphan_adjunct_counts: dict[str, int] = field(default_factory=dict)
    workload_profile_file: str | None = None

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "provider": _provider_value(self.provider),
                "version": self.version,
                "anchor_kind": self.anchor_kind,
                "default_element_kind": self.default_element_kind,
                "first_seen": self.first_seen,
                "last_seen": self.last_seen,
                "bundle_scope_count": self.bundle_scope_count,
                "sample_count": self.sample_count,
                "anchor_profile_family_id": self.anchor_profile_family_id,
                "bundle_scope_identities": self.bundle_scope_identities,
                "profile_family_ids": self.profile_family_ids,
                "elements": [element.to_dict() for element in self.elements],
                "orphan_adjunct_counts": self.orphan_adjunct_counts,
                "workload_profile_file": self.workload_profile_file,
            }
        )

    @classmethod
    def from_dict(cls, data: JSONDocument) -> SchemaVersionPackage:
        return cls(
            provider=_provider_value(str(data["provider"])),
            version=str(data["version"]),
            anchor_kind=str(data["anchor_kind"]),
            default_element_kind=str(data.get("default_element_kind", data["anchor_kind"])),
            first_seen=str(data["first_seen"]),
            last_seen=str(data["last_seen"]),
            bundle_scope_count=_int_value(data.get("bundle_scope_count")),
            sample_count=_int_value(data.get("sample_count")),
            anchor_profile_family_id=str(data.get("anchor_profile_family_id", "")),
            bundle_scope_identities=_string_list(data.get("bundle_scope_identities")),
            profile_family_ids=_string_list(data.get("profile_family_ids")),
            elements=[SchemaElementManifest.from_dict(item) for item in json_document_list(data.get("elements"))],
            orphan_adjunct_counts=_string_int_dict(data.get("orphan_adjunct_counts")),
            workload_profile_file=_string_or_none(data.get("workload_profile_file")),
        )

    def element(self, element_kind: str | None = None) -> SchemaElementManifest | None:
        target = element_kind or self.default_element_kind
        return next((item for item in self.elements if item.element_kind == target), None)

    def matches_bundle_scope(self, scope: str, element_kind: str | None = None) -> bool:
        from polylogue.schemas.observation_identity import bundle_scope_identity

        element = self.element(element_kind)
        identities = set(self.bundle_scope_identities)
        if element is not None:
            identities.update(element.bundle_scope_identities)
        return bundle_scope_identity(scope) in identities


@dataclass
class SchemaPackageCatalog:
    provider: str | Provider
    packages: list[SchemaVersionPackage] = field(default_factory=list)
    generated_at: str = ""
    latest_version: str | None = None
    default_version: str | None = None
    recommended_version: str | None = None
    orphan_adjunct_counts: dict[str, int] = field(default_factory=dict)
    selection_rationale: JSONDocument = field(default_factory=dict)
    observation_outcomes: JSONDocument = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> JSONDocument:
        data = json_document(
            {
                "provider": _provider_value(self.provider),
                "generated_at": self.generated_at,
                "latest_version": self.latest_version,
                "default_version": self.default_version,
                "recommended_version": self.recommended_version,
                "orphan_adjunct_counts": self.orphan_adjunct_counts,
                "packages": [package.to_dict() for package in self.packages],
            }
        )
        if self.selection_rationale:
            data["selection_rationale"] = self.selection_rationale
        if self.observation_outcomes:
            data["observation_outcomes"] = self.observation_outcomes
        return json_document(data)

    @classmethod
    def from_dict(cls, data: JSONDocument) -> SchemaPackageCatalog:
        return cls(
            provider=_provider_value(str(data["provider"])),
            generated_at=str(data.get("generated_at", "")),
            latest_version=_string_or_none(data.get("latest_version")),
            default_version=_string_or_none(data.get("default_version")),
            recommended_version=_string_or_none(data.get("recommended_version")),
            orphan_adjunct_counts=_string_int_dict(data.get("orphan_adjunct_counts")),
            selection_rationale=json_document(data.get("selection_rationale")),
            observation_outcomes=json_document(data.get("observation_outcomes")),
            packages=[SchemaVersionPackage.from_dict(item) for item in json_document_list(data.get("packages"))],
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
    reason: SchemaResolutionReason
    profile_score: float | None = None


__all__ = [
    "SchemaElementManifest",
    "SchemaPackageCatalog",
    "SchemaResolution",
    "SchemaResolutionReason",
    "SchemaVersionPackage",
]
