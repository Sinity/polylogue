"""Typed result models for artifact-proof and corpus-verification workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderSchemaVerification:
    """Per-provider schema verification summary."""

    provider: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    drift_records: int = 0
    skipped_no_schema: int = 0
    decode_errors: int = 0
    quarantined_records: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "provider": self.provider,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "drift_records": self.drift_records,
            "skipped_no_schema": self.skipped_no_schema,
            "decode_errors": self.decode_errors,
            "quarantined_records": self.quarantined_records,
        }


@dataclass
class SchemaVerificationReport:
    """Aggregate report for schema verification over raw corpus."""

    providers: dict[str, ProviderSchemaVerification]
    max_samples: int | None
    total_records: int
    record_limit: int | None = None
    record_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_samples": self.max_samples if self.max_samples is not None else "all",
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "total_records": self.total_records,
            "providers": {provider: stats.to_dict() for provider, stats in sorted(self.providers.items())},
        }


@dataclass
class ProviderArtifactProof:
    """Per-provider proof of raw artifact support and linkage."""

    provider: str
    total_records: int = 0
    contract_backed_records: int = 0
    unsupported_parseable_records: int = 0
    recognized_non_parseable_records: int = 0
    unknown_records: int = 0
    decode_errors: int = 0
    artifact_counts: dict[str, int] = field(default_factory=dict)
    package_versions: dict[str, int] = field(default_factory=dict)
    element_kinds: dict[str, int] = field(default_factory=dict)
    resolution_reasons: dict[str, int] = field(default_factory=dict)
    linked_sidecars: int = 0
    orphan_sidecars: int = 0
    subagent_streams: int = 0
    streams_with_sidecars: int = 0
    sidecar_agent_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "total_records": self.total_records,
            "contract_backed_records": self.contract_backed_records,
            "unsupported_parseable_records": self.unsupported_parseable_records,
            "recognized_non_parseable_records": self.recognized_non_parseable_records,
            "unknown_records": self.unknown_records,
            "decode_errors": self.decode_errors,
            "artifact_counts": dict(sorted(self.artifact_counts.items())),
            "package_versions": dict(sorted(self.package_versions.items())),
            "element_kinds": dict(sorted(self.element_kinds.items())),
            "resolution_reasons": dict(sorted(self.resolution_reasons.items())),
            "linked_sidecars": self.linked_sidecars,
            "orphan_sidecars": self.orphan_sidecars,
            "subagent_streams": self.subagent_streams,
            "streams_with_sidecars": self.streams_with_sidecars,
            "sidecar_agent_types": dict(sorted(self.sidecar_agent_types.items())),
        }


@dataclass
class ArtifactProofReport:
    """Aggregate proof report over the raw artifact corpus."""

    providers: dict[str, ProviderArtifactProof]
    total_records: int
    record_limit: int | None = None
    record_offset: int = 0

    @property
    def contract_backed_records(self) -> int:
        return sum(stats.contract_backed_records for stats in self.providers.values())

    @property
    def unsupported_parseable_records(self) -> int:
        return sum(stats.unsupported_parseable_records for stats in self.providers.values())

    @property
    def recognized_non_parseable_records(self) -> int:
        return sum(stats.recognized_non_parseable_records for stats in self.providers.values())

    @property
    def unknown_records(self) -> int:
        return sum(stats.unknown_records for stats in self.providers.values())

    @property
    def decode_errors(self) -> int:
        return sum(stats.decode_errors for stats in self.providers.values())

    @property
    def linked_sidecars(self) -> int:
        return sum(stats.linked_sidecars for stats in self.providers.values())

    @property
    def orphan_sidecars(self) -> int:
        return sum(stats.orphan_sidecars for stats in self.providers.values())

    @property
    def subagent_streams(self) -> int:
        return sum(stats.subagent_streams for stats in self.providers.values())

    @property
    def streams_with_sidecars(self) -> int:
        return sum(stats.streams_with_sidecars for stats in self.providers.values())

    @property
    def artifact_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stats in self.providers.values():
            for kind, count in stats.artifact_counts.items():
                counts[kind] = counts.get(kind, 0) + count
        return dict(sorted(counts.items()))

    @property
    def package_versions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stats in self.providers.values():
            for version, count in stats.package_versions.items():
                counts[version] = counts.get(version, 0) + count
        return dict(sorted(counts.items()))

    @property
    def element_kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stats in self.providers.values():
            for element_kind, count in stats.element_kinds.items():
                counts[element_kind] = counts.get(element_kind, 0) + count
        return dict(sorted(counts.items()))

    @property
    def resolution_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stats in self.providers.values():
            for reason, count in stats.resolution_reasons.items():
                counts[reason] = counts.get(reason, 0) + count
        return dict(sorted(counts.items()))

    @property
    def is_clean(self) -> bool:
        return self.unsupported_parseable_records == 0 and self.unknown_records == 0 and self.decode_errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_limit": self.record_limit if self.record_limit is not None else "all",
            "record_offset": self.record_offset,
            "total_records": self.total_records,
            "summary": {
                "contract_backed_records": self.contract_backed_records,
                "unsupported_parseable_records": self.unsupported_parseable_records,
                "recognized_non_parseable_records": self.recognized_non_parseable_records,
                "unknown_records": self.unknown_records,
                "decode_errors": self.decode_errors,
                "linked_sidecars": self.linked_sidecars,
                "orphan_sidecars": self.orphan_sidecars,
                "subagent_streams": self.subagent_streams,
                "streams_with_sidecars": self.streams_with_sidecars,
                "artifact_counts": self.artifact_counts,
                "package_versions": self.package_versions,
                "element_kinds": self.element_kinds,
                "resolution_reasons": self.resolution_reasons,
                "clean": self.is_clean,
            },
            "providers": {provider: stats.to_dict() for provider, stats in sorted(self.providers.items())},
        }


__all__ = [
    "ArtifactProofReport",
    "ProviderArtifactProof",
    "ProviderSchemaVerification",
    "SchemaVerificationReport",
]
