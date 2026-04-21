"""Typed result models for artifact-proof and corpus-verification workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, TypedDict

CountPayload: TypeAlias = dict[str, int]
AllLiteral: TypeAlias = Literal["all"]
LimitPayload: TypeAlias = int | AllLiteral
ArtifactProofCountKind: TypeAlias = Literal[
    "artifact_counts",
    "package_versions",
    "element_kinds",
    "resolution_reasons",
]


class ProviderSchemaVerificationPayload(TypedDict):
    provider: str
    total_records: int
    valid_records: int
    invalid_records: int
    drift_records: int
    skipped_no_schema: int
    decode_errors: int
    quarantined_records: int


class SchemaVerificationReportPayload(TypedDict):
    max_samples: LimitPayload
    record_limit: LimitPayload
    record_offset: int
    total_records: int
    providers: dict[str, ProviderSchemaVerificationPayload]


class ProviderArtifactProofPayload(TypedDict):
    provider: str
    total_records: int
    contract_backed_records: int
    unsupported_parseable_records: int
    recognized_non_parseable_records: int
    unknown_records: int
    decode_errors: int
    artifact_counts: CountPayload
    package_versions: CountPayload
    element_kinds: CountPayload
    resolution_reasons: CountPayload
    linked_sidecars: int
    orphan_sidecars: int
    subagent_streams: int
    streams_with_sidecars: int
    sidecar_agent_types: CountPayload


class ArtifactProofSummaryPayload(TypedDict):
    contract_backed_records: int
    unsupported_parseable_records: int
    recognized_non_parseable_records: int
    unknown_records: int
    decode_errors: int
    linked_sidecars: int
    orphan_sidecars: int
    subagent_streams: int
    streams_with_sidecars: int
    artifact_counts: CountPayload
    package_versions: CountPayload
    element_kinds: CountPayload
    resolution_reasons: CountPayload
    clean: bool


class ArtifactProofReportPayload(TypedDict):
    record_limit: LimitPayload
    record_offset: int
    total_records: int
    summary: ArtifactProofSummaryPayload
    providers: dict[str, ProviderArtifactProofPayload]


def _limit_payload(value: int | None) -> LimitPayload:
    return value if value is not None else "all"


def _sorted_counts(counts: Mapping[str, int]) -> CountPayload:
    return dict(sorted(counts.items()))


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

    def to_dict(self) -> ProviderSchemaVerificationPayload:
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

    def to_dict(self) -> SchemaVerificationReportPayload:
        return {
            "max_samples": _limit_payload(self.max_samples),
            "record_limit": _limit_payload(self.record_limit),
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
    artifact_counts: CountPayload = field(default_factory=dict)
    package_versions: CountPayload = field(default_factory=dict)
    element_kinds: CountPayload = field(default_factory=dict)
    resolution_reasons: CountPayload = field(default_factory=dict)
    linked_sidecars: int = 0
    orphan_sidecars: int = 0
    subagent_streams: int = 0
    streams_with_sidecars: int = 0
    sidecar_agent_types: CountPayload = field(default_factory=dict)

    def to_dict(self) -> ProviderArtifactProofPayload:
        return {
            "provider": self.provider,
            "total_records": self.total_records,
            "contract_backed_records": self.contract_backed_records,
            "unsupported_parseable_records": self.unsupported_parseable_records,
            "recognized_non_parseable_records": self.recognized_non_parseable_records,
            "unknown_records": self.unknown_records,
            "decode_errors": self.decode_errors,
            "artifact_counts": _sorted_counts(self.artifact_counts),
            "package_versions": _sorted_counts(self.package_versions),
            "element_kinds": _sorted_counts(self.element_kinds),
            "resolution_reasons": _sorted_counts(self.resolution_reasons),
            "linked_sidecars": self.linked_sidecars,
            "orphan_sidecars": self.orphan_sidecars,
            "subagent_streams": self.subagent_streams,
            "streams_with_sidecars": self.streams_with_sidecars,
            "sidecar_agent_types": _sorted_counts(self.sidecar_agent_types),
        }


def _provider_artifact_counts(stats: ProviderArtifactProof, *, kind: ArtifactProofCountKind) -> Mapping[str, int]:
    if kind == "artifact_counts":
        return stats.artifact_counts
    if kind == "package_versions":
        return stats.package_versions
    if kind == "element_kinds":
        return stats.element_kinds
    return stats.resolution_reasons


def _aggregate_counts(providers: Mapping[str, ProviderArtifactProof], *, kind: ArtifactProofCountKind) -> CountPayload:
    aggregated: CountPayload = {}
    for stats in providers.values():
        for key, count in _provider_artifact_counts(stats, kind=kind).items():
            aggregated[key] = aggregated.get(key, 0) + count
    return _sorted_counts(aggregated)


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
    def artifact_counts(self) -> CountPayload:
        return _aggregate_counts(self.providers, kind="artifact_counts")

    @property
    def package_versions(self) -> CountPayload:
        return _aggregate_counts(self.providers, kind="package_versions")

    @property
    def element_kinds(self) -> CountPayload:
        return _aggregate_counts(self.providers, kind="element_kinds")

    @property
    def resolution_reasons(self) -> CountPayload:
        return _aggregate_counts(self.providers, kind="resolution_reasons")

    @property
    def is_clean(self) -> bool:
        return self.unsupported_parseable_records == 0 and self.unknown_records == 0 and self.decode_errors == 0

    def to_dict(self) -> ArtifactProofReportPayload:
        return {
            "record_limit": _limit_payload(self.record_limit),
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
