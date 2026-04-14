"""Canonical typed state models for the acquire/validate/parse/prepare pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polylogue.types import Provider, ValidationStatus

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedConversation


@dataclass(slots=True)
class AcquireResult:
    """Result of an acquisition operation."""

    acquired: int = 0
    skipped: int = 0
    errors: int = 0
    raw_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "acquired": self.acquired,
            "skipped": self.skipped,
            "errors": self.errors,
        }


@dataclass(frozen=True, slots=True)
class ValidatedRawRecord:
    """Typed validation outcome for one raw record."""

    raw_id: str
    parseable: bool
    validation_status: ValidationStatus
    validation_error: str | None
    parse_error: str | None
    canonical_provider: Provider
    payload_provider: Provider | None
    drift_count: int = 0


@dataclass(slots=True)
class ValidateResult:
    """Result of validating a batch of raw records."""

    validated: int = 0
    invalid: int = 0
    drift: int = 0
    skipped_no_schema: int = 0
    errors: int = 0
    records: list[ValidatedRawRecord] = field(default_factory=list)
    drift_counts: dict[str, int] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "validated": self.validated,
            "invalid": self.invalid,
            "drift": self.drift,
            "skipped_no_schema": self.skipped_no_schema,
            "errors": self.errors,
        }

    @property
    def parseable_raw_ids(self) -> list[str]:
        return [record.raw_id for record in self.records if record.parseable]

    @property
    def invalid_raw_ids(self) -> list[str]:
        return [record.raw_id for record in self.records if not record.parseable]

    def merge(self, other: ValidateResult) -> None:
        self.validated += other.validated
        self.invalid += other.invalid
        self.drift += other.drift
        self.skipped_no_schema += other.skipped_no_schema
        self.errors += other.errors
        self.records.extend(other.records)
        for provider, count in other.drift_counts.items():
            self.drift_counts[provider] = self.drift_counts.get(provider, 0) + count


@dataclass(frozen=True, slots=True)
class ParsedConversationArtifact:
    """One parsed conversation associated with its raw source artifact."""

    conversation: ParsedConversation
    source_name: str
    raw_id: str
    payload_provider: Provider | str | None


__all__ = [
    "AcquireResult",
    "ParsedConversationArtifact",
    "ValidateResult",
    "ValidatedRawRecord",
]
