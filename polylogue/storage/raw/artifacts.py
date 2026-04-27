"""Shared raw-ingest backlog semantics for planning and reparse flows."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.storage.raw.models import RawConversationState
from polylogue.types import ValidationStatus

_PARSEABLE_VALIDATION_STATUSES = (
    ValidationStatus.PASSED,
    ValidationStatus.SKIPPED,
)


@dataclass(frozen=True, slots=True)
class RawBacklogQuerySpec:
    """SQL selection contract for one persisted raw backlog view."""

    require_unparsed: bool
    require_unvalidated: bool = False
    validation_statuses: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class RawIngestArtifactState:
    """Canonical persisted raw-state semantics for validation and parse planning."""

    parsed_at: str | None = None
    parse_error: str | None = None
    validation_status: ValidationStatus | None = None

    @classmethod
    def from_state(cls, state: RawConversationState | None) -> RawIngestArtifactState:
        if state is None:
            return cls()
        return cls(
            parsed_at=state.parsed_at,
            parse_error=state.parse_error,
            validation_status=state.validation_status,
        )

    @property
    def parsed(self) -> bool:
        return self.parsed_at is not None

    @property
    def validated(self) -> bool:
        return self.validation_status is not None

    @property
    def quarantined(self) -> bool:
        return not self.parsed and self.parse_error is not None

    def needs_validation_backlog(self, *, force_reparse: bool = False) -> bool:
        return self.validation_status is None and (force_reparse or not self.parsed)

    def needs_parse_backlog(self, *, force_reparse: bool = False) -> bool:
        if force_reparse:
            return True
        return (not self.parsed) and self.validation_status in _PARSEABLE_VALIDATION_STATUSES


def validation_backlog_query_spec(*, force_reparse: bool = False) -> RawBacklogQuerySpec:
    return RawBacklogQuerySpec(
        require_unparsed=not force_reparse,
        require_unvalidated=True,
    )


def parse_backlog_query_spec(*, force_reparse: bool = False) -> RawBacklogQuerySpec:
    return RawBacklogQuerySpec(
        require_unparsed=not force_reparse,
        validation_statuses=None if force_reparse else tuple(status.value for status in _PARSEABLE_VALIDATION_STATUSES),
    )


__all__ = [
    "RawBacklogQuerySpec",
    "RawIngestArtifactState",
    "parse_backlog_query_spec",
    "validation_backlog_query_spec",
]
