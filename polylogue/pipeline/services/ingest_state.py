"""Typed ingest state transitions for acquire/validate/parse orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class IngestPhase(str, Enum):
    """Phases for ingestion stage orchestration."""

    INIT = "init"
    ACQUIRED = "acquired"
    VALIDATED = "validated"
    PARSED = "parsed"


def _dedupe_ids(raw_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(raw_ids))


@dataclass(slots=True)
class IngestState:
    """Tracks acquire → validate → parse state with transition guards."""

    source_names: tuple[str, ...]
    parse_requested: bool
    phase: IngestPhase = IngestPhase.INIT
    acquired_raw_ids: list[str] = field(default_factory=list)
    validation_raw_ids: list[str] = field(default_factory=list)
    parseable_raw_ids: list[str] = field(default_factory=list)
    parse_raw_ids: list[str] = field(default_factory=list)

    def record_acquired(self, raw_ids: Iterable[str]) -> None:
        self._expect_phase(IngestPhase.INIT, "record acquired raw IDs")
        self.acquired_raw_ids = _dedupe_ids(raw_ids)
        self.phase = IngestPhase.ACQUIRED

    def record_validation_candidates(self, raw_ids: Iterable[str]) -> None:
        self._expect_phase(IngestPhase.ACQUIRED, "record validation candidates")
        self.validation_raw_ids = _dedupe_ids(raw_ids)

    def record_validation_result(self, parseable_raw_ids: Iterable[str] | None) -> None:
        self._expect_phase(IngestPhase.ACQUIRED, "record validation result")
        parseable = _dedupe_ids(parseable_raw_ids or [])
        allowed = set(self.validation_raw_ids)
        unexpected = [raw_id for raw_id in parseable if raw_id not in allowed]
        if unexpected:
            raise ValueError(
                "Validation result contains raw IDs outside validation candidates: "
                + ", ".join(unexpected[:5])
            )
        self.parseable_raw_ids = parseable
        self.phase = IngestPhase.VALIDATED

    def record_parse_candidates(
        self,
        raw_ids: Iterable[str],
        *,
        persisted_validated_raw_ids: Iterable[str] = (),
    ) -> None:
        self._expect_phase(IngestPhase.VALIDATED, "record parse candidates")
        parse_ids = _dedupe_ids(raw_ids)
        allowed = set(self.validation_raw_ids) | set(persisted_validated_raw_ids)
        unexpected = [raw_id for raw_id in parse_ids if raw_id not in allowed]
        if unexpected:
            raise ValueError(
                "Parse candidates contain raw IDs outside validation candidates: "
                + ", ".join(unexpected[:5])
            )
        self.parse_raw_ids = parse_ids

    def record_parse_completed(self) -> None:
        self._expect_phase(IngestPhase.VALIDATED, "record parse completion")
        self.phase = IngestPhase.PARSED

    def _expect_phase(self, expected: IngestPhase, action: str) -> None:
        if self.phase != expected:
            raise RuntimeError(
                f"Cannot {action}: expected phase {expected.value}, got {self.phase.value}"
            )


__all__ = ["IngestState", "IngestPhase"]
