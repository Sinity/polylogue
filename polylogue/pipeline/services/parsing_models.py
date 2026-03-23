from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.pipeline.services.acquisition import AcquireResult
    from polylogue.pipeline.services.validation import ValidateResult


class IngestPhase(str, Enum):
    """Phases for acquire → validate → parse orchestration."""

    INIT = "init"
    ACQUIRED = "acquired"
    VALIDATED = "validated"
    PARSED = "parsed"


def _dedupe_ids(raw_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(raw_ids))


@dataclass(slots=True)
class IngestState:
    """Tracks ingest-state transitions and validates phase ordering."""

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


class ParseResult:
    """Result of an async parsing operation."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }
        self.changed_counts: dict[str, int] = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
        }
        self.processed_ids: set[str] = set()
        self.parse_failures: int = 0
        self._lock = asyncio.Lock()

    async def merge_result(
        self,
        conversation_id: str,
        result_counts: dict[str, int],
        content_changed: bool,
    ) -> None:
        """Merge a single conversation's result into the aggregate."""
        ingest_changed = (
            result_counts["conversations"]
            + result_counts["messages"]
            + result_counts["attachments"]
        ) > 0

        async with self._lock:
            if ingest_changed or content_changed:
                self.processed_ids.add(conversation_id)
            if content_changed:
                self.changed_counts["conversations"] += 1
            if result_counts["messages"]:
                self.changed_counts["messages"] += result_counts["messages"]
            if result_counts["attachments"]:
                self.changed_counts["attachments"] += result_counts["attachments"]
            for key, value in result_counts.items():
                if key in self.counts:
                    self.counts[key] += value


@dataclass
class IngestResult:
    """Result of acquire -> validate -> parse orchestration."""

    acquire_result: AcquireResult
    validation_result: ValidateResult | None
    parse_result: ParseResult
    parse_raw_ids: list[str]


__all__ = ["IngestPhase", "IngestResult", "IngestState", "ParseResult"]
