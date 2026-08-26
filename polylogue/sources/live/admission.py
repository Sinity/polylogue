"""Receipt-bound source admission primitives.

This module is deliberately independent of SQLite and blob storage.  It is the
small state machine shared by ordinary and bounded streaming acquisition: the
caller supplies the durable observation/commit callbacks, while this module
owns the rules that make a receipt resumable (or force a safe reacquisition).
Operational offsets are intentionally not part of the receipt.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum


class AdmissionDisposition(StrEnum):
    OBSERVED = "observed"
    ACCEPTED = "accepted"
    RETRYABLE_FAILURE = "retryable_failure"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    RESOURCE_LIMIT = "resource_limit"
    REPLACED = "replaced"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class ContinuationDecision(StrEnum):
    RESUME = "resume"
    REACQUIRE = "reacquire"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class SourceCoordinates:
    """Stable source identity; path is only a coordinate, never identity."""

    source_id: str
    coordinate: str


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Exact artifact identity captured by one attempt."""

    sha256: str
    size_bytes: int

    @classmethod
    def from_bytes(cls, payload: bytes) -> ArtifactIdentity:
        return cls(hashlib.sha256(payload).hexdigest(), len(payload))


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    """Declared bounds for one admission unit."""

    max_bytes: int
    max_duration_ms: int

    def __post_init__(self) -> None:
        if self.max_bytes < 0 or self.max_duration_ms < 0:
            raise ValueError("resource envelope bounds must be non-negative")


@dataclass(frozen=True, slots=True)
class SemanticFrontier:
    """The only evidence that can authorize continuation after restart."""

    source_revision: str
    stable_body_bytes: int
    semantics_version: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.stable_body_bytes < 0:
            raise ValueError("frontier byte boundary must be non-negative")


@dataclass(frozen=True, slots=True)
class AdmissionAttempt:
    """Immutable attempt binding all evidence needed for later decisions."""

    attempt_id: str
    coordinates: SourceCoordinates
    artifact: ArtifactIdentity
    source_law: str
    parser_identity: str
    start_frontier: SemanticFrontier | None
    envelope: ResourceEnvelope


@dataclass(frozen=True, slots=True)
class AdmissionReceipt:
    attempt: AdmissionAttempt
    disposition: AdmissionDisposition
    frontier: SemanticFrontier | None = None
    diagnostic: str = ""
    retryable: bool = False


def continuation(
    prior: AdmissionReceipt | None,
    *,
    artifact: ArtifactIdentity,
    source_law: str,
    parser_identity: str,
    frontier_evidence: SemanticFrontier | None,
) -> ContinuationDecision:
    """Decide whether a prior receipt permits resume.

    Equality is intentional: a changed artifact, source law, parser, or
    missing/mismatched frontier is not an append proof and must reacquire.
    """
    if prior is None or prior.disposition is not AdmissionDisposition.ACCEPTED:
        return ContinuationDecision.REACQUIRE
    if prior.attempt.artifact != artifact:
        return ContinuationDecision.REACQUIRE
    if prior.attempt.source_law != source_law or prior.attempt.parser_identity != parser_identity:
        return ContinuationDecision.REACQUIRE
    if prior.frontier is None or frontier_evidence != prior.frontier:
        return ContinuationDecision.REACQUIRE
    return ContinuationDecision.RESUME


class AdmissionState:
    """One-attempt transition guard; durable callbacks observe its receipts."""

    def __init__(self, attempt: AdmissionAttempt, publish: Callable[[AdmissionReceipt], None]) -> None:
        self.attempt = attempt
        self._publish = publish
        self._released = False
        self._terminal: AdmissionReceipt | None = None
        self.observe()

    @property
    def terminal(self) -> AdmissionReceipt | None:
        return self._terminal

    @property
    def ownership_released(self) -> bool:
        return self._released

    def observe(self) -> AdmissionReceipt:
        receipt = AdmissionReceipt(self.attempt, AdmissionDisposition.OBSERVED)
        self._publish(receipt)
        return receipt

    def finish(
        self,
        disposition: AdmissionDisposition,
        *,
        frontier: SemanticFrontier | None = None,
        diagnostic: str = "",
        retryable: bool = False,
    ) -> AdmissionReceipt:
        if self._terminal is not None:
            return self._terminal
        if disposition is AdmissionDisposition.ACCEPTED and frontier is None:
            raise ValueError("accepted admission requires a semantic frontier")
        receipt = AdmissionReceipt(self.attempt, disposition, frontier, diagnostic[:4096], retryable)
        self._publish(receipt)
        self._terminal = receipt
        self._released = True
        return receipt

    def fail_retryably(
        self,
        diagnostic: str,
        *,
        disposition: AdmissionDisposition = AdmissionDisposition.RETRYABLE_FAILURE,
    ) -> AdmissionReceipt:
        return self.finish(disposition, diagnostic=diagnostic, retryable=True)


@dataclass(frozen=True, slots=True)
class AdmissionUnit:
    source_id: str
    work: Callable[[], object]


class FairAdmissionScheduler:
    """Bounded round-robin scheduler; a yielding source cannot starve siblings."""

    def __init__(self, *, max_units: int = 1) -> None:
        if max_units < 1:
            raise ValueError("max_units must be positive")
        self.max_units = max_units
        self._queue: deque[AdmissionUnit] = deque()

    def add(self, unit: AdmissionUnit) -> None:
        self._queue.append(unit)

    def run(self) -> Iterable[tuple[str, object]]:
        active = 0
        while self._queue:
            unit = self._queue.popleft()
            if active >= self.max_units:
                active = 0
            active += 1
            result = unit.work()
            yield unit.source_id, result
            active -= 1


__all__ = [
    "AdmissionAttempt",
    "AdmissionDisposition",
    "AdmissionReceipt",
    "AdmissionState",
    "AdmissionUnit",
    "ArtifactIdentity",
    "ContinuationDecision",
    "FairAdmissionScheduler",
    "ResourceEnvelope",
    "SemanticFrontier",
    "SourceCoordinates",
    "continuation",
]
