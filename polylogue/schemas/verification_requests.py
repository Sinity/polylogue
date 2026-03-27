"""Typed request objects for artifact-proof and corpus-verification workflows."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.protocols import ProgressCallback


@dataclass(frozen=True)
class ArtifactObservationQuery:
    providers: list[str] | None = None
    support_statuses: list[str] | None = None
    artifact_kinds: list[str] | None = None
    record_limit: int | None = None
    record_offset: int = 0


@dataclass(frozen=True)
class ArtifactProofRequest:
    providers: list[str] | None = None
    record_limit: int | None = None
    record_offset: int = 0


@dataclass(frozen=True)
class SchemaVerificationRequest:
    providers: list[str] | None = None
    max_samples: int | None = None
    record_limit: int | None = None
    record_offset: int = 0
    quarantine_malformed: bool = False
    progress_callback: ProgressCallback | None = None


def bounded_window(
    record_limit: int | None,
    record_offset: int,
) -> tuple[int | None, int]:
    bounded_limit = max(1, int(record_limit)) if record_limit is not None else None
    bounded_offset = max(0, int(record_offset))
    return bounded_limit, bounded_offset


__all__ = [
    "ArtifactObservationQuery",
    "ArtifactProofRequest",
    "SchemaVerificationRequest",
    "bounded_window",
]
