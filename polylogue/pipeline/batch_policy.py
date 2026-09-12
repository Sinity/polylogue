"""Pure shape selection shared by live batches and offline generation builds.

The policy reads only queue facts and destination ownership. Keeping it below
the daemon package lets replay use the same decision without importing daemon
startup surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

__all__ = [
    "BatchShape",
    "COLD_BACKLOG_MIN_QUEUE_AGE_S",
    "COLD_BACKLOG_MIN_QUEUE_DEPTH",
    "IngestMode",
    "LIVE_TRICKLE_MAX_BYTES",
    "LIVE_TRICKLE_MAX_FILES",
    "WriteDestination",
    "select_cold_build_shape",
    "select_batch_shape",
]

LIVE_TRICKLE_MAX_FILES = 4
LIVE_TRICKLE_MAX_BYTES = 16 * 1024 * 1024
COLD_BACKLOG_MAX_FILES = 256
COLD_BACKLOG_MAX_BYTES = 256 * 1024 * 1024
COLD_BACKLOG_MIN_QUEUE_DEPTH = 64
COLD_BACKLOG_MIN_QUEUE_AGE_S = 60.0


class IngestMode(Enum):
    LIVE_TRICKLE = "live_trickle"
    COLD_BACKLOG = "cold_backlog"


@dataclass(frozen=True, slots=True)
class WriteDestination:
    """What a batch writes, and whether it owns a disposable generation."""

    tier: Literal["source", "index", "embeddings", "user", "audit", "ops"]
    owned_rebuildable_generation: bool = False

    @property
    def admits_bulk_pragmas(self) -> bool:
        return self.owned_rebuildable_generation and self.tier in {"index", "embeddings"}


@dataclass(frozen=True, slots=True)
class BatchShape:
    """The shape one ingest batch takes, and why."""

    mode: IngestMode
    max_files: int
    max_bytes: int
    bulk_pragmas: bool
    archive_wide_derivations: bool
    defer_secondary_indexes: bool = False
    fresh_build: bool = False

    @property
    def reason(self) -> str:
        return f"{self.mode.value}(files<={self.max_files}, bytes<={self.max_bytes}, bulk={self.bulk_pragmas})"


def select_batch_shape(
    *,
    queue_depth: int,
    queue_age_s: float,
    destination: WriteDestination,
    at_admitted_input_boundary: bool = False,
    archive_empty: bool = False,
) -> BatchShape:
    """Choose the batch shape for the queue as it stands."""
    if queue_depth < 0:
        raise ValueError("queue_depth must be non-negative")
    if queue_age_s < 0:
        raise ValueError("queue_age_s must be non-negative")

    cold = queue_depth >= COLD_BACKLOG_MIN_QUEUE_DEPTH and queue_age_s >= COLD_BACKLOG_MIN_QUEUE_AGE_S
    if cold:
        return select_cold_build_shape(
            destination=destination,
            archive_empty=archive_empty,
            at_admitted_input_boundary=at_admitted_input_boundary,
        )
    return BatchShape(
        mode=IngestMode.LIVE_TRICKLE,
        max_files=LIVE_TRICKLE_MAX_FILES,
        max_bytes=LIVE_TRICKLE_MAX_BYTES,
        bulk_pragmas=False,
        archive_wide_derivations=at_admitted_input_boundary,
    )


def select_cold_build_shape(
    *,
    destination: WriteDestination,
    archive_empty: bool,
    at_admitted_input_boundary: bool = True,
) -> BatchShape:
    """Return the cold-build shape for an already-admitted offline rebuild."""
    fresh_owned_generation = destination.admits_bulk_pragmas and archive_empty
    return BatchShape(
        mode=IngestMode.COLD_BACKLOG,
        max_files=COLD_BACKLOG_MAX_FILES,
        max_bytes=COLD_BACKLOG_MAX_BYTES,
        bulk_pragmas=destination.admits_bulk_pragmas,
        archive_wide_derivations=at_admitted_input_boundary,
        defer_secondary_indexes=fresh_owned_generation,
        fresh_build=fresh_owned_generation,
    )
