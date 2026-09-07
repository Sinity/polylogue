"""The one decision point between a cold backlog and a live trickle.

There is no cold path and no live path in the code: both ends run the same
functions with different parameters, and this module is the only place that
reads the difference. Four files or 16 MiB per chunk is right for a session file
appended a line at a time and wrong for 46k files sitting on disk; the policy
says which shape applies, and nothing downstream re-decides it.

The policy is pure. It reads queue depth, queue age and the destination's own
ownership, and returns a shape. It opens nothing, holds nothing, and is
therefore testable without an archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

__all__ = [
    "BatchShape",
    "IngestMode",
    "LIVE_TRICKLE_MAX_BYTES",
    "LIVE_TRICKLE_MAX_FILES",
    "WriteDestination",
    "select_batch_shape",
]

#: The live-trickle ceiling, unchanged: a chunk is at most four files or 16 MiB
#: so an interactive read never waits behind a long hold.
LIVE_TRICKLE_MAX_FILES = 4
LIVE_TRICKLE_MAX_BYTES = 16 * 1024 * 1024

#: The cold-backlog ceiling. Bounded by bytes in flight rather than file count,
#: so one whale cannot inflate memory and a thousand small files are one batch.
COLD_BACKLOG_MAX_FILES = 256
COLD_BACKLOG_MAX_BYTES = 256 * 1024 * 1024

#: A backlog is cold when it is both deep and old: depth alone is a burst of
#: live appends, and age alone is one stale file nobody has touched.
COLD_BACKLOG_MIN_QUEUE_DEPTH = 64
COLD_BACKLOG_MIN_QUEUE_AGE_S = 60.0


class IngestMode(Enum):
    LIVE_TRICKLE = "live_trickle"
    COLD_BACKLOG = "cold_backlog"


@dataclass(frozen=True, slots=True)
class WriteDestination:
    """What the batch is about to write into, and who owns it.

    ``owned_rebuildable_generation`` is the only licence for bulk pragmas. An
    owned inactive index generation has one writer, no readers, and is discarded
    wholesale if the pass raises, which is what makes ``synchronous=OFF`` and a
    memory journal safe there. A durable tier — source, user, audit — never
    qualifies, however deep the backlog: the tradeoff those pragmas make is
    "corruption is acceptable because this artifact is disposable", and that is
    false for every durable tier by definition.
    """

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
    #: Archive-wide derivations run at a quiescence boundary, never per batch.
    #: Their cost is a function of the archive, so charging it to each chunk is
    #: the quadratic term the scope split removes.
    archive_wide_derivations: bool

    @property
    def reason(self) -> str:
        return f"{self.mode.value}(files<={self.max_files}, bytes<={self.max_bytes}, bulk={self.bulk_pragmas})"


def select_batch_shape(
    *,
    queue_depth: int,
    queue_age_s: float,
    destination: WriteDestination,
    at_admitted_input_boundary: bool = False,
) -> BatchShape:
    """Choose the batch shape for the queue as it stands.

    ``at_admitted_input_boundary`` is what licenses archive-wide derivations: a
    stable boundary over admitted input, not an observation that the queue
    looked empty for an instant. A queue that drains and refills between the
    check and the work would otherwise run an archive-wide pass against a moving
    input set and bind its output to a frame that never existed.
    """
    if queue_depth < 0:
        raise ValueError("queue_depth must be non-negative")
    if queue_age_s < 0:
        raise ValueError("queue_age_s must be non-negative")

    cold = queue_depth >= COLD_BACKLOG_MIN_QUEUE_DEPTH and queue_age_s >= COLD_BACKLOG_MIN_QUEUE_AGE_S
    if cold:
        return BatchShape(
            mode=IngestMode.COLD_BACKLOG,
            max_files=COLD_BACKLOG_MAX_FILES,
            max_bytes=COLD_BACKLOG_MAX_BYTES,
            bulk_pragmas=destination.admits_bulk_pragmas,
            archive_wide_derivations=at_admitted_input_boundary,
        )
    return BatchShape(
        mode=IngestMode.LIVE_TRICKLE,
        max_files=LIVE_TRICKLE_MAX_FILES,
        max_bytes=LIVE_TRICKLE_MAX_BYTES,
        bulk_pragmas=False,
        archive_wide_derivations=at_admitted_input_boundary,
    )
