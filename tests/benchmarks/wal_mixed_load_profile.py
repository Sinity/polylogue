"""Single authority for the WAL/read-frame mixed-load workload contract.

The profile names every workload and every metric the checkpoint and
read-frame policy is calibrated against, so a run that omits one is a
declaration error rather than a quietly narrower measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MixedLoadWorkload:
    name: str
    route: str
    #: Whether the workload runs concurrently with the others rather than alone.
    concurrent: bool = False


MIXED_LOAD_WORKLOADS: tuple[MixedLoadWorkload, ...] = (
    MixedLoadWorkload("interactive-read", "storage.read_frame", concurrent=True),
    MixedLoadWorkload("incremental-ingest", "storage.write_coordinator.publication", concurrent=True),
    MixedLoadWorkload("candidate-construction", "storage.bulk_build_writer", concurrent=True),
    MixedLoadWorkload("recurring-checkpoint", "storage.wal_checkpoint.recurring"),
)

PROFILE_METRICS: tuple[str, ...] = (
    # WAL trajectory across the mixed load.
    "wal_bytes_start",
    "wal_bytes_peak",
    "wal_bytes_end",
    # What the recurring checkpoint actually did.
    "checkpoint_mode",
    "checkpoint_escalation",
    "checkpoint_log_pages",
    "checkpointed_pages",
    "checkpoint_busy_pages",
    "checkpoint_elapsed_ms",
    "checkpoint_blockers",
    # Writer hold, split so checkpoint time cannot hide inside publication.
    "checkpoint_writer_hold_ms",
    "publication_writer_hold_ms",
    "checkpoint_hold_budget_ms",
    "over_budget_holds",
    # Read-frame population and lifetime.
    "read_connections",
    "read_frame_rebinds",
    "max_read_frame_age_s",
    # Memory.
    "peak_rss_kib",
)


def profile_manifest() -> dict[str, object]:
    """Return the reproducible workload/metric declaration."""
    return {
        "workloads": [asdict(workload) for workload in MIXED_LOAD_WORKLOADS],
        "metrics": list(PROFILE_METRICS),
    }


def record_metrics(benchmark: Any, **metrics: int | float | str) -> None:
    """Attach the declared mixed-load metrics to pytest-benchmark output."""
    unknown = set(metrics) - set(PROFILE_METRICS)
    if unknown:
        raise ValueError(f"unknown WAL mixed-load profile metrics: {sorted(unknown)}")
    missing = set(PROFILE_METRICS) - set(metrics)
    if missing:
        raise ValueError(f"WAL mixed-load profile did not record: {sorted(missing)}")
    extra_info = getattr(benchmark, "extra_info", None)
    if extra_info is not None:
        extra_info.update(metrics)


__all__ = [
    "MIXED_LOAD_WORKLOADS",
    "PROFILE_METRICS",
    "MixedLoadWorkload",
    "profile_manifest",
    "record_metrics",
]
