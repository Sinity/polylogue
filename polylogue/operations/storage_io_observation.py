"""Process-local storage I/O observations for product status surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.storage.io_phase_metrics import UNAVAILABLE_SQLITE_INTERNAL_PHASES, io_phase_snapshot


@dataclass(frozen=True, slots=True)
class IoPhaseObservation:
    tier: str
    phase: str
    inside_writer_lease: bool
    succeeded: bool
    count: int
    elapsed_ns: int


@dataclass(frozen=True, slots=True)
class StorageIoObservation:
    samples: tuple[IoPhaseObservation, ...]
    unavailable_phases: tuple[str, ...]


def storage_io_observation() -> StorageIoObservation:
    """Translate the bounded substrate counters into one surface-neutral view."""
    return StorageIoObservation(
        samples=tuple(
            IoPhaseObservation(
                tier=item.tier,
                phase=item.phase,
                inside_writer_lease=item.inside_writer_lease,
                succeeded=item.succeeded,
                count=item.count,
                elapsed_ns=item.elapsed_ns,
            )
            for item in io_phase_snapshot()
        ),
        unavailable_phases=UNAVAILABLE_SQLITE_INTERNAL_PHASES,
    )
