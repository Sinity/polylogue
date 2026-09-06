"""Durable halt state for every unit of work the daemon schedules.

A halted unit has exactly two inseparable consequences, and they are one
state read from one place:

1. Unschedulable. Selection excludes it. Nothing downstream of a halt
   acquires the write lease, forms a batch, or consumes a budget. Refusing
   at execution time is the defect this exists to prevent -- work that
   cannot succeed must never have been planned.
2. Reported. The halt carries a typed reason and the frame that produced
   it, is published as the unit's status observation, and a daemon holding
   one is not ``ok``.

The store is a small JSON document under the archive root rather than a
tier table: a halt must outlive the process that recorded it, but it is
neither derived from nor authority over archive content.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

__all__ = [
    "HALT_STORE_RELATIVE_PATH",
    "HaltReason",
    "HaltRecord",
    "HaltRegistry",
    "UnitKind",
    "unit_id",
]

HALT_STORE_RELATIVE_PATH = Path("daemon") / "halts.json"

_STORE_VERSION = 1


class UnitKind(str, Enum):
    """The kinds of work that can hold a halt."""

    SERVICE = "service"
    SOURCE = "source"
    INTAKE_CLASS = "intake_class"
    DERIVATION = "derivation"


class HaltReason(str, Enum):
    """Why a unit stopped being schedulable.

    Every member is terminal for the current process' plan. A reason that
    would resolve on its own is retryable backlog, not a halt.
    """

    TERMINAL_REFUSAL = "terminal_refusal"
    """The unit refused further work until restart or operator action."""

    POISON_ITEM = "poison_item"
    """Repeated bounded attempts on the same item all failed."""

    SCHEMA_INCOMPATIBLE = "schema_incompatible"
    """The unit's storage shape cannot be read by this build."""

    PREREQUISITE_FAILED = "prerequisite_failed"
    """A declared prerequisite could not be satisfied."""

    OPERATOR_HALT = "operator_halt"
    """Halted deliberately from outside."""


@dataclass(frozen=True, slots=True)
class HaltRecord:
    """One durable halt."""

    unit: str
    reason: HaltReason
    message: str
    frame: str
    """The identity of the run that produced the halt (generation, pid, run id)."""

    halted_at: str

    def as_dict(self) -> dict[str, str]:
        return {
            "unit": self.unit,
            "reason": self.reason.value,
            "message": self.message,
            "frame": self.frame,
            "halted_at": self.halted_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HaltRecord:
        return cls(
            unit=str(payload["unit"]),
            reason=HaltReason(str(payload["reason"])),
            message=str(payload.get("message", "")),
            frame=str(payload.get("frame", "")),
            halted_at=str(payload.get("halted_at", "")),
        )


def unit_id(kind: UnitKind, name: str) -> str:
    """Return the stable identity a halt is recorded against."""
    return f"{kind.value}:{name}"


class HaltRegistry:
    """Durable halt state for one archive root.

    Reads are served from an in-memory view refreshed from disk on
    construction and on every mutation, so a selection-time check costs no
    filesystem work.
    """

    def __init__(self, archive_root: Path | str) -> None:
        self._path = Path(archive_root) / HALT_STORE_RELATIVE_PATH
        self._lock = threading.Lock()
        self._records: dict[str, HaltRecord] = {}
        self.reload()

    @property
    def path(self) -> Path:
        return self._path

    def reload(self) -> None:
        """Re-read durable state, discarding the in-memory view."""
        with self._lock:
            self._records = _read_records(self._path)

    def halt(
        self,
        unit: str,
        *,
        reason: HaltReason,
        message: str,
        frame: str,
    ) -> HaltRecord:
        """Record *unit* as halted and return the durable record.

        Re-halting an already-halted unit keeps the first record: the frame
        that produced the halt is the one worth reporting.
        """
        with self._lock:
            existing = self._records.get(unit)
            if existing is not None:
                return existing
            record = HaltRecord(
                unit=unit,
                reason=reason,
                message=message,
                frame=frame,
                halted_at=datetime.now(UTC).isoformat(),
            )
            self._records[unit] = record
            _write_records(self._path, self._records)
            return record

    def clear(self, unit: str) -> bool:
        """Remove *unit*'s halt. Returns whether one was present."""
        with self._lock:
            if unit not in self._records:
                return False
            del self._records[unit]
            _write_records(self._path, self._records)
            return True

    def is_halted(self, unit: str) -> bool:
        with self._lock:
            return unit in self._records

    def record_for(self, unit: str) -> HaltRecord | None:
        with self._lock:
            return self._records.get(unit)

    def halted_units(self) -> tuple[HaltRecord, ...]:
        """Every halt, ordered by unit id."""
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda record: record.unit))

    def __iter__(self) -> Iterator[HaltRecord]:
        return iter(self.halted_units())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def _read_records(path: Path) -> dict[str, HaltRecord]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        # An unreadable halt store must not be read as "nothing is halted".
        raise
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _STORE_VERSION:
        return {}
    entries = payload.get("halts")
    if not isinstance(entries, list):
        return {}
    records: dict[str, HaltRecord] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "unit" not in entry or "reason" not in entry:
            continue
        try:
            record = HaltRecord.from_dict(entry)
        except ValueError:
            continue
        records[record.unit] = record
    return records


def _write_records(path: Path, records: Mapping[str, HaltRecord]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "version": _STORE_VERSION,
        "halts": [record.as_dict() for record in sorted(records.values(), key=lambda item: item.unit)],
    }
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".halts-", suffix=".json")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
