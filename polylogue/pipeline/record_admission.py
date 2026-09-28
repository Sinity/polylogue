"""Admit decoded export records into an ingest batch."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Records admitted from one export, and whether enumeration finished."""

    admitted: tuple[Mapping[str, Any], ...]
    refused: tuple[tuple[int, str], ...] = field(default_factory=tuple)
    enumeration_complete: bool = False


def admit_records(records: Iterable[object]) -> AdmissionResult:
    """Admit every mapping record that carries a session id."""
    admitted: list[Mapping[str, Any]] = []
    refused: list[tuple[int, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        if "session_id" not in record:
            refused.append((index, "missing_session_id"))
            continue
        admitted.append(record)
    return AdmissionResult(admitted=tuple(admitted), refused=tuple(refused), enumeration_complete=True)


def admit_export_records(records: Iterable[object]) -> AdmissionResult:
    """Previous name of :func:`admit_records`, kept so older callers keep working."""
    return admit_records(records)


legacy_admit_records = admit_export_records
