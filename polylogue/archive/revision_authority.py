"""Durable source-revision evidence and conservative legacy classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Literal


class RawRevisionKind(StrEnum):
    FULL = "full"
    APPEND = "append"
    UNKNOWN = "unknown"


class RawRevisionAuthority(StrEnum):
    ASSERTED = "asserted"
    BYTE_PROVEN = "byte_proven"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class RawRevisionEnvelope:
    """Evidence captured with raw bytes, before any derived write occurs."""

    logical_source_key: str
    kind: RawRevisionKind
    source_revision: str
    acquisition_generation: int
    predecessor_raw_id: str | None = None
    baseline_raw_id: str | None = None
    append_start_offset: int | None = None
    append_end_offset: int | None = None
    authority: RawRevisionAuthority = RawRevisionAuthority.ASSERTED

    def __post_init__(self) -> None:
        if not self.logical_source_key or not self.source_revision:
            raise ValueError("revision envelope identity must be non-empty")
        if self.acquisition_generation < 0:
            raise ValueError("acquisition_generation must be non-negative")
        offsets = (self.append_start_offset, self.append_end_offset)
        if self.kind is RawRevisionKind.APPEND:
            if self.baseline_raw_id is None or None in offsets:
                raise ValueError("append evidence requires baseline and offsets")
            assert self.append_start_offset is not None and self.append_end_offset is not None
            if self.append_start_offset < 0 or self.append_end_offset <= self.append_start_offset:
                raise ValueError("append offsets must describe a non-empty forward range")
        elif any(value is not None for value in offsets):
            raise ValueError("only append evidence may carry byte offsets")


@dataclass(frozen=True)
class HistoricalRawRevision:
    raw_id: str
    payload: bytes


@dataclass(frozen=True)
class HistoricalRevisionDecision:
    raw_id: str
    authority: RawRevisionAuthority
    relation: Literal["baseline", "predecessor", "ambiguous"]
    predecessor_raw_id: str | None = None


def append_source_revision(predecessor_revision: str, payload_hash: str) -> str:
    """Return the exact content fingerprint committed by the append cursor."""
    return sha256(f"{predecessor_revision}\0{payload_hash}".encode()).hexdigest()


def classify_historical_full_revisions(
    revisions: list[HistoricalRawRevision],
) -> list[HistoricalRevisionDecision]:
    """Prove a unique byte-prefix chain; quarantine every ambiguous cohort.

    Acquisition time, source path, provider timestamps, and raw-id ordering are
    intentionally absent. Equal or divergent payloads do not establish which
    capture is newer.
    """
    if not revisions:
        return []
    by_id = {revision.raw_id: revision for revision in revisions}
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {raw_id: [] for raw_id in by_id}
    for child in revisions:
        candidates = [
            parent.raw_id
            for parent in revisions
            if parent.raw_id != child.raw_id
            and len(parent.payload) < len(child.payload)
            and child.payload.startswith(parent.payload)
        ]
        maximal = [
            candidate
            for candidate in candidates
            if not any(
                candidate != other
                and len(by_id[candidate].payload) < len(by_id[other].payload)
                and by_id[other].payload.startswith(by_id[candidate].payload)
                for other in candidates
            )
        ]
        parents[child.raw_id] = maximal
        for parent in maximal:
            children[parent].append(child.raw_id)
    roots = [raw_id for raw_id, parent_ids in parents.items() if not parent_ids]
    leaves = [raw_id for raw_id, child_ids in children.items() if not child_ids]
    unique_chain = (
        len(roots) == 1
        and len(leaves) == 1
        and all(len(parent_ids) <= 1 for parent_ids in parents.values())
        and all(len(child_ids) <= 1 for child_ids in children.values())
    )
    if not unique_chain:
        return [
            HistoricalRevisionDecision(
                raw_id=revision.raw_id, authority=RawRevisionAuthority.QUARANTINED, relation="ambiguous"
            )
            for revision in revisions
        ]
    root = roots[0]
    decisions: list[HistoricalRevisionDecision] = []
    current: str | None = root
    while current is not None:
        predecessor: str | None = parents[current][0] if parents[current] else None
        relation: Literal["baseline", "predecessor"] = "baseline" if not parents[current] else "predecessor"
        decisions.append(
            HistoricalRevisionDecision(
                raw_id=current,
                authority=RawRevisionAuthority.BYTE_PROVEN,
                relation=relation,
                predecessor_raw_id=predecessor,
            )
        )
        current = children[current][0] if children[current] else None
    return decisions


__all__ = [
    "HistoricalRawRevision",
    "HistoricalRevisionDecision",
    "RawRevisionAuthority",
    "RawRevisionEnvelope",
    "RawRevisionKind",
    "append_source_revision",
    "classify_historical_full_revisions",
]
