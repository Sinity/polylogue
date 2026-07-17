"""Durable source-revision evidence and conservative legacy classification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import BinaryIO, Literal


class RawRevisionKind(StrEnum):
    FULL = "full"
    APPEND = "append"
    UNKNOWN = "unknown"


class RawRevisionAuthority(StrEnum):
    ASSERTED = "asserted"
    BYTE_PROVEN = "byte_proven"
    QUARANTINED = "quarantined"


BYTE_AUTHORITY_CENSUS_DETAIL = "append fragments are governed by byte revision authority"


@dataclass(frozen=True)
class RawRevisionEnvelope:
    """Evidence captured with raw bytes, before any derived write occurs."""

    logical_source_key: str
    kind: RawRevisionKind
    source_revision: str
    acquisition_generation: int
    predecessor_source_revision: str | None = None
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
            if self.predecessor_source_revision is None or None in offsets:
                raise ValueError("append evidence requires predecessor revision and offsets")
            assert self.append_start_offset is not None and self.append_end_offset is not None
            if self.append_start_offset < 0 or self.append_end_offset <= self.append_start_offset:
                raise ValueError("append offsets must describe a non-empty forward range")
            raw_predecessors = (self.predecessor_raw_id, self.baseline_raw_id)
            if self.authority is RawRevisionAuthority.QUARANTINED:
                if any(value is not None for value in raw_predecessors):
                    raise ValueError("quarantined append evidence may not claim raw predecessors")
            elif any(value is None for value in raw_predecessors):
                raise ValueError("replay-eligible append evidence requires baseline and raw predecessor")
        elif self.predecessor_source_revision is not None or any(value is not None for value in offsets):
            raise ValueError("only append evidence may carry predecessor revision or byte offsets")


@dataclass(frozen=True)
class HistoricalRawRevision:
    raw_id: str
    payload: bytes


@dataclass(frozen=True)
class HistoricalRawRevisionStream:
    """A retained full revision whose bytes can be compared without loading it."""

    raw_id: str
    payload_size: int
    open_payload: Callable[[], BinaryIO]


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


def _stream_size(revision: HistoricalRawRevisionStream) -> int:
    size = 0
    with revision.open_payload() as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
    return size


def _stream_is_prefix(
    parent: HistoricalRawRevisionStream,
    child: HistoricalRawRevisionStream,
    *,
    parent_size: int,
    child_size: int,
) -> bool:
    """Return whether *parent* is an exact proper byte prefix of *child*."""
    if parent_size >= child_size:
        return False
    remaining = parent_size
    with parent.open_payload() as parent_handle, child.open_payload() as child_handle:
        while remaining:
            chunk = parent_handle.read(min(1024 * 1024, remaining))
            if not chunk or child_handle.read(len(chunk)) != chunk:
                return False
            remaining -= len(chunk)
        return parent_handle.read(1) == b""


def classify_historical_full_revision_streams(
    revisions: list[HistoricalRawRevisionStream],
) -> list[HistoricalRevisionDecision]:
    """Stream the same unique-prefix proof as the eager byte classifier."""
    if not revisions:
        return []
    actual_sizes = {revision.raw_id: _stream_size(revision) for revision in revisions}
    ordered = sorted(revisions, key=lambda revision: (actual_sizes[revision.raw_id], revision.raw_id))
    if len({actual_sizes[revision.raw_id] for revision in ordered}) != len(ordered):
        return [
            HistoricalRevisionDecision(
                raw_id=revision.raw_id, authority=RawRevisionAuthority.QUARANTINED, relation="ambiguous"
            )
            for revision in revisions
        ]
    decisions: list[HistoricalRevisionDecision] = []
    previous: HistoricalRawRevisionStream | None = None
    for current in ordered:
        predecessor = previous.raw_id if previous is not None else None
        if previous is not None and not _stream_is_prefix(
            previous,
            current,
            parent_size=actual_sizes[previous.raw_id],
            child_size=actual_sizes[current.raw_id],
        ):
            return [
                HistoricalRevisionDecision(
                    raw_id=revision.raw_id, authority=RawRevisionAuthority.QUARANTINED, relation="ambiguous"
                )
                for revision in revisions
            ]
        relation: Literal["baseline", "predecessor"] = "baseline" if predecessor is None else "predecessor"
        decisions.append(
            HistoricalRevisionDecision(
                raw_id=current.raw_id,
                authority=RawRevisionAuthority.BYTE_PROVEN,
                relation=relation,
                predecessor_raw_id=predecessor,
            )
        )
        previous = current
    return decisions


__all__ = [
    "HistoricalRawRevision",
    "HistoricalRawRevisionStream",
    "HistoricalRevisionDecision",
    "RawRevisionAuthority",
    "RawRevisionEnvelope",
    "RawRevisionKind",
    "append_source_revision",
    "classify_historical_full_revisions",
    "classify_historical_full_revision_streams",
]
