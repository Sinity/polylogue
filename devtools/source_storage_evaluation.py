"""Post-reindex source-retention versus CDC evaluation lab.

This module is deliberately an in-memory candidate model.  It is a measurement
and law surface, not a second source-tier implementation: production admission
is exercised by the tests that consume this catalog, while neither candidate
adds durable schema or changes the daemon writer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

WORKLOAD_NAMES = (
    "append",
    "leading-record-rewrite",
    "truncation",
    "rotation",
    "relocation",
    "duplicate-export",
    "corruption",
    "missing-chunks",
    "privacy-deletion",
)


@dataclass(frozen=True, slots=True)
class WorkloadCase:
    name: str
    observations: tuple[bytes, ...]
    source_paths: tuple[str, ...]


def frozen_workload() -> tuple[WorkloadCase, ...]:
    """Return the privacy-safe, deterministic post-reindex workload matrix."""
    base = b"record-000\nrecord-001\nrecord-002\n"
    return (
        WorkloadCase("append", (base, base + b"record-003\n"), ("/a.jsonl",) * 2),
        WorkloadCase("leading-record-rewrite", (base, b"record-REWRITTEN\n" + base[11:]), ("/b.jsonl",) * 2),
        WorkloadCase("truncation", (base, base[:22]), ("/c.jsonl",) * 2),
        WorkloadCase("rotation", (base, b"rotated-000\n"), ("/c.jsonl", "/c.jsonl.1")),
        WorkloadCase("relocation", (base, base), ("/d.jsonl", "/moved/d.jsonl")),
        WorkloadCase("duplicate-export", (base, base), ("/e.jsonl", "/export.jsonl")),
        WorkloadCase("corruption", (base, base[:-2] + b"XX"), ("/f.jsonl",) * 2),
        WorkloadCase("missing-chunks", (base, base + b"record-003\n"), ("/g.jsonl",) * 2),
        WorkloadCase("privacy-deletion", (base, base + b"private-marker\n"), ("/private.jsonl",) * 2),
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class Candidate(Protocol):
    def admit(self, payload: bytes, *, observation_id: str) -> None: ...

    def read(self, observation_id: str) -> bytes: ...

    def erase(self, observation_id: str) -> None: ...


@dataclass
class FrontierCandidate:
    """Whole observations with a bounded semantic-frontier retention policy."""

    payloads: dict[str, bytes] = field(default_factory=dict)
    hashes: dict[str, int] = field(default_factory=dict)

    def admit(self, payload: bytes, *, observation_id: str) -> None:
        self.payloads[observation_id] = payload
        self.hashes[_sha(payload)] = self.hashes.get(_sha(payload), 0) + 1

    def read(self, observation_id: str) -> bytes:
        return self.payloads[observation_id]

    def erase(self, observation_id: str) -> None:
        payload = self.payloads.pop(observation_id)
        digest = _sha(payload)
        self.hashes[digest] -= 1
        if self.hashes[digest] == 0:
            del self.hashes[digest]

    @property
    def stored_bytes(self) -> int:
        return sum(len(payload) for payload in self.payloads.values())


def _cdc_chunks(payload: bytes) -> tuple[bytes, ...]:
    """Small deterministic CDC profile suitable for a bounded comparison."""
    if not payload:
        return (b"",)
    chunks: list[bytes] = []
    start = 0
    rolling = 0
    for index, byte in enumerate(payload, start=1):
        rolling = ((rolling << 5) ^ byte ^ (rolling >> 2)) & 0xFFFF
        boundary = index - start >= 8 and (rolling & 0x1F) == 0
        if boundary or index - start >= 32:
            chunks.append(payload[start:index])
            start = index
    if start < len(payload):
        chunks.append(payload[start:])
    return tuple(chunks)


@dataclass
class CdcCandidate:
    manifests: dict[str, tuple[tuple[str, ...], str, int]] = field(default_factory=dict)
    chunks: dict[str, bytes] = field(default_factory=dict)

    def admit(self, payload: bytes, *, observation_id: str) -> None:
        refs = tuple(_sha(chunk) for chunk in _cdc_chunks(payload))
        self.manifests[observation_id] = (refs, _sha(payload), len(payload))
        for ref, chunk in zip(refs, _cdc_chunks(payload), strict=True):
            self.chunks.setdefault(ref, chunk)

    def read(self, observation_id: str) -> bytes:
        try:
            refs, expected_hash, expected_length = self.manifests[observation_id]
            payload = b"".join(self.chunks[ref] for ref in refs)
            if len(payload) != expected_length or _sha(payload) != expected_hash:
                raise ValueError(f"manifest {observation_id!r} failed reconstruction integrity")
            return payload
        except KeyError as exc:
            raise ValueError(f"manifest {observation_id!r} has a missing chunk") from exc

    def erase(self, observation_id: str) -> None:
        self.manifests.pop(observation_id)
        live = {ref for refs, _, _ in self.manifests.values() for ref in refs}
        self.chunks = {ref: chunk for ref, chunk in self.chunks.items() if ref in live}

    @property
    def stored_bytes(self) -> int:
        return sum(len(chunk) for chunk in self.chunks.values())


@dataclass(frozen=True, slots=True)
class Comparison:
    workload_count: int
    frontier_bytes: int
    cdc_bytes: int
    frontier_write_bytes: int
    cdc_write_bytes: int
    laws: tuple[str, ...]


def compare_frozen_workload() -> Comparison:
    frontier = FrontierCandidate()
    cdc = CdcCandidate()
    writes = {"frontier": 0, "cdc": 0}
    laws: list[str] = []
    for case in frozen_workload():
        for index, payload in enumerate(case.observations):
            observation_id = f"{case.name}:{index}"
            frontier.admit(payload, observation_id=observation_id)
            cdc.admit(payload, observation_id=observation_id)
            writes["frontier"] += len(payload)
            writes["cdc"] += sum(len(chunk) for chunk in _cdc_chunks(payload))
            if frontier.read(observation_id) != cdc.read(observation_id):
                raise AssertionError(f"reconstruction law failed for {case.name}")
        laws.append(f"{case.name}:reconstruction")
    return Comparison(
        workload_count=len(frozen_workload()),
        frontier_bytes=frontier.stored_bytes,
        cdc_bytes=cdc.stored_bytes,
        frontier_write_bytes=writes["frontier"],
        cdc_write_bytes=writes["cdc"],
        laws=tuple(laws),
    )


__all__ = [
    "CdcCandidate",
    "Comparison",
    "FrontierCandidate",
    "WORKLOAD_NAMES",
    "compare_frozen_workload",
    "frozen_workload",
]
