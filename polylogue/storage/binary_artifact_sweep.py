"""Read-only sweep: find raw rows whose bytes are a non-session binary format.

polylogue-hbtj2: the authority-dataflow audit
(``/realm/data/derived/reports/polylogue-authority-dataflow-2026-08-02.html``,
"the residue, dissected") found ~450 MB of SQLite databases (Hermes
``state.db``/``verification_evidence.db``, Codex ``state_5.sqlite``)
sitting in ``raw_sessions`` and misread as "genuinely diverged" session
revisions by a byte-diff classifier that was never designed to reason
about opaque binary snapshots. This module answers, read-only, "which
already-ingested rows are binary-shaped by content" -- reusing
``inspect_raw_artifact`` (the same classifier ``raw_artifacts`` census
uses) purely as a pure function, never invoking its persistence layer.

Separate from this report: an explicit, ``--apply``-gated actuator
(``devtools/binary_artifact_reclassify_apply.py``) that actually persists
``raw_artifacts`` rows for the flagged content. This module only classifies
and counts; it never mutates ``source.db``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord
from polylogue.storage.sqlite.queries.mappers_archive import _row_to_raw_session

_BINARY_KINDS = frozenset({ArtifactKind.BINARY_DATABASE.value, ArtifactKind.BINARY_DOCUMENT.value})


@dataclass(frozen=True, slots=True)
class BinaryArtifactCandidate:
    raw_id: str
    origin: str
    source_path: str
    blob_size: int
    artifact_kind: str
    classification_reason: str
    recognized_session_parser: bool
    """True when a dedicated, content-verified session parser exists for this
    exact shape (currently: Hermes state.db / verification_evidence.db).
    These rows are NOT classified as ``binary_*`` (they legitimately parse as
    sessions) -- ``recognized_session_parser`` candidates are reported
    separately, informationally, so the sweep is honest about the one
    existing feature this bead's stricter detection deliberately leaves
    alone rather than silently ignoring it.
    """


@dataclass(slots=True)
class BinaryArtifactSweepPlan:
    scanned_count: int = 0
    binary_candidates: list[BinaryArtifactCandidate] = field(default_factory=list)
    recognized_session_candidates: list[BinaryArtifactCandidate] = field(default_factory=list)

    @property
    def binary_bytes(self) -> int:
        return sum(c.blob_size for c in self.binary_candidates)

    @property
    def recognized_session_bytes(self) -> int:
        return sum(c.blob_size for c in self.recognized_session_candidates)

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.binary_candidates:
            counts[candidate.artifact_kind] = counts.get(candidate.artifact_kind, 0) + 1
        return counts

    def by_origin(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.binary_candidates:
            counts[candidate.origin] = counts.get(candidate.origin, 0) + 1
        return counts


def _iter_raw_session_rows(conn: sqlite3.Connection, *, limit: int | None) -> Iterator[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    last_rowid = 0
    scanned = 0
    while True:
        batch = 250 if limit is None else min(250, limit - scanned)
        if batch <= 0:
            return
        rows = conn.execute(
            "SELECT rowid AS raw_rowid, * FROM raw_sessions WHERE rowid > ? ORDER BY rowid LIMIT ?",
            (last_rowid, batch),
        ).fetchall()
        if not rows:
            return
        for row in rows:
            last_rowid = max(last_rowid, int(row["raw_rowid"]))
            scanned += 1
            yield row
        if limit is not None and scanned >= limit:
            return


def _classify(record: RawSessionRecord) -> ArtifactObservationRecord:
    return inspect_raw_artifact(record)


def scan_binary_artifacts(
    conn: sqlite3.Connection,
    *,
    blob_store: BlobStore | None = None,
    limit: int | None = None,
) -> BinaryArtifactSweepPlan:
    """Read-only: classify every raw row, reporting binary-shaped miscaptures.

    *conn* should be opened read-only (``?mode=ro``) by the caller -- this
    function never writes. *blob_store* is accepted for symmetry with other
    reconciliation reports but is not currently required (``inspect_raw_artifact``
    resolves its own blob store); reserved for a future in-process override.
    """
    del blob_store
    plan = BinaryArtifactSweepPlan()
    for row in _iter_raw_session_rows(conn, limit=limit):
        plan.scanned_count += 1
        record = _row_to_raw_session(row)
        observation = _classify(record)
        if observation.artifact_kind in _BINARY_KINDS:
            plan.binary_candidates.append(
                BinaryArtifactCandidate(
                    raw_id=record.raw_id,
                    origin=row["origin"],
                    source_path=record.source_path,
                    blob_size=record.blob_size,
                    artifact_kind=observation.artifact_kind,
                    classification_reason=observation.classification_reason,
                    recognized_session_parser=False,
                )
            )
        elif observation.parse_as_session and "Hermes" in observation.classification_reason:
            plan.recognized_session_candidates.append(
                BinaryArtifactCandidate(
                    raw_id=record.raw_id,
                    origin=row["origin"],
                    source_path=record.source_path,
                    blob_size=record.blob_size,
                    artifact_kind=observation.artifact_kind,
                    classification_reason=observation.classification_reason,
                    recognized_session_parser=True,
                )
            )
    return plan


__all__ = [
    "BinaryArtifactCandidate",
    "BinaryArtifactSweepPlan",
    "scan_binary_artifacts",
]
