"""Read-only sweep: find raw rows that should reclassify as sidecar artifacts.

polylogue-omsw: two Claude Code artifact families were historically admitted
as independent ``claude-code-session`` rows in ``raw_sessions`` instead of
being recognized as non-conversational sidecars:

* ``tool-results/<name>.<ext>`` tool-call-overflow content
  (``ArtifactKind.TOOL_RESULT_SIDECAR``) -- joined back to its owning
  ``tool_result`` block by ``sources/live/tool_result_sidecars.py``, never an
  independent session.
* ``projects/<proj>/<uuid>.jsonl`` files whose only records are
  ``file-history-snapshot``/``progress`` checkpoints
  (``ArtifactKind.FILE_HISTORY_SNAPSHOT``) -- pure filesystem-checkpoint
  activity that never carried a single chat turn.

Both classification gaps are now closed in ``archive.artifact_taxonomy``
(``classify_artifact``/``classify_artifact_path``), so freshly-acquired rows
of either shape are no longer misclassified. This module answers, read-only,
"which already-ingested rows are affected by content acquired before that
fix" -- reusing ``inspect_raw_artifact`` (the same classifier ``raw_artifacts``
census uses) purely as a pure function, never invoking its persistence layer.

Separate from this report: the existing durable-tier reclassification
actuator (``devtools/tool_result_history_reclassify_apply.py``, following the
same non-destructive pattern as ``devtools/binary_artifact_reclassify_apply.py``
for the earlier binary-database miscapture) that actually persists
``raw_artifacts`` rows for the flagged content via
``materialize_artifact_observations`` -- the same durable-observation
machinery every other reclassification actuator in this family uses. This
module only classifies and counts; it never mutates ``source.db``. Per
this repo's durable-tier evidence retention precedent (the 2026-07-22
hook-inflation postmortem), reclassification never deletes ``raw_sessions``
rows -- it only adds/refreshes a ``raw_artifacts`` observation alongside them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord
from polylogue.storage.sqlite.queries.mappers_archive import _row_to_raw_session

_TARGET_KINDS = frozenset({ArtifactKind.TOOL_RESULT_SIDECAR.value, ArtifactKind.FILE_HISTORY_SNAPSHOT.value})


@dataclass(frozen=True, slots=True)
class ToolResultHistoryCandidate:
    raw_id: str
    origin: str
    source_path: str
    blob_size: int
    artifact_kind: str
    classification_reason: str


@dataclass(slots=True)
class ToolResultHistorySweepPlan:
    scanned_count: int = 0
    candidates: list[ToolResultHistoryCandidate] = field(default_factory=list)

    @property
    def candidate_bytes(self) -> int:
        return sum(c.blob_size for c in self.candidates)

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            counts[candidate.artifact_kind] = counts.get(candidate.artifact_kind, 0) + 1
        return counts


def _iter_claude_code_raw_session_rows(conn: sqlite3.Connection, *, limit: int | None) -> Iterator[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    last_rowid = 0
    scanned = 0
    while True:
        batch = 250 if limit is None else min(250, limit - scanned)
        if batch <= 0:
            return
        rows = conn.execute(
            """
            SELECT rowid AS raw_rowid, *
            FROM raw_sessions
            WHERE rowid > ? AND origin = 'claude-code-session'
            ORDER BY rowid
            LIMIT ?
            """,
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


def scan_tool_result_and_file_history_artifacts(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> ToolResultHistorySweepPlan:
    """Read-only: classify claude-code-session raw rows, reporting sidecar miscaptures.

    *conn* should be opened read-only (``?mode=ro``) by the caller -- this
    function never writes.
    """
    plan = ToolResultHistorySweepPlan()
    for row in _iter_claude_code_raw_session_rows(conn, limit=limit):
        plan.scanned_count += 1
        record = _row_to_raw_session(row)
        observation = _classify(record)
        if observation.artifact_kind in _TARGET_KINDS:
            plan.candidates.append(
                ToolResultHistoryCandidate(
                    raw_id=record.raw_id,
                    origin=row["origin"],
                    source_path=record.source_path,
                    blob_size=record.blob_size,
                    artifact_kind=observation.artifact_kind,
                    classification_reason=observation.classification_reason,
                )
            )
    return plan


__all__ = [
    "ToolResultHistoryCandidate",
    "ToolResultHistorySweepPlan",
    "scan_tool_result_and_file_history_artifacts",
]
