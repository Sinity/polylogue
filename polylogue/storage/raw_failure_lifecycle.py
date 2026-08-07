"""Read-only lifecycle projection for retained raw failures.

``raw_sessions`` is the ground-truth failure universe.  A parser diagnostic is
not a lifecycle decision: only a matching ``raw_artifacts`` observation can
explain a failed raw as deferred or terminal.  This module keeps that rule in
one substrate helper so status, preflight, and maintenance gates cannot drift.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.core.raw_failure_evidence import (
    RawFailureEvidenceKind,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

RawFailureLifecycle = Literal["deferred", "terminal", "unexplained"]
RawFailureLifecycleState = Literal["healthy", "degraded", "blocked", "unavailable"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RawFailureLifecycleSnapshot:
    """Counts and bounded evidence from one source-tier read snapshot."""

    available: bool
    parse_failures: int = 0
    validation_failures: int = 0
    deferred: int = 0
    terminal: int = 0
    unexplained: int = 0
    by_origin: tuple[tuple[str, int], ...] = ()
    by_artifact_kind: tuple[tuple[str, int], ...] = ()
    samples: tuple[dict[str, str | None], ...] = ()
    reason: str | None = None

    @property
    def blocking(self) -> bool:
        """Whether failed source evidence lacks a typed lifecycle explanation."""
        return not self.available or self.unexplained > 0

    @property
    def state(self) -> RawFailureLifecycleState:
        """Return the public claim state for this source-tier evidence."""
        if not self.available:
            return "unavailable"
        if self.unexplained > 0:
            return "blocked"
        if self.parse_failures > 0 or self.validation_failures > 0:
            return "degraded"
        return "healthy"

    @property
    def healthy(self) -> bool:
        """Whether the source proves a clean, zero-failure lifecycle."""
        return self.state == "healthy"

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "parse_failures": self.parse_failures,
            "validation_failures": self.validation_failures,
            "deferred": self.deferred,
            "terminal": self.terminal,
            "unexplained": self.unexplained,
            "by_origin": dict(self.by_origin),
            "by_artifact_kind": dict(self.by_artifact_kind),
            "samples": [dict(sample) for sample in self.samples],
            "reason": self.reason,
            "state": self.state,
        }


def _lifecycle(
    artifact_kind: object,
    support_status: object,
    *,
    validation_failed: bool,
) -> RawFailureLifecycle:
    """Classify an artifact only when its closed evidence is self-consistent."""
    if validation_failed:
        return "unexplained"
    if artifact_kind is None or support_status is None:
        return "unexplained"
    kind = str(artifact_kind)
    support = str(support_status)
    try:
        evidence_kind = RawFailureEvidenceKind(kind)
    except ValueError:
        return "unexplained"
    if evidence_kind.lifecycle not in {"deferred", "terminal"}:
        return "unexplained"
    if evidence_kind.support_status.value != support:
        return "unexplained"
    return "deferred" if evidence_kind.lifecycle == "deferred" else "terminal"


def read_raw_failure_lifecycle(source_db: Path, *, sample_limit: int = 10) -> RawFailureLifecycleSnapshot:
    """Read and classify every failed raw without opening a write connection."""
    if not source_db.exists():
        return RawFailureLifecycleSnapshot(False, reason=f"source.db not found: {source_db}")
    try:
        conn = open_readonly_connection(source_db)
    except (OSError, sqlite3.Error) as exc:
        logger.warning("could not open source.db read-only", exc_info=exc)
        return RawFailureLifecycleSnapshot(False, reason=f"could not open source.db read-only: {exc}")
    try:
        conn.execute("BEGIN")
        raw_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_sessions'"
        ).fetchone()
        if raw_table is None:
            return RawFailureLifecycleSnapshot(False, reason="source.db is missing raw_sessions")
        parse_failures = int(
            conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE parse_error IS NOT NULL AND TRIM(parse_error) != ''"
            ).fetchone()[0]
            or 0
        )
        validation_failures = int(
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE validation_status = 'failed'").fetchone()[0] or 0
        )
        has_artifacts = (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_artifacts'").fetchone()
            is not None
        )
        sample_limit = max(0, sample_limit)
        if has_artifacts:
            failed_cte = """
            WITH latest_artifact AS (
                SELECT raw_id, origin, source_path, source_index, artifact_kind, support_status,
                       ROW_NUMBER() OVER (
                           PARTITION BY raw_id, origin, source_path, source_index
                           ORDER BY last_observed_at_ms DESC, artifact_id DESC
                       ) AS artifact_rank
                FROM raw_artifacts
            ), failed AS (
                SELECT r.raw_id, r.origin, r.validation_status,
                       a.artifact_kind, a.support_status, r.acquired_at_ms
                FROM raw_sessions AS r
                LEFT JOIN latest_artifact AS a
                  ON a.raw_id = r.raw_id
                 AND a.origin = r.origin
                 AND a.source_path = r.source_path
                 AND a.source_index = r.source_index
                 AND a.artifact_rank = 1
                WHERE (r.parse_error IS NOT NULL AND TRIM(r.parse_error) != '')
                   OR r.validation_status = 'failed'
            )
            """
        else:
            failed_cte = """
            WITH failed AS (
                SELECT r.raw_id, r.origin, r.validation_status,
                       NULL AS artifact_kind, NULL AS support_status, r.acquired_at_ms
                FROM raw_sessions AS r
                WHERE (r.parse_error IS NOT NULL AND TRIM(r.parse_error) != '')
                   OR r.validation_status = 'failed'
            )
            """
        summary_rows = conn.execute(
            failed_cte
            + """
            SELECT origin, validation_status, artifact_kind, support_status, COUNT(*) AS failure_count
            FROM failed
            GROUP BY origin, validation_status, artifact_kind, support_status
            ORDER BY origin, validation_status, artifact_kind, support_status
            """
        ).fetchall()
        sample_rows = conn.execute(
            failed_cte
            + """
            SELECT raw_id, origin, validation_status, artifact_kind, support_status
            FROM failed
            ORDER BY acquired_at_ms DESC, raw_id DESC
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("could not read raw failure lifecycle", exc_info=exc)
        return RawFailureLifecycleSnapshot(False, reason=f"could not read raw failure lifecycle: {exc}")
    finally:
        conn.close()

    by_origin: Counter[str] = Counter()
    by_artifact_kind: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    samples: list[dict[str, str | None]] = []
    for row in summary_rows:
        origin = str(row[0] or "unknown")
        artifact_kind = str(row[2]) if row[2] is not None else None
        support_status = str(row[3]) if row[3] is not None else None
        validation_failed = str(row[1] or "") == "failed"
        lifecycle = _lifecycle(artifact_kind, support_status, validation_failed=validation_failed)
        count = int(row[4])
        counts[lifecycle] += count
        by_origin[origin] += count
        by_artifact_kind[artifact_kind or "<none>"] += count
    for row in sample_rows:
        origin = str(row[1] or "unknown")
        artifact_kind = str(row[3]) if row[3] is not None else None
        support_status = str(row[4]) if row[4] is not None else None
        samples.append(
            {
                "raw_id": str(row[0]),
                "origin": origin,
                "artifact_kind": artifact_kind,
                "support_status": support_status,
                "lifecycle": _lifecycle(
                    artifact_kind,
                    support_status,
                    validation_failed=str(row[2] or "") == "failed",
                ),
            }
        )
    return RawFailureLifecycleSnapshot(
        available=True,
        parse_failures=parse_failures,
        validation_failures=validation_failures,
        deferred=counts["deferred"],
        terminal=counts["terminal"],
        unexplained=counts["unexplained"],
        by_origin=tuple(sorted(by_origin.items())),
        by_artifact_kind=tuple(sorted(by_artifact_kind.items())),
        samples=tuple(samples),
    )


__all__ = ["RawFailureLifecycleSnapshot", "RawFailureLifecycleState", "read_raw_failure_lifecycle"]
