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
    RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS,
    raw_failure_outcome_code,
    validated_raw_failure_evidence_kind,
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
    classification_reason: object = None,
) -> RawFailureLifecycle:
    """Classify an artifact only when its closed evidence is self-consistent."""
    evidence_kind = validated_raw_failure_evidence_kind(
        artifact_kind,
        support_status,
        validation_failed=validation_failed,
        classification_reason=classification_reason,
        outcome_code=raw_failure_outcome_code(classification_reason),
    )
    if evidence_kind is None:
        return "unexplained"
    if evidence_kind.lifecycle == "deferred":
        return "deferred"
    if evidence_kind.lifecycle == "terminal":
        return "terminal"
    return "unexplained"


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
        failed_cte = """
        WITH failed AS (
            SELECT r.raw_id, r.origin, r.source_path, r.source_index,
                   r.validation_status, r.acquired_at_ms
            FROM raw_sessions AS r
            WHERE (r.parse_error IS NOT NULL AND TRIM(r.parse_error) != '')
               OR r.validation_status = 'failed'
        )
        """
        typed_failure_placeholders = ", ".join("(?, ?)" for _ in RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS)
        valid_failure_artifacts_cte = (
            """
        , valid_failure_artifacts AS (
            SELECT a.*
            FROM raw_artifacts AS a
            WHERE (a.artifact_kind, a.support_status) IN ("""
            + typed_failure_placeholders
            + """)
        )
        """
        )
        latest_artifact_join = """
        LEFT JOIN valid_failure_artifacts AS a
          ON a.raw_id IS f.raw_id
         AND a.origin IS f.origin
         AND a.source_path IS f.source_path
         AND a.source_index IS f.source_index
         AND NOT EXISTS (
             SELECT 1
             FROM valid_failure_artifacts AS newer
              WHERE newer.raw_id IS a.raw_id
               AND newer.origin IS a.origin
               AND newer.source_path IS a.source_path
               AND newer.source_index IS a.source_index
               AND (newer.last_observed_at_ms > a.last_observed_at_ms
                    OR (newer.last_observed_at_ms = a.last_observed_at_ms
                        AND newer.artifact_id > a.artifact_id))
         )
        """
        if has_artifacts:
            summary_sql = (
                failed_cte
                + valid_failure_artifacts_cte
                + """
                SELECT f.origin, f.validation_status, a.artifact_kind, a.support_status,
                       a.classification_reason,
                       COUNT(*) AS failure_count
                FROM failed AS f
                """
                + latest_artifact_join
                + """
                GROUP BY f.origin, f.validation_status, a.artifact_kind, a.support_status,
                         a.classification_reason
                ORDER BY f.origin, f.validation_status, a.artifact_kind, a.support_status,
                         a.classification_reason
                """
            )
            sample_sql = (
                failed_cte
                + valid_failure_artifacts_cte
                + """
                , sampled AS (
                    SELECT f.raw_id, f.origin, f.validation_status, f.acquired_at_ms,
                           a.artifact_kind, a.support_status, a.classification_reason
                    FROM failed AS f
                    """
                + latest_artifact_join
                + """
                    ORDER BY CASE
                        WHEN a.artifact_kind IS NOT NULL THEN 0
                        ELSE 2
                    END,
                    f.acquired_at_ms DESC, f.raw_id DESC
                    LIMIT ?
                )
                SELECT raw_id, origin, validation_status, artifact_kind, support_status,
                       classification_reason
                FROM sampled
                """
            )
        else:
            summary_sql = (
                failed_cte
                + """
                SELECT f.origin, f.validation_status, NULL, NULL, NULL, COUNT(*) AS failure_count
                FROM failed AS f
                GROUP BY f.origin, f.validation_status
                ORDER BY f.origin, f.validation_status
                """
            )
            sample_sql = (
                failed_cte
                + """
                SELECT raw_id, origin, validation_status, NULL, NULL, NULL
                FROM failed
                ORDER BY acquired_at_ms DESC, raw_id DESC
                LIMIT ?
                """
            )
        typed_failure_params = tuple(
            value for pair in RAW_FAILURE_LIFECYCLE_EVIDENCE_SUPPORT_STATUS_PAIRS for value in pair
        )
        summary_rows = conn.execute(summary_sql, typed_failure_params if has_artifacts else ()).fetchall()
        sample_params: tuple[object, ...] = typed_failure_params + (sample_limit,) if has_artifacts else (sample_limit,)
        sample_rows = conn.execute(sample_sql, sample_params).fetchall()
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
        classification_reason = row[4]
        validation_failed = str(row[1] or "") == "failed"
        lifecycle = _lifecycle(
            artifact_kind,
            support_status,
            validation_failed=validation_failed,
            classification_reason=classification_reason,
        )
        count = int(row[5])
        counts[lifecycle] += count
        by_origin[origin] += count
        by_artifact_kind[artifact_kind or "<none>"] += count
    for row in sample_rows:
        origin = str(row[1] or "unknown")
        artifact_kind = str(row[3]) if row[3] is not None else None
        support_status = str(row[4]) if row[4] is not None else None
        classification_reason = row[5]
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
                    classification_reason=classification_reason,
                ),
                "classification_reason": str(classification_reason) if classification_reason is not None else None,
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
