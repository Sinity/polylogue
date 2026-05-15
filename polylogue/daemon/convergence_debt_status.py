"""Status projection for durable daemon convergence debt."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.storage.sqlite.connection_profile import open_readonly_connection


class ConvergenceDebtStageSummary(BaseModel):
    stage: str
    failed_count: int = 0
    retry_due_count: int = 0


class ConvergenceDebtItem(BaseModel):
    stage: str
    subject_type: str
    subject_id: str
    status: str
    failure_count: int = 0
    last_failed_at: str
    next_retry_at: str | None = None
    retry_due: bool = False
    last_error: str | None = None


class ConvergenceDebtSummary(BaseModel):
    failed_count: int = 0
    retry_due_count: int = 0
    stage_summaries: list[ConvergenceDebtStageSummary] = Field(default_factory=list)
    recent: list[ConvergenceDebtItem] = Field(default_factory=list)


def convergence_debt_summary_info(dbf: Path) -> ConvergenceDebtSummary:
    """Return durable post-ingest convergence debt snapshots."""
    if not dbf.exists():
        return ConvergenceDebtSummary()
    try:
        conn = open_readonly_connection(dbf)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'live_convergence_debt'"
            ).fetchone()
            if has_table is None:
                return ConvergenceDebtSummary()
            stage_rows = conn.execute(
                """
                SELECT stage, next_retry_at
                FROM live_convergence_debt
                WHERE status = 'failed'
                ORDER BY stage
                """
            ).fetchall()
            recent_rows = conn.execute(
                """
                SELECT stage, subject_type, subject_id, status, failure_count,
                       last_failed_at, next_retry_at, last_error
                FROM live_convergence_debt
                WHERE status = 'failed'
                ORDER BY last_failed_at DESC
                LIMIT 10
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return ConvergenceDebtSummary()

    now = datetime.now(UTC)
    failed_by_stage: dict[str, int] = {}
    retry_due_by_stage: dict[str, int] = {}
    for row in stage_rows:
        stage = _required_str(row[0])
        failed_by_stage[stage] = failed_by_stage.get(stage, 0) + 1
        if _retry_due(_optional_str(row[1]), now=now):
            retry_due_by_stage[stage] = retry_due_by_stage.get(stage, 0) + 1
    stage_summaries = [
        ConvergenceDebtStageSummary(
            stage=stage,
            failed_count=failed_count,
            retry_due_count=retry_due_by_stage.get(stage, 0),
        )
        for stage, failed_count in sorted(failed_by_stage.items(), key=lambda item: (-item[1], item[0]))
    ]
    recent = [
        ConvergenceDebtItem(
            stage=_required_str(row[0]),
            subject_type=_required_str(row[1]),
            subject_id=_required_str(row[2]),
            status=_required_str(row[3]),
            failure_count=_row_int(row[4]),
            last_failed_at=_required_str(row[5]),
            next_retry_at=_optional_str(row[6]),
            retry_due=_retry_due(_optional_str(row[6]), now=now),
            last_error=_optional_str(row[7]),
        )
        for row in recent_rows
    ]
    return ConvergenceDebtSummary(
        failed_count=sum(item.failed_count for item in stage_summaries),
        retry_due_count=sum(item.retry_due_count for item in stage_summaries),
        stage_summaries=stage_summaries,
        recent=recent,
    )


def _required_str(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _row_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _retry_due(next_retry_at: str | None, *, now: datetime) -> bool:
    if not next_retry_at:
        return True
    try:
        retry_at = datetime.fromisoformat(next_retry_at)
    except ValueError:
        return True
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return retry_at <= now


__all__ = [
    "ConvergenceDebtItem",
    "ConvergenceDebtStageSummary",
    "ConvergenceDebtSummary",
    "convergence_debt_summary_info",
]
