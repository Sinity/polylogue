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


class ConvergenceDebtFamilySummary(BaseModel):
    """Convergence-debt counts bucketed by inferred source family (#1226)."""

    family: str
    failed_count: int = 0


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
    family_summaries: list[ConvergenceDebtFamilySummary] = Field(default_factory=list)
    recent: list[ConvergenceDebtItem] = Field(default_factory=list)


def convergence_debt_summary_info(dbf: Path) -> ConvergenceDebtSummary:
    """Return durable post-ingest convergence debt snapshots."""
    ops_summary = _archive_convergence_debt_summary_info(dbf, dbf.with_name("ops.db"))
    if ops_summary is not None:
        return ops_summary
    return ConvergenceDebtSummary()


def _archive_convergence_debt_summary_info(dbf: Path, ops_db: Path) -> ConvergenceDebtSummary | None:
    """Return the archive ops convergence-debt projection when populated."""
    if not ops_db.exists():
        return None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'convergence_debt'"
            ).fetchone()
            if has_table is None:
                return None
            rows = conn.execute(
                """
                SELECT stage, target_type, target_id, status, attempts,
                       updated_at_ms, last_error, next_retry_at
                FROM convergence_debt
                ORDER BY updated_at_ms DESC, priority DESC, debt_id DESC
                """
            ).fetchall()
            if not rows:
                return None
        finally:
            conn.close()
    except sqlite3.Error:
        return None

    items = [
        ConvergenceDebtItem(
            stage=_required_str(row[0]),
            subject_type=_required_str(row[1]),
            subject_id=_required_str(row[2]),
            status=_required_str(row[3]),
            failure_count=_row_int(row[4]),
            last_failed_at=_iso_from_epoch_ms(row[5]),
            next_retry_at=_optional_str(row[7]),
            retry_due=False,
            last_error=_optional_str(row[6]),
        )
        for row in rows
        if _required_str(row[3]) == "failed"
    ]
    if not items:
        return ConvergenceDebtSummary()

    now = datetime.now(UTC)
    recent = [item.model_copy(update={"retry_due": _retry_due(item.next_retry_at, now=now)}) for item in items[:10]]
    failed_by_stage: dict[str, int] = {}
    retry_due_by_stage: dict[str, int] = {}
    for item in items:
        failed_by_stage[item.stage] = failed_by_stage.get(item.stage, 0) + 1
        if _retry_due(item.next_retry_at, now=now):
            retry_due_by_stage[item.stage] = retry_due_by_stage.get(item.stage, 0) + 1
    stage_summaries = [
        ConvergenceDebtStageSummary(
            stage=stage,
            failed_count=failed_count,
            retry_due_count=retry_due_by_stage.get(stage, 0),
        )
        for stage, failed_count in sorted(failed_by_stage.items(), key=lambda item: (-item[1], item[0]))
    ]
    return _summary_from_parts(stage_summaries=stage_summaries, recent=recent)


def _summary_from_parts(
    *,
    stage_summaries: list[ConvergenceDebtStageSummary],
    recent: list[ConvergenceDebtItem],
) -> ConvergenceDebtSummary:
    # Per-family rollup over the recent items so polylogue status and the
    # /health envelope show which source family the debt belongs to. This
    # is the same view the convergence-debt alert (see
    # polylogue/daemon/convergence_debt_alert.py) thresholds against, so
    # operators can correlate alert messages with status output directly.
    from polylogue.daemon.convergence_debt_alert import source_family_for_subject

    family_counts: dict[str, int] = {}
    for item in recent:
        family = source_family_for_subject(item.subject_type, item.subject_id)
        family_counts[family] = family_counts.get(family, 0) + 1
    family_summaries = [
        ConvergenceDebtFamilySummary(family=family, failed_count=count)
        for family, count in sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    return ConvergenceDebtSummary(
        failed_count=sum(item.failed_count for item in stage_summaries),
        retry_due_count=sum(item.retry_due_count for item in stage_summaries),
        stage_summaries=stage_summaries,
        family_summaries=family_summaries,
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


def _iso_from_epoch_ms(value: object) -> str:
    epoch_ms = _row_int(value)
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat()


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
    "ConvergenceDebtFamilySummary",
    "ConvergenceDebtItem",
    "ConvergenceDebtStageSummary",
    "ConvergenceDebtSummary",
    "convergence_debt_summary_info",
]
