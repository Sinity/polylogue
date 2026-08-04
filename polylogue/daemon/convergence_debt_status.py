"""Status projection for durable daemon convergence debt."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.core.payload_coercion import optional_str as _optional_str
from polylogue.core.payload_coercion import required_str as _required_str
from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.logging import get_logger
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)


class ConvergenceDebtStageSummary(BaseModel):
    stage: str
    failed_count: int = 0
    deferred_count: int = 0
    retry_due_count: int = 0


class ConvergenceDebtFamilySummary(BaseModel):
    """Convergence-debt counts bucketed by inferred source family (#1226)."""

    family: str
    failed_count: int = 0
    deferred_count: int = 0


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
    available: bool = True
    error: str | None = None
    failed_count: int = 0
    deferred_count: int = 0
    retry_due_count: int = 0
    stage_summaries: list[ConvergenceDebtStageSummary] = Field(default_factory=list)
    family_summaries: list[ConvergenceDebtFamilySummary] = Field(default_factory=list)
    recent: list[ConvergenceDebtItem] = Field(default_factory=list)


def convergence_debt_summary_info(dbf: Path, *, ops_db: Path | None = None) -> ConvergenceDebtSummary:
    """Return durable post-ingest convergence debt snapshots."""
    resolved_ops_db = ops_db if ops_db is not None else dbf.with_name("ops.db")
    return _archive_convergence_debt_summary_info(dbf, resolved_ops_db)


def _archive_convergence_debt_summary_info(dbf: Path, ops_db: Path) -> ConvergenceDebtSummary:
    """Return the archive ops convergence-debt projection when populated."""
    if not ops_db.exists():
        return ConvergenceDebtSummary(available=False, error=f"convergence debt database is missing: {ops_db}")
    try:
        conn = open_readonly_connection(ops_db)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'convergence_debt'"
            ).fetchone()
            if has_table is None:
                return ConvergenceDebtSummary(
                    available=False,
                    error="convergence debt table is unavailable",
                )
            rows = conn.execute(
                """
                SELECT stage, target_type, target_id, status, attempts,
                       updated_at_ms, last_error, next_retry_at
                FROM convergence_debt
                ORDER BY updated_at_ms DESC, priority DESC, debt_id DESC
                """
            ).fetchall()
            if not rows:
                return ConvergenceDebtSummary()
        finally:
            conn.close()
    except Exception as exc:
        # An unreadable authoritative debt ledger is unknown, not empty. The
        # claim guard must be able to distinguish this from a genuinely empty
        # convergence_debt table (polylogue-cpf.4).
        logger.warning("convergence-debt summary query failed for %s: %s", ops_db, exc, exc_info=True)
        return ConvergenceDebtSummary(available=False, error=f"convergence debt status unavailable: {exc}")

    try:
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
        ]
        if not items:
            return ConvergenceDebtSummary()

        now = datetime.now(UTC)
        recent = [
            item.model_copy(update={"retry_due": item.status == "failed" and _retry_due(item.next_retry_at, now=now)})
            for item in items[:10]
        ]
        failed_by_stage: dict[str, int] = {}
        deferred_by_stage: dict[str, int] = {}
        retry_due_by_stage: dict[str, int] = {}
        for item in items:
            if item.status == "failed":
                failed_by_stage[item.stage] = failed_by_stage.get(item.stage, 0) + 1
                if _retry_due(item.next_retry_at, now=now):
                    retry_due_by_stage[item.stage] = retry_due_by_stage.get(item.stage, 0) + 1
            elif item.status == "deferred":
                deferred_by_stage[item.stage] = deferred_by_stage.get(item.stage, 0) + 1
        stage_summaries = [
            ConvergenceDebtStageSummary(
                stage=stage,
                failed_count=failed_by_stage.get(stage, 0),
                deferred_count=deferred_by_stage.get(stage, 0),
                retry_due_count=retry_due_by_stage.get(stage, 0),
            )
            for stage in sorted(
                set(failed_by_stage) | set(deferred_by_stage),
                key=lambda stage: (-(failed_by_stage.get(stage, 0) + deferred_by_stage.get(stage, 0)), stage),
            )
        ]
        return _summary_from_parts(stage_summaries=stage_summaries, recent=recent)
    except Exception as exc:
        logger.warning("convergence-debt summary projection failed for %s: %s", ops_db, exc, exc_info=True)
        return ConvergenceDebtSummary(available=False, error=f"convergence debt status unavailable: {exc}")


def _summary_from_parts(
    *,
    stage_summaries: list[ConvergenceDebtStageSummary],
    recent: list[ConvergenceDebtItem],
) -> ConvergenceDebtSummary:
    # Per-family rollup over the recent items so polylogue ops status and the
    # /health envelope show which source family the debt belongs to. This
    # is the same view the convergence-debt alert (see
    # polylogue/daemon/convergence_debt_alert.py) thresholds against, so
    # operators can correlate alert messages with status output directly.
    from polylogue.daemon.convergence_debt_alert import source_family_for_subject

    family_counts: dict[str, dict[str, int]] = {}
    for item in recent:
        family = source_family_for_subject(item.subject_type, item.subject_id)
        counts = family_counts.setdefault(family, {"failed": 0, "deferred": 0})
        if item.status in counts:
            counts[item.status] += 1
    family_summaries = [
        ConvergenceDebtFamilySummary(
            family=family,
            failed_count=counts["failed"],
            deferred_count=counts["deferred"],
        )
        for family, counts in sorted(
            family_counts.items(), key=lambda kv: (-(kv[1]["failed"] + kv[1]["deferred"]), kv[0])
        )
    ]

    return ConvergenceDebtSummary(
        failed_count=sum(item.failed_count for item in stage_summaries),
        deferred_count=sum(item.deferred_count for item in stage_summaries),
        retry_due_count=sum(item.retry_due_count for item in stage_summaries),
        stage_summaries=stage_summaries,
        family_summaries=family_summaries,
        recent=recent,
    )


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
