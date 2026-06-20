"""Unified archive debt projection for operator surfaces."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.daemon.convergence_debt_status import convergence_debt_summary_info
from polylogue.daemon.embedding_readiness import embedding_readiness_info
from polylogue.daemon.fts_status import fts_readiness_info
from polylogue.maintenance.targets import MAINTENANCE_TARGET_NAMES
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.surfaces.payloads import (
    ArchiveDebtActionPayload,
    ArchiveDebtKind,
    ArchiveDebtListPayload,
    ArchiveDebtRowPayload,
    ArchiveDebtSeverity,
    ArchiveDebtTotalsPayload,
)

_CONVERGENCE_STAGE_MAINTENANCE_TARGETS = {
    "fts": "dangling_fts",
    "embed": "message_embeddings",
    "insights": "session_insights",
    "session_insights": "session_insights",
}


def archive_debt_list(
    *,
    archive_root: Path,
    kinds: Iterable[str] | None = None,
    only_actionable: bool = False,
    limit: int | None = None,
    exact_fts: bool = False,
) -> ArchiveDebtListPayload:
    """Return a unified archive debt report across current readiness providers."""
    generated_at = datetime.now(UTC).isoformat()
    selected_kinds = _selected_kinds(kinds)
    index_db = archive_root / "index.db"
    rows: list[ArchiveDebtRowPayload] = []

    if _include("archive-tier", selected_kinds):
        rows.extend(_tier_rows(archive_root))
    if _include("convergence", selected_kinds):
        rows.extend(_convergence_rows(index_db))
    if _include("embedding", selected_kinds):
        rows.extend(_embedding_rows(index_db))
    if _include("fts", selected_kinds):
        rows.extend(_fts_rows(index_db, exact=exact_fts))

    rows = sorted(rows, key=_row_sort_key)
    if only_actionable:
        rows = [row for row in rows if row.status == "actionable"]
    if limit is not None:
        rows = rows[: max(0, limit)]
    return ArchiveDebtListPayload(
        generated_at=generated_at,
        archive_root=str(archive_root),
        rows=tuple(rows),
        totals=_totals(rows),
        caveats=(
            "Rows are composed from existing readiness projections; exact FTS reconciliation requires --exact-fts.",
        )
        if not exact_fts
        else (),
    )


def _selected_kinds(kinds: Iterable[str] | None) -> set[str] | None:
    if kinds is None:
        return None
    selected = {kind for kind in kinds if kind}
    return selected or None


def _include(kind: ArchiveDebtKind, selected: set[str] | None) -> bool:
    return selected is None or kind in selected


def _tier_rows(archive_root: Path) -> list[ArchiveDebtRowPayload]:
    rows: list[ArchiveDebtRowPayload] = []
    for tier, spec in ARCHIVE_TIER_SPECS.items():
        path = archive_root / spec.filename
        subject_ref = f"archive-tier:{tier.value}"
        if not path.exists():
            rows.append(
                ArchiveDebtRowPayload(
                    debt_ref=f"debt:archive-tier:{tier.value}:missing",
                    kind="archive-tier",
                    stage="archive-layout",
                    subject_ref=subject_ref,
                    severity=_tier_missing_severity(spec.durability),
                    status="actionable",
                    owner="ops",
                    summary=f"{spec.filename} is missing",
                    details=f"Expected {spec.durability} archive tier at {path}.",
                    evidence_refs=(f"file:{path}",),
                    actions=(
                        ArchiveDebtActionPayload(
                            label="Initialize archive tiers",
                            command=("polylogue", "ops", "maintenance", "archive-init"),
                        ),
                    ),
                )
            )
            continue
        version = _read_user_version(path)
        if version != spec.version:
            rows.append(
                ArchiveDebtRowPayload(
                    debt_ref=f"debt:archive-tier:{tier.value}:version",
                    kind="archive-tier",
                    stage="archive-layout",
                    subject_ref=subject_ref,
                    severity="critical",
                    status="blocked" if version is None else "actionable",
                    owner="ops",
                    summary=f"{spec.filename} schema version is not current",
                    details=f"Expected PRAGMA user_version {spec.version}, observed {version!r}.",
                    evidence_refs=(f"file:{path}",),
                    caveats=("Unreadable SQLite files are reported as version debt.",) if version is None else (),
                    actions=(
                        ArchiveDebtActionPayload(
                            label="Reset and rebuild archive",
                            command=("polylogue", "ops", "reset", "--database"),
                        ),
                    ),
                )
            )
    return rows


def _tier_missing_severity(durability: str) -> ArchiveDebtSeverity:
    if durability in {"irreplaceable", "human"}:
        return "critical"
    return "warning"


def _read_user_version(path: Path) -> int | None:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        return _int_value(row[0] if row is not None else None)
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _convergence_rows(index_db: Path) -> list[ArchiveDebtRowPayload]:
    summary = convergence_debt_summary_info(index_db)
    rows: list[ArchiveDebtRowPayload] = []
    for item in summary.recent:
        subject_ref = f"{item.subject_type}:{item.subject_id}"
        actions = _convergence_actions(item.stage) if item.retry_due else ()
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref=f"debt:convergence:{item.stage}:{item.subject_type}:{item.subject_id}",
                kind="convergence",
                stage=item.stage,
                subject_ref=subject_ref,
                severity="critical" if item.retry_due else "warning",
                status="actionable" if item.retry_due else "blocked",
                owner="daemon",
                summary=f"Convergence debt failed for {subject_ref}",
                details=item.last_error,
                source_family=_source_family(item.subject_type, item.subject_id),
                observed_at=item.last_failed_at,
                evidence_refs=(f"ops-db:{index_db.with_name('ops.db')}",),
                actions=actions,
            )
        )
    return rows


def _convergence_actions(stage: str) -> tuple[ArchiveDebtActionPayload, ...]:
    target = _CONVERGENCE_STAGE_MAINTENANCE_TARGETS.get(stage)
    if target is None or target not in MAINTENANCE_TARGET_NAMES:
        return ()
    return (
        ArchiveDebtActionPayload(
            label="Run maintenance",
            command=("polylogue", "ops", "maintenance", "run", "--target", target),
        ),
    )


def _source_family(subject_type: str, subject_id: str) -> str:
    from polylogue.daemon.convergence_debt_alert import source_family_for_subject

    return source_family_for_subject(subject_type, subject_id)


def _embedding_rows(index_db: Path) -> list[ArchiveDebtRowPayload]:
    info = embedding_readiness_info(index_db, detail=False)
    rows: list[ArchiveDebtRowPayload] = []
    config_enabled = _bool_value(info.get("embedding_config_enabled"))
    has_key = _bool_value(info.get("embedding_has_voyage_key"))
    enabled = _bool_value(info.get("embedding_enabled"))
    pending = _int_value(info.get("embedding_pending_count")) or 0
    pending_messages = _int_value(info.get("embedding_pending_message_count")) or 0
    stale = _int_value(info.get("embedding_stale_count")) or 0
    failures = _int_value(info.get("embedding_failure_count")) or 0

    if config_enabled and not has_key:
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref="debt:embedding:configuration:missing-voyage-key",
                kind="embedding",
                stage="configuration",
                subject_ref="embedding:configuration",
                severity="critical",
                status="blocked",
                owner="ops",
                summary="Embedding is enabled but no Voyage API key is configured",
                actions=(
                    ArchiveDebtActionPayload(
                        label="Configure embeddings",
                        command=("polylogue", "ops", "embed", "enable"),
                    ),
                ),
            )
        )
    if failures:
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref="debt:embedding:catchup:failures",
                kind="embedding",
                stage="catchup",
                subject_ref="embedding:catchup",
                severity="critical",
                status="actionable" if enabled else "blocked",
                owner="daemon",
                summary=f"{failures} embedding catch-up failure(s) recorded",
                evidence_refs=(f"archive-tier:{index_db.with_name('embeddings.db')}",),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Inspect embedding status",
                        command=("polylogue", "ops", "embed", "status", "--detail"),
                    ),
                ),
            )
        )
    if pending or stale:
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref="debt:embedding:catchup:backlog",
                kind="embedding",
                stage="catchup",
                subject_ref="embedding:pending",
                severity="warning",
                status="actionable" if enabled else "blocked",
                owner="daemon",
                summary=f"{pending} session(s) pending embedding catch-up",
                details=f"Pending messages: {pending_messages}; stale messages: {stale}.",
                evidence_refs=(f"archive-tier:{index_db.with_name('embeddings.db')}",),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Run embedding backfill",
                        command=("polylogue", "ops", "embed", "backfill"),
                    ),
                )
                if enabled
                else (),
            )
        )
    return rows


def _fts_rows(index_db: Path, *, exact: bool) -> list[ArchiveDebtRowPayload]:
    info = fts_readiness_info(index_db, exact=exact)
    surfaces = info.get("surfaces")
    if not isinstance(surfaces, Mapping):
        if _bool_value(info.get("invariant_ready")):
            return []
        return [
            ArchiveDebtRowPayload(
                debt_ref="debt:fts:readiness:unknown",
                kind="fts",
                stage="readiness",
                subject_ref="fts:*",
                severity="warning",
                status="actionable",
                owner="ops",
                summary="FTS readiness could not be proven",
                evidence_refs=(f"archive-tier:{index_db}",),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Run FTS maintenance",
                        command=("polylogue", "ops", "maintenance", "run", "--target", "dangling_fts"),
                    ),
                ),
            )
        ]
    rows: list[ArchiveDebtRowPayload] = []
    for name, raw_surface in surfaces.items():
        if not isinstance(raw_surface, Mapping):
            continue
        surface = raw_surface
        if _bool_value(surface.get("ready")):
            continue
        source_exists = _bool_value(surface.get("source_exists"))
        exists = _bool_value(surface.get("exists"))
        triggers_present = _bool_value(surface.get("triggers_present"))
        missing = _int_value(surface.get("missing_rows")) or 0
        excess = _int_value(surface.get("excess_rows")) or 0
        duplicate = _int_value(surface.get("duplicate_rows")) or 0
        problems: list[str] = []
        if source_exists and not exists:
            problems.append("index table missing")
        if exists and not triggers_present:
            problems.append("sync triggers missing")
        if missing:
            problems.append(f"{missing} missing row(s)")
        if excess:
            problems.append(f"{excess} excess row(s)")
        if duplicate:
            problems.append(f"{duplicate} duplicate row(s)")
        if not problems:
            problems.append("freshness not ready")
        severity: ArchiveDebtSeverity = "critical" if not triggers_present or not exists else "warning"
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref=f"debt:fts:{name}",
                kind="fts",
                stage="index-freshness",
                subject_ref=f"fts:{name}",
                severity=severity,
                status="actionable",
                owner="ops",
                summary=f"{name} is not query-ready",
                details="; ".join(problems),
                evidence_refs=(f"archive-tier:{index_db}",),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Run FTS maintenance",
                        command=("polylogue", "ops", "maintenance", "run", "--target", "dangling_fts"),
                    ),
                ),
            )
        )
    return rows


def _totals(rows: list[ArchiveDebtRowPayload]) -> ArchiveDebtTotalsPayload:
    return ArchiveDebtTotalsPayload(
        total=len(rows),
        critical=sum(1 for row in rows if row.severity == "critical"),
        warning=sum(1 for row in rows if row.severity == "warning"),
        info=sum(1 for row in rows if row.severity == "info"),
        actionable=sum(1 for row in rows if row.status == "actionable"),
        blocked=sum(1 for row in rows if row.status == "blocked"),
    )


def _row_sort_key(row: ArchiveDebtRowPayload) -> tuple[int, int, str, str]:
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    status_order = {"actionable": 0, "blocked": 1, "open": 2}
    return (
        severity_order[row.severity],
        status_order[row.status],
        row.kind,
        row.debt_ref,
    )


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _bool_value(value: Any) -> bool:
    return bool(value)


__all__ = ["archive_debt_list"]
