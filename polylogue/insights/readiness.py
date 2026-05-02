"""Typed insight materialization readiness reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import aiosqlite
from pydantic import Field

from polylogue.insights.archive_models import ARCHIVE_INSIGHT_CONTRACT_VERSION, ArchiveInsightModel
from polylogue.maintenance.targets import build_maintenance_target_catalog
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.runtime.store_constants import (
    SESSION_ENRICHMENT_VERSION,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
)

InsightReadinessVerdict = Literal["ready", "partial", "empty", "missing", "stale", "legacy", "unknown"]
_REPAIR_HINT = build_maintenance_target_catalog().repair_hint(("session_products",), include_run_all=True)


class InsightReadinessQuery(ArchiveInsightModel):
    insights: tuple[str, ...] = ()
    provider: str | None = None
    since: str | None = None
    until: str | None = None


class InsightStorageArtifact(ArchiveInsightModel):
    name: str
    present: bool
    ready: bool | None = None


class InsightVersionCoverage(ArchiveInsightModel):
    field: str
    current_version: int
    versions: dict[str, int] = Field(default_factory=dict)
    legacy_count: int = 0


class InsightProviderCoverage(ArchiveInsightModel):
    provider_name: str
    row_count: int
    min_time: str | None = None
    max_time: str | None = None


class InsightReadinessEntry(ArchiveInsightModel):
    insight_name: str
    display_name: str
    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    verdict: InsightReadinessVerdict = "unknown"
    row_count: int = 0
    expected_row_count: int | None = None
    missing_count: int = 0
    stale_count: int = 0
    orphan_count: int = 0
    legacy_incompatible_count: int = 0
    storage_artifacts: tuple[InsightStorageArtifact, ...] = ()
    ready_flags: dict[str, bool] = Field(default_factory=dict)
    provider_coverage: tuple[InsightProviderCoverage, ...] = ()
    version_coverage: tuple[InsightVersionCoverage, ...] = ()
    schema_contract_issues: tuple[str, ...] = ()
    min_time: str | None = None
    max_time: str | None = None
    repair_command: str = _REPAIR_HINT
    evidence: tuple[str, ...] = ()


class InsightReadinessReport(ArchiveInsightModel):
    checked_at: str
    aggregate_verdict: InsightReadinessVerdict
    total_conversations: int = 0
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    insights: tuple[InsightReadinessEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class InsightReadinessSpec:
    insight_name: str
    display_name: str
    table_name: str | None
    row_count_attr: str
    expected_count_attr: str | None = None
    missing_count_attr: str | None = None
    stale_count_attr: str | None = None
    orphan_count_attr: str | None = None
    ready_flags: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    provider_column: str | None = "provider_name"
    time_column: str | None = "source_updated_at"
    version_fields: tuple[tuple[str, int], ...] = (("materializer_version", SESSION_INSIGHT_MATERIALIZER_VERSION),)


_SPECS: tuple[InsightReadinessSpec, ...] = (
    InsightReadinessSpec(
        insight_name="session_profiles",
        display_name="Session Profiles",
        table_name="session_profiles",
        row_count_attr="profile_row_count",
        expected_count_attr="total_conversations",
        missing_count_attr="missing_profile_row_count",
        stale_count_attr="stale_profile_row_count",
        orphan_count_attr="orphan_profile_row_count",
        ready_flags=(
            "profile_rows_ready",
            "profile_merged_fts_ready",
            "profile_evidence_fts_ready",
            "profile_inference_fts_ready",
        ),
        artifacts=(
            "session_profiles",
            "session_profiles_fts",
            "session_profile_evidence_fts",
            "session_profile_inference_fts",
        ),
        time_column="source_updated_at",
        version_fields=(
            ("materializer_version", SESSION_INSIGHT_MATERIALIZER_VERSION),
            ("inference_version", SESSION_INFERENCE_VERSION),
        ),
    ),
    InsightReadinessSpec(
        insight_name="session_enrichments",
        display_name="Session Enrichments",
        table_name="session_profiles",
        row_count_attr="profile_row_count",
        expected_count_attr="total_conversations",
        missing_count_attr="missing_profile_row_count",
        stale_count_attr="stale_profile_row_count",
        orphan_count_attr="orphan_profile_row_count",
        ready_flags=("profile_rows_ready", "profile_enrichment_fts_ready"),
        artifacts=("session_profiles", "session_profile_enrichment_fts"),
        time_column="source_updated_at",
        version_fields=(
            ("materializer_version", SESSION_INSIGHT_MATERIALIZER_VERSION),
            ("enrichment_version", SESSION_ENRICHMENT_VERSION),
        ),
    ),
    InsightReadinessSpec(
        insight_name="session_work_events",
        display_name="Work Events",
        table_name="session_work_events",
        row_count_attr="work_event_inference_count",
        expected_count_attr="expected_work_event_inference_count",
        stale_count_attr="stale_work_event_inference_count",
        orphan_count_attr="orphan_work_event_inference_count",
        ready_flags=("work_event_inference_rows_ready", "work_event_inference_fts_ready"),
        artifacts=("session_work_events", "session_work_events_fts"),
        time_column="start_time",
        version_fields=(
            ("materializer_version", SESSION_INSIGHT_MATERIALIZER_VERSION),
            ("inference_version", SESSION_INFERENCE_VERSION),
        ),
    ),
    InsightReadinessSpec(
        insight_name="session_phases",
        display_name="Session Phases",
        table_name="session_phases",
        row_count_attr="phase_inference_count",
        expected_count_attr="expected_phase_inference_count",
        stale_count_attr="stale_phase_inference_count",
        orphan_count_attr="orphan_phase_inference_count",
        ready_flags=("phase_inference_rows_ready",),
        artifacts=("session_phases",),
        time_column="start_time",
        version_fields=(
            ("materializer_version", SESSION_INSIGHT_MATERIALIZER_VERSION),
            ("inference_version", SESSION_INFERENCE_VERSION),
        ),
    ),
    InsightReadinessSpec(
        insight_name="work_threads",
        display_name="Work Threads",
        table_name="work_threads",
        row_count_attr="thread_count",
        expected_count_attr="root_threads",
        stale_count_attr="stale_thread_count",
        orphan_count_attr="orphan_thread_count",
        ready_flags=("threads_ready", "threads_fts_ready"),
        artifacts=("work_threads", "work_threads_fts"),
        provider_column=None,
        time_column="end_time",
    ),
    InsightReadinessSpec(
        insight_name="session_tag_rollups",
        display_name="Session Tag Rollups",
        table_name="session_tag_rollups",
        row_count_attr="tag_rollup_count",
        expected_count_attr="expected_tag_rollup_count",
        stale_count_attr="stale_tag_rollup_count",
        ready_flags=("tag_rollups_ready",),
        artifacts=("session_tag_rollups",),
        time_column="bucket_day",
    ),
    InsightReadinessSpec(
        insight_name="day_session_summaries",
        display_name="Day Session Summaries",
        table_name="day_session_summaries",
        row_count_attr="day_summary_count",
        expected_count_attr="expected_day_summary_count",
        stale_count_attr="stale_day_summary_count",
        ready_flags=("day_summaries_ready",),
        artifacts=("day_session_summaries",),
        time_column="day",
    ),
    InsightReadinessSpec(
        insight_name="week_session_summaries",
        display_name="Week Session Summaries",
        table_name="day_session_summaries",
        row_count_attr="day_summary_count",
        expected_count_attr="expected_day_summary_count",
        stale_count_attr="stale_day_summary_count",
        ready_flags=("week_summaries_ready",),
        artifacts=("day_session_summaries",),
        time_column="day",
    ),
    InsightReadinessSpec(
        insight_name="provider_analytics",
        display_name="Provider Analytics",
        table_name="conversations",
        row_count_attr="total_conversations",
        ready_flags=(),
        artifacts=("conversations",),
        provider_column="provider_name",
        time_column="updated_at",
        version_fields=(),
    ),
)

_SPEC_BY_NAME = {spec.insight_name: spec for spec in _SPECS}
_ALIASES = {
    **{spec.insight_name.replace("_", "-"): spec.insight_name for spec in _SPECS},
    "profiles": "session_profiles",
    "enrichments": "session_enrichments",
    "work-events": "session_work_events",
    "phases": "session_phases",
    "threads": "work_threads",
    "tags": "session_tag_rollups",
    "day-summaries": "day_session_summaries",
    "week-summaries": "week_session_summaries",
    "analytics": "provider_analytics",
}


def known_insight_readiness_names() -> tuple[str, ...]:
    return tuple(spec.insight_name for spec in _SPECS)


def normalize_insight_readiness_name(value: str) -> str:
    normalized = value.strip().replace("-", "_")
    if normalized in _SPEC_BY_NAME:
        return normalized
    alias = _ALIASES.get(value.strip()) or _ALIASES.get(value.strip().replace("_", "-"))
    if alias is not None:
        return alias
    raise ValueError(f"Unknown insight readiness target: {value}")


def _count(status: SessionInsightStatusSnapshot, attr: str | None) -> int:
    if attr is None:
        return 0
    return int(getattr(status, attr))


def _artifact_ready(status: SessionInsightStatusSnapshot, artifact_name: str) -> bool | None:
    mapping = {
        "session_profiles": "profile_rows_ready",
        "session_profiles_fts": "profile_merged_fts_ready",
        "session_profile_evidence_fts": "profile_evidence_fts_ready",
        "session_profile_inference_fts": "profile_inference_fts_ready",
        "session_profile_enrichment_fts": "profile_enrichment_fts_ready",
        "session_work_events": "work_event_inference_rows_ready",
        "session_work_events_fts": "work_event_inference_fts_ready",
        "session_phases": "phase_inference_rows_ready",
        "work_threads": "threads_ready",
        "work_threads_fts": "threads_fts_ready",
        "session_tag_rollups": "tag_rollups_ready",
        "day_session_summaries": "day_summaries_ready",
    }
    attr = mapping.get(artifact_name)
    return bool(getattr(status, attr)) if attr is not None else None


def _entry_verdict(
    *,
    table_present: bool,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    legacy_count: int,
    ready_flags: dict[str, bool],
) -> InsightReadinessVerdict:
    if not table_present:
        return "missing"
    if legacy_count:
        return "legacy"
    if stale_count or orphan_count:
        return "stale"
    if missing_count or (expected_row_count is not None and row_count < expected_row_count):
        return "partial"
    if row_count == 0:
        return "empty"
    if ready_flags and all(ready_flags.values()):
        return "ready"
    if not ready_flags:
        return "ready"
    return "unknown"


def _aggregate_verdict(entries: tuple[InsightReadinessEntry, ...]) -> InsightReadinessVerdict:
    verdicts = {entry.verdict for entry in entries}
    priority: tuple[InsightReadinessVerdict, ...] = ("legacy", "stale", "partial", "missing", "unknown", "empty")
    for verdict in priority:
        if verdict in verdicts:
            return verdict
    return "ready"


async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    row = await (
        await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,))
    ).fetchone()
    return bool(row)


async def _table_columns(conn: aiosqlite.Connection, table: str) -> set[str]:
    rows = await (await conn.execute(f"PRAGMA table_info({table})")).fetchall()
    return {str(row[1]) for row in rows}


def _where_clause(
    spec: InsightReadinessSpec,
    query: InsightReadinessQuery,
    columns: set[str],
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if query.provider and spec.provider_column and spec.provider_column in columns:
        clauses.append(f"{spec.provider_column} = ?")
        params.append(query.provider)
    if query.since and spec.time_column and spec.time_column in columns:
        clauses.append(f"{spec.time_column} >= ?")
        params.append(query.since)
    if query.until and spec.time_column and spec.time_column in columns:
        clauses.append(f"{spec.time_column} <= ?")
        params.append(query.until)
    return (" WHERE " + " AND ".join(clauses), params) if clauses else ("", params)


async def _provider_coverage(
    conn: aiosqlite.Connection,
    spec: InsightReadinessSpec,
    query: InsightReadinessQuery,
    *,
    table_present: bool,
    columns: set[str],
) -> tuple[InsightProviderCoverage, ...]:
    if (
        not table_present
        or spec.table_name is None
        or spec.provider_column is None
        or spec.provider_column not in columns
    ):
        return ()
    where, params = _where_clause(spec, query, columns)
    time_min = f"MIN({spec.time_column})" if spec.time_column and spec.time_column in columns else "NULL"
    time_max = f"MAX({spec.time_column})" if spec.time_column and spec.time_column in columns else "NULL"
    sql = (
        f"SELECT {spec.provider_column} AS provider_name, COUNT(*) AS row_count, "
        f"{time_min} AS min_time, {time_max} AS max_time "
        f"FROM {spec.table_name}{where} GROUP BY {spec.provider_column} ORDER BY {spec.provider_column}"
    )
    rows = await (await conn.execute(sql, tuple(params))).fetchall()
    return tuple(
        InsightProviderCoverage(
            provider_name=str(row["provider_name"] or "unknown"),
            row_count=int(row["row_count"]),
            min_time=str(row["min_time"]) if row["min_time"] is not None else None,
            max_time=str(row["max_time"]) if row["max_time"] is not None else None,
        )
        for row in rows
    )


async def _version_coverage(
    conn: aiosqlite.Connection,
    spec: InsightReadinessSpec,
    *,
    table_present: bool,
    columns: set[str],
) -> tuple[InsightVersionCoverage, ...]:
    if not table_present or spec.table_name is None or not spec.version_fields:
        return ()
    coverage: list[InsightVersionCoverage] = []
    for field, current_version in spec.version_fields:
        if field not in columns:
            continue
        rows = await (
            await conn.execute(
                f"SELECT {field} AS version, COUNT(*) AS row_count FROM {spec.table_name} GROUP BY {field}"
            )
        ).fetchall()
        versions = {str(row["version"]): int(row["row_count"]) for row in rows}
        legacy_count = sum(count for version, count in versions.items() if version != str(current_version))
        coverage.append(
            InsightVersionCoverage(
                field=field,
                current_version=current_version,
                versions=versions,
                legacy_count=legacy_count,
            )
        )
    return tuple(coverage)


def _schema_contract_issues(spec: InsightReadinessSpec, columns: set[str]) -> tuple[str, ...]:
    issues: list[str] = []
    if spec.provider_column is not None and spec.provider_column not in columns:
        issues.append(f"missing provider column: {spec.provider_column}")
    if spec.time_column is not None and spec.time_column not in columns:
        issues.append(f"missing time column: {spec.time_column}")
    for field, _current_version in spec.version_fields:
        if field not in columns:
            issues.append(f"missing version field: {field}")
    return tuple(issues)


def _evidence(
    *,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    legacy_count: int,
    schema_contract_issues: tuple[str, ...],
    ready_flags: dict[str, bool],
) -> tuple[str, ...]:
    values = [f"rows={row_count}"]
    if expected_row_count is not None:
        values.append(f"expected={expected_row_count}")
    if missing_count:
        values.append(f"missing={missing_count}")
    if stale_count:
        values.append(f"stale={stale_count}")
    if orphan_count:
        values.append(f"orphan={orphan_count}")
    if legacy_count:
        values.append(f"legacy={legacy_count}")
    values.extend(f"schema_issue={issue}" for issue in schema_contract_issues)
    values.extend(f"{key}={value}" for key, value in sorted(ready_flags.items()))
    return tuple(values)


async def _entry(
    conn: aiosqlite.Connection,
    status: SessionInsightStatusSnapshot,
    spec: InsightReadinessSpec,
    query: InsightReadinessQuery,
) -> InsightReadinessEntry:
    table_present = bool(spec.table_name and await _table_exists(conn, spec.table_name))
    row_count = _count(status, spec.row_count_attr)
    expected_row_count = _count(status, spec.expected_count_attr) if spec.expected_count_attr is not None else None
    missing_count = _count(status, spec.missing_count_attr)
    stale_count = _count(status, spec.stale_count_attr)
    orphan_count = _count(status, spec.orphan_count_attr)
    ready_flags = {flag: bool(getattr(status, flag)) for flag in spec.ready_flags}
    columns = await _table_columns(conn, spec.table_name) if table_present and spec.table_name is not None else set()
    schema_contract_issues = _schema_contract_issues(spec, columns) if table_present else ()
    version_coverage = await _version_coverage(conn, spec, table_present=table_present, columns=columns)
    version_legacy_count = sum(version.legacy_count for version in version_coverage)
    schema_legacy_count = row_count if schema_contract_issues else 0
    legacy_count = max(version_legacy_count, schema_legacy_count)
    provider_coverage = await _provider_coverage(conn, spec, query, table_present=table_present, columns=columns)
    artifacts: list[InsightStorageArtifact] = []
    for artifact in spec.artifacts:
        artifacts.append(
            InsightStorageArtifact(
                name=artifact,
                present=await _table_exists(conn, artifact),
                ready=_artifact_ready(status, artifact),
            )
        )
    min_time = min((item.min_time for item in provider_coverage if item.min_time), default=None)
    max_time = max((item.max_time for item in provider_coverage if item.max_time), default=None)
    verdict = _entry_verdict(
        table_present=table_present,
        row_count=row_count,
        expected_row_count=expected_row_count,
        missing_count=missing_count,
        stale_count=stale_count,
        orphan_count=orphan_count,
        legacy_count=legacy_count,
        ready_flags=ready_flags,
    )
    return InsightReadinessEntry(
        insight_name=spec.insight_name,
        display_name=spec.display_name,
        verdict=verdict,
        row_count=row_count,
        expected_row_count=expected_row_count,
        missing_count=missing_count,
        stale_count=stale_count,
        orphan_count=orphan_count,
        legacy_incompatible_count=legacy_count,
        storage_artifacts=tuple(artifacts),
        ready_flags=ready_flags,
        provider_coverage=provider_coverage,
        version_coverage=version_coverage,
        schema_contract_issues=schema_contract_issues,
        min_time=min_time,
        max_time=max_time,
        evidence=_evidence(
            row_count=row_count,
            expected_row_count=expected_row_count,
            missing_count=missing_count,
            stale_count=stale_count,
            orphan_count=orphan_count,
            legacy_count=legacy_count,
            schema_contract_issues=schema_contract_issues,
            ready_flags=ready_flags,
        ),
    )


async def build_insight_readiness_report(
    conn: aiosqlite.Connection,
    status: SessionInsightStatusSnapshot,
    query: InsightReadinessQuery | None = None,
) -> InsightReadinessReport:
    request = query or InsightReadinessQuery()
    selected = tuple(normalize_insight_readiness_name(insight) for insight in request.insights)
    specs = tuple(_SPEC_BY_NAME[name] for name in selected) if selected else _SPECS
    entries: list[InsightReadinessEntry] = []
    for spec in specs:
        entries.append(await _entry(conn, status, spec, request))
    insights = tuple(entries)
    return InsightReadinessReport(
        checked_at=datetime.now(timezone.utc).isoformat(),
        aggregate_verdict=_aggregate_verdict(insights),
        total_conversations=status.total_conversations,
        provider=request.provider,
        since=request.since,
        until=request.until,
        insights=insights,
    )


__all__ = [
    "InsightProviderCoverage",
    "InsightReadinessEntry",
    "InsightReadinessQuery",
    "InsightReadinessReport",
    "InsightReadinessVerdict",
    "InsightStorageArtifact",
    "InsightVersionCoverage",
    "build_insight_readiness_report",
    "known_insight_readiness_names",
    "normalize_insight_readiness_name",
]
