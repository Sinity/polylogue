"""Typed insight materialization readiness reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, cast

import aiosqlite
from pydantic import Field

from polylogue.archive.query.spec import parse_query_date
from polylogue.insights.archive_models import ARCHIVE_INSIGHT_CONTRACT_VERSION, ArchiveInsightModel
from polylogue.maintenance.targets import build_maintenance_target_catalog
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.table_existence import table_exists_async as _table_exists

InsightReadinessVerdict = Literal[
    "ready", "partial", "empty", "missing", "stale", "incompatible", "degraded", "unknown"
]
_REPAIR_HINT = build_maintenance_target_catalog().repair_hint(("session_insights",), include_run_all=True)


def _origin_value(origin: str | None) -> str | None:
    from polylogue.storage.sqlite.archive_tiers.archive import _origin_value as _impl

    return _impl(origin)


def _readiness_query_ms(field: str, value: str | None) -> int | None:
    parsed = parse_query_date(field, value)
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def _iso_from_ms(value: object) -> str | None:
    if value is None:
        return None
    epoch_ms = int(cast("int | float | str", value))
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


class InsightReadinessQuery(ArchiveInsightModel):
    insights: tuple[str, ...] = ()
    origin: str | None = None
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
    incompatible_count: int = 0


class InsightOriginCoverage(ArchiveInsightModel):
    origin: str
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
    incompatible_count: int = 0
    degraded_count: int = 0
    fallback_reason_counts: dict[str, int] = Field(default_factory=dict)
    storage_artifacts: tuple[InsightStorageArtifact, ...] = ()
    ready_flags: dict[str, bool] = Field(default_factory=dict)
    origin_coverage: tuple[InsightOriginCoverage, ...] = ()
    version_coverage: tuple[InsightVersionCoverage, ...] = ()
    schema_contract_issues: tuple[str, ...] = ()
    min_time: str | None = None
    max_time: str | None = None
    repair_command: str = _REPAIR_HINT
    evidence: tuple[str, ...] = ()


class InsightReadinessReport(ArchiveInsightModel):
    checked_at: str
    aggregate_verdict: InsightReadinessVerdict
    total_sessions: int = 0
    origin: str | None = None
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
    missing_count_attrs: tuple[str, ...] = ()
    stale_count_attr: str | None = None
    orphan_count_attr: str | None = None
    ready_flags: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    # insight tables are keyed by ``session_id`` and carry no
    # ``source_name``/``source_updated_at``/``materializer_version`` columns of
    # their own. Provider and time coverage derive from a join to ``sessions``;
    # the join is enabled per spec because some specs (e.g. archive coverage
    # over ``sessions`` itself) already expose those columns directly.
    provider_via_session: bool = True
    fallback_payload_columns: tuple[str, ...] = ()
    empty_is_ready: bool = False


_SPECS: tuple[InsightReadinessSpec, ...] = (
    InsightReadinessSpec(
        insight_name="session_profiles",
        display_name="Session Profiles",
        table_name="session_profiles",
        row_count_attr="profile_row_count",
        expected_count_attr="total_sessions",
        missing_count_attr="missing_profile_row_count",
        missing_count_attrs=("missing_session_profile_materialization_count",),
        stale_count_attr="stale_profile_row_count",
        orphan_count_attr="orphan_profile_row_count",
        ready_flags=("profile_rows_ready",),
        artifacts=("session_profiles",),
    ),
    InsightReadinessSpec(
        insight_name="session_work_events",
        display_name="Work Events",
        table_name="session_work_events",
        row_count_attr="work_event_inference_count",
        expected_count_attr="expected_work_event_inference_count",
        missing_count_attr="missing_work_event_materialization_count",
        stale_count_attr="stale_work_event_inference_count",
        orphan_count_attr="orphan_work_event_inference_count",
        ready_flags=("work_event_inference_rows_ready",),
        artifacts=("session_work_events",),
        fallback_payload_columns=("inference_json",),
    ),
    InsightReadinessSpec(
        insight_name="session_phases",
        display_name="Session Phases",
        table_name="session_phases",
        row_count_attr="phase_count",
        expected_count_attr="expected_phase_count",
        missing_count_attr="missing_phase_materialization_count",
        stale_count_attr="stale_phase_count",
        orphan_count_attr="orphan_phase_count",
        ready_flags=("phase_rows_ready",),
        artifacts=("session_phases",),
    ),
    # polylogue-dab/itvd: session_runs/session_observed_events/
    # session_context_snapshots are source-derived CTE relations
    # (run_projection_relations.py), not tables, so they can never appear in
    # sqlite_master. table_name points at the always-present `sessions`
    # table purely so the presence gate (`table_present` in `_entry()`)
    # reports true and the real row_count/ready_flags (already computed in
    # `status` from the CTE) actually drive the verdict, instead of
    # permanently reporting "missing". `artifacts` intentionally keeps the
    # legacy table name -- it is genuinely, permanently absent, and
    # InsightStorageArtifact.present reporting that honestly is useful
    # diagnostic info distinguishing "no cache table" from "not ready".
    InsightReadinessSpec(
        insight_name="session_runs",
        display_name="Session Runs",
        table_name="sessions",
        row_count_attr="run_count",
        ready_flags=("run_rows_ready",),
        artifacts=("session_runs",),
        empty_is_ready=True,
    ),
    InsightReadinessSpec(
        insight_name="session_observed_events",
        display_name="Observed Events",
        table_name="sessions",
        row_count_attr="observed_event_count",
        ready_flags=("observed_event_rows_ready",),
        artifacts=("session_observed_events",),
        empty_is_ready=True,
    ),
    InsightReadinessSpec(
        insight_name="session_context_snapshots",
        display_name="Context Snapshots",
        table_name="sessions",
        row_count_attr="context_snapshot_count",
        ready_flags=("context_snapshot_rows_ready",),
        artifacts=("session_context_snapshots",),
        empty_is_ready=True,
    ),
    InsightReadinessSpec(
        insight_name="threads",
        display_name="Work Threads",
        table_name="threads",
        row_count_attr="thread_count",
        expected_count_attr="root_threads",
        missing_count_attr="missing_thread_materialization_count",
        stale_count_attr="stale_thread_count",
        orphan_count_attr="orphan_thread_count",
        ready_flags=("threads_ready",),
        artifacts=("threads",),
        provider_via_session=False,
    ),
    InsightReadinessSpec(
        insight_name="session_tag_rollups",
        display_name="Session Tag Rollups",
        table_name="session_tags",
        row_count_attr="tag_rollup_count",
        expected_count_attr="expected_tag_rollup_count",
        stale_count_attr="stale_tag_rollup_count",
        ready_flags=("tag_rollups_ready",),
        artifacts=("session_tags",),
    ),
    InsightReadinessSpec(
        insight_name="archive_coverage",
        display_name="Archive Coverage",
        table_name="sessions",
        row_count_attr="total_sessions",
        ready_flags=(),
        artifacts=("sessions",),
    ),
)

_SPEC_BY_NAME = {spec.insight_name: spec for spec in _SPECS}
_ALIASES = {
    **{spec.insight_name.replace("_", "-"): spec.insight_name for spec in _SPECS},
    "profiles": "session_profiles",
    "work-events": "session_work_events",
    "phases": "session_phases",
    "runs": "session_runs",
    "run-projection": "session_runs",
    "observed-events": "session_observed_events",
    "context-snapshots": "session_context_snapshots",
    "threads": "threads",
    "tags": "session_tag_rollups",
    "coverage": "archive_coverage",
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


def _missing_count(status: SessionInsightStatusSnapshot, spec: InsightReadinessSpec) -> int:
    return _count(status, spec.missing_count_attr) + sum(_count(status, attr) for attr in spec.missing_count_attrs)


def _artifact_ready(status: SessionInsightStatusSnapshot, artifact_name: str) -> bool | None:
    mapping = {
        "session_profiles": "profile_rows_ready",
        "session_work_events": "work_event_inference_rows_ready",
        "session_phases": "phase_rows_ready",
        "session_runs": "run_rows_ready",
        "session_observed_events": "observed_event_rows_ready",
        "session_context_snapshots": "context_snapshot_rows_ready",
        "threads": "threads_ready",
        "session_tags": "tag_rollups_ready",
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
    incompatible_count: int,
    degraded_count: int,
    ready_flags: dict[str, bool],
    empty_is_ready: bool = False,
) -> InsightReadinessVerdict:
    if not table_present:
        return "missing"
    if incompatible_count:
        return "incompatible"
    if stale_count or orphan_count:
        return "stale"
    if missing_count or (expected_row_count is not None and row_count < expected_row_count):
        return "partial"
    if degraded_count:
        return "degraded"
    if ready_flags and all(ready_flags.values()):
        return "ready"
    if row_count == 0:
        return "ready" if empty_is_ready else "empty"
    if not ready_flags:
        return "ready"
    return "unknown"


def _aggregate_verdict(entries: tuple[InsightReadinessEntry, ...]) -> InsightReadinessVerdict:
    verdicts = {entry.verdict for entry in entries}
    priority: tuple[InsightReadinessVerdict, ...] = (
        "incompatible",
        "stale",
        "partial",
        "missing",
        "degraded",
        "unknown",
        "empty",
    )
    for verdict in priority:
        if verdict in verdicts:
            return verdict
    return "ready"


async def _table_columns(conn: aiosqlite.Connection, table: str) -> set[str]:
    rows = await (await conn.execute(f"PRAGMA table_info({table})")).fetchall()
    return {str(row[1]) for row in rows}


def _normalize_origin_filter(origin: str | None) -> str | None:
    if not origin:
        return None
    return _origin_value(origin)


def _where_clause(
    spec: InsightReadinessSpec,
    query: InsightReadinessQuery,
) -> tuple[str, list[object]]:
    """Build a origin/time filter expressed against the joined ``sessions``.

    Archive insight tables key everything on ``session_id``; origin identity
    lives on ``sessions.origin`` and recency on ``sessions.sort_key_ms``, so
    every filter clause references the ``s.`` alias from the session join.
    """
    clauses: list[str] = []
    params: list[object] = []
    origin = _normalize_origin_filter(query.origin)
    if origin is not None:
        clauses.append("s.origin = ?")
        params.append(origin)
    if query.since:
        since_ms = _readiness_query_ms("since", query.since)
        if since_ms is not None:
            clauses.append("s.sort_key_ms >= ?")
            params.append(since_ms)
    if query.until:
        until_ms = _readiness_query_ms("until", query.until)
        if until_ms is not None:
            clauses.append("s.sort_key_ms <= ?")
            params.append(until_ms)
    return (" WHERE " + " AND ".join(clauses), params) if clauses else ("", params)


async def _origin_coverage(
    conn: aiosqlite.Connection,
    spec: InsightReadinessSpec,
    query: InsightReadinessQuery,
    *,
    table_present: bool,
) -> tuple[InsightOriginCoverage, ...]:
    if not table_present or spec.table_name is None or not spec.provider_via_session:
        return ()
    where, params = _where_clause(spec, query)
    sql = (
        "SELECT s.origin AS origin, COUNT(*) AS row_count, "
        "MIN(s.sort_key_ms) AS min_time_ms, MAX(s.sort_key_ms) AS max_time_ms "
        f"FROM {spec.table_name} AS t "
        "JOIN sessions AS s ON s.session_id = t.session_id"
        f"{where} GROUP BY s.origin ORDER BY s.origin"
    )
    rows = await (await conn.execute(sql, tuple(params))).fetchall()
    return tuple(
        InsightOriginCoverage(
            origin=str(row["origin"]) if row["origin"] is not None else "unknown",
            row_count=int(row["row_count"]),
            min_time=_iso_from_ms(row["min_time_ms"]),
            max_time=_iso_from_ms(row["max_time_ms"]),
        )
        for row in rows
    )


async def _fallback_coverage(
    conn: aiosqlite.Connection,
    spec: InsightReadinessSpec,
    *,
    table_present: bool,
    columns: set[str],
) -> tuple[int, dict[str, int]]:
    """Count rows whose payload carries a non-empty ``fallback_reasons`` array.

    Returns ``(degraded_row_count, reason_totals)``. The row count is the
    number of rows where at least one declared payload column reports any
    fallback reason. ``reason_totals`` sums occurrences per reason across
    every inspected payload column. The query uses ``json_extract`` and
    ``json_each`` so each row contributes at most one count to
    ``degraded_row_count`` regardless of how many payload columns flag it.
    """

    if not table_present or spec.table_name is None or not spec.fallback_payload_columns:
        return (0, {})
    present_columns = tuple(column for column in spec.fallback_payload_columns if column in columns)
    if not present_columns:
        return (0, {})
    any_terms = " OR ".join(
        f"json_array_length(COALESCE(json_extract({column}, '$.fallback_reasons'), '[]')) > 0"
        for column in present_columns
    )
    any_sql = f"SELECT COUNT(*) AS degraded FROM {spec.table_name} WHERE {any_terms}"
    degraded_row = await (await conn.execute(any_sql)).fetchone()
    degraded_row_count = int(degraded_row["degraded"]) if degraded_row is not None else 0

    reason_totals: dict[str, int] = {}
    for column in present_columns:
        reason_sql = (
            f"SELECT value AS reason, COUNT(*) AS occurrences FROM {spec.table_name}, "
            f"json_each(COALESCE(json_extract({column}, '$.fallback_reasons'), '[]')) GROUP BY value"
        )
        rows = await (await conn.execute(reason_sql)).fetchall()
        for row in rows:
            reason = str(row["reason"])
            reason_totals[reason] = reason_totals.get(reason, 0) + int(row["occurrences"])
    return (degraded_row_count, dict(sorted(reason_totals.items())))


def _schema_contract_issues(spec: InsightReadinessSpec, columns: set[str]) -> tuple[str, ...]:
    """Report structural schema drift for an archive insight table.

    has no per-row version columns and derives origin/time
    via the ``sessions`` join, so the only contract a present table can break
    is its own primary ``session_id`` key (every archive insight table is keyed
    on it). A missing ``session_id`` column means the table is not the expected
    archive shape and its rows cannot be trusted.
    """
    if spec.provider_via_session and "session_id" not in columns:
        return (f"missing session_id column: {spec.table_name}",)
    return ()


def _evidence(
    *,
    row_count: int,
    expected_row_count: int | None,
    missing_count: int,
    stale_count: int,
    orphan_count: int,
    incompatible_count: int,
    degraded_count: int,
    fallback_reason_counts: dict[str, int],
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
    if incompatible_count:
        values.append(f"incompatible={incompatible_count}")
    if degraded_count:
        values.append(f"degraded={degraded_count}")
    values.extend(f"fallback_reason={reason}={count}" for reason, count in fallback_reason_counts.items())
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
    missing_count = _missing_count(status, spec)
    stale_count = _count(status, spec.stale_count_attr)
    orphan_count = _count(status, spec.orphan_count_attr)
    ready_flags = {flag: bool(getattr(status, flag)) for flag in spec.ready_flags}
    columns = await _table_columns(conn, spec.table_name) if table_present and spec.table_name is not None else set()
    schema_contract_issues = _schema_contract_issues(spec, columns) if table_present else ()
    version_coverage: tuple[InsightVersionCoverage, ...] = ()
    incompatible_count = row_count if schema_contract_issues else 0
    origin_coverage = await _origin_coverage(conn, spec, query, table_present=table_present)
    degraded_count, fallback_reason_counts = await _fallback_coverage(
        conn, spec, table_present=table_present, columns=columns
    )
    artifacts: list[InsightStorageArtifact] = []
    for artifact in spec.artifacts:
        artifacts.append(
            InsightStorageArtifact(
                name=artifact,
                present=await _table_exists(conn, artifact),
                ready=_artifact_ready(status, artifact),
            )
        )
    min_time = min((item.min_time for item in origin_coverage if item.min_time), default=None)
    max_time = max((item.max_time for item in origin_coverage if item.max_time), default=None)
    verdict = _entry_verdict(
        table_present=table_present,
        row_count=row_count,
        expected_row_count=expected_row_count,
        missing_count=missing_count,
        stale_count=stale_count,
        orphan_count=orphan_count,
        incompatible_count=incompatible_count,
        degraded_count=degraded_count,
        ready_flags=ready_flags,
        empty_is_ready=spec.empty_is_ready,
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
        incompatible_count=incompatible_count,
        degraded_count=degraded_count,
        fallback_reason_counts=fallback_reason_counts,
        storage_artifacts=tuple(artifacts),
        ready_flags=ready_flags,
        origin_coverage=origin_coverage,
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
            incompatible_count=incompatible_count,
            degraded_count=degraded_count,
            fallback_reason_counts=fallback_reason_counts,
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
        total_sessions=status.total_sessions,
        origin=request.origin,
        since=request.since,
        until=request.until,
        insights=insights,
    )


__all__ = [
    "InsightOriginCoverage",
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
