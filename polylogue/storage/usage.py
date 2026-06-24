"""Provider usage accounting diagnostics over the archive index tier.

The report in this module intentionally keeps three evidence streams separate:
provider event rows, provider cumulative counters, and derived/priced model
rollups.  It is an audit surface, not a billing estimator.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UsageCounters:
    """Token counters with provider-native cache/reasoning lanes preserved."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_row(
        cls,
        row: sqlite3.Row,
        *,
        input_key: str,
        output_key: str,
        cached_input_key: str,
        cache_write_key: str,
        reasoning_output_key: str,
        total_key: str,
    ) -> UsageCounters:
        return cls(
            input_tokens=_int(row[input_key]),
            output_tokens=_int(row[output_key]),
            cached_input_tokens=_int(row[cached_input_key]),
            cache_write_tokens=_int(row[cache_write_key]),
            reasoning_output_tokens=_int(row[reasoning_output_key]),
            total_tokens=_int(row[total_key]),
        )

    def plus(self, other: UsageCounters) -> UsageCounters:
        return UsageCounters(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens + other.reasoning_output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def is_zero(self) -> bool:
        return not any(self.to_dict().values())

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class OriginUsageReport:
    """Usage evidence summary for one archive origin."""

    origin: str
    session_count: int = 0
    message_count: int = 0
    transcript_word_count: int = 0
    provider_event_session_count: int = 0
    provider_event_count: int = 0
    token_count_event_count: int = 0
    message_usage_event_count: int = 0
    zero_token_event_count: int = 0
    missing_model_event_count: int = 0
    multi_model_session_count: int = 0
    priced_model_row_count: int = 0
    origin_reported_model_row_count: int = 0
    estimated_model_row_count: int = 0
    provider_request_usage: UsageCounters = field(default_factory=UsageCounters)
    provider_cumulative_usage: UsageCounters = field(default_factory=UsageCounters)
    model_rollup_usage: UsageCounters = field(default_factory=UsageCounters)
    sample_missing_model_sessions: tuple[str, ...] = ()
    sample_zero_token_sessions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "session_count": self.session_count,
            "message_count": self.message_count,
            "transcript_word_count": self.transcript_word_count,
            "provider_event_session_count": self.provider_event_session_count,
            "provider_event_count": self.provider_event_count,
            "token_count_event_count": self.token_count_event_count,
            "message_usage_event_count": self.message_usage_event_count,
            "zero_token_event_count": self.zero_token_event_count,
            "missing_model_event_count": self.missing_model_event_count,
            "multi_model_session_count": self.multi_model_session_count,
            "priced_model_row_count": self.priced_model_row_count,
            "origin_reported_model_row_count": self.origin_reported_model_row_count,
            "estimated_model_row_count": self.estimated_model_row_count,
            "provider_request_usage": self.provider_request_usage.to_dict(),
            "provider_cumulative_usage": self.provider_cumulative_usage.to_dict(),
            "model_rollup_usage": self.model_rollup_usage.to_dict(),
            "sample_missing_model_sessions": list(self.sample_missing_model_sessions),
            "sample_zero_token_sessions": list(self.sample_zero_token_sessions),
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True, slots=True)
class ProviderUsageReport:
    """Archive-level provider usage accounting report."""

    archive_root: str
    origins: tuple[OriginUsageReport, ...]
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": self.archive_root,
            "origins": [origin.to_dict() for origin in self.origins],
            "caveats": list(self.caveats),
        }


def provider_usage_report_for_archive_root(
    archive_root: Path,
    *,
    origin: str | None = None,
    limit: int | None = 25,
) -> ProviderUsageReport:
    """Read ``archive_root/index.db`` and return a provider usage audit report."""

    index_db = Path(archive_root) / "index.db"
    if not index_db.exists():
        return ProviderUsageReport(
            archive_root=str(archive_root),
            origins=(),
            caveats=(f"index.db not found at {index_db}",),
        )
    uri = index_db.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return provider_usage_report_from_connection(conn, archive_root=archive_root, origin=origin, limit=limit)
    finally:
        conn.close()


def provider_usage_report_from_connection(
    conn: sqlite3.Connection,
    *,
    archive_root: Path | str = "",
    origin: str | None = None,
    limit: int | None = 25,
) -> ProviderUsageReport:
    """Return a provider usage report for an already-open index connection."""

    conn.row_factory = sqlite3.Row
    caveats: list[str] = [
        "provider usage events, transcript text volume, and model rollups are separate evidence streams",
        "this report does not query provider billing and is not a precise cost report",
    ]
    if not _table_exists(conn, "sessions"):
        return ProviderUsageReport(
            archive_root=str(archive_root),
            origins=(),
            caveats=tuple(caveats + ["sessions table is missing"]),
        )

    base_by_origin = _base_session_stats(conn, origin)
    model_by_origin = _model_rollup_stats(conn, origin) if _table_exists(conn, "session_model_usage") else {}
    model_counts_by_origin = _model_row_counts(conn, origin) if _table_exists(conn, "session_model_usage") else {}
    multi_model_by_origin = (
        _multi_model_session_counts(conn, origin) if _table_exists(conn, "session_model_usage") else {}
    )

    if _table_exists(conn, "session_provider_usage_events"):
        event_by_origin = _provider_event_stats(conn, origin)
        cumulative_by_origin = _provider_cumulative_usage(conn, origin)
        missing_samples = _sample_event_sessions(conn, origin, limit, missing_model=True)
        zero_samples = _sample_event_sessions(conn, origin, limit, zero_token=True)
    else:
        event_by_origin = {}
        cumulative_by_origin = {}
        missing_samples = {}
        zero_samples = {}
        caveats.append("session_provider_usage_events table is missing; rebuild the index tier with the current schema")

    origins = sorted(set(base_by_origin) | set(event_by_origin) | set(model_by_origin))
    reports: list[OriginUsageReport] = []
    for origin_name in origins:
        base = base_by_origin.get(origin_name, {})
        events = event_by_origin.get(origin_name, {})
        model_counts = model_counts_by_origin.get(origin_name, {})
        provider_request_usage = events.get("provider_request_usage")
        if not isinstance(provider_request_usage, UsageCounters):
            provider_request_usage = UsageCounters()
        origin_caveats = _origin_caveats(
            session_count=_int(base.get("session_count")),
            provider_event_session_count=_int(events.get("provider_event_session_count")),
            missing_model_event_count=_int(events.get("missing_model_event_count")),
            zero_token_event_count=_int(events.get("zero_token_event_count")),
            multi_model_session_count=multi_model_by_origin.get(origin_name, 0),
            token_count_event_count=_int(events.get("token_count_event_count")),
            message_usage_event_count=_int(events.get("message_usage_event_count")),
        )
        reports.append(
            OriginUsageReport(
                origin=origin_name,
                session_count=_int(base.get("session_count")),
                message_count=_int(base.get("message_count")),
                transcript_word_count=_int(base.get("transcript_word_count")),
                provider_event_session_count=_int(events.get("provider_event_session_count")),
                provider_event_count=_int(events.get("provider_event_count")),
                token_count_event_count=_int(events.get("token_count_event_count")),
                message_usage_event_count=_int(events.get("message_usage_event_count")),
                zero_token_event_count=_int(events.get("zero_token_event_count")),
                missing_model_event_count=_int(events.get("missing_model_event_count")),
                multi_model_session_count=multi_model_by_origin.get(origin_name, 0),
                priced_model_row_count=_int(model_counts.get("priced_model_row_count")),
                origin_reported_model_row_count=_int(model_counts.get("origin_reported_model_row_count")),
                estimated_model_row_count=_int(model_counts.get("estimated_model_row_count")),
                provider_request_usage=provider_request_usage,
                provider_cumulative_usage=cumulative_by_origin.get(origin_name, UsageCounters()),
                model_rollup_usage=model_by_origin.get(origin_name, UsageCounters()),
                sample_missing_model_sessions=tuple(missing_samples.get(origin_name, ())),
                sample_zero_token_sessions=tuple(zero_samples.get(origin_name, ())),
                caveats=tuple(origin_caveats),
            )
        )

    if origin is not None and not reports:
        caveats.append(f"no sessions found for origin {origin!r}")
    return ProviderUsageReport(archive_root=str(archive_root), origins=tuple(reports), caveats=tuple(caveats))


def _base_session_stats(conn: sqlite3.Connection, origin: str | None) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        f"""
        SELECT origin,
               COUNT(*) AS session_count,
               COALESCE(SUM(message_count), 0) AS message_count,
               COALESCE(SUM(word_count), 0) AS transcript_word_count
        FROM sessions
        {_where_origin(origin)}
        GROUP BY origin
        ORDER BY origin
        """,
        _origin_args(origin),
    ).fetchall()
    return {
        str(row["origin"]): {
            "session_count": _int(row["session_count"]),
            "message_count": _int(row["message_count"]),
            "transcript_word_count": _int(row["transcript_word_count"]),
        }
        for row in rows
    }


def _model_rollup_stats(conn: sqlite3.Connection, origin: str | None) -> dict[str, UsageCounters]:
    rows = conn.execute(
        f"""
        SELECT s.origin AS origin,
               COALESCE(SUM(u.input_tokens), 0) AS input_tokens,
               COALESCE(SUM(u.output_tokens), 0) AS output_tokens,
               COALESCE(SUM(u.cache_read_tokens), 0) AS cached_input_tokens,
               COALESCE(SUM(u.cache_write_tokens), 0) AS cache_write_tokens,
               0 AS reasoning_output_tokens,
               COALESCE(SUM(u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_write_tokens), 0) AS total_tokens
        FROM session_model_usage u
        JOIN sessions s ON s.session_id = u.session_id
        {_where_origin(origin, table_alias="s")}
        GROUP BY s.origin
        ORDER BY s.origin
        """,
        _origin_args(origin),
    ).fetchall()
    return {
        str(row["origin"]): UsageCounters.from_row(
            row,
            input_key="input_tokens",
            output_key="output_tokens",
            cached_input_key="cached_input_tokens",
            cache_write_key="cache_write_tokens",
            reasoning_output_key="reasoning_output_tokens",
            total_key="total_tokens",
        )
        for row in rows
    }


def _model_row_counts(conn: sqlite3.Connection, origin: str | None) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        f"""
        SELECT s.origin AS origin,
               COALESCE(SUM(CASE WHEN u.cost_provenance = 'priced' THEN 1 ELSE 0 END), 0) AS priced_model_row_count,
               COALESCE(SUM(CASE WHEN u.cost_provenance = 'origin_reported' THEN 1 ELSE 0 END), 0) AS origin_reported_model_row_count,
               COALESCE(SUM(CASE WHEN u.cost_provenance = 'estimated' THEN 1 ELSE 0 END), 0) AS estimated_model_row_count
        FROM session_model_usage u
        JOIN sessions s ON s.session_id = u.session_id
        {_where_origin(origin, table_alias="s")}
        GROUP BY s.origin
        ORDER BY s.origin
        """,
        _origin_args(origin),
    ).fetchall()
    return {
        str(row["origin"]): {
            "priced_model_row_count": _int(row["priced_model_row_count"]),
            "origin_reported_model_row_count": _int(row["origin_reported_model_row_count"]),
            "estimated_model_row_count": _int(row["estimated_model_row_count"]),
        }
        for row in rows
    }


def _multi_model_session_counts(conn: sqlite3.Connection, origin: str | None) -> dict[str, int]:
    rows = conn.execute(
        f"""
        SELECT origin, COUNT(*) AS session_count
        FROM (
            SELECT s.origin AS origin, u.session_id AS session_id, COUNT(DISTINCT u.model_name) AS model_count
            FROM session_model_usage u
            JOIN sessions s ON s.session_id = u.session_id
            {_where_origin(origin, table_alias="s")}
            GROUP BY s.origin, u.session_id
            HAVING model_count > 1
        )
        GROUP BY origin
        """,
        _origin_args(origin),
    ).fetchall()
    return {str(row["origin"]): _int(row["session_count"]) for row in rows}


def _provider_event_stats(conn: sqlite3.Connection, origin: str | None) -> dict[str, dict[str, object]]:
    columns = _table_columns(conn, "session_provider_usage_events")
    select_parts = [
        "s.origin AS origin",
        "COUNT(*) AS provider_event_count",
        "COUNT(DISTINCT e.session_id) AS provider_event_session_count",
        "COALESCE(SUM(CASE WHEN e.provider_event_type = 'token_count' THEN 1 ELSE 0 END), 0) AS token_count_event_count",
        "COALESCE(SUM(CASE WHEN e.provider_event_type = 'message_usage' THEN 1 ELSE 0 END), 0) AS message_usage_event_count",
        "COALESCE(SUM(CASE WHEN e.model_name IS NULL OR TRIM(e.model_name) = '' THEN 1 ELSE 0 END), 0) AS missing_model_event_count",
    ]
    last_cols = _counter_columns(columns, prefix="last")
    total_cols = _counter_columns(columns, prefix="total")
    zero_predicate = " AND ".join([f"COALESCE({expr}, 0) = 0" for expr in (*last_cols.values(), *total_cols.values())])
    select_parts.append(f"COALESCE(SUM(CASE WHEN {zero_predicate} THEN 1 ELSE 0 END), 0) AS zero_token_event_count")
    for public_name, expr in last_cols.items():
        select_parts.append(f"COALESCE(SUM({expr}), 0) AS {public_name}")
    rows = conn.execute(
        f"""
        SELECT {", ".join(select_parts)}
        FROM session_provider_usage_events e
        JOIN sessions s ON s.session_id = e.session_id
        {_where_origin(origin, table_alias="s")}
        GROUP BY s.origin
        ORDER BY s.origin
        """,
        _origin_args(origin),
    ).fetchall()
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        result[str(row["origin"])] = {
            "provider_event_count": _int(row["provider_event_count"]),
            "provider_event_session_count": _int(row["provider_event_session_count"]),
            "token_count_event_count": _int(row["token_count_event_count"]),
            "message_usage_event_count": _int(row["message_usage_event_count"]),
            "missing_model_event_count": _int(row["missing_model_event_count"]),
            "zero_token_event_count": _int(row["zero_token_event_count"]),
            "provider_request_usage": UsageCounters.from_row(
                row,
                input_key="input_tokens",
                output_key="output_tokens",
                cached_input_key="cached_input_tokens",
                cache_write_key="cache_write_tokens",
                reasoning_output_key="reasoning_output_tokens",
                total_key="total_tokens",
            ),
        }
    return result


def _provider_cumulative_usage(conn: sqlite3.Connection, origin: str | None) -> dict[str, UsageCounters]:
    columns = _table_columns(conn, "session_provider_usage_events")
    total_cols = _counter_columns(columns, prefix="total")
    total_predicate = " OR ".join([f"COALESCE({expr}, 0) > 0" for expr in total_cols.values()])
    rows = conn.execute(
        f"""
        SELECT s.origin AS origin,
               e.session_id AS session_id,
               COALESCE(NULLIF(TRIM(e.model_name), ''), '__unknown_model__') AS model_key,
               e.position AS position,
               {total_cols["input_tokens"]} AS input_tokens,
               {total_cols["output_tokens"]} AS output_tokens,
               {total_cols["cached_input_tokens"]} AS cached_input_tokens,
               {total_cols["cache_write_tokens"]} AS cache_write_tokens,
               {total_cols["reasoning_output_tokens"]} AS reasoning_output_tokens,
               {total_cols["total_tokens"]} AS total_tokens
        FROM session_provider_usage_events e
        JOIN sessions s ON s.session_id = e.session_id
        {_where_origin(origin, table_alias="s")}
          {"AND" if origin is not None else "WHERE"} ({total_predicate})
        ORDER BY s.origin, e.session_id, model_key, e.position
        """,
        _origin_args(origin),
    ).fetchall()
    latest: dict[tuple[str, str, str], UsageCounters] = {}
    for row in rows:
        latest[(str(row["origin"]), str(row["session_id"]), str(row["model_key"]))] = UsageCounters.from_row(
            row,
            input_key="input_tokens",
            output_key="output_tokens",
            cached_input_key="cached_input_tokens",
            cache_write_key="cache_write_tokens",
            reasoning_output_key="reasoning_output_tokens",
            total_key="total_tokens",
        )
    by_origin: dict[str, UsageCounters] = defaultdict(UsageCounters)
    for (origin_name, _session_id, _model_key), counters in latest.items():
        by_origin[origin_name] = by_origin[origin_name].plus(counters)
    return dict(by_origin)


def _sample_event_sessions(
    conn: sqlite3.Connection,
    origin: str | None,
    limit: int | None,
    *,
    missing_model: bool = False,
    zero_token: bool = False,
) -> dict[str, tuple[str, ...]]:
    if limit is not None and limit <= 0:
        return {}
    columns = _table_columns(conn, "session_provider_usage_events")
    predicates: list[str] = []
    if missing_model:
        predicates.append("(e.model_name IS NULL OR TRIM(e.model_name) = '')")
    if zero_token:
        last_cols = _counter_columns(columns, prefix="last")
        total_cols = _counter_columns(columns, prefix="total")
        predicates.append(
            "("
            + " AND ".join([f"COALESCE({expr}, 0) = 0" for expr in (*last_cols.values(), *total_cols.values())])
            + ")"
        )
    if not predicates:
        return {}
    rows = conn.execute(
        f"""
        SELECT DISTINCT s.origin AS origin, e.session_id AS session_id
        FROM session_provider_usage_events e
        JOIN sessions s ON s.session_id = e.session_id
        {_where_origin(origin, table_alias="s")}
          {"AND" if origin is not None else "WHERE"} {" AND ".join(predicates)}
        ORDER BY s.origin, e.session_id
        {"LIMIT ?" if limit is not None else ""}
        """,
        (*_origin_args(origin), *(() if limit is None else (limit,))),
    ).fetchall()
    by_origin: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_origin[str(row["origin"])].append(str(row["session_id"]))
    return {key: tuple(value) for key, value in by_origin.items()}


def _counter_columns(columns: set[str], *, prefix: str) -> dict[str, str]:
    raw = {
        "input_tokens": f"e.{prefix}_input_tokens",
        "output_tokens": f"e.{prefix}_output_tokens",
        "cached_input_tokens": f"e.{prefix}_cached_input_tokens",
        "cache_write_tokens": f"e.{prefix}_cache_write_tokens",
        "reasoning_output_tokens": f"e.{prefix}_reasoning_output_tokens",
        "total_tokens": "e.total_tokens" if prefix == "total" else "e.last_total_tokens",
    }
    result: dict[str, str] = {}
    for public_name, expression in raw.items():
        column_name = expression.split(".", 1)[1]
        result[public_name] = expression if column_name in columns else "0"
    return result


def _origin_caveats(
    *,
    session_count: int,
    provider_event_session_count: int,
    missing_model_event_count: int,
    zero_token_event_count: int,
    multi_model_session_count: int,
    token_count_event_count: int,
    message_usage_event_count: int,
) -> list[str]:
    caveats: list[str] = []
    if session_count and provider_event_session_count < session_count:
        caveats.append(
            "some sessions have no provider usage event rows; transcript words and model rollups cover different evidence"
        )
    if missing_model_event_count:
        caveats.append("some provider events have no model; multi-model attribution is intentionally not guessed")
    if zero_token_event_count:
        caveats.append("zero-token provider events are preserved as meaningful provider telemetry")
    if multi_model_session_count:
        caveats.append("multi-model sessions are present; inspect per-model rows before treating totals as one model")
    if token_count_event_count and message_usage_event_count:
        caveats.append("provider event rows mix cumulative Codex token_count and per-message Claude usage semantics")
    return caveats


def _where_origin(origin: str | None, *, table_alias: str | None = None) -> str:
    if origin is None:
        return ""
    qualifier = f"{table_alias}." if table_alias else ""
    return f"WHERE {qualifier}origin = ?"


def _origin_args(origin: str | None) -> tuple[str, ...]:
    return () if origin is None else (origin,)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({name})")}


def _int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


__all__ = [
    "OriginUsageReport",
    "ProviderUsageReport",
    "UsageCounters",
    "provider_usage_report_for_archive_root",
    "provider_usage_report_from_connection",
]
