"""Provider usage accounting diagnostics over the archive index tier.

The report in this module intentionally keeps three evidence streams separate:
provider event rows, provider cumulative counters, and derived/priced model
rollups.  It is an audit surface, not a billing estimator.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from polylogue.archive.semantic.pricing import (
    CATALOG_EFFECTIVE_DATE,
    CATALOG_PROVENANCE,
    PRICING,
    _normalize_model,
    estimate_cost,
)
from polylogue.core.enums import Origin, Provider

UsageReportDetail = Literal["headline", "full"]


@dataclass(frozen=True, slots=True)
class ProviderUsageCoverage:
    """Declared provider-usage telemetry coverage for one archive origin."""

    origin: str
    provider: str
    status: str
    evidence_stream: str
    event_types: tuple[str, ...] = ()
    request_semantics: str = ""
    cumulative_semantics: str = ""
    cache_semantics: str = ""
    notes: tuple[str, ...] = ()
    rebuild_guidance: str = (
        "rebuild index.db from source.db/raw archives so usage events and rollups are materialized from source evidence"
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "provider": self.provider,
            "status": self.status,
            "evidence_stream": self.evidence_stream,
            "event_types": list(self.event_types),
            "request_semantics": self.request_semantics,
            "cumulative_semantics": self.cumulative_semantics,
            "cache_semantics": self.cache_semantics,
            "notes": list(self.notes),
            "rebuild_guidance": self.rebuild_guidance,
        }


_PROVIDER_USAGE_COVERAGE: tuple[ProviderUsageCoverage, ...] = (
    ProviderUsageCoverage(
        origin=Origin.CLAUDE_CODE_SESSION.value,
        provider=Provider.CLAUDE_CODE.value,
        status="exact",
        evidence_stream="provider_reported_usage",
        event_types=("message_usage",),
        request_semantics="Claude Code message.usage rows are per message/request observations.",
        cumulative_semantics="Session rollups are derived by summing message usage rows; Claude Code does not supply a separate cumulative session high-water event.",
        cache_semantics="cache_read_input_tokens and cache_creation_input_tokens are preserved as cached_input and cache_write lanes and are not folded into generic input/output.",
        notes=("Exact token telemetry is available only where exported records include message.usage.",),
    ),
    ProviderUsageCoverage(
        origin=Origin.CODEX_SESSION.value,
        provider=Provider.CODEX.value,
        status="exact",
        evidence_stream="provider_reported_usage",
        event_types=("token_count",),
        request_semantics="Codex last_token_usage is request/current-window telemetry and can be summed by request when present.",
        cumulative_semantics="Codex total_token_usage is cumulative and session-global; rollups take the latest total per session to avoid double-counting.",
        cache_semantics="cached_input_tokens and cache write/cache creation aliases are preserved as separate lanes and are not folded into generic input/output.",
        notes=("model_context_window is carried on token_count events when the provider supplies it.",),
    ),
    ProviderUsageCoverage(
        origin=Origin.CHATGPT_EXPORT.value,
        provider=Provider.CHATGPT.value,
        status="estimate_only",
        evidence_stream="transcript_text_estimate",
        request_semantics="ChatGPT exports do not carry reliable per-request token counters.",
        cumulative_semantics="Provider/account UI totals are external summaries and are not reconstructed from transcript text.",
        cache_semantics="No cache read/write token lanes are available in ChatGPT export rows.",
        notes=("Cost-looking metadata is not treated as exact token telemetry.",),
    ),
    ProviderUsageCoverage(
        origin=Origin.CLAUDE_AI_EXPORT.value,
        provider=Provider.CLAUDE_AI.value,
        status="estimate_only",
        evidence_stream="transcript_text_estimate",
        request_semantics="Claude.ai exports preserve transcript content, not provider usage counters.",
        cumulative_semantics="No cumulative provider usage window is present in the export shape.",
        cache_semantics="No cache read/write token lanes are available in Claude.ai export rows.",
    ),
    ProviderUsageCoverage(
        origin=Origin.AISTUDIO_DRIVE.value,
        provider=Provider.GEMINI.value,
        status="partial",
        evidence_stream="message_token_fields",
        request_semantics="AI Studio/Gemini exports may carry message-level tokenCount output counters on some records.",
        cumulative_semantics="No provider cumulative session usage window is available from Drive prompt exports.",
        cache_semantics="No cache read/write token lanes are available from Drive prompt exports.",
        notes=("Input tokens and cache semantics are missing unless a future export shape supplies them explicitly.",),
    ),
    ProviderUsageCoverage(
        origin=Origin.GEMINI_CLI_SESSION.value,
        provider=Provider.GEMINI_CLI.value,
        status="partial",
        evidence_stream="message_token_fields",
        request_semantics="Local Gemini CLI documents may carry generic usage/tokens dictionaries per message.",
        cumulative_semantics="No provider cumulative session usage window is available.",
        cache_semantics="Generic cache_read/cache_write keys are preserved when present, but their provider semantics are not independently verified.",
    ),
    ProviderUsageCoverage(
        origin=Origin.HERMES_SESSION.value,
        provider=Provider.HERMES.value,
        status="partial",
        evidence_stream="message_token_fields",
        request_semantics="Hermes local-agent documents may carry generic usage/tokens dictionaries per message.",
        cumulative_semantics="No provider cumulative session usage window is available.",
        cache_semantics="Generic cache_read/cache_write keys are preserved when present, but their provider semantics are not independently verified.",
    ),
    ProviderUsageCoverage(
        origin=Origin.ANTIGRAVITY_SESSION.value,
        provider=Provider.ANTIGRAVITY.value,
        status="unsupported",
        evidence_stream="transcript_text_only",
        request_semantics="No provider usage telemetry parser is implemented for this origin.",
        cumulative_semantics="No cumulative provider usage window is available.",
        cache_semantics="No cache read/write token lanes are available.",
    ),
    ProviderUsageCoverage(
        origin=Origin.UNKNOWN_EXPORT.value,
        provider=Provider.UNKNOWN.value,
        status="unsupported",
        evidence_stream="transcript_text_only",
        request_semantics="Unknown exports are parsed for transcript content only.",
        cumulative_semantics="No cumulative provider usage window is available.",
        cache_semantics="No cache read/write token lanes are available.",
    ),
)

_PROVIDER_USAGE_COVERAGE_BY_ORIGIN = {item.origin: item for item in _PROVIDER_USAGE_COVERAGE}


def provider_usage_coverage_matrix() -> tuple[ProviderUsageCoverage, ...]:
    """Return the declared provider usage coverage matrix."""

    return _PROVIDER_USAGE_COVERAGE


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
    detail_level: str = "full"
    provider: str = "unknown"
    declared_coverage: str = "unsupported"
    coverage_state: str = "unsupported"
    coverage_basis: str = ""
    evidence_stream: str = "transcript_text_only"
    request_semantics: str = ""
    cumulative_semantics: str = ""
    cache_semantics: str = ""
    rebuild_guidance: str = ""
    session_count: int = 0
    message_count: int = 0
    transcript_word_count: int = 0
    raw_session_count: int = 0
    raw_parse_error_count: int = 0
    acquired_not_materialized_count: int = 0
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
    stale_rollup_session_count: int = 0
    provider_request_usage: UsageCounters = field(default_factory=UsageCounters)
    provider_cumulative_usage: UsageCounters = field(default_factory=UsageCounters)
    model_rollup_grain: str = "physical_session"
    model_rollup_usage: UsageCounters = field(default_factory=UsageCounters)
    logical_model_rollup_grain: str = "logical_session_model_high_water"
    logical_model_rollup_usage: UsageCounters = field(default_factory=UsageCounters)
    sample_missing_model_sessions: tuple[str, ...] = ()
    sample_zero_token_sessions: tuple[str, ...] = ()
    sample_acquired_not_materialized_raw_ids: tuple[str, ...] = ()
    sample_stale_rollup_sessions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "detail_level": self.detail_level,
            "provider": self.provider,
            "declared_coverage": self.declared_coverage,
            "coverage_state": self.coverage_state,
            "coverage_basis": self.coverage_basis,
            "evidence_stream": self.evidence_stream,
            "request_semantics": self.request_semantics,
            "cumulative_semantics": self.cumulative_semantics,
            "cache_semantics": self.cache_semantics,
            "rebuild_guidance": self.rebuild_guidance,
            "session_count": self.session_count,
            "message_count": self.message_count,
            "transcript_word_count": self.transcript_word_count,
            "raw_session_count": self.raw_session_count,
            "raw_parse_error_count": self.raw_parse_error_count,
            "acquired_not_materialized_count": self.acquired_not_materialized_count,
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
            "stale_rollup_session_count": self.stale_rollup_session_count,
            "provider_request_usage": self.provider_request_usage.to_dict(),
            "provider_cumulative_usage": self.provider_cumulative_usage.to_dict(),
            "model_rollup_grain": self.model_rollup_grain,
            "model_rollup_usage": self.model_rollup_usage.to_dict(),
            "logical_model_rollup_grain": self.logical_model_rollup_grain,
            "logical_model_rollup_usage": self.logical_model_rollup_usage.to_dict(),
            "sample_missing_model_sessions": list(self.sample_missing_model_sessions),
            "sample_zero_token_sessions": list(self.sample_zero_token_sessions),
            "sample_acquired_not_materialized_raw_ids": list(self.sample_acquired_not_materialized_raw_ids),
            "sample_stale_rollup_sessions": list(self.sample_stale_rollup_sessions),
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True, slots=True)
class PricingLaneReport:
    """Fast repricing headline for one ``session_model_usage`` provenance lane."""

    provenance: str
    row_count: int = 0
    session_count: int = 0
    matched_model_row_count: int = 0
    unmatched_model_row_count: int = 0
    usage: UsageCounters = field(default_factory=UsageCounters)
    stored_cost_usd: float = 0.0
    catalog_api_equivalent_usd: float = 0.0
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "provenance": self.provenance,
            "row_count": self.row_count,
            "session_count": self.session_count,
            "matched_model_row_count": self.matched_model_row_count,
            "unmatched_model_row_count": self.unmatched_model_row_count,
            "usage": self.usage.to_dict(),
            "stored_cost_usd": round(self.stored_cost_usd, 6),
            "catalog_api_equivalent_usd": round(self.catalog_api_equivalent_usd, 6),
            "caveats": list(self.caveats),
        }


@dataclass(slots=True)
class _PricingLaneAccumulator:
    row_count: int = 0
    session_count: int = 0
    matched_model_row_count: int = 0
    unmatched_model_row_count: int = 0
    usage: UsageCounters = field(default_factory=UsageCounters)
    stored_cost_usd: float = 0.0
    catalog_api_equivalent_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class ProviderUsageReport:
    """Archive-level provider usage accounting report."""

    archive_root: str
    origins: tuple[OriginUsageReport, ...]
    detail_level: str = "full"
    model_rollup_grain: str = "physical_session"
    model_rollup_usage: UsageCounters = field(default_factory=UsageCounters)
    logical_model_rollup_grain: str = "logical_session_model_high_water"
    logical_model_rollup_usage: UsageCounters = field(default_factory=UsageCounters)
    pricing_catalog_provenance: str = CATALOG_PROVENANCE
    pricing_catalog_effective_date: str = CATALOG_EFFECTIVE_DATE
    pricing_lanes: tuple[PricingLaneReport, ...] = ()
    pricing_grain: str = "physical_session"
    logical_pricing_lanes: tuple[PricingLaneReport, ...] = ()
    logical_pricing_grain: str = "logical_session_model_high_water"
    stored_provider_priced_usd: float = 0.0
    catalog_api_equivalent_usd: float = 0.0
    logical_catalog_api_equivalent_usd: float = 0.0
    caveats: tuple[str, ...] = ()
    coverage_matrix: tuple[ProviderUsageCoverage, ...] = _PROVIDER_USAGE_COVERAGE

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": self.archive_root,
            "detail_level": self.detail_level,
            "coverage_matrix": [item.to_dict() for item in self.coverage_matrix],
            "model_rollup_grain": self.model_rollup_grain,
            "model_rollup_usage": self.model_rollup_usage.to_dict(),
            "logical_model_rollup_grain": self.logical_model_rollup_grain,
            "logical_model_rollup_usage": self.logical_model_rollup_usage.to_dict(),
            "pricing_catalog_provenance": self.pricing_catalog_provenance,
            "pricing_catalog_effective_date": self.pricing_catalog_effective_date,
            "pricing_lanes": [lane.to_dict() for lane in self.pricing_lanes],
            "pricing_grain": self.pricing_grain,
            "logical_pricing_lanes": [lane.to_dict() for lane in self.logical_pricing_lanes],
            "logical_pricing_grain": self.logical_pricing_grain,
            "stored_provider_priced_usd": round(self.stored_provider_priced_usd, 6),
            "catalog_api_equivalent_usd": round(self.catalog_api_equivalent_usd, 6),
            "logical_catalog_api_equivalent_usd": round(self.logical_catalog_api_equivalent_usd, 6),
            "origins": [origin.to_dict() for origin in self.origins],
            "caveats": list(self.caveats),
        }


def provider_usage_report_for_archive_root(
    archive_root: Path,
    *,
    origin: str | None = None,
    limit: int | None = 25,
    detail: UsageReportDetail = "full",
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
        return provider_usage_report_from_connection(
            conn, archive_root=archive_root, origin=origin, limit=limit, detail=detail
        )
    finally:
        conn.close()


def provider_usage_report_from_connection(
    conn: sqlite3.Connection,
    *,
    archive_root: Path | str = "",
    origin: str | None = None,
    limit: int | None = 25,
    detail: UsageReportDetail = "full",
) -> ProviderUsageReport:
    """Return a provider usage report for an already-open index connection."""

    conn.row_factory = sqlite3.Row
    caveats: list[str] = [
        "provider usage events, transcript text volume, and model rollups are separate evidence streams",
        "this report does not query provider billing and is not a precise cost report",
    ]
    if detail == "headline":
        caveats.append(
            "headline detail computes session/source/model-rollup totals only; run with detail='full' for provider-event, cumulative, sample, and stale-rollup diagnostics"
        )
    if not _table_exists(conn, "sessions"):
        return ProviderUsageReport(
            archive_root=str(archive_root),
            origins=(),
            detail_level=detail,
            caveats=tuple(caveats + ["sessions table is missing"]),
        )

    base_by_origin = _base_session_stats(conn, origin)
    model_table_present = _table_exists(conn, "session_model_usage")
    usage_event_table_present = _table_exists(conn, "session_provider_usage_events")
    model_by_origin = _model_rollup_stats(conn, origin) if model_table_present else {}
    logical_model_by_origin = _logical_model_rollup_stats(conn, origin) if model_table_present else {}
    model_counts_by_origin = _model_row_counts(conn, origin) if model_table_present else {}
    multi_model_by_origin = _multi_model_session_counts(conn, origin) if model_table_present else {}
    pricing_lanes = _pricing_lane_reports(conn, origin) if model_table_present else ()
    logical_pricing_lanes = _pricing_lane_reports(conn, origin, logical=True) if model_table_present else ()
    raw_by_origin, raw_samples, raw_caveat = _source_raw_stats(
        conn, archive_root=Path(archive_root), origin=origin, limit=limit
    )
    if raw_caveat:
        caveats.append(raw_caveat)

    event_by_origin: dict[str, dict[str, object]]
    cumulative_by_origin: dict[str, UsageCounters]
    missing_samples: dict[str, tuple[str, ...]]
    zero_samples: dict[str, tuple[str, ...]]
    stale_by_origin: dict[str, int]
    stale_samples: dict[str, tuple[str, ...]]
    if detail == "headline":
        event_by_origin = {}
        cumulative_by_origin = {}
        missing_samples = {}
        zero_samples = {}
        stale_by_origin = {}
        stale_samples = {}
    elif usage_event_table_present:
        event_by_origin = _provider_event_stats(conn, origin)
        cumulative_by_origin = _provider_cumulative_usage(conn, origin)
        missing_samples = _sample_event_sessions(conn, origin, limit, missing_model=True)
        zero_samples = _sample_event_sessions(conn, origin, limit, zero_token=True)
        stale_by_origin, stale_samples = (
            _stale_provider_rollup_stats(conn, origin, limit) if model_table_present else ({}, {})
        )
    else:
        event_by_origin = {}
        cumulative_by_origin = {}
        missing_samples = {}
        zero_samples = {}
        stale_by_origin = {}
        stale_samples = {}
        caveats.append("session_provider_usage_events table is missing; rebuild the index tier with the current schema")

    origins = sorted(set(base_by_origin) | set(event_by_origin) | set(model_by_origin) | set(raw_by_origin))
    reports: list[OriginUsageReport] = []
    for origin_name in origins:
        base = base_by_origin.get(origin_name, {})
        raw = raw_by_origin.get(origin_name, {})
        events = event_by_origin.get(origin_name, {})
        model_counts = model_counts_by_origin.get(origin_name, {})
        provider_request_usage = events.get("provider_request_usage")
        if not isinstance(provider_request_usage, UsageCounters):
            provider_request_usage = UsageCounters()
        model_rollup_usage = model_by_origin.get(origin_name, UsageCounters())
        coverage = _coverage_for_origin(origin_name)
        session_count = _int(base.get("session_count"))
        provider_event_session_count = _int(events.get("provider_event_session_count"))
        acquired_not_materialized_count = _int(raw.get("acquired_not_materialized_count"))
        stale_rollup_session_count = _int(stale_by_origin.get(origin_name))
        coverage_state, coverage_basis = _coverage_state(
            coverage,
            session_count=session_count,
            provider_event_session_count=provider_event_session_count,
            model_rollup_usage=model_rollup_usage,
            acquired_not_materialized_count=acquired_not_materialized_count,
            stale_rollup_session_count=stale_rollup_session_count,
        )
        if detail == "headline":
            coverage_state = "headline_not_audited"
            coverage_basis = (
                "headline detail does not scan provider usage events or stale rollup diagnostics; "
                "model rollup totals are still computed from session_model_usage"
            )
        origin_caveats = _origin_caveats(
            coverage=coverage,
            coverage_state=coverage_state,
            session_count=session_count,
            provider_event_session_count=provider_event_session_count,
            missing_model_event_count=_int(events.get("missing_model_event_count")),
            zero_token_event_count=_int(events.get("zero_token_event_count")),
            multi_model_session_count=multi_model_by_origin.get(origin_name, 0),
            token_count_event_count=_int(events.get("token_count_event_count")),
            message_usage_event_count=_int(events.get("message_usage_event_count")),
            raw_parse_error_count=_int(raw.get("raw_parse_error_count")),
            acquired_not_materialized_count=acquired_not_materialized_count,
            stale_rollup_session_count=stale_rollup_session_count,
        )
        reports.append(
            OriginUsageReport(
                origin=origin_name,
                detail_level=detail,
                provider=coverage.provider,
                declared_coverage=coverage.status,
                coverage_state=coverage_state,
                coverage_basis=coverage_basis,
                evidence_stream=coverage.evidence_stream,
                request_semantics=coverage.request_semantics,
                cumulative_semantics=coverage.cumulative_semantics,
                cache_semantics=coverage.cache_semantics,
                rebuild_guidance=coverage.rebuild_guidance,
                session_count=session_count,
                message_count=_int(base.get("message_count")),
                transcript_word_count=_int(base.get("transcript_word_count")),
                raw_session_count=_int(raw.get("raw_session_count")),
                raw_parse_error_count=_int(raw.get("raw_parse_error_count")),
                acquired_not_materialized_count=acquired_not_materialized_count,
                provider_event_session_count=provider_event_session_count,
                provider_event_count=_int(events.get("provider_event_count")),
                token_count_event_count=_int(events.get("token_count_event_count")),
                message_usage_event_count=_int(events.get("message_usage_event_count")),
                zero_token_event_count=_int(events.get("zero_token_event_count")),
                missing_model_event_count=_int(events.get("missing_model_event_count")),
                multi_model_session_count=multi_model_by_origin.get(origin_name, 0),
                priced_model_row_count=_int(model_counts.get("priced_model_row_count")),
                origin_reported_model_row_count=_int(model_counts.get("origin_reported_model_row_count")),
                estimated_model_row_count=_int(model_counts.get("estimated_model_row_count")),
                stale_rollup_session_count=stale_rollup_session_count,
                provider_request_usage=provider_request_usage,
                provider_cumulative_usage=cumulative_by_origin.get(origin_name, UsageCounters()),
                model_rollup_grain="physical_session",
                model_rollup_usage=model_rollup_usage,
                logical_model_rollup_grain="logical_session_model_high_water",
                logical_model_rollup_usage=logical_model_by_origin.get(origin_name, UsageCounters()),
                sample_missing_model_sessions=tuple(missing_samples.get(origin_name, ())),
                sample_zero_token_sessions=tuple(zero_samples.get(origin_name, ())),
                sample_acquired_not_materialized_raw_ids=tuple(raw_samples.get(origin_name, ())),
                sample_stale_rollup_sessions=tuple(stale_samples.get(origin_name, ())),
                caveats=tuple(origin_caveats),
            )
        )

    if origin is not None and not reports:
        caveats.append(f"no sessions found for origin {origin!r}")
        caveats.append(f"no raw rows found for origin {origin!r}")
    return ProviderUsageReport(
        archive_root=str(archive_root),
        origins=tuple(reports),
        detail_level=detail,
        model_rollup_usage=_sum_usage_counters(row.model_rollup_usage for row in reports),
        logical_model_rollup_usage=_sum_usage_counters(row.logical_model_rollup_usage for row in reports),
        pricing_lanes=pricing_lanes,
        logical_pricing_lanes=logical_pricing_lanes,
        stored_provider_priced_usd=sum(lane.stored_cost_usd for lane in pricing_lanes if lane.provenance == "priced"),
        catalog_api_equivalent_usd=sum(lane.catalog_api_equivalent_usd for lane in pricing_lanes),
        logical_catalog_api_equivalent_usd=sum(lane.catalog_api_equivalent_usd for lane in logical_pricing_lanes),
        caveats=tuple(caveats),
    )


def _sum_usage_counters(counters: Iterable[UsageCounters]) -> UsageCounters:
    total = UsageCounters()
    for counter in counters:
        total = total.plus(counter)
    return total


def _coverage_for_origin(origin: str) -> ProviderUsageCoverage:
    return _PROVIDER_USAGE_COVERAGE_BY_ORIGIN.get(
        origin,
        ProviderUsageCoverage(
            origin=origin,
            provider=Provider.UNKNOWN.value,
            status="unsupported",
            evidence_stream="transcript_text_only",
            request_semantics="No provider usage telemetry parser is declared for this origin.",
            cumulative_semantics="No cumulative provider usage window is available.",
            cache_semantics="No cache read/write token lanes are available.",
        ),
    )


def _coverage_state(
    coverage: ProviderUsageCoverage,
    *,
    session_count: int,
    provider_event_session_count: int,
    model_rollup_usage: UsageCounters,
    acquired_not_materialized_count: int,
    stale_rollup_session_count: int,
) -> tuple[str, str]:
    if acquired_not_materialized_count:
        return (
            "acquired_not_materialized",
            "source.db has raw rows without parse errors that are not represented in index.db sessions",
        )
    if stale_rollup_session_count:
        return (
            "stale_rollup",
            "provider usage events exist, but session_model_usage no longer matches the event-derived rollup",
        )
    if session_count <= 0:
        return "no_sessions", "no materialized sessions for this origin"
    if coverage.status == "exact":
        if provider_event_session_count <= 0:
            return (
                "missing_provider_telemetry",
                "this origin supports exact provider telemetry, but no provider usage event rows are materialized",
            )
        if provider_event_session_count < session_count:
            return (
                "partial_provider_telemetry",
                "some materialized sessions have provider usage event rows and some do not",
            )
        return (
            "exact_provider_telemetry",
            "all materialized sessions for this origin have provider usage event rows",
        )
    if coverage.status == "partial":
        if model_rollup_usage.is_zero():
            return (
                "partial_telemetry_unobserved",
                "this origin can carry partial message token fields, but none are materialized in model rollups",
            )
        return (
            "partial_provider_telemetry",
            "message-level token fields are materialized, but exact provider request/cumulative semantics are incomplete",
        )
    if coverage.status == "estimate_only":
        return (
            "estimate_only",
            "exact provider token telemetry is unavailable; transcript text counts remain estimate-only evidence",
        )
    return (
        "unsupported",
        "no reliable provider token telemetry is supported for this origin",
    )


def _source_raw_stats(
    conn: sqlite3.Connection,
    archive_root: Path,
    origin: str | None,
    limit: int | None,
) -> tuple[dict[str, dict[str, int]], dict[str, tuple[str, ...]], str | None]:
    alias = _source_schema_alias(conn)
    if alias is None:
        return {}, {}, "source.db raw_sessions unavailable; acquired-not-materialized coverage cannot be checked"
    alias_sql = _quote_identifier(alias)
    try:
        rows = conn.execute(
            f"""
            SELECT r.origin AS origin,
                   COUNT(*) AS raw_session_count,
                   COALESCE(SUM(CASE
                       WHEN r.parse_error IS NOT NULL AND TRIM(r.parse_error) != '' THEN 1 ELSE 0
                   END), 0) AS raw_parse_error_count,
                   COALESCE(SUM(CASE
                       WHEN (r.parse_error IS NULL OR TRIM(r.parse_error) = '')
                        AND s.session_id IS NULL THEN 1 ELSE 0
                   END), 0) AS acquired_not_materialized_count
            FROM {alias_sql}.raw_sessions AS r
            LEFT JOIN sessions AS s ON s.raw_id = r.raw_id
            {_where_origin(origin, table_alias="r")}
            GROUP BY r.origin
            ORDER BY r.origin
            """,
            _origin_args(origin),
        ).fetchall()
        candidates = _acquired_not_materialized_raw_rows(conn, alias_sql, origin)
        actionable_by_origin: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for candidate in candidates:
            if _raw_row_is_known_non_session_artifact(archive_root, candidate):
                continue
            actionable_by_origin[str(candidate["origin"])].append(candidate)
        stats = {
            str(row["origin"]): {
                "raw_session_count": _int(row["raw_session_count"]),
                "raw_parse_error_count": _int(row["raw_parse_error_count"]),
                "acquired_not_materialized_count": len(actionable_by_origin.get(str(row["origin"]), ())),
            }
            for row in rows
        }
        samples = _sample_acquired_not_materialized_raw_ids(actionable_by_origin, limit)
    except sqlite3.Error as exc:
        return {}, {}, f"source.db raw_sessions coverage check failed: {exc}"
    return stats, samples, None


def _acquired_not_materialized_raw_rows(
    conn: sqlite3.Connection,
    alias_sql: str,
    origin: str | None,
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            f"""
        SELECT r.origin AS origin,
               r.raw_id AS raw_id,
               r.source_path AS source_path,
               r.blob_hash AS blob_hash,
               r.parsed_at_ms AS parsed_at_ms
        FROM {alias_sql}.raw_sessions AS r
        LEFT JOIN sessions AS s ON s.raw_id = r.raw_id
        {_where_origin(origin, table_alias="r")}
          {"AND" if origin is not None else "WHERE"} (r.parse_error IS NULL OR TRIM(r.parse_error) = '')
          AND s.session_id IS NULL
        ORDER BY r.origin, r.raw_id
        """,
            _origin_args(origin),
        ).fetchall()
    )


def _sample_acquired_not_materialized_raw_ids(
    rows_by_origin: dict[str, list[sqlite3.Row]],
    limit: int | None,
) -> dict[str, tuple[str, ...]]:
    if limit is not None and limit <= 0:
        return {}
    by_origin: dict[str, list[str]] = defaultdict(list)
    for origin, rows in rows_by_origin.items():
        selected = rows if limit is None else rows[:limit]
        by_origin[origin].extend(str(row["raw_id"]) for row in selected)
    return {key: tuple(value) for key, value in by_origin.items()}


def _raw_row_is_known_non_session_artifact(archive_root: Path, row: sqlite3.Row) -> bool:
    if str(row["origin"]) != Origin.CODEX_SESSION.value:
        return False
    if row["parsed_at_ms"] is None:
        return False
    return _raw_jsonl_type_set(_raw_blob_path(archive_root, row), limit=8) == {"session_meta"}


def _raw_blob_path(archive_root: Path, row: sqlite3.Row) -> Path:
    blob_hash = row["blob_hash"]
    digest = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash)
    return archive_root / "blob" / digest[:2] / digest[2:]


def _raw_jsonl_type_set(path: Path, *, limit: int) -> set[str]:
    types: set[str] = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return set()
                if isinstance(payload, dict) and isinstance(payload.get("type"), str):
                    types.add(str(payload["type"]))
                if len(types) >= limit:
                    break
    except OSError:
        return set()
    return types


def _stale_provider_rollup_stats(
    conn: sqlite3.Connection,
    origin: str | None,
    limit: int | None,
) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
    expected = _expected_provider_model_rollups(conn, origin)
    if not expected:
        return {}, {}
    actual = _actual_model_rollups(conn, origin)
    origin_by_session = _origin_by_session(conn, origin)
    stale_by_origin: dict[str, set[str]] = defaultdict(set)
    for session_id, expected_by_model in expected.items():
        for model_name, expected_tokens in expected_by_model.items():
            if actual.get((session_id, model_name)) != expected_tokens:
                origin_name = origin_by_session.get(session_id)
                if origin_name:
                    stale_by_origin[origin_name].add(session_id)
    counts = {origin_name: len(session_ids) for origin_name, session_ids in stale_by_origin.items()}
    samples = {
        origin_name: tuple(sorted(session_ids)[:limit]) if limit is not None else tuple(sorted(session_ids))
        for origin_name, session_ids in stale_by_origin.items()
    }
    return counts, samples


def _expected_provider_model_rollups(
    conn: sqlite3.Connection,
    origin: str | None,
) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    models_by_session = _models_by_session(conn, origin)
    if not models_by_session:
        return {}
    rows = conn.execute(
        f"""
        SELECT s.origin AS origin, e.session_id AS session_id, e.model_name AS model_name, e.position AS position,
               e.last_input_tokens AS last_input_tokens,
               e.last_output_tokens AS last_output_tokens,
               e.last_cached_input_tokens AS last_cached_input_tokens,
               e.last_cache_write_tokens AS last_cache_write_tokens,
               e.last_reasoning_output_tokens AS last_reasoning_output_tokens,
               e.last_total_tokens AS last_total_tokens,
               e.total_input_tokens AS total_input_tokens,
               e.total_output_tokens AS total_output_tokens,
               e.total_cached_input_tokens AS total_cached_input_tokens,
               e.total_cache_write_tokens AS total_cache_write_tokens,
               e.total_reasoning_output_tokens AS total_reasoning_output_tokens,
               e.total_tokens AS total_tokens
        FROM session_provider_usage_events AS e
        JOIN sessions AS s ON s.session_id = e.session_id
        {_where_origin(origin, table_alias="s")}
          {"AND" if origin is not None else "WHERE"} e.provider_event_type = 'token_count'
        ORDER BY e.session_id, e.position
        """,
        _origin_args(origin),
    ).fetchall()
    latest_total_by_session: dict[str, tuple[str, tuple[int, int, int, int, int]]] = {}
    summed_last_by_model: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        session_id = str(row["session_id"])
        model_name = str(row["model_name"]).strip() if row["model_name"] else ""
        existing_models = models_by_session.get(session_id, ())
        if not model_name and len(existing_models) == 1:
            model_name = existing_models[0]
        if not model_name:
            continue
        last_values = (
            _int(row["last_input_tokens"]),
            _int(row["last_output_tokens"]),
            _int(row["last_cached_input_tokens"]),
            _int(row["last_cache_write_tokens"]),
            _int(row["last_reasoning_output_tokens"]),
            _int(row["last_total_tokens"]),
        )
        total_values = (
            _int(row["total_input_tokens"]),
            _int(row["total_output_tokens"]),
            _int(row["total_cached_input_tokens"]),
            _int(row["total_cache_write_tokens"]),
            _int(row["total_reasoning_output_tokens"]),
            _int(row["total_tokens"]),
        )
        if any(total_values):
            latest_total_by_session[session_id] = (model_name, total_values[:5])
            continue
        key = (session_id, model_name)
        if any(last_values):
            bucket = summed_last_by_model.setdefault(key, [0, 0, 0, 0, 0])
            bucket[0] += last_values[0]
            bucket[1] += last_values[1]
            bucket[2] += last_values[2]
            bucket[3] += last_values[3]
            bucket[4] += last_values[4]
    expected: dict[str, dict[str, tuple[int, int, int, int]]] = defaultdict(dict)
    # Same disjoint-lane mapping the materializer applies, so this audit's
    # "expected" rollup matches the corrected session_model_usage rows instead
    # of flagging false drift (cached is subtracted out of input; reasoning is
    # already inside output). See _provider_usage_disjoint_lanes for the
    # corpus-verified Codex token semantics.
    from polylogue.storage.sqlite.archive_tiers.write import _provider_usage_disjoint_lanes

    for session_id, (model_name, total_tuple) in latest_total_by_session.items():
        expected[session_id][model_name] = _provider_usage_disjoint_lanes(
            total_tuple[0], total_tuple[1], total_tuple[2], total_tuple[3]
        )
    for (session_id, model_name), last_totals in summed_last_by_model.items():
        if session_id in latest_total_by_session:
            continue
        expected[session_id][model_name] = _provider_usage_disjoint_lanes(
            last_totals[0], last_totals[1], last_totals[2], last_totals[3]
        )
    return {session_id: dict(rows) for session_id, rows in expected.items()}


def _actual_model_rollups(
    conn: sqlite3.Connection,
    origin: str | None,
) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    rows = conn.execute(
        f"""
        SELECT u.session_id AS session_id, u.model_name AS model_name,
               u.input_tokens AS input_tokens, u.output_tokens AS output_tokens,
               u.cache_read_tokens AS cache_read_tokens, u.cache_write_tokens AS cache_write_tokens
        FROM session_model_usage AS u
        JOIN sessions AS s ON s.session_id = u.session_id
        {_where_origin(origin, table_alias="s")}
        """,
        _origin_args(origin),
    ).fetchall()
    return {
        (str(row["session_id"]), str(row["model_name"])): (
            _int(row["input_tokens"]),
            _int(row["output_tokens"]),
            _int(row["cache_read_tokens"]),
            _int(row["cache_write_tokens"]),
        )
        for row in rows
    }


def _models_by_session(conn: sqlite3.Connection, origin: str | None) -> dict[str, tuple[str, ...]]:
    rows = conn.execute(
        f"""
        SELECT u.session_id AS session_id, u.model_name AS model_name
        FROM session_model_usage AS u
        JOIN sessions AS s ON s.session_id = u.session_id
        {_where_origin(origin, table_alias="s")}
        ORDER BY u.session_id, u.model_name
        """,
        _origin_args(origin),
    ).fetchall()
    result: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        model_name = str(row["model_name"]).strip() if row["model_name"] else ""
        if model_name:
            result[str(row["session_id"])].append(model_name)
    return {session_id: tuple(models) for session_id, models in result.items()}


def _origin_by_session(conn: sqlite3.Connection, origin: str | None) -> dict[str, str]:
    rows = conn.execute(
        f"""
        SELECT session_id, origin
        FROM sessions
        {_where_origin(origin)}
        """,
        _origin_args(origin),
    ).fetchall()
    return {str(row["session_id"]): str(row["origin"]) for row in rows}


def _source_schema_alias(conn: sqlite3.Connection) -> str | None:
    for alias in ("source_tier", "source_debt", "source", "usage_source_tier"):
        if _table_exists_in_schema(conn, alias, "raw_sessions"):
            return alias
    source_db = _sibling_source_db(conn)
    if source_db is None or not source_db.exists():
        return None
    try:
        conn.execute("ATTACH DATABASE ? AS usage_source_tier", (str(source_db),))
    except sqlite3.Error:
        return None
    return "usage_source_tier" if _table_exists_in_schema(conn, "usage_source_tier", "raw_sessions") else None


def _sibling_source_db(conn: sqlite3.Connection) -> Path | None:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row[1]) != "main":
            continue
        path_text = str(row[2] or "")
        if not path_text:
            return None
        return Path(path_text).with_name("source.db")
    return None


def _quote_identifier(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def _table_exists_in_schema(conn: sqlite3.Connection, schema: str, name: str) -> bool:
    try:
        row = conn.execute(
            f"SELECT 1 FROM {_quote_identifier(schema)}.sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


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


def _logical_model_rollup_stats(conn: sqlite3.Connection, origin: str | None) -> dict[str, UsageCounters]:
    """Return logical-session/model high-water usage by origin.

    ``session_model_usage`` is the physical-session evidence stream.  Logical
    accounting collapses continuation/fork/replay chains by grouping rows under
    ``session_profiles.logical_session_id`` and taking the highest observed lane
    value for each logical-session/model pair.  Summing those high-water rows
    gives consumers a labeled logical view without erasing the physical rows.
    """

    rows = conn.execute(
        f"""
        WITH logical_model AS (
            SELECT s.origin AS origin,
                   COALESCE(p.logical_session_id, u.session_id) AS logical_session_id,
                   u.model_name AS model_name,
                   MAX(u.input_tokens) AS input_tokens,
                   MAX(u.output_tokens) AS output_tokens,
                   MAX(u.cache_read_tokens) AS cached_input_tokens,
                   MAX(u.cache_write_tokens) AS cache_write_tokens
            FROM session_model_usage u
            JOIN sessions s ON s.session_id = u.session_id
            LEFT JOIN session_profiles p ON p.session_id = u.session_id
            {_where_origin(origin, table_alias="s")}
            GROUP BY s.origin, logical_session_id, u.model_name
        )
        SELECT origin,
               COALESCE(SUM(input_tokens), 0) AS input_tokens,
               COALESCE(SUM(output_tokens), 0) AS output_tokens,
               COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
               COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
               0 AS reasoning_output_tokens,
               COALESCE(SUM(input_tokens + output_tokens + cached_input_tokens + cache_write_tokens), 0) AS total_tokens
        FROM logical_model
        GROUP BY origin
        ORDER BY origin
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


def _pricing_lane_reports(
    conn: sqlite3.Connection,
    origin: str | None,
    *,
    logical: bool = False,
) -> tuple[PricingLaneReport, ...]:
    if logical:
        session_count_rows = conn.execute(
            f"""
            WITH logical_model AS (
                SELECT COALESCE(u.cost_provenance, 'unknown') AS provenance,
                       COALESCE(p.logical_session_id, u.session_id) AS logical_session_id,
                       COALESCE(NULLIF(TRIM(u.model_name), ''), '') AS model_name
                FROM session_model_usage u
                JOIN sessions s ON s.session_id = u.session_id
                LEFT JOIN session_profiles p ON p.session_id = u.session_id
                {_where_origin(origin, table_alias="s")}
                GROUP BY provenance, logical_session_id, model_name
            )
            SELECT provenance,
                   COUNT(DISTINCT logical_session_id) AS session_count
            FROM logical_model
            GROUP BY provenance
            """,
            _origin_args(origin),
        ).fetchall()
    else:
        session_count_rows = conn.execute(
            f"""
            SELECT COALESCE(u.cost_provenance, 'unknown') AS provenance,
                   COUNT(DISTINCT u.session_id) AS session_count
            FROM session_model_usage u
            JOIN sessions s ON s.session_id = u.session_id
            {_where_origin(origin, table_alias="s")}
            GROUP BY provenance
            """,
            _origin_args(origin),
        ).fetchall()
    session_counts = {str(row["provenance"] or "unknown"): _int(row["session_count"]) for row in session_count_rows}
    if logical:
        rows = conn.execute(
            f"""
            WITH logical_model AS (
                SELECT COALESCE(u.cost_provenance, 'unknown') AS provenance,
                       COALESCE(p.logical_session_id, u.session_id) AS logical_session_id,
                       COALESCE(NULLIF(TRIM(u.model_name), ''), '') AS model_name,
                       MAX(u.input_tokens) AS input_tokens,
                       MAX(u.output_tokens) AS output_tokens,
                       MAX(u.cache_read_tokens) AS cached_input_tokens,
                       MAX(u.cache_write_tokens) AS cache_write_tokens
                FROM session_model_usage u
                JOIN sessions s ON s.session_id = u.session_id
                LEFT JOIN session_profiles p ON p.session_id = u.session_id
                {_where_origin(origin, table_alias="s")}
                GROUP BY provenance, logical_session_id, model_name
            )
            SELECT provenance,
                   model_name,
                   COUNT(*) AS row_count,
                   COALESCE(SUM(input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                   COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                   0 AS reasoning_output_tokens,
                   COALESCE(SUM(input_tokens + output_tokens + cached_input_tokens + cache_write_tokens), 0)
                     AS total_tokens,
                   0.0 AS stored_cost_usd
            FROM logical_model
            GROUP BY provenance, model_name
            ORDER BY provenance, model_name
            """,
            _origin_args(origin),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT COALESCE(u.cost_provenance, 'unknown') AS provenance,
                   COALESCE(NULLIF(TRIM(u.model_name), ''), '') AS model_name,
                   COUNT(*) AS row_count,
                   COALESCE(SUM(u.input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(u.output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(u.cache_read_tokens), 0) AS cached_input_tokens,
                   COALESCE(SUM(u.cache_write_tokens), 0) AS cache_write_tokens,
                   0 AS reasoning_output_tokens,
                   COALESCE(SUM(u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_write_tokens), 0) AS total_tokens,
                   COALESCE(SUM(u.cost_usd), 0.0) AS stored_cost_usd
            FROM session_model_usage u
            JOIN sessions s ON s.session_id = u.session_id
            {_where_origin(origin, table_alias="s")}
            GROUP BY provenance, model_name
            ORDER BY provenance, model_name
            """,
            _origin_args(origin),
        ).fetchall()
    by_provenance: dict[str, _PricingLaneAccumulator] = {}
    caveats_by_provenance: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        provenance = str(row["provenance"] or "unknown")
        bucket = by_provenance.setdefault(provenance, _PricingLaneAccumulator())
        model_name = str(row["model_name"] or "")
        usage = UsageCounters.from_row(
            row,
            input_key="input_tokens",
            output_key="output_tokens",
            cached_input_key="cached_input_tokens",
            cache_write_key="cache_write_tokens",
            reasoning_output_key="reasoning_output_tokens",
            total_key="total_tokens",
        )
        row_count = _int(row["row_count"])
        bucket.row_count += row_count
        bucket.session_count = session_counts.get(provenance, 0)
        bucket.usage = bucket.usage.plus(usage)
        stored_cost = float(row["stored_cost_usd"] or 0.0)
        bucket.stored_cost_usd += stored_cost
        catalog_cost = 0.0
        if provenance == "priced" and stored_cost > 0 and not logical:
            catalog_cost = stored_cost
            bucket.matched_model_row_count += row_count
        elif model_name:
            normalized = _normalize_model(model_name)
            if normalized in PRICING:
                catalog_cost = estimate_cost(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cached_input_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    model=normalized,
                )
                bucket.matched_model_row_count += row_count
            else:
                bucket.unmatched_model_row_count += row_count
                caveats_by_provenance[provenance].add("missing_price")
        else:
            bucket.unmatched_model_row_count += row_count
            caveats_by_provenance[provenance].add("missing_model")
        if usage.cached_input_tokens and catalog_cost == 0.0 and provenance != "priced":
            caveats_by_provenance[provenance].add("unpriced_cache_read_or_missing_price")
        bucket.catalog_api_equivalent_usd += catalog_cost

    result: list[PricingLaneReport] = []
    for provenance, bucket in sorted(
        by_provenance.items(),
        key=lambda item: (0 if item[0] == "priced" else 1 if item[0] == "origin_reported" else 2, item[0]),
    ):
        result.append(
            PricingLaneReport(
                provenance=provenance,
                row_count=bucket.row_count,
                session_count=bucket.session_count,
                matched_model_row_count=bucket.matched_model_row_count,
                unmatched_model_row_count=bucket.unmatched_model_row_count,
                usage=bucket.usage,
                stored_cost_usd=round(bucket.stored_cost_usd, 6),
                catalog_api_equivalent_usd=round(bucket.catalog_api_equivalent_usd, 6),
                caveats=tuple(sorted(caveats_by_provenance.get(provenance, ()))),
            )
        )
    return tuple(result)


def _provider_event_stats(conn: sqlite3.Connection, origin: str | None) -> dict[str, dict[str, object]]:
    columns = _table_columns(conn, "session_provider_usage_events")
    origin_select = "? AS origin" if origin is not None else "s.origin AS origin"
    join_sessions = "" if origin is not None else "JOIN sessions s ON s.session_id = e.session_id"
    where_clause = _event_origin_where(origin)
    args = _event_origin_args(origin)
    select_parts = [
        origin_select,
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
        {join_sessions}
        {where_clause}
        GROUP BY origin
        ORDER BY origin
        """,
        args,
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
    origin_select = "? AS origin" if origin is not None else "s.origin AS origin"
    join_sessions = "" if origin is not None else "JOIN sessions s ON s.session_id = e.session_id"
    origin_filter = _event_origin_predicate(origin)
    where_parts = [part for part in (origin_filter, f"({total_predicate})") if part]
    where_clause = "WHERE " + " AND ".join(where_parts)
    rows = conn.execute(
        f"""
        SELECT {origin_select},
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
        {join_sessions}
        {where_clause}
        ORDER BY origin, e.session_id, e.position
        """,
        (*_event_origin_args(origin),),
    ).fetchall()
    # The cumulative total_* is session-global, so dedupe to one latest
    # cumulative per (origin, session) — the highest-position event — rather
    # than per model. Partitioning by model and summing double-counts because
    # each model's latest cumulative already includes prior models' tokens
    # (#2472). ORDER BY ... e.position makes the last write per session win.
    latest: dict[tuple[str, str], UsageCounters] = {}
    for row in rows:
        latest[(str(row["origin"]), str(row["session_id"]))] = UsageCounters.from_row(
            row,
            input_key="input_tokens",
            output_key="output_tokens",
            cached_input_key="cached_input_tokens",
            cache_write_key="cache_write_tokens",
            reasoning_output_key="reasoning_output_tokens",
            total_key="total_tokens",
        )
    by_origin: dict[str, UsageCounters] = defaultdict(UsageCounters)
    for (origin_name, _session_id), counters in latest.items():
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
        SELECT DISTINCT {"? AS origin" if origin is not None else "s.origin AS origin"}, e.session_id AS session_id
        FROM session_provider_usage_events e
        {"" if origin is not None else "JOIN sessions s ON s.session_id = e.session_id"}
        WHERE {" AND ".join([part for part in (_event_origin_predicate(origin), *predicates) if part])}
        ORDER BY origin, e.session_id
        {"LIMIT ?" if limit is not None else ""}
        """,
        (*_event_origin_args(origin), *(() if limit is None else (limit,))),
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
    coverage: ProviderUsageCoverage,
    coverage_state: str,
    session_count: int,
    provider_event_session_count: int,
    missing_model_event_count: int,
    zero_token_event_count: int,
    multi_model_session_count: int,
    token_count_event_count: int,
    message_usage_event_count: int,
    raw_parse_error_count: int,
    acquired_not_materialized_count: int,
    stale_rollup_session_count: int,
) -> list[str]:
    caveats: list[str] = []
    if coverage.status == "estimate_only":
        caveats.append("exact provider telemetry unavailable; transcript text counts are estimate-only")
    elif coverage.status == "unsupported":
        caveats.append("provider usage telemetry unsupported for this origin")
    elif coverage.status == "partial":
        caveats.append("provider usage telemetry is partial; request, cumulative, and cache semantics are incomplete")
    if coverage_state == "missing_provider_telemetry":
        caveats.append("exact provider telemetry is supported for this origin but no usage events are materialized")
    if session_count and provider_event_session_count < session_count and coverage.status == "exact":
        caveats.append(
            "some sessions have no provider usage event rows; transcript words and model rollups cover different evidence"
        )
    if acquired_not_materialized_count:
        caveats.append(
            "raw rows without parse errors are acquired but not materialized; usage coverage is incomplete until index.db is rebuilt"
        )
    if raw_parse_error_count:
        caveats.append(
            "some raw rows have parse errors; usage telemetry in those rows is unavailable until parsing succeeds"
        )
    if stale_rollup_session_count:
        caveats.append(
            "some model rollups are stale relative to provider usage events; rebuild usage materialization from source evidence"
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


def _event_origin_where(origin: str | None) -> str:
    predicate = _event_origin_predicate(origin)
    return "" if not predicate else f"WHERE {predicate}"


def _event_origin_predicate(origin: str | None) -> str:
    if origin is None:
        return ""
    return "e.session_id >= ? AND e.session_id < ?"


def _event_origin_args(origin: str | None) -> tuple[str, ...]:
    if origin is None:
        return ()
    prefix = f"{origin}:"
    return (origin, prefix, f"{origin};")


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
    "ProviderUsageCoverage",
    "ProviderUsageReport",
    "UsageCounters",
    "provider_usage_coverage_matrix",
    "provider_usage_report_for_archive_root",
    "provider_usage_report_from_connection",
]
