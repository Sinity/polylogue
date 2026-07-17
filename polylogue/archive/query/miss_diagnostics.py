"""Observable diagnostics for query no-result states."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.enums import Origin
from polylogue.core.json import JSONDocument
from polylogue.readiness import VerifyStatus, get_readiness

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from polylogue.archive.query.source_freshness import NamedSourceFreshness
    from polylogue.config import Config

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class QueryMissReason:
    """One observed reason a query may have returned no sessions."""

    code: str
    severity: Severity
    summary: str
    detail: str | None = None
    count: int | None = None

    def to_dict(self) -> JSONDocument:
        payload: JSONDocument = {
            "code": self.code,
            "severity": self.severity,
            "summary": self.summary,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.count is not None:
            payload["count"] = self.count
        return payload


@dataclass(frozen=True, slots=True)
class QueryMissDiagnostics:
    """Structured no-result diagnosis shared by CLI and MCP surfaces."""

    message: str
    filters: tuple[str, ...]
    reasons: tuple[QueryMissReason, ...]
    archive_session_count: int | None = None
    raw_session_count: int | None = None

    def to_dict(self) -> JSONDocument:
        payload: JSONDocument = {
            "message": self.message,
            "filters": list(self.filters),
            "reasons": [reason.to_dict() for reason in self.reasons],
        }
        if self.archive_session_count is not None:
            payload["archive_session_count"] = self.archive_session_count
        if self.raw_session_count is not None:
            payload["raw_session_count"] = self.raw_session_count
        return payload

    def human_reason_lines(self) -> list[str]:
        """Return concise human-facing reason lines."""
        lines: list[str] = []
        for reason in self.reasons:
            lines.append(reason.summary)
            if reason.detail:
                lines.append(f"  {reason.detail}")
        return lines


def _async_method(obj: object, method_name: str) -> Callable[..., Awaitable[object]] | None:
    candidate = getattr(obj, method_name, None)
    if not callable(candidate):
        return None
    return cast(Callable[..., Awaitable[object]], candidate)


async def _call_optional(repository: object, method_name: str, *args: object, **kwargs: object) -> object | None:
    method = _async_method(repository, method_name)
    if method is None:
        return None
    try:
        return await method(*args, **kwargs)
    except Exception:
        logger.exception("_call_optional: repository method `%s` failed", method_name)
        return None


def _int_value(value: object) -> int | None:
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


def _origins(stats: object | None) -> dict[str, int]:
    value = getattr(stats, "origins", None)
    if not isinstance(value, Mapping):
        value = getattr(stats, "providers", None)
    if not isinstance(value, Mapping):
        return {}
    origins: dict[str, int] = {}
    for key, count in value.items():
        parsed = _int_value(count)
        if parsed is not None:
            origins[str(key)] = parsed
    return origins


def _selected_origin(selection: SessionQuerySpec) -> str | None:
    if len(selection.origins) != 1 or selection.excluded_origins:
        return None
    return Origin.from_string(selection.origins[0]).value


def _archive_count_for_selection(stats: object | None, selection: SessionQuerySpec) -> int | None:
    if len(selection.origins) == 1 and not selection.excluded_origins:
        origins = _origins(stats)
        if origins:
            origin = selection.origins[0]
            return origins.get(origin, 0)
    return _int_value(getattr(stats, "total_sessions", None))


def _query_miss_message(filters: tuple[str, ...]) -> str:
    return "No sessions matched filters." if filters else "No sessions matched."


def _readiness_index_reason(config: Config | None, selection: SessionQuerySpec) -> QueryMissReason | None:
    if config is None:
        return None
    if not isinstance(getattr(config, "db_path", None), Path):
        return None
    try:
        plan = selection.to_plan()
    except Exception:
        logger.exception("_readiness_index_reason: selection.to_plan() failed")
        return None
    if not plan.fts_terms:
        return None
    try:
        report = get_readiness(config, probe_only=True)
    except Exception:
        logger.exception("_readiness_index_reason: get_readiness() failed")
        return None
    index_check = next((check for check in report.checks if check.name == "index"), None)
    if index_check is None or index_check.status is VerifyStatus.OK:
        return None
    return QueryMissReason(
        code="message_index_degraded",
        severity="warning",
        summary="Message search index is not ready.",
        detail=index_check.summary or None,
        count=index_check.count,
    )


async def _action_read_model_reason(
    repository: object,
    selection: SessionQuerySpec,
) -> QueryMissReason | None:
    del repository, selection
    return None


def _archive_empty_reason(archive_count: int | None) -> QueryMissReason | None:
    if archive_count != 0:
        return None
    return QueryMissReason(
        code="archive_empty",
        severity="info",
        summary="The selected archive scope has no materialized sessions.",
        count=0,
    )


def _raw_backlog_reason(raw_count: int | None, archive_count: int | None) -> QueryMissReason | None:
    if raw_count is None or raw_count <= 0:
        return None
    if archive_count is not None and archive_count > 0:
        return None
    return QueryMissReason(
        code="raw_ingest_backlog",
        severity="warning",
        summary="Raw ingested sessions exist but are not materialized into searchable sessions.",
        count=raw_count,
    )


def _fallback_reason(archive_count: int | None, reasons: list[QueryMissReason]) -> QueryMissReason | None:
    if reasons:
        return None
    if archive_count is None or archive_count > 0:
        return QueryMissReason(
            code="no_matching_session",
            severity="info",
            summary="The archive is reachable, but no materialized session matched this selection.",
            count=archive_count,
        )
    return None


async def diagnose_query_miss(
    repository: object,
    selection: SessionQuerySpec,
    *,
    config: Config | None = None,
) -> QueryMissDiagnostics:
    """Build a best-effort diagnosis for an empty query result."""
    filters = tuple(selection.describe())
    stats = await _call_optional(repository, "get_archive_stats")
    archive_count = _archive_count_for_selection(stats, selection)
    raw_count_result = await _call_optional(
        repository,
        "get_raw_session_count",
        origin=_selected_origin(selection),
    )
    raw_count = _int_value(raw_count_result)

    reasons: list[QueryMissReason] = []
    readiness_reason = _readiness_index_reason(config, selection)
    if readiness_reason is not None:
        reasons.append(readiness_reason)
    action_reason = await _action_read_model_reason(repository, selection)
    if action_reason is not None:
        reasons.append(action_reason)
    archive_reason = _archive_empty_reason(archive_count)
    if archive_reason is not None:
        reasons.append(archive_reason)
    backlog_reason = _raw_backlog_reason(raw_count, archive_count)
    if backlog_reason is not None:
        reasons.append(backlog_reason)
    fallback_reason = _fallback_reason(archive_count, reasons)
    if fallback_reason is not None:
        reasons.append(fallback_reason)

    return QueryMissDiagnostics(
        message=_query_miss_message(filters),
        filters=filters,
        reasons=tuple(reasons),
        archive_session_count=archive_count,
        raw_session_count=raw_count,
    )


_NAMED_SOURCE_MISS_COPY: dict[str, tuple[str, Severity, str]] = {
    "unseen": (
        "named_source_unseen",
        "info",
        "The named source has not been acquired into the raw archive.",
    ),
    "acquired-unparsed": (
        "named_source_acquired_unparsed",
        "warning",
        "The named source was acquired, but its accepted raw revision is not parsed.",
    ),
    "parsed-unindexed": (
        "named_source_parsed_unindexed",
        "warning",
        "The named source was parsed, but its accepted raw revision is not indexed.",
    ),
    "indexed-unconverged": (
        "named_source_indexed_unconverged",
        "warning",
        "The named source is indexed, but FTS or insight evidence has not converged.",
    ),
    "searchable": (
        "named_source_searchable",
        "info",
        "The named source is searchable; the miss is downstream of source freshness.",
    ),
}


def diagnose_named_source_miss(freshness: NamedSourceFreshness) -> QueryMissDiagnostics:
    """Translate one exact-source projection into the shared miss envelope."""
    stage_object = freshness.stage
    stage = str(getattr(stage_object, "value", stage_object))
    code, severity, summary = _NAMED_SOURCE_MISS_COPY.get(
        stage,
        (
            "named_source_unknown",
            "warning",
            "The named source freshness stage is unavailable.",
        ),
    )
    operational_object = freshness.operational_state
    operational = str(getattr(operational_object, "value", operational_object))
    operational_reason_object = getattr(freshness, "operational_reason", None)
    operational_reason = getattr(operational_reason_object, "value", operational_reason_object)
    detail_parts = [
        f"source_path={freshness.source_path}",
        f"stage={stage}",
        f"operational_state={operational}",
    ]
    if operational_reason is not None:
        detail_parts.append(f"operational_reason={operational_reason}")
    source_stat = getattr(freshness, "source_stat", None)
    if source_stat is not None:
        detail_parts.append(f"source_exists={getattr(source_stat, 'exists', None)}")
        stat_error = getattr(source_stat, "error", None)
        if stat_error:
            detail_parts.append(f"source_stat_error={stat_error}")
    if freshness.cursor.excluded:
        detail_parts.append("cursor_excluded=true")
    if freshness.cursor.pending_bytes is not None:
        detail_parts.append(f"pending_bytes={freshness.cursor.pending_bytes}")
    cursor_ahead = getattr(freshness.cursor, "cursor_ahead_bytes", None)
    if cursor_ahead:
        detail_parts.append(f"cursor_ahead_bytes={cursor_ahead}")
    if freshness.index.broken_head:
        detail_parts.append("broken_head=true")
    source_session_count = freshness.index.session_count_lower_bound
    raw_sample_count = len(freshness.raw_revisions)
    detail_parts.append(f"source_indexed_session_lower_bound={source_session_count}")
    detail_parts.append(f"source_raw_revision_sample_count={raw_sample_count}")
    if freshness.retry.reason:
        detail_parts.append(f"reason={freshness.retry.reason}")
    reason = QueryMissReason(
        code=code,
        severity=severity,
        summary=summary,
        detail="; ".join(detail_parts),
    )
    return QueryMissDiagnostics(
        message="No session matched the named source.",
        filters=(f"source_path:{freshness.source_path}",),
        reasons=(reason,),
    )


__all__ = [
    "QueryMissDiagnostics",
    "QueryMissReason",
    "diagnose_named_source_miss",
    "diagnose_query_miss",
]
