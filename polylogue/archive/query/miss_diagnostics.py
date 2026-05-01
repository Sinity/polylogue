"""Observable diagnostics for query no-result states."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from polylogue.archive.query.retrieval_candidates import uses_action_read_model
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.core.json import JSONDocument
from polylogue.readiness import VerifyStatus, get_readiness

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from polylogue.config import Config

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class QueryMissReason:
    """One observed reason a query may have returned no conversations."""

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
    archive_conversation_count: int | None = None
    raw_conversation_count: int | None = None

    def to_dict(self) -> JSONDocument:
        payload: JSONDocument = {
            "message": self.message,
            "filters": list(self.filters),
            "reasons": [reason.to_dict() for reason in self.reasons],
        }
        if self.archive_conversation_count is not None:
            payload["archive_conversation_count"] = self.archive_conversation_count
        if self.raw_conversation_count is not None:
            payload["raw_conversation_count"] = self.raw_conversation_count
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


def _providers(stats: object | None) -> dict[str, int]:
    value = getattr(stats, "providers", None)
    if not isinstance(value, Mapping):
        return {}
    providers: dict[str, int] = {}
    for key, count in value.items():
        parsed = _int_value(count)
        if parsed is not None:
            providers[str(key)] = parsed
    return providers


def _selected_provider(selection: ConversationQuerySpec) -> str | None:
    if len(selection.providers) != 1 or selection.excluded_providers:
        return None
    return selection.providers[0].value


def _archive_count_for_selection(stats: object | None, selection: ConversationQuerySpec) -> int | None:
    provider = _selected_provider(selection)
    if provider is not None:
        providers = _providers(stats)
        if providers:
            return providers.get(provider, 0)
    return _int_value(getattr(stats, "total_conversations", None))


def _query_miss_message(filters: tuple[str, ...]) -> str:
    return "No conversations matched filters." if filters else "No conversations matched."


def _readiness_index_reason(config: Config | None, selection: ConversationQuerySpec) -> QueryMissReason | None:
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


def _state_ready(state: object) -> bool:
    ready = getattr(state, "ready", True)
    return bool(ready)


def _state_repair_count(state: object) -> int | None:
    return _int_value(getattr(state, "repair_item_count", None))


def _state_repair_detail(state: object) -> str | None:
    repair_detail = getattr(state, "repair_detail", None)
    if not callable(repair_detail):
        return None
    detail = repair_detail()
    return str(detail) if detail else None


async def _action_read_model_reason(
    repository: object,
    selection: ConversationQuerySpec,
) -> QueryMissReason | None:
    try:
        plan = selection.to_plan()
    except Exception:
        logger.exception("_action_read_model_reason: selection.to_plan() failed")
        return None
    if not uses_action_read_model(plan):
        return None
    state = await _call_optional(repository, "get_action_event_artifact_state")
    if state is None or _state_ready(state):
        return None
    return QueryMissReason(
        code="action_read_model_degraded",
        severity="warning",
        summary="Action-event read model is not ready.",
        detail=_state_repair_detail(state),
        count=_state_repair_count(state),
    )


def _archive_empty_reason(archive_count: int | None) -> QueryMissReason | None:
    if archive_count != 0:
        return None
    return QueryMissReason(
        code="archive_empty",
        severity="info",
        summary="The selected archive scope has no materialized conversations.",
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
        summary="Raw ingested conversations exist but are not materialized into searchable conversations.",
        count=raw_count,
    )


def _fallback_reason(archive_count: int | None, reasons: list[QueryMissReason]) -> QueryMissReason | None:
    if reasons:
        return None
    if archive_count is None or archive_count > 0:
        return QueryMissReason(
            code="no_matching_conversation",
            severity="info",
            summary="The archive is reachable, but no materialized conversation matched this selection.",
            count=archive_count,
        )
    return None


async def diagnose_query_miss(
    repository: object,
    selection: ConversationQuerySpec,
    *,
    config: Config | None = None,
) -> QueryMissDiagnostics:
    """Build a best-effort diagnosis for an empty query result."""
    filters = tuple(selection.describe())
    stats = await _call_optional(repository, "get_archive_stats")
    archive_count = _archive_count_for_selection(stats, selection)
    raw_count_result = await _call_optional(
        repository,
        "get_raw_conversation_count",
        provider=_selected_provider(selection),
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
        archive_conversation_count=archive_count,
        raw_conversation_count=raw_count,
    )


__all__ = [
    "QueryMissDiagnostics",
    "QueryMissReason",
    "diagnose_query_miss",
]
