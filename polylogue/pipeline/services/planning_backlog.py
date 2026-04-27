"""Backlog helpers for ingest planning."""

from __future__ import annotations

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.raw.artifacts import parse_backlog_query_spec, validation_backlog_query_spec


def dedupe_ids(raw_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_id in raw_ids:
        if raw_id in seen:
            continue
        seen.add(raw_id)
        ordered.append(raw_id)
    return ordered


async def collect_validation_backlog(
    backend: SQLiteBackend,
    *,
    source_names: list[str] | None,
    exclude_raw_ids: list[str] | None = None,
    force_reparse: bool = False,
) -> list[str]:
    exclude = set(exclude_raw_ids or [])
    backlog_validate_ids: list[str] = []
    query_spec = validation_backlog_query_spec(force_reparse=force_reparse)
    async for raw_id in backend.iter_raw_ids(
        source_names=source_names,
        require_unparsed=query_spec.require_unparsed,
        require_unvalidated=query_spec.require_unvalidated,
        validation_statuses=list(query_spec.validation_statuses)
        if query_spec.validation_statuses is not None
        else None,
    ):
        if raw_id not in exclude:
            backlog_validate_ids.append(raw_id)
    return backlog_validate_ids


async def collect_parse_backlog(
    backend: SQLiteBackend,
    *,
    source_names: list[str] | None,
    exclude_raw_ids: list[str] | None = None,
    force_reparse: bool = False,
) -> list[str]:
    exclude = set(exclude_raw_ids or [])
    backlog_parse_ids: list[str] = []
    query_spec = parse_backlog_query_spec(force_reparse=force_reparse)
    async for raw_id in backend.iter_raw_ids(
        source_names=source_names,
        require_unparsed=query_spec.require_unparsed,
        validation_statuses=list(query_spec.validation_statuses)
        if query_spec.validation_statuses is not None
        else None,
    ):
        if raw_id not in exclude:
            backlog_parse_ids.append(raw_id)
    return dedupe_ids(backlog_parse_ids)


__all__ = ["collect_parse_backlog", "collect_validation_backlog", "dedupe_ids"]
