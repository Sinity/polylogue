"""Terminal unit-query execution over the archive."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.archive.query.expression import QueryUnitSource
from polylogue.archive.query.spec import (
    normalize_action_sequence,
    normalize_action_terms,
    normalize_tool_terms,
    optional_int,
    optional_message_type,
    optional_text,
    parse_query_date,
    split_csv,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import (
    ActionQueryRowPayload,
    AssertionQueryRowPayload,
    BlockQueryRowPayload,
    MessageQueryRowPayload,
    QueryUnitEnvelope,
    build_query_unit_envelope,
)


def _bool_param(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _epoch_ms(field: str, value: object) -> int | None:
    if isinstance(value, int):
        return value
    if value is None:
        return None
    parsed = parse_query_date(field, str(value))
    if parsed is None:
        return None
    return int(parsed.timestamp() * 1000)


def query_unit_session_filters(**params: object) -> dict[str, object]:
    """Normalize shared session filters for terminal query-unit rows.

    Terminal unit-source execution returns row-level
    results, but callers still need the same surrounding session scope as the
    normal session query surfaces.  This helper is the single cross-surface
    adapter into ``ArchiveStore.query_*``'s ``session_filters`` argument.
    """

    origin = optional_text(params.get("origin"))
    origins = split_csv(params.get("origins"))
    if not origins and origin is None:
        origins = split_csv(params.get("source"))
    excluded_origins = split_csv(params.get("excluded_origins") or params.get("exclude_origin"))
    tags = tuple(tag.lower() for tag in split_csv(params.get("tags") or params.get("tag")))
    excluded_tags = tuple(tag.lower() for tag in split_csv(params.get("excluded_tags") or params.get("exclude_tag")))
    repo_names = split_csv(params.get("repo_names") or params.get("repo"))
    has_types = split_csv(params.get("has_types") or params.get("has_type"))
    since_ms = params.get("since_ms")
    until_ms = params.get("until_ms")
    return {
        "origin": origin,
        "origins": origins,
        "excluded_origins": excluded_origins,
        "tags": tags,
        "excluded_tags": excluded_tags,
        "repo_names": repo_names,
        "has_types": has_types,
        "has_tool_use": _bool_param(params.get("has_tool_use") or params.get("filter_has_tool_use")),
        "has_thinking": _bool_param(params.get("has_thinking") or params.get("filter_has_thinking")),
        "has_paste": _bool_param(params.get("has_paste") or params.get("filter_has_paste")),
        "tool_terms": normalize_tool_terms(params.get("tool_terms") or params.get("tool")),
        "excluded_tool_terms": normalize_tool_terms(params.get("excluded_tool_terms") or params.get("exclude_tool")),
        "action_terms": normalize_action_terms("action", params.get("action_terms") or params.get("action")),
        "excluded_action_terms": normalize_action_terms(
            "exclude_action", params.get("excluded_action_terms") or params.get("exclude_action")
        ),
        "action_sequence": normalize_action_sequence(
            "action_sequence", params.get("action_sequence") or params.get("sequence")
        ),
        "action_text_terms": split_csv(params.get("action_text_terms") or params.get("action_text")),
        "referenced_paths": split_csv(params.get("referenced_paths") or params.get("referenced_path")),
        "cwd_prefix": optional_text(params.get("cwd_prefix")),
        "typed_only": _bool_param(params.get("typed_only")),
        "message_type": optional_message_type(params.get("message_type")),
        "title": optional_text(params.get("title")),
        "min_messages": optional_int(params.get("min_messages")),
        "max_messages": optional_int(params.get("max_messages")),
        "min_words": optional_int(params.get("min_words")),
        "max_words": optional_int(params.get("max_words")),
        "since_ms": int(since_ms) if isinstance(since_ms, int) else _epoch_ms("since", params.get("since")),
        "until_ms": int(until_ms) if isinstance(until_ms, int) else _epoch_ms("until", params.get("until")),
    }


def query_unit_rows(
    archive: ArchiveStore,
    source: QueryUnitSource,
    *,
    query: str,
    limit: int,
    offset: int = 0,
    session_filters: Mapping[str, object] | None = None,
) -> QueryUnitEnvelope:
    """Execute an explicit unit-source query."""

    fetch_limit = limit + 1
    if source.unit == "message":
        message_rows = archive.query_messages(
            source.predicate,
            limit=fetch_limit,
            offset=offset,
            session_filters=session_filters,
        )
        return build_query_unit_envelope(
            tuple(MessageQueryRowPayload.from_row(row) for row in message_rows[:limit]),
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(message_rows) > limit,
        )
    if source.unit == "action":
        action_rows = archive.query_actions(
            source.predicate,
            limit=fetch_limit,
            offset=offset,
            session_filters=session_filters,
        )
        return build_query_unit_envelope(
            tuple(ActionQueryRowPayload.from_row(row) for row in action_rows[:limit]),
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(action_rows) > limit,
        )
    if source.unit == "assertion":
        assertion_rows = archive.query_assertions(
            source.predicate,
            limit=fetch_limit,
            offset=offset,
            session_filters=session_filters,
        )
        return build_query_unit_envelope(
            tuple(AssertionQueryRowPayload.from_row(row) for row in assertion_rows[:limit]),
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(assertion_rows) > limit,
        )
    block_rows = archive.query_blocks(
        source.predicate,
        limit=fetch_limit,
        offset=offset,
        session_filters=session_filters,
    )
    return build_query_unit_envelope(
        tuple(BlockQueryRowPayload.from_row(row) for row in block_rows[:limit]),
        unit=source.unit,
        query=query,
        limit=limit,
        offset=offset,
        has_next=len(block_rows) > limit,
    )


__all__ = ["query_unit_rows", "query_unit_session_filters"]
