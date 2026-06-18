"""Archive helpers shared by MCP tools and resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from polylogue.archive.query.spec import parse_query_date
from polylogue.paths import archive_file_set_index_available_for_paths, archive_file_set_root_for_paths
from polylogue.surfaces.payloads import TargetRefPayload, reader_anchor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.archive.blackboard import BlackboardNote
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec
    from polylogue.config import Config
    from polylogue.mcp.payloads import (
        MCPBlackboardNotePayload,
        MCPMessagePayload,
        MCPMessagesListPayload,
        MCPPaginatedQueryResultPayload,
        MCPSessionSummaryPayload,
    )
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSearchHit,
        ArchiveSessionSummary,
        ArchiveStore,
    )
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveBlockRow, ArchiveMessageRow, ArchiveSessionEnvelope
    from polylogue.surfaces.payloads import QueryUnitEnvelope, SearchEnvelope, SessionSearchHitPayload


def _real_path(value: object) -> Path | None:
    return value if isinstance(value, Path) else None


class ArchiveQueryFilters(TypedDict):
    origin: str | None
    origins: tuple[str, ...]
    excluded_origins: tuple[str, ...]
    tags: tuple[str, ...]
    excluded_tags: tuple[str, ...]
    repo_names: tuple[str, ...]
    has_types: tuple[str, ...]
    has_tool_use: bool
    has_thinking: bool
    has_paste: bool
    tool_terms: tuple[str, ...]
    excluded_tool_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    excluded_action_terms: tuple[str, ...]
    action_sequence: tuple[str, ...]
    action_text_terms: tuple[str, ...]
    referenced_paths: tuple[str, ...]
    cwd_prefix: str | None
    typed_only: bool
    message_type: str | None
    title: str | None
    min_messages: int | None
    max_messages: int | None
    min_words: int | None
    max_words: int | None
    since_ms: int | None
    until_ms: int | None
    since_session_id: str | None


def active_archive_root(config: Config) -> Path | None:
    """Return the archive root that owns the active archive index."""
    archive_root = _real_path(config.archive_root)
    db_anchor = _real_path(config.db_path)
    if archive_root is None or db_anchor is None:
        return None
    return archive_file_set_root_for_paths(archive_root_path=archive_root, db_anchor=db_anchor)


def archive_index_active_paths(
    *,
    archive_root: Path,
    db_anchor_path: Path,
) -> bool:
    """Path-oriented variant for call sites that do not carry Config."""
    if not isinstance(archive_root, Path):
        return False
    return archive_file_set_index_available_for_paths(archive_root_path=archive_root, db_anchor=db_anchor_path)


def archive_query_filters(spec: SessionQuerySpec) -> ArchiveQueryFilters:
    """Translate the shared query spec into archive index filter kwargs."""
    return {
        "origin": None,
        "origins": spec.origins,
        "excluded_origins": spec.excluded_origins,
        "tags": spec.tags,
        "excluded_tags": spec.excluded_tags,
        "repo_names": spec.repo_names,
        "has_types": spec.has_types,
        "has_tool_use": spec.filter_has_tool_use,
        "has_thinking": spec.filter_has_thinking,
        "has_paste": spec.filter_has_paste,
        "tool_terms": spec.tool_terms,
        "excluded_tool_terms": spec.excluded_tool_terms,
        "action_terms": spec.action_terms,
        "excluded_action_terms": spec.excluded_action_terms,
        "action_sequence": spec.action_sequence,
        "action_text_terms": spec.action_text_terms,
        "referenced_paths": spec.referenced_path,
        "cwd_prefix": spec.cwd_prefix,
        "typed_only": spec.typed_only,
        "message_type": spec.message_type,
        "title": spec.title,
        "min_messages": spec.min_messages,
        "max_messages": spec.max_messages,
        "min_words": spec.min_words,
        "max_words": spec.max_words,
        "since_ms": _date_ms(spec.since),
        "until_ms": _date_ms(spec.until),
        "since_session_id": spec.since_session_id,
    }


def _archive_text_query(spec: SessionQuerySpec) -> str | None:
    terms = (*spec.query_terms, *spec.contains_terms)
    text = " ".join(term for term in terms if term).strip()
    return text or None


def archive_summary_payload(summary: ArchiveSessionSummary) -> MCPSessionSummaryPayload:
    """Project an archive session summary into the generic MCP summary shape."""
    from polylogue.mcp.payloads import MCPSessionSummaryPayload

    session_id = summary.session_id
    return MCPSessionSummaryPayload(
        id=session_id,
        origin=summary.origin,
        title=summary.title or "(untitled)",
        message_count=summary.message_count,
        target_ref=TargetRefPayload.session(session_id),
        anchor=reader_anchor("session", session_id),
        created_at=_parse_archive_datetime(summary.created_at),
        updated_at=_parse_archive_datetime(summary.updated_at),
    )


def blackboard_note_payload(note: BlackboardNote) -> MCPBlackboardNotePayload:
    """Project a decoded blackboard note into its MCP payload shape (#1697)."""
    from polylogue.mcp.payloads import MCPBlackboardNotePayload

    return MCPBlackboardNotePayload(
        note_id=note.note_id,
        kind=note.kind,
        title=note.title,
        content=note.content,
        scope_repo=note.scope_repo,
        target_type=note.target_type,
        target_id=note.target_id,
        created_at_ms=note.created_at_ms,
        updated_at_ms=note.updated_at_ms,
    )


def _project_text_block(text: str | None, projection: ContentProjectionSpec) -> str | None:
    if text is None:
        return None
    if (
        projection.include_prose
        and projection.include_code
        and projection.include_reasoning
        and projection.include_system_noise
    ):
        return text
    from polylogue.archive.message.models import Message
    from polylogue.archive.message.roles import Role
    from polylogue.archive.semantic.content_projection import project_message_content

    projected = project_message_content(
        [Message(id="archive-block", role=Role.ASSISTANT, text=text, blocks=[])],
        projection,
    )
    if not projected:
        return None
    return projected[0].text


def _project_archive_message(
    message: ArchiveMessageRow,
    projection: ContentProjectionSpec | None,
) -> ArchiveMessageRow | None:
    if projection is None or projection.is_default():
        return message
    tool_semantics = {
        block.tool_id: block.semantic_type
        for block in message.blocks
        if block.block_type == "tool_use" and block.tool_id and block.semantic_type
    }
    blocks: list[ArchiveBlockRow] = []
    for block in message.blocks:
        if block.block_type == "thinking" and not projection.include_reasoning:
            continue
        if block.block_type == "code" and not projection.include_code:
            continue
        if block.block_type == "tool_use" and not projection.include_tool_calls:
            continue
        if block.block_type == "tool_result":
            semantic_type = tool_semantics.get(block.tool_id or "", block.semantic_type or "")
            if semantic_type == "file_read" and not (projection.include_file_reads and projection.include_tool_outputs):
                continue
            if semantic_type != "file_read" and not projection.include_tool_outputs:
                continue
        if block.block_type in {"image", "document", "file"} and not projection.include_attachments:
            continue
        if block.block_type == "text":
            text = _project_text_block(block.text, projection)
            if text is None:
                continue
            blocks.append(replace(block, text=text))
            continue
        if (
            block.block_type not in {"thinking", "code", "tool_use", "tool_result", "image", "document", "file"}
            and not projection.include_prose
        ):
            continue
        blocks.append(block)
    if not blocks and projection.filters_content():
        return None
    return replace(message, blocks=tuple(blocks))


def archive_message_payload(message: ArchiveMessageRow, *, session_id: str) -> MCPMessagePayload:
    """Project one archive message into the generic MCP message shape."""
    from polylogue.mcp.payloads import MCPMessagePayload

    text = "\n\n".join(block.text for block in message.blocks if block.text)
    content_blocks: list[dict[str, object]] = [
        {
            "type": block.block_type,
            "text": block.text or "",
            "block_id": block.block_id,
            **({"tool_name": block.tool_name} if block.tool_name else {}),
            **({"tool_id": block.tool_id} if block.tool_id else {}),
            **({"semantic_type": block.semantic_type} if block.semantic_type else {}),
        }
        for block in message.blocks
    ]
    return MCPMessagePayload(
        id=message.message_id,
        role=message.role,
        text=text,
        target_ref=TargetRefPayload.message(session_id=session_id, message_id=message.message_id),
        anchor=reader_anchor("message", message.message_id),
        timestamp=_parse_archive_datetime(message.occurred_at),
        message_type=message.message_type,
        content_blocks=content_blocks,
        branch_index=message.variant_index,
        has_paste=message.has_paste,
        has_tool_use=message.has_tool_use,
        has_thinking=message.has_thinking,
    )


def archive_session_list_payload(
    archive: ArchiveStore,
    spec: SessionQuerySpec,
    *,
    config: Config | None = None,
    archive_root: Path | None = None,
    default_limit: int = 10,
) -> MCPPaginatedQueryResultPayload:
    """Build the generic MCP list-sessions envelope from the archive."""
    from polylogue.mcp.payloads import MCPPaginatedQueryResultPayload

    limit = spec.limit or default_limit
    offset = max(0, spec.offset)
    if spec.similar_session_id is not None:
        from polylogue.archive.query.archive_execution import archive_search_hits
        from polylogue.archive.query.spec import query_spec_to_plan

        resolved_archive_root = archive_root or archive.archive_root
        plan = query_spec_to_plan(replace(spec, limit=limit, offset=offset))
        pairs, _resolved_lane = archive_search_hits(
            plan,
            archive_root=resolved_archive_root,
            config=config,
            default_limit=default_limit,
        )
        summaries = [summary for _hit, summary in pairs]
        total = offset + len(summaries) + (1 if len(summaries) == limit else 0)
        next_offset = offset + len(summaries) if len(summaries) == limit else None
        return MCPPaginatedQueryResultPayload(
            items=tuple(archive_summary_payload(summary) for summary in summaries),
            total=total,
            limit=limit,
            offset=offset,
            next_offset=next_offset,
        )
    filters = archive_query_filters(spec)
    text_query = _archive_text_query(spec)
    if text_query is None:
        summaries = archive.list_summaries(
            limit=limit,
            offset=offset,
            sort=_sort_value(spec.sort),
            reverse=spec.reverse,
            sample=bool(spec.sample),
            **filters,
        )
        total = archive.count_sessions(**filters)
    else:
        summaries = [
            archive.read_summary(hit.session_id)
            for hit in archive.search_summaries(
                text_query,
                limit=limit,
                offset=offset,
                sort=_sort_value(spec.sort),
                reverse=spec.reverse,
                **filters,
            )
        ]
        total = archive.count_search_sessions(text_query, **filters)
    next_offset = offset + len(summaries) if len(summaries) == limit and offset + limit < total else None
    return MCPPaginatedQueryResultPayload(
        items=tuple(archive_summary_payload(summary) for summary in summaries),
        total=total,
        limit=limit,
        offset=offset,
        next_offset=next_offset,
    )


def archive_search_payload(
    archive: ArchiveStore,
    spec: SessionQuerySpec,
    *,
    query: str,
    limit: int,
    offset: int,
    retrieval_lane: str,
    sort: str | None,
    config: Config | None = None,
    archive_root: Path | None = None,
) -> SearchEnvelope:
    """Build the generic MCP search envelope from archive block search."""
    from polylogue.surfaces.payloads import build_search_envelope

    if spec.similar_session_id is not None:
        from polylogue.archive.query.archive_execution import archive_search_hits
        from polylogue.archive.query.spec import query_spec_to_plan

        resolved_archive_root = archive_root or archive.archive_root
        plan = query_spec_to_plan(replace(spec, limit=limit, offset=offset))
        pairs, resolved_lane = archive_search_hits(
            plan,
            archive_root=resolved_archive_root,
            config=config,
            default_limit=limit,
        )
        return build_search_envelope(
            tuple(archive_search_hit_payload(hit, archive=archive) for hit, _summary in pairs),
            total=offset + len(pairs) + (1 if len(pairs) == limit else 0),
            limit=limit,
            offset=offset,
            query=query,
            retrieval_lane=resolved_lane,
            sort=sort,
        )

    filters = archive_query_filters(spec)
    hits = archive.search_summaries(
        query,
        limit=limit,
        offset=offset,
        sort=_sort_value(spec.sort),
        reverse=spec.reverse,
        **filters,
    )
    return build_search_envelope(
        tuple(archive_search_hit_payload(hit, archive=archive) for hit in hits),
        total=len(hits),
        limit=limit,
        offset=offset,
        query=query,
        retrieval_lane=retrieval_lane,
        sort=sort,
    )


def archive_query_unit_payload(
    archive: ArchiveStore,
    *,
    expression: str,
    limit: int,
    offset: int,
    session_filters: Mapping[str, object] | None = None,
) -> QueryUnitEnvelope:
    """Build the shared terminal query-unit envelope from an archive."""
    from polylogue.archive.query.expression import ExpressionCompileError, parse_unit_source_expression
    from polylogue.archive.query.unit_results import query_unit_rows

    source = parse_unit_source_expression(expression)
    if source is None:
        raise ExpressionCompileError(
            "query_units requires an explicit messages/actions/blocks/assertions where expression",
            field=None,
        )
    return query_unit_rows(
        archive, source, query=expression, limit=limit, offset=offset, session_filters=session_filters
    )


def archive_messages_payload(
    session: ArchiveSessionEnvelope,
    *,
    roles: Sequence[str] = (),
    message_type: str | None = None,
    content_projection: ContentProjectionSpec | None = None,
    limit: int,
    offset: int,
) -> MCPMessagesListPayload:
    """Build the generic MCP message-list envelope from an archive session."""
    from polylogue.mcp.payloads import MCPMessagesListPayload

    role_filter = frozenset(roles)
    messages = [
        projected
        for message in session.messages
        if (not role_filter or message.role in role_filter)
        and (message_type is None or message.message_type == message_type)
        for projected in (_project_archive_message(message, content_projection),)
        if projected is not None
    ]
    page = messages[offset : offset + limit]
    return MCPMessagesListPayload(
        session_id=session.session_id,
        messages=tuple(archive_message_payload(message, session_id=session.session_id) for message in page),
        total=len(messages),
        limit=limit,
        offset=offset,
    )


def archive_search_hit_payload(hit: ArchiveSessionSearchHit, *, archive: ArchiveStore) -> SessionSearchHitPayload:
    """Project an archive FTS hit into the generic search-hit payload."""
    from polylogue.surfaces.payloads import (
        SessionSearchHitPayload,
        SessionSearchMatchPayload,
        reader_message_actions,
    )

    summary = archive.read_summary(hit.session_id)
    return SessionSearchHitPayload(
        session=archive_summary_payload(summary),
        match=SessionSearchMatchPayload(
            rank=hit.rank,
            retrieval_lane="dialogue",
            match_surface="message",
            target_ref=TargetRefPayload.message(session_id=hit.session_id, message_id=hit.message_id),
            anchor=reader_anchor("message", hit.message_id),
            actions=reader_message_actions(),
            message_id=hit.message_id,
            snippet=hit.snippet,
            score=None,
            score_kind=None,
        ),
    )


def _date_ms(value: str | None) -> int | None:
    parsed = parse_query_date("date", value)
    return int(parsed.timestamp() * 1000) if parsed is not None else None


def _parse_archive_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _sort_value(sort: object) -> str | None:
    if sort is None:
        return None
    value = getattr(sort, "value", sort)
    return str(value)


__all__ = [
    "ArchiveQueryFilters",
    "active_archive_root",
    "archive_index_active_paths",
    "archive_session_list_payload",
    "archive_message_payload",
    "archive_messages_payload",
    "archive_query_filters",
    "archive_query_unit_payload",
    "archive_search_hit_payload",
    "archive_search_payload",
    "archive_summary_payload",
    "blackboard_note_payload",
]
