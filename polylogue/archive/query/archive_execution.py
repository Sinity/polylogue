"""Execution of session-query plans over ``ArchiveStore``.

The fluent :class:`~polylogue.archive.filter.filters.SessionFilter` no
longer has a competing storage route. Its five terminal operations
(``list``/``list_summaries``/``first``/``count``/``delete``) translate the
canonical :class:`~polylogue.archive.query.plan.SessionQueryPlan` into the
``ArchiveStore`` filter kwarg set, fetch session summaries/envelopes from
the archive, and re-apply the plan's residual post-filters
(topology, predicates, negative terms, action sequences) that the SQL layer
does not push down.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, TypedDict

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.domain_models import Session, SessionSummary
from polylogue.core.sources import origin_from_provider
from polylogue.types import Provider, SessionId

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from polylogue.archive.message.models import Message
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.config import Config
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSearchHit,
        ArchiveSessionSummary,
        ArchiveStore,
    )
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveMessageRow, ArchiveSessionEnvelope


_ORIGIN_TO_PROVIDER = {
    "claude-code-session": Provider.CLAUDE_CODE,
    "codex-session": Provider.CODEX,
    "gemini-cli-session": Provider.GEMINI_CLI,
    "hermes-session": Provider.HERMES,
    "antigravity-session": Provider.ANTIGRAVITY,
    "chatgpt-export": Provider.CHATGPT,
    "claude-ai-export": Provider.CLAUDE_AI,
    "aistudio-drive": Provider.GEMINI,
    "unknown-export": Provider.UNKNOWN,
}


def _provider_for_origin(origin: str) -> Provider:
    return _ORIGIN_TO_PROVIDER.get(origin, Provider.UNKNOWN)


def _reject_unexecutable_session_seed(plan: SessionQueryPlan) -> None:
    """Fail typed for ``near:id:<ref>`` until session-seeded vectors are wired.

    ``similar_session_id`` carries the intent "rank sessions by vector
    similarity to a stored session's embeddings". Executing it requires a
    storage primitive that does not exist yet: ``VectorProvider`` only exposes
    ``query(text, limit)`` (which re-embeds a text string), with no way to fetch
    a session's stored ``message_embeddings`` and vector-search against them.

    Until that primitive lands, the plan compiles and round-trips (so surfaces
    and completion already know the field), but executing it raises a typed
    error here instead of silently broadening to an unfiltered listing — which
    is what every execution branch below would otherwise do, because none of
    them inspect ``similar_session_id``.
    """
    if plan.similar_session_id is None:
        return
    from polylogue.archive.query.expression import ExpressionCompileError

    raise ExpressionCompileError(
        "near:id: session-seeded similarity is not executable yet — it needs a "
        "VectorProvider primitive that vector-searches a stored session's "
        'embeddings (see issue #1842). Use near:"text" for text-seeded '
        "similarity in the meantime.",
        field="near",
    )


def _parse_archive_datetime(value: str | None) -> datetime | None:
    from datetime import datetime as _datetime

    if value is None:
        return None
    return _datetime.fromisoformat(value.replace("Z", "+00:00"))


def _datetime_to_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    return int(value.timestamp() * 1000)


class _ArchiveFilterKwargs(TypedDict):
    """The SQL-pushable filter kwarg set shared by every ``ArchiveStore`` reader."""

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


def _plan_filter_kwargs(plan: SessionQueryPlan) -> _ArchiveFilterKwargs:
    """Translate the SQL-pushable subset of a plan into ``ArchiveStore`` kwargs."""
    message_type = MessageType.normalize(plan.message_type).value if plan.message_type is not None else None
    return {
        "origins": plan.origins,
        "excluded_origins": plan.excluded_origins,
        "tags": plan.tags,
        "excluded_tags": plan.excluded_tags,
        "repo_names": plan.repo_names,
        "has_types": plan.has_types,
        "has_tool_use": plan.filter_has_tool_use,
        "has_thinking": plan.filter_has_thinking,
        "has_paste": plan.filter_has_paste,
        "tool_terms": plan.tool_terms,
        "excluded_tool_terms": plan.excluded_tool_terms,
        "action_terms": plan.action_terms,
        "excluded_action_terms": plan.excluded_action_terms,
        "action_sequence": plan.action_sequence,
        "action_text_terms": plan.action_text_terms,
        "referenced_paths": plan.referenced_path,
        "cwd_prefix": plan.cwd_prefix,
        "typed_only": plan.typed_only,
        "message_type": message_type,
        "title": plan.title,
        "min_messages": plan.min_messages,
        "max_messages": plan.max_messages,
        "min_words": plan.min_words,
        "max_words": plan.max_words,
        "since_ms": _datetime_to_ms(plan.since),
        "until_ms": _datetime_to_ms(plan.until),
        "since_session_id": plan.since_session_id,
    }


def _summary_to_domain(summary: ArchiveSessionSummary) -> SessionSummary:
    return SessionSummary(
        id=SessionId(summary.session_id),
        origin=origin_from_provider(summary.provider),
        title=summary.title,
        created_at=_parse_archive_datetime(summary.created_at),
        updated_at=_parse_archive_datetime(summary.updated_at),
        working_directories=tuple(summary.working_directories),
        git_branch=summary.git_branch,
        git_repository_url=summary.git_repository_url,
        message_count=summary.message_count,
        tags_m2m=summary.tags,
    )


def _maybe_parse_json_object(value: str | None) -> dict[str, object] | None:
    """Decode a stored JSON object column back into a mapping for domain blocks."""
    if not value:
        return None
    import json

    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _message_to_domain(message: ArchiveMessageRow, *, provider: Provider) -> Message:
    from polylogue.archive.message.models import Message

    text = "\n\n".join(block.text for block in message.blocks if block.text) or None
    content_blocks: list[dict[str, object]] = [
        {
            key: value
            for key, value in {
                "id": block.block_id,
                "type": block.block_type,
                "text": block.text,
                "tool_name": block.tool_name,
                "tool_id": block.tool_id,
                "semantic_type": block.semantic_type,
                "tool_input": _maybe_parse_json_object(block.tool_input),
                "metadata": _maybe_parse_json_object(block.metadata),
            }.items()
            if value is not None
        }
        for block in message.blocks
    ]
    return Message(
        id=message.message_id,
        role=Role.normalize(message.role),
        text=text,
        timestamp=_parse_archive_datetime(message.occurred_at),
        provider=provider,
        blocks=content_blocks,
        message_type=MessageType.normalize(message.message_type),
        has_tool_use=message.has_tool_use,
        has_thinking=message.has_thinking,
        has_paste=message.has_paste,
        duration_ms=message.duration_ms,
        branch_index=message.variant_index,
        parent_id=message.parent_message_id,
    )


def _session_to_session(session: ArchiveSessionEnvelope) -> Session:
    from polylogue.archive.session.branch_type import BranchType

    provider = _provider_for_origin(session.origin)
    messages = [_message_to_domain(message, provider=provider) for message in session.messages]
    timestamps = [message.timestamp for message in messages if message.timestamp is not None]
    return Session(
        id=SessionId(session.session_id),
        origin=origin_from_provider(provider),
        title=session.title,
        messages=MessageCollection(messages=messages),
        created_at=min(timestamps) if timestamps else None,
        updated_at=max(timestamps) if timestamps else None,
        working_directories=tuple(session.working_directories),
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
        parent_id=SessionId(session.parent_session_id) if session.parent_session_id else None,
        branch_type=BranchType(session.branch_type) if session.branch_type else None,
    )


def _plan_text_query(plan: SessionQueryPlan) -> str | None:
    terms = (*plan.query_terms, *plan.contains_terms)
    text = " ".join(term for term in terms if term).strip()
    return text or None


def _fetch_limit(plan: SessionQueryPlan, *, default: int) -> int:
    limit = plan.limit if plan.limit is not None else default
    if plan.has_post_filters():
        # Post-filters discard rows after fetch; over-fetch so the page still
        # fills.  Unbounded plans already fetch everything.
        return limit * 4 if limit < 1_000_000 else limit
    return limit


def _semantic_hits(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
) -> list[ArchiveSessionSearchHit]:
    """Resolve the vector leg of a semantic/hybrid plan.

    Graceful-degradation contract (#1743): when no vector provider can be
    constructed for the active archive — sqlite-vec/Voyage not configured, the
    archive holds no embeddings, or the configured embeddings are unusable — the
    semantic leg yields no hits instead of raising. The caller (``_archive_summaries``)
    therefore returns an empty result set for a pure-semantic request and falls
    back to the lexical leg for hybrid, never surfacing a hard error to read
    surfaces over an embeddings-less archive.
    """
    from polylogue.storage.search_providers import create_vector_provider

    text = plan.similar_text or _plan_text_query(plan) or ""
    if not text:
        return []
    vector_provider = plan.vector_provider
    if vector_provider is None and config is not None:
        vector_provider = create_vector_provider(config, db_path=archive_root / "embeddings.db")
    if vector_provider is None:
        # No vector backend → empty semantic leg (graceful degradation, #1743).
        return []
    limit = plan.limit if plan.limit is not None else 50
    scored = vector_provider.query(text, limit=max(limit + plan.offset, limit) * 3)
    return archive.semantic_summaries(
        scored,
        limit=max(limit + plan.offset, limit) * 3,
        offset=0,
        **_plan_filter_kwargs(plan),
    )


def _archive_summaries(
    plan: SessionQueryPlan,
    archive: ArchiveStore,
    *,
    config: Config | None,
    archive_root: Path,
    default_limit: int,
) -> list[ArchiveSessionSummary]:
    _reject_unexecutable_session_seed(plan)
    filter_kwargs = _plan_filter_kwargs(plan)
    limit = _fetch_limit(plan, default=default_limit)
    sort = plan.sort
    reverse = plan.reverse

    if plan.similar_text is not None or plan.retrieval_lane in {"semantic", "hybrid"}:
        hits = _semantic_hits(plan, archive, config=config, archive_root=archive_root)
        return _summaries_from_hits(archive, hits)

    query_text = _plan_text_query(plan)
    if query_text is not None:
        hits = archive.search_summaries(
            query_text,
            limit=limit,
            offset=plan.offset,
            sort=sort,
            reverse=reverse,
            **filter_kwargs,
        )
        return _summaries_from_hits(archive, hits)

    return archive.list_summaries(
        limit=limit,
        offset=plan.offset,
        sort=sort,
        reverse=reverse,
        sample=plan.sample is not None,
        **filter_kwargs,
    )


def _summaries_from_hits(archive: ArchiveStore, hits: list[ArchiveSessionSearchHit]) -> list[ArchiveSessionSummary]:
    summaries: list[ArchiveSessionSummary] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.session_id in seen:
            continue
        seen.add(hit.session_id)
        try:
            summaries.append(archive.read_summary(hit.session_id))
        except KeyError:
            continue
    return summaries


def _open_archive(archive_root: Path) -> ArchiveStore:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    return ArchiveStore.open_existing(archive_root)


async def list_summaries_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
    default_limit: int = 50,
) -> builtins.list[SessionSummary]:
    rank_first = bool(plan.fts_terms and plan.sort is None)
    with _open_archive(archive_root) as archive:
        archive_rows = _archive_summaries(
            plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
        )
    summaries = [_summary_to_domain(summary) for summary in archive_rows]
    filtered = plan._apply_common_filters(summaries, sql_pushed=True)
    ordered = filtered if rank_first else plan._sort_summaries(filtered)
    return plan._finalize(ordered)


async def list_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
    default_limit: int = 50,
) -> builtins.list[Session]:
    rank_first = bool(plan.fts_terms and plan.sort is None)
    with _open_archive(archive_root) as archive:
        archive_rows = _archive_summaries(
            plan,
            archive,
            config=config,
            archive_root=archive_root,
            default_limit=default_limit,
        )
        sessions = [_session_to_session(archive.read_session(summary.session_id)) for summary in archive_rows]
    filtered = plan._apply_full_filters(sessions, sql_pushed=True)
    ordered = filtered if rank_first else plan._sort_sessions(filtered)
    return plan._finalize(ordered)


async def first_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
) -> Session | None:
    results = await list_archive(plan.with_limit(1), archive_root=archive_root, config=config)
    return results[0] if results else None


async def count_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
) -> int:
    _reject_unexecutable_session_seed(plan)
    if not plan.has_post_filters() and plan.similar_text is None and plan.retrieval_lane not in {"semantic", "hybrid"}:
        filter_kwargs = _plan_filter_kwargs(plan)
        query_text = _plan_text_query(plan)
        with _open_archive(archive_root) as archive:
            if query_text is not None:
                return int(archive.count_search_sessions(query_text, **filter_kwargs))
            return int(archive.count_sessions(**filter_kwargs))

    unbounded = plan.with_limit(None)
    if unbounded.can_use_summaries():
        rows = await list_summaries_archive(
            unbounded,
            archive_root=archive_root,
            config=config,
            default_limit=1_000_000,
        )
        return len(rows)
    sessions = await list_archive(
        unbounded,
        archive_root=archive_root,
        config=config,
        default_limit=1_000_000,
    )
    return len(sessions)


def archive_search_hits(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
    default_limit: int = 50,
) -> tuple[list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]], str]:
    """Resolve a search plan to archive session hits paired with summaries.

    Returns ``(hits, resolved_lane)`` where each hit carries its
    :class:`ArchiveSessionSearchHit` plus the session summary, and ``resolved_lane``
    is the concrete lane that ran (``dialogue``/``semantic``/``hybrid``).
    """
    from polylogue.storage.search_providers import create_vector_provider, reciprocal_rank_fusion

    _reject_unexecutable_session_seed(plan)
    text = plan.similar_text or _plan_text_query(plan) or ""
    limit = plan.limit if plan.limit is not None else default_limit
    offset = plan.offset
    filter_kwargs = _plan_filter_kwargs(plan)
    with _open_archive(archive_root) as archive:
        if plan.similar_text is None and plan.retrieval_lane in {"auto", "dialogue"}:
            hits = archive.search_summaries(
                text,
                limit=limit,
                offset=offset,
                sort=plan.sort,
                reverse=plan.reverse,
                **filter_kwargs,
            )
            return _pair_hits(archive, hits), "dialogue"

        vector_provider = plan.vector_provider
        if vector_provider is None and config is not None:
            vector_provider = create_vector_provider(config, db_path=archive_root / "embeddings.db")
        if vector_provider is None:
            # Graceful-degradation contract (#1743): a semantic or hybrid request
            # against an archive with no usable vector backend yields an empty
            # result set rather than raising. The resolved lane is preserved so
            # the caller still reports which lane was requested; a pure-semantic
            # request returns no hits and a hybrid request degrades to no fused
            # rows (the lexical leg below is skipped only because there is no
            # semantic leg to fuse it with).
            return [], "semantic" if plan.retrieval_lane != "hybrid" else "hybrid"

        semantic_query = plan.similar_text or text
        pool = max(limit + offset, limit) * 3
        scored = vector_provider.query(semantic_query, limit=pool)
        semantic_hits = archive.semantic_summaries(scored, limit=pool, offset=0, **filter_kwargs)
        if plan.retrieval_lane != "hybrid":
            return _pair_hits(archive, semantic_hits[offset : offset + limit]), "semantic"

        lexical_hits = archive.search_summaries(
            text,
            limit=pool,
            offset=0,
            sort=plan.sort,
            reverse=plan.reverse,
            **filter_kwargs,
        )
        from dataclasses import replace as _replace

        hit_by_session: dict[str, ArchiveSessionSearchHit] = {}
        for hit in [*lexical_hits, *semantic_hits]:
            hit_by_session.setdefault(hit.session_id, hit)
        fused = reciprocal_rank_fusion(
            [(hit.session_id, 0.0) for hit in lexical_hits],
            [(hit.session_id, 0.0) for hit in semantic_hits],
        )
        page = fused[offset : offset + limit]
        ranked = [
            _replace(hit_by_session[session_id], rank=offset + index)
            for index, (session_id, _score) in enumerate(page, start=1)
            if session_id in hit_by_session
        ]
        return _pair_hits(archive, ranked), "hybrid"


def _pair_hits(
    archive: ArchiveStore,
    hits: list[ArchiveSessionSearchHit],
) -> list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]]:
    paired: list[tuple[ArchiveSessionSearchHit, ArchiveSessionSummary]] = []
    for hit in hits:
        try:
            paired.append((hit, archive.read_summary(hit.session_id)))
        except KeyError:
            continue
    return paired


async def delete_archive(
    plan: SessionQueryPlan,
    *,
    archive_root: Path,
    config: Config | None,
) -> int:
    if plan.can_use_summaries():
        summaries = await list_summaries_archive(
            plan,
            archive_root=archive_root,
            config=config,
        )
        session_ids = tuple(str(summary.id) for summary in summaries)
    else:
        sessions = await list_archive(plan, archive_root=archive_root, config=config)
        session_ids = tuple(str(session.id) for session in sessions)
    if not session_ids:
        return 0
    with _open_archive(archive_root) as archive:
        return archive.delete_sessions(session_ids)


__all__ = [
    "count_archive",
    "delete_archive",
    "first_archive",
    "list_archive",
    "list_summaries_archive",
]
