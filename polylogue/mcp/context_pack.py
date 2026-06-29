"""Context-pack session selection helpers.

The context-pack *payload* is no longer assembled here: compilation collapsed
onto the shared ``compile_context`` / ``ContextImage`` engine (see
``polylogue.context.compiler`` and ``Polylogue.context_pack_payload``). What
remains is the recall-oriented query-algebra **selection** used to pick which
sessions a context pack should cover.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypedDict

from polylogue.mcp.archive_support import archive_index_active_paths, archive_query_filters
from polylogue.storage.sqlite.archive_tiers.archive import (
    ArchiveSessionSearchHit,
    ArchiveSessionSummary,
    ArchiveStore,
)

if TYPE_CHECKING:
    from polylogue.archive.query.spec import SessionQuerySpec


def _clamp_context_pack_limit(value: int | object) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(1, min(value, 20))
    if isinstance(value, str | bytes | bytearray):
        return max(1, min(int(value), 20))
    return 1


@dataclass(frozen=True, slots=True)
class ContextPackSelection:
    sessions: list[Any]
    match_strategy: str
    relaxed_filters: tuple[str, ...] = ()
    query_total: int = 0


class ArchiveContextPackFilters(TypedDict):
    origins: tuple[str, ...]
    excluded_origins: tuple[str, ...]
    tags: tuple[str, ...]
    excluded_tags: tuple[str, ...]
    repo_names: tuple[str, ...]
    cwd_prefix: str | None
    since_ms: int | None
    until_ms: int | None


@dataclass(frozen=True, slots=True)
class _ContextPackQueryAttempt:
    query: str | None
    project_path: str | None
    project_repo: str | None
    strategy: str
    relaxed_filters: tuple[str, ...] = ()


def _context_pack_recall_terms(query: str | None) -> tuple[str, ...]:
    if not query:
        return ()
    from polylogue.storage.search.query_support import extract_match_terms

    terms = extract_match_terms(query)
    # Single-letter FTS terms produce noisy archaeology packs and pure boolean
    # operators are already stripped by extract_match_terms().
    return tuple(term for term in terms if len(term) > 1)


def _context_pack_query_attempts(
    *,
    query: str | None,
    project_path: str | None,
    project_repo: str | None,
) -> tuple[_ContextPackQueryAttempt, ...]:
    attempts = [
        _ContextPackQueryAttempt(
            query=query,
            project_path=project_path,
            project_repo=project_repo,
            strategy="strict",
        )
    ]
    terms = _context_pack_recall_terms(query)
    if len(terms) <= 1:
        return tuple(attempts)

    attempts.extend(
        _ContextPackQueryAttempt(
            query=term,
            project_path=project_path,
            project_repo=project_repo,
            strategy="term_recall",
        )
        for term in terms
    )
    if project_path or project_repo:
        relaxed = tuple(
            name for name, value in (("project_path", project_path), ("project_repo", project_repo)) if value
        )
        attempts.extend(
            _ContextPackQueryAttempt(
                query=term,
                project_path=None,
                project_repo=None,
                strategy="relaxed_project_term_recall",
                relaxed_filters=relaxed,
            )
            for term in terms
        )
    return tuple(attempts)


async def select_context_pack_sessions(
    query_sessions: Callable[[SessionQuerySpec], Awaitable[Sequence[Any]]],
    clamp_limit: Callable[[int | object], int],
    *,
    project_path: str | None,
    project_repo: str | None,
    since: str | None,
    until: str | None,
    origin: str | None,
    query: str | None,
    limit: int,
) -> ContextPackSelection:
    """Select sessions for a context pack with recall-oriented fallback.

    The context-pack surface is an archaeology/reorientation tool. A pasted
    investigative query often contains many alternative identifiers; treating
    it as one strict FTS conjunction produces false "no history" answers. We
    still run the strict request first, then fall back to single-term recall
    only when strict selection returns no sessions.
    """
    from polylogue.mcp.query_contracts import MCPSessionQueryRequest

    def _spec(attempt: _ContextPackQueryAttempt) -> SessionQuerySpec:
        return MCPSessionQueryRequest(
            query=attempt.query,
            origin=origin,
            since=since,
            until=until,
            cwd_prefix=attempt.project_path,
            repo=attempt.project_repo,
            sort="date",
            reverse=True,
            limit=limit,
        ).build_spec(clamp_limit)

    attempts = _context_pack_query_attempts(
        query=query,
        project_path=project_path,
        project_repo=project_repo,
    )
    strict = list(await query_sessions(_spec(attempts[0])))
    if strict:
        return ContextPackSelection(sessions=strict[:limit], match_strategy="strict", query_total=len(strict))

    for strategy in ("term_recall", "relaxed_project_term_recall"):
        merged: list[Any] = []
        seen: set[str] = set()
        relaxed_filters: tuple[str, ...] = ()
        for attempt in attempts:
            if attempt.strategy != strategy:
                continue
            for session in await query_sessions(_spec(attempt)):
                conv_id = str(getattr(session, "id", ""))
                if conv_id and conv_id in seen:
                    continue
                if conv_id:
                    seen.add(conv_id)
                merged.append(session)
                if len(merged) >= limit:
                    break
            relaxed_filters = attempt.relaxed_filters
            if len(merged) >= limit:
                break
        if merged:
            return ContextPackSelection(
                sessions=merged,
                match_strategy=strategy,
                relaxed_filters=relaxed_filters,
                query_total=len(merged),
            )

    return ContextPackSelection(sessions=[], match_strategy="strict", query_total=0)


def archive_context_pack_active(
    *,
    archive_root: Path,
    db_anchor_path: Path,
) -> bool:
    """Return whether context-pack should read archive index data."""
    return archive_index_active_paths(
        archive_root=archive_root,
        db_anchor_path=db_anchor_path,
    )


def query_archive_context_pack(
    archive: ArchiveStore,
    spec: SessionQuerySpec,
    *,
    default_limit: int,
) -> list[SimpleNamespace]:
    """Project archive sessions into the context-pack summary surface."""
    query = " ".join(spec.query_terms).strip()
    kwargs = archive_context_pack_filters(spec)
    if query:
        rows: list[ArchiveSessionSummary | ArchiveSessionSearchHit] = list(
            archive.search_summaries(
                query,
                limit=spec.limit or default_limit,
                offset=spec.offset,
                sort="date",
                reverse=spec.reverse,
                **kwargs,
            )
        )
    else:
        rows = list(
            archive.list_summaries(
                limit=spec.limit or default_limit,
                offset=spec.offset,
                sort="date",
                reverse=spec.reverse,
                **kwargs,
            )
        )

    summaries: list[ArchiveSessionSummary] = []
    for row in dedupe_archive_context_pack_rows(rows):
        if isinstance(row, ArchiveSessionSearchHit):
            try:
                summaries.append(archive.read_summary(row.session_id))
            except KeyError:
                continue
        else:
            summaries.append(row)
    return [archive_context_pack_summary(row) for row in summaries]


def archive_context_pack_filters(spec: SessionQuerySpec) -> ArchiveContextPackFilters:
    filters = archive_query_filters(spec)
    return {
        "origins": filters["origins"],
        "excluded_origins": filters["excluded_origins"],
        "tags": filters["tags"],
        "excluded_tags": filters["excluded_tags"],
        "repo_names": filters["repo_names"],
        "cwd_prefix": filters["cwd_prefix"],
        "since_ms": filters["since_ms"],
        "until_ms": filters["until_ms"],
    }


def archive_context_pack_summary(row: ArchiveSessionSummary) -> SimpleNamespace:
    return SimpleNamespace(
        id=row.session_id,
        origin=row.origin,
        title=row.title,
        display_title=row.title,
        created_at=_parse_archive_datetime(row.created_at),
        updated_at=_parse_archive_datetime(row.updated_at),
        message_count=row.message_count,
        messages=(),
        tool_use_count=0,
    )


def dedupe_archive_context_pack_rows(
    rows: list[ArchiveSessionSummary | ArchiveSessionSearchHit],
) -> list[ArchiveSessionSummary | ArchiveSessionSearchHit]:
    deduped: list[ArchiveSessionSummary | ArchiveSessionSearchHit] = []
    seen: set[str] = set()
    for row in rows:
        session_id = row.session_id
        if session_id in seen:
            continue
        seen.add(session_id)
        deduped.append(row)
    return deduped


def _parse_archive_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
