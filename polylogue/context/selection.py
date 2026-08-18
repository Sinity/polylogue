"""Context-image session selection helpers.

The context-image compiler owns payload assembly. This module owns the small
query-selection lens that chooses seed sessions for multi-session handoff
images, including recall-oriented fallback for archaeology queries.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.archive.query.spec import SessionQuerySpec


def clamp_context_image_limit(value: int | object) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(1, min(value, 20))
    if isinstance(value, str | bytes | bytearray):
        return max(1, min(int(value), 20))
    return 1


@dataclass(frozen=True, slots=True)
class ContextImageSelection:
    sessions: list[Any]
    match_strategy: str
    relaxed_filters: tuple[str, ...] = ()
    query_total: int = 0


@dataclass(frozen=True, slots=True)
class _ContextImageQueryAttempt:
    query: str | None
    project_path: str | None
    project_repo: str | None
    strategy: str
    relaxed_filters: tuple[str, ...] = ()


def _context_image_recall_terms(query: str | None) -> tuple[str, ...]:
    if not query:
        return ()
    from polylogue.storage.search.query_support import extract_match_terms

    terms = extract_match_terms(query)
    # Single-letter FTS terms produce noisy archaeology images and pure boolean
    # operators are already stripped by extract_match_terms().
    return tuple(term for term in terms if len(term) > 1)


def _context_image_query_attempts(
    *,
    query: str | None,
    project_path: str | None,
    project_repo: str | None,
) -> tuple[_ContextImageQueryAttempt, ...]:
    attempts = [
        _ContextImageQueryAttempt(
            query=query,
            project_path=project_path,
            project_repo=project_repo,
            strategy="strict",
        )
    ]
    terms = _context_image_recall_terms(query)
    if len(terms) <= 1:
        return tuple(attempts)

    attempts.extend(
        _ContextImageQueryAttempt(
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
            _ContextImageQueryAttempt(
                query=term,
                project_path=None,
                project_repo=None,
                strategy="relaxed_project_term_recall",
                relaxed_filters=relaxed,
            )
            for term in terms
        )
    return tuple(attempts)


async def select_context_image_sessions(
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
) -> ContextImageSelection:
    """Select sessions for a context image with recall-oriented fallback."""
    from polylogue.mcp.query_contracts import MCPSessionQueryRequest

    def _spec(attempt: _ContextImageQueryAttempt) -> SessionQuerySpec:
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

    attempts = _context_image_query_attempts(
        query=query,
        project_path=project_path,
        project_repo=project_repo,
    )
    strict = list(await query_sessions(_spec(attempts[0])))
    if strict:
        return ContextImageSelection(sessions=strict[:limit], match_strategy="strict", query_total=len(strict))

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
            return ContextImageSelection(
                sessions=merged,
                match_strategy=strategy,
                relaxed_filters=relaxed_filters,
                query_total=len(merged),
            )

    return ContextImageSelection(sessions=[], match_strategy="strict", query_total=0)
