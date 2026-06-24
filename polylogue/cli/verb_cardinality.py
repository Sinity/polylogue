"""Shared cardinality guards for query-result action verbs.

Verbs that act on query results (``mark``, ``delete``) need to enforce
cardinality contracts before performing mutations:

- **singleton** (default): the operation requires exactly one matched session.
- ``--all``: explicit opt-in to act on every matched session.
- ``--first``: silently act on the first matched session only.

:func:`check_cardinality` is the single shared enforcement point.  All three
verbs import it so tests can verify the shared path without repeating
assertions.

:func:`probe_session_ids_for_verb` and :func:`resolve_session_ids_for_verb` use
the same filter-chain infrastructure as the ``read`` and ``export`` paths so
guards and resolved sets are always consistent with what the user would see
from ``find QUERY``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from polylogue.archive.filter.filters import SessionFilter
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.shared.types import AppEnv


class CardinalityError(click.UsageError):
    """Raised when a verb's cardinality constraint is violated."""


def check_cardinality(
    count: int,
    *,
    allow_all: bool,
    first_only: bool,
    operation: str = "operate on sessions",
    multi_match_hint: str | None = None,
) -> None:
    """Enforce the singleton / ``--all`` / ``--first`` cardinality contract.

    Rules:

    - ``count == 0``: always raises — nothing to act on.
    - ``count == 1``: always passes — unambiguous singleton.
    - ``count > 1`` and ``allow_all``: passes — caller acts on all results.
    - ``count > 1`` and ``first_only``: passes — caller acts on ``results[0]``.
    - ``count > 1`` otherwise: raises :class:`CardinalityError`.

    Args:
        count: number of matched sessions.
        allow_all: ``True`` when ``--all`` was supplied by the user.
        first_only: ``True`` when ``--first`` was supplied by the user.
        operation: human-readable verb label used in the error message.
        multi_match_hint: optional guidance for verbs that do not expose both
            ``--first`` and ``--all``.

    Raises:
        CardinalityError: when the cardinality constraint is not satisfied.
    """
    if count == 0:
        raise CardinalityError(f"No sessions matched; cannot {operation}.")
    if count == 1:
        return
    if allow_all or first_only:
        return
    hint = multi_match_hint or "Use --first to act on the first match only, or --all to act on all."
    raise CardinalityError(f"'{operation}' matched {count} sessions. {hint}")


def _reject_sample_for_mutating_verb(request: RootModeRequest) -> None:
    # ``--sample`` is a display-window operation (random subset applied during
    # result windowing); verb guard/resolution paths deliberately inspect the
    # COMPLETE matched set so cardinality checks and mutations act on the same
    # rows. Honoring ``--sample`` here would mean a destructive verb silently
    # operated on the full match while the operator believed the blast radius was
    # capped at N, so reject the combination instead of ignoring it.
    if request.query_spec().sample is not None:
        raise click.UsageError(
            "Root query does not combine --sample with a mutating verb "
            "(delete/mark): these operate on the complete matched set, so "
            "--sample would be silently ignored. Narrow the query (e.g. an "
            "id:/since: filter) to scope the blast radius explicitly."
        )


def probe_session_ids_for_verb(env: AppEnv, request: RootModeRequest, *, limit: int) -> list[str]:
    """Resolve a bounded ID prefix for cheap zero/one/many verb guards."""
    from polylogue.api.sync.bridge import run_coroutine_sync

    _reject_sample_for_mutating_verb(request)
    return run_coroutine_sync(_async_probe_ids(env, request, limit=limit))


def resolve_session_ids_for_verb(env: AppEnv, request: RootModeRequest) -> list[str]:
    """Resolve session IDs for a verb that needs to inspect the matched set.

    This is the shared resolution path used by ``mark``, ``delete`` (for the
    cardinality pre-check), and any future verb that must know the result set
    before mutating it.  Uses the same filter-chain as the ``read`` / export
    paths so results are always consistent with ``find QUERY``.

    Returns IDs in the query's natural order (most-recent first by default).
    """
    from polylogue.api.sync.bridge import run_coroutine_sync

    _reject_sample_for_mutating_verb(request)
    return run_coroutine_sync(_async_resolve_ids(env, request))


def _filter_chain_for_request(env: AppEnv, request: RootModeRequest) -> SessionFilter:
    from polylogue.cli.query import _create_query_vector_provider
    from polylogue.paths._roots import archive_file_set_root_for_paths

    config = env.config
    spec = request.query_spec()
    archive_root = archive_file_set_root_for_paths(
        archive_root_path=config.archive_root,
        db_anchor=config.db_path,
    )
    vector_provider = _create_query_vector_provider(config, db_path=archive_root / "embeddings.db")
    return spec.build_filter(config, vector_provider=vector_provider)


async def _async_probe_ids(env: AppEnv, request: RootModeRequest, *, limit: int) -> list[str]:
    bounded = request.with_param_updates(limit=limit)
    filter_chain = _filter_chain_for_request(env, bounded)
    if filter_chain.can_use_summaries():
        summaries = await filter_chain.list_summaries()
        return [str(s.id) for s in summaries[:limit]]
    sessions = await filter_chain.list()
    return [str(s.id) for s in sessions[:limit]]


async def _async_resolve_ids(env: AppEnv, request: RootModeRequest) -> list[str]:
    """Async implementation of session-ID resolution for verb cardinality.

    Uses the compiled DSL spec (``request.query_spec()``) so that field clauses
    such as ``repo:polylogue`` or ``since:7d`` are resolved to structured filters
    rather than being passed as literal FTS text.
    """
    filter_chain = _filter_chain_for_request(env, request)
    # Resolve the COMPLETE matched set, not a single page: this list drives the
    # cardinality guard and the actual delete/mark, so a paged list_summaries()
    # (default limit 50) would let ``delete --yes --all`` silently skip every
    # match beyond the first page (#1873).
    if filter_chain.can_use_summaries():
        summaries = await filter_chain.list_all_summaries()
        return [str(s.id) for s in summaries]
    sessions = await filter_chain.list_all()
    return [str(s.id) for s in sessions]


__all__ = [
    "CardinalityError",
    "check_cardinality",
    "probe_session_ids_for_verb",
    "resolve_session_ids_for_verb",
]
