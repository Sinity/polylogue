"""Shared cardinality guards for query-result action verbs.

Verbs that act on query results (``mark``, ``delete``) need to enforce
cardinality contracts before performing mutations:

- **singleton** (default): the operation requires exactly one matched session.
- ``--all``: explicit opt-in to act on every matched session.
- ``--first``: silently act on the first matched session only.

:func:`check_cardinality` is the single shared enforcement point.  All three
verbs import it so tests can verify the shared path without repeating
assertions.

:func:`probe_session_ids_for_verb` and :func:`resolve_session_ids_for_verb`
answer through the declared ``cli.query`` operation -- the same read
``find QUERY`` runs -- so a guard and the set it guards can never disagree with
what the operator was shown.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import click

from polylogue.cli.contextual_errors import (
    AmbiguousSelectionError,
    ContextualCliError,
    EmptySelectionError,
    NextAction,
    ambiguous_selection_actions,
)

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.shared.types import AppEnv


class CardinalityError(ContextualCliError):
    """Raised when a verb's cardinality constraint is violated."""


class EmptyCardinalityError(EmptySelectionError, CardinalityError):
    """Nothing matched, so the verb has nothing to act on."""


class AmbiguousCardinalityError(AmbiguousSelectionError, CardinalityError):
    """Several sessions matched where the verb needs exactly one."""


def check_cardinality(
    count: int,
    *,
    allow_all: bool,
    first_only: bool,
    operation: str = "operate on sessions",
    multi_match_hint: str | None = None,
    candidates: Sequence[str] = (),
    bounded: bool = False,
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
        raise EmptyCardinalityError(f"No sessions matched; cannot {operation}.")
    if count == 1:
        return
    if allow_all or first_only:
        return
    hint = multi_match_hint or "Use --first to act on the first match only, or --all to act on all."
    refs = tuple(str(candidate) for candidate in candidates)
    actions = ambiguous_selection_actions(operation, refs[0] if refs else None)
    if multi_match_hint is None:
        actions = (
            *actions,
            NextAction("Act on the first match only", f"polylogue find <QUERY> then {operation} --first"),
            NextAction("Act on every match", f"polylogue find <QUERY> then {operation} --all"),
        )
    raise AmbiguousCardinalityError(
        f"'{operation}' matched {count} sessions. {hint}",
        candidates=refs,
        next_actions=actions,
        bounded=bounded,
    )


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
    from polylogue.cli.session_rows import query_session_ids

    _reject_sample_for_mutating_verb(request)
    return query_session_ids(env.config, request, limit=limit)[:limit]


def resolve_session_ids_for_verb(env: AppEnv, request: RootModeRequest) -> list[str]:
    """Resolve session IDs for a verb that needs to inspect the matched set.

    The shared resolution path used by ``mark`` and ``delete`` for their
    cardinality pre-check. It is the declared ``cli.query`` operation, the same
    one ``find QUERY`` runs, so a guard can never disagree with what the
    operator was shown.

    Returns IDs in the query's natural order (most-recent first by default).
    """
    from polylogue.cli.session_rows import query_complete_session_ids

    _reject_sample_for_mutating_verb(request)
    return query_complete_session_ids(env.config, request)


__all__ = [
    "AmbiguousCardinalityError",
    "CardinalityError",
    "EmptyCardinalityError",
    "check_cardinality",
    "probe_session_ids_for_verb",
    "resolve_session_ids_for_verb",
]
