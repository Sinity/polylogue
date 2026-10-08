"""Shared cardinality guards for query-result action verbs.

Verbs that act on query results (``mark``, ``delete``) need to enforce
cardinality contracts before performing mutations:

- **singleton** (default): the operation requires exactly one matched session.
- ``--all``: explicit opt-in to act on every matched session.
- ``--first``: silently act on the first matched session only.

:func:`check_cardinality` is the shared enforcement point for verbs that
resolve their own selection. The mutating verbs (``mark``, ``delete``) resolve
their selection in the resident daemon's complete walk instead;
:func:`ambiguous_resident_selection` renders that walk's ambiguity refusal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from polylogue.cli.contextual_errors import (
    AmbiguousSelectionError,
    ContextualCliError,
    EmptySelectionError,
    NextAction,
    ambiguous_selection_actions,
)

if TYPE_CHECKING:
    from polylogue.cli.root_request import RootModeRequest


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


class WideningSelectorError(ContextualCliError):
    """A mutating verb was given a selector it could not honour exactly.

    A destructive verb acts on the complete matched set, resolved once to full
    session ids. A display-window selector (``--limit``, ``--offset``,
    ``--cursor``, ``--sample``) or a contradictory pair (``--latest`` with
    ``--all``) names a different set than that, so the verb refuses instead of
    silently acting on more sessions than the operator chose.
    """

    default_next_actions: ClassVar[tuple[NextAction, ...]] = (
        NextAction("See the complete set the verb acts on", "polylogue find <QUERY>"),
        NextAction("Narrow the query instead of windowing it", "polylogue find 'id:<REF>' then <VERB>"),
    )


def _reject_window_selectors(request: RootModeRequest) -> None:
    """Refuse display-window selectors on a mutating verb's selection.

    Verb resolution inspects the COMPLETE matched set so the cardinality
    guard, the preview and the mutation act on the same rows. A window
    selector would otherwise be ignored by that walk, and the verb would act
    on the whole match while the operator believed the blast radius was the
    window: ``--sample``/``--limit``/``limit N`` bound a page, and
    ``--offset``/``--cursor`` skip into it.
    """

    spec = request.query_spec()
    windows = [
        name
        for name, present in (
            ("--sample", spec.sample is not None),
            ("--limit", spec.limit is not None),
            ("--offset", spec.offset > 0),
            ("--cursor", spec.cursor is not None),
        )
        if present
    ]
    if windows:
        raise WideningSelectorError(
            f"Root query does not combine {', '.join(windows)} with a mutating verb "
            "(delete/mark): these operate on the complete matched set, so the window "
            "would be silently ignored. Narrow the query (e.g. an id:/since: filter) "
            "to scope the blast radius explicitly."
        )


def require_exact_mutation_selection(request: RootModeRequest, *, allow_all: bool, operation: str) -> None:
    """Refuse selector combinations whose target set is not one exact set.

    ``--latest`` selects one session; ``--all`` asks for every match. Combined
    they are contradictory, and resolving either reading silently is how
    ``--latest --all`` came to delete the whole filtered archive. The verb owns
    the ``--all`` flag, so it states the combination here before any read.
    """

    _reject_window_selectors(request)
    if allow_all and request.query_spec().latest:
        raise WideningSelectorError(
            f"'{operation}' does not combine --latest with --all: --latest selects one "
            "session and --all every match. Drop --all to act on the latest session, "
            "or drop --latest to act on every match.",
            next_actions=(
                NextAction("Act on the latest session only", f"polylogue --latest find <QUERY> then {operation}"),
                NextAction("Preview every match", "polylogue find <QUERY> then delete --dry-run --all"),
            ),
        )


def ambiguous_resident_selection(operation: str, candidates: Sequence[str]) -> AmbiguousCardinalityError:
    """Refuse a daemon ``selection_ambiguous`` outcome in the verb's own voice.

    The resident selection walk for a single-target mutation stops at its
    second row, so *candidates* is a bounded sample, never the whole match.
    The next actions are the verb's own: ``mark`` accepts ``--first`` and
    ``--all``; ``delete`` accepts only ``--all``.
    """
    verb = "mark" if operation == "mutation.session.mark" else "delete"
    refs = tuple(candidates)
    actions = (
        *ambiguous_selection_actions(verb, refs[0] if refs else None),
        *(
            (NextAction("Act on the first match only", "polylogue find <QUERY> then mark --first"),)
            if verb == "mark"
            else ()
        ),
        NextAction("Act on every match", f"polylogue find <QUERY> then {verb} --all"),
    )
    return AmbiguousCardinalityError(
        f"'{verb}' matched more than one session.",
        candidates=refs,
        next_actions=actions,
        bounded=True,
    )


__all__ = [
    "AmbiguousCardinalityError",
    "CardinalityError",
    "EmptyCardinalityError",
    "WideningSelectorError",
    "ambiguous_resident_selection",
    "check_cardinality",
    "require_exact_mutation_selection",
]
