"""CLI refusals that carry the next action with them.

A refusal is useful only when the operator can act on it, so the next action
travels with the error class rather than being restated at each raise site.
Click renders :meth:`format_message`, which appends the declared actions, so a
route that raises one of these cannot deliver a bare sentence to the terminal.

Every subclass declares :attr:`default_next_actions`; an instance built without
explicit actions inherits them, which is why an error of this family always
names at least one thing to do.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import click

#: How many refs an ambiguity refusal lists, and the probe depth that feeds it.
#: Bounded because the refusal is read by a human or parsed by one consumer,
#: not because the selection is bounded.
AMBIGUITY_CANDIDATE_LIMIT = 10


@dataclass(frozen=True, slots=True)
class NextAction:
    """One concrete step, optionally with the command that performs it."""

    summary: str
    command: str | None = None

    def render(self) -> str:
        return f"  - {self.summary}: {self.command}" if self.command else f"  - {self.summary}"


class ContextualCliError(click.UsageError):
    """A typed CLI refusal that renders its own next actions."""

    default_next_actions: ClassVar[tuple[NextAction, ...]] = (
        NextAction("See what the current selection matches", "polylogue find <QUERY>"),
    )

    def __init__(
        self,
        message: str,
        *,
        next_actions: Sequence[NextAction] = (),
        ctx: click.Context | None = None,
    ) -> None:
        super().__init__(message, ctx=ctx)
        self.next_actions: tuple[NextAction, ...] = tuple(next_actions) or self.default_next_actions

    def format_message(self) -> str:
        return "\n".join((self.message, "Next:", *(action.render() for action in self.next_actions)))


class EmptySelectionError(ContextualCliError):
    """The query matched nothing, so the verb has nothing to act on."""

    default_next_actions: ClassVar[tuple[NextAction, ...]] = (
        NextAction("Show why the selection is empty", "polylogue find <QUERY> --why"),
        NextAction("Drop the narrowest filter and re-run", "polylogue find <QUERY>"),
    )


class AmbiguousSelectionError(ContextualCliError):
    """The query matched several sessions where the verb needs one.

    The candidate refs are part of the message so a non-interactive consumer
    can resolve the ambiguity from the error alone instead of re-querying.
    """

    default_next_actions: ClassVar[tuple[NextAction, ...]] = (
        NextAction("Select one candidate by ref", "polylogue find id:<REF> then <VERB>"),
    )

    def __init__(
        self,
        message: str,
        *,
        candidates: Sequence[str] = (),
        next_actions: Sequence[NextAction] = (),
        bounded: bool = False,
        ctx: click.Context | None = None,
    ) -> None:
        self.candidates: tuple[str, ...] = tuple(candidates)
        self.bounded = bounded
        super().__init__(message, next_actions=next_actions, ctx=ctx)

    def format_message(self) -> str:
        lines = [self.message]
        if self.candidates:
            lines.append("Candidates:" if not self.bounded else f"First {len(self.candidates)} candidates:")
            lines.extend(f"  {ref}" for ref in self.candidates)
        lines.append("Next:")
        lines.extend(action.render() for action in self.next_actions)
        return "\n".join(lines)


def ambiguous_selection_actions(operation: str, first_candidate: str | None) -> tuple[NextAction, ...]:
    """Return the next actions for an ambiguous *operation*, ref-specific."""
    ref = first_candidate or "<REF>"
    return (
        NextAction(f"Run {operation} against one session", f"polylogue find id:{ref} then {operation}"),
        NextAction("Pick one interactively", "polylogue select"),
    )


__all__ = [
    "AMBIGUITY_CANDIDATE_LIMIT",
    "AmbiguousSelectionError",
    "ContextualCliError",
    "EmptySelectionError",
    "NextAction",
    "ambiguous_selection_actions",
]
