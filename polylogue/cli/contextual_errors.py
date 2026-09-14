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

import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import click

#: How many refs an ambiguity refusal lists, and the probe depth that feeds it.
#: Bounded because the refusal is read by a human or parsed by one consumer,
#: not because the selection is bounded.
AMBIGUITY_CANDIDATE_LIMIT = 10


def display_ref(ref: str) -> str:
    """Render an archive-derived ref with control characters made visible.

    A session ref carries the provider's own native id verbatim
    (``session_id = origin || ':' || native_id``), and no import path
    restricts what bytes a provider may put there. An id carrying ESC or CR
    can repaint or rewrite the surrounding terminal line, so a refusal
    listing candidates would be spoofable by the very export it is refusing
    to disambiguate. Escaping is visible rather than silent: the operator
    sees ``\\x1b`` and knows the ref is not the plain text it resembles.
    """
    return "".join(
        character if character.isprintable() or character == " " else f"\\x{ord(character):02x}" for character in ref
    )


def ref_command_argument(ref: str) -> str:
    """Quote *ref* as one shell word for a command the operator may paste.

    The next action of a refusal is deliberately copy-pasteable, which makes
    any provider-controlled text inside it a shell-injection vector: an
    imported conversation id of ``innocent; touch /tmp/pwned #`` otherwise
    renders as ``polylogue find id:innocent; touch /tmp/pwned # then delete``
    and runs the injected command as the archive owner on paste.

    Control characters are escaped before quoting, so a ref that cannot be
    displayed honestly also cannot be pasted as if it were intact -- the
    command shown is then explicitly not the literal ref, which is the
    correct signal, not a silently mangled one.
    """
    return shlex.quote(f"id:{display_ref(ref)}")


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
        NextAction("Select one candidate by ref", "polylogue find id:'<REF>' then <VERB>"),
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
            lines.extend(f"  {display_ref(ref)}" for ref in self.candidates)
        lines.append("Next:")
        lines.extend(action.render() for action in self.next_actions)
        return "\n".join(lines)


def ambiguous_selection_actions(operation: str, first_candidate: str | None) -> tuple[NextAction, ...]:
    """Return the next actions for an ambiguous *operation*, ref-specific."""
    argument = ref_command_argument(first_candidate) if first_candidate else "id:'<REF>'"
    return (
        NextAction(f"Run {operation} against one session", f"polylogue find {argument} then {operation}"),
        NextAction("Pick one interactively", "polylogue select"),
    )


__all__ = [
    "AMBIGUITY_CANDIDATE_LIMIT",
    "AmbiguousSelectionError",
    "ContextualCliError",
    "EmptySelectionError",
    "NextAction",
    "ambiguous_selection_actions",
    "display_ref",
    "ref_command_argument",
]
