"""Parser-decision diagnostics for the query-first CLI.

When ``--diagnose`` is set on the root group, this module emits a short
human-readable explanation of how the parser routed the current invocation:

- which token was treated as a subcommand (if any),
- which tokens were captured as query terms (and fell through to query-first
  dispatch),
- which verb (if any) was applied to the matched sessions.

The output goes to stderr so it never contaminates stdout pipelines (JSON
output, exports, etc.).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence

import click

from polylogue.cli.verb_names import QUERY_VERB_NAMES, VERB_NAMES


def _format_tokens(tokens: Iterable[str]) -> str:
    """Render a token list as a shell-friendly inline quotation."""
    rendered = [f'"{tok}"' if " " in tok else tok for tok in tokens]
    return " ".join(rendered) if rendered else "<none>"


def emit_parser_decision(
    argv: Sequence[str],
    *,
    has_subcommand: bool,
    subcommand: str | None,
    query_terms: Sequence[str],
    verb: str | None,
    registered_commands: Sequence[str],
) -> None:
    """Print a parser-decision explanation to stderr.

    The output stays short and one-screen by design so it does not bury
    the actual command output. It always begins with ``[diagnose]`` so the
    line is easy to grep out.
    """

    def line(text: str) -> None:
        click.echo(f"[diagnose] {text}", err=True)

    line(f"argv: {_format_tokens(argv)}")
    if subcommand is not None:
        line(f"matched subcommand: {subcommand!r}")
        line("dispatching to subcommand; query-first fallback not used.")
        return

    if query_terms:
        line("no registered subcommand matched leading positional token(s).")
        line(f"interpreting as search query: {_format_tokens(query_terms)}")
        if verb is not None:
            line(f"verb applied to matched sessions: {verb!r}")
        else:
            line("no verb supplied; default render selected for the matched set.")
        line(
            "this is the query-first default. "
            "pass `--help` for the full subcommand list, or quote your query like "
            '`polylogue "your search"` to make the intent explicit.'
        )
        return

    if has_subcommand and verb is not None:
        line(f"verb mode: {verb!r}")
        return

    line("no positional tokens; showing status / stats.")


def looks_like_subcommand_typo(token: str, registered: Sequence[str]) -> list[str]:
    """Return up to three close-match suggestions for ``token`` from ``registered``."""
    import difflib

    return difflib.get_close_matches(token, list(registered), n=3, cutoff=0.6)


def format_unknown_subcommand_hint(token: str, registered: Sequence[str]) -> str:
    """Return the actionable hint shown when a bare token did not match any subcommand.

    The hint always offers the query-first interpretation; if the token looks
    like a typo of a real subcommand, it also offers ``did-you-mean`` style
    suggestions before the query suggestion.
    """
    suggestions = looks_like_subcommand_typo(token, registered)
    parts: list[str] = []
    if suggestions:
        parts.append("did you mean: " + ", ".join(f"`polylogue {s}`" for s in suggestions) + "?")
    parts.append(
        f'did you mean to search? try: `polylogue "{token}"` '
        "(query-first dispatch interprets any unrecognized leading token as a search query)."
    )
    parts.append("run `polylogue --help` to list every registered subcommand.")
    return "\n  ".join(parts)


def is_query_verb(name: str) -> bool:
    """Return True if ``name`` is one of the query-first verbs (list/count/etc)."""
    return name in QUERY_VERB_NAMES or name in VERB_NAMES


__all__ = [
    "emit_parser_decision",
    "format_unknown_subcommand_hint",
    "is_query_verb",
    "looks_like_subcommand_typo",
]


def _stderr() -> object:
    """Expose sys.stderr through a helper so it can be redirected in tests."""
    return sys.stderr
