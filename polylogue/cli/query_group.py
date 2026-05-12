"""Lightweight Click group for the query-first CLI.

This module intentionally only imports ``click`` so that ``--help``,
tab-completion, and every subcommand never pay the heavy
archive/storage/operations import cost (~2.5 s).  The full query
execution logic lives in ``query.py`` and is only imported when a
query actually runs.
"""

from __future__ import annotations

import click


def _split_query_mode_args(group: click.Group, args: list[str]) -> tuple[list[str], tuple[str, ...], bool]:
    """Split args into (click_args, query_terms, has_subcommand).

    Lives here (not in query.py) so that parse_args can use it without
    importing the heavy query.py module.
    """
    from polylogue.cli.query_verbs import VERB_NAMES

    option_arity = _option_arity(group)
    option_args: list[str] = []
    query_terms: list[str] = []
    index = 0

    while index < len(args):
        arg = args[index]
        if arg == "--":
            query_terms.extend(args[index + 1 :])
            break
        if arg.startswith("-"):
            option_args.append(arg)
            nargs = option_arity.get(arg, 0)
            option_args.extend(_iter_option_values(args, index, nargs))
            index += nargs + 1
            continue
        if not query_terms and arg in group.commands and arg not in VERB_NAMES:
            return args, (), True
        if arg in VERB_NAMES:
            misplaced = _find_root_option_after_verb(group, arg, list(args[index + 1 :]))
            if misplaced is not None:
                raise click.UsageError(
                    f"Query filters and root output flags must appear before the verb. Move {misplaced} before `{arg}`."
                )
            verb_args = option_args + [arg] + list(args[index + 1 :])
            return verb_args, tuple(query_terms), True
        query_terms.append(arg)
        index += 1

    return option_args, tuple(query_terms), False


def _option_arity(group: click.Group) -> dict[str, int]:
    """Return a mapping of long-option -> nargs for all root group params."""
    arity: dict[str, int] = {}
    for param in group.params:
        if isinstance(param, click.Option):
            n = param.nargs if param.nargs != 1 else 0
            if param.name:
                arity[f"--{param.name}"] = n
            for opt in param.opts:
                if opt.startswith("--"):
                    arity[opt] = n
    return arity


def _iter_option_values(args: list[str], index: int, nargs: int):  # type: ignore[no-untyped-def]
    """Yield the next ``nargs`` positional args from the arg list."""
    for j in range(1, nargs + 1):
        yield args[index + j]


def _find_root_option_after_verb(group: click.Group, verb: str, remaining: list[str]) -> str | None:
    """Find a root group option that appears after the verb."""
    root_opts: set[str] = set()
    for param in group.params:
        if isinstance(param, click.Option):
            for opt in param.opts:
                root_opts.add(opt)
    for arg in remaining:
        if arg in root_opts:
            return arg
    return None


class QueryFirstGroupBase(click.Group):
    """Custom Click group that routes to query mode by default."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse args, preserving raw query terms."""
        parse_args, query_terms, has_subcommand = _split_query_mode_args(self, args)
        ctx.meta["polylogue_has_subcommand"] = has_subcommand
        if not has_subcommand:
            ctx.meta["polylogue_query_terms"] = query_terms
        return list(super().parse_args(ctx, parse_args))

    def invoke(self, ctx: click.Context) -> object:
        """Invoke the group, dispatching to query or stats mode if no subcommand."""
        if ctx.meta.get("polylogue_has_subcommand", False):
            return super().invoke(ctx)

        assert self.callback is not None, "QueryFirstGroup requires a callback"
        with ctx:
            ctx.invoke(self.callback, **ctx.params)

        self.handle_default_mode(ctx)
        return None

    def handle_default_mode(self, ctx: click.Context) -> None:
        """Dispatch no-subcommand mode for subclasses."""
        raise NotImplementedError
