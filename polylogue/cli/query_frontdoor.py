"""Query-first CLI routing primitives."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import click

from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.types import AppEnv

_ROOT_GLOBAL_OPTIONS = frozenset({"--plain", "--verbose", "-v"})


def _option_arity(group: click.Group) -> dict[str, int]:
    value_options: dict[str, int] = {}
    for param in group.params:
        if isinstance(param, click.Option) and not param.is_flag:
            nargs = param.nargs if param.nargs > 0 else 1
            for opt in param.opts + param.secondary_opts:
                value_options[opt] = nargs
    return value_options


def _matches_option(option: str, token: str) -> bool:
    return token == option or token.startswith(f"{option}=")


def _is_root_global_option(token: str) -> bool:
    return any(_matches_option(option, token) for option in _ROOT_GLOBAL_OPTIONS)


def _iter_option_values(args: list[str], start: int, nargs: int) -> Iterable[str]:
    for offset in range(1, nargs + 1):
        if start + offset < len(args):
            yield args[start + offset]


def _split_query_mode_args(group: click.Group, args: list[str]) -> tuple[list[str], tuple[str, ...], bool]:
    option_arity = _option_arity(group)
    option_args: list[str] = []
    query_terms: list[str] = []
    query_mode_locked = False
    index = 0

    while index < len(args):
        arg = args[index]
        if arg == "--":
            query_terms.extend(args[index + 1 :])
            break
        if arg.startswith("-"):
            option_args.append(arg)
            nargs = option_arity.get(arg, 0)
            if not _is_root_global_option(arg):
                query_mode_locked = True
            option_args.extend(_iter_option_values(args, index, nargs))
            index += nargs + 1
            continue
        if not query_terms and not query_mode_locked and arg in group.commands:
            return args, (), True
        query_terms.append(arg)
        index += 1

    return option_args, tuple(query_terms), False


def handle_query_mode(
    ctx: click.Context,
    *,
    show_stats: Any,
) -> None:
    """Handle query mode: display stats or perform search."""
    from polylogue.cli.query import execute_query

    env: AppEnv = ctx.obj
    request = RootModeRequest.from_context(ctx)

    if request.should_show_stats():
        show_stats(env, verbose=bool(request.params.get("verbose", False)))
        return

    execute_query(env, request.query_params())


class QueryFirstGroupBase(click.Group):
    """Custom Click group that routes to query mode by default."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse args, preserving raw query terms instead of rewriting them as hidden options."""
        parse_args, query_terms, has_subcommand = _split_query_mode_args(self, args)
        ctx.meta["polylogue_has_subcommand"] = has_subcommand
        if not has_subcommand:
            ctx.meta["polylogue_query_terms"] = query_terms
        return list(super().parse_args(ctx, parse_args))

    def invoke(self, ctx: click.Context) -> Any:
        """Invoke the group, dispatching to query or stats mode if no subcommand."""
        if ctx.meta.get("polylogue_has_subcommand", False):
            return super().invoke(ctx)

        assert self.callback is not None, "QueryFirstGroup requires a callback"
        with ctx:
            ctx.invoke(self.callback, **ctx.params)

        self.handle_default_mode(ctx)

    def handle_default_mode(self, ctx: click.Context) -> None:
        """Dispatch no-subcommand mode for subclasses."""
        raise NotImplementedError


__all__ = ["QueryFirstGroupBase", "handle_query_mode"]
