"""Query-first CLI routing primitives."""

from __future__ import annotations

from typing import Any

import click

from polylogue.cli.types import AppEnv
from polylogue.lib.query_spec import ConversationQuerySpec


def handle_query_mode(
    ctx: click.Context,
    *,
    show_stats: Any,
) -> None:
    """Handle query mode: display stats or perform search."""
    from polylogue.cli.query import execute_query

    env: AppEnv = ctx.obj
    params = ctx.params

    query_terms = params.get("query_term", ())
    params_copy = dict(params)
    params_copy["query"] = query_terms
    query_spec = ConversationQuerySpec.from_params(params_copy)
    has_filters = query_spec.has_filters()

    has_output_mode = any(
        params.get(key)
        for key in (
            "list_mode",
            "limit",
            "stats_only",
            "stats_by",
            "count_only",
            "stream",
            "dialogue_only",
        )
    )

    has_modifiers = any(
        params.get(key)
        for key in (
            "add_tag",
            "set_meta",
            "delete_matched",
        )
    )

    if not query_terms and not has_filters and not has_output_mode and not has_modifiers:
        show_stats(env, verbose=params.get("verbose", False))
        return

    execute_query(env, params_copy)


class QueryFirstGroupBase(click.Group):
    """Custom Click group that routes to query mode by default."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse args, converting positional args to hidden query-term options."""
        first_arg_idx = None
        for index, arg in enumerate(args):
            if not arg.startswith("-"):
                first_arg_idx = index
                break

        if first_arg_idx is not None and args[first_arg_idx] in self.commands:
            ctx.ensure_object(dict)
            ctx.obj["_has_subcommand"] = True
            return super().parse_args(ctx, args)

        value_options: dict[str, int] = {}
        for param in self.params:
            if isinstance(param, click.Option) and not param.is_flag:
                nargs = param.nargs if param.nargs > 0 else 1
                for opt in param.opts + param.secondary_opts:
                    value_options[opt] = nargs

        new_args: list[str] = []
        index = 0
        while index < len(args):
            arg = args[index]
            if arg.startswith("-"):
                new_args.append(arg)
                nargs = value_options.get(arg, 0)
                for _ in range(nargs):
                    if index + 1 < len(args):
                        index += 1
                        new_args.append(args[index])
            else:
                new_args.extend(["--query-term", arg])
            index += 1

        return list(super().parse_args(ctx, new_args))

    def invoke(self, ctx: click.Context) -> Any:
        """Invoke the group, dispatching to query or stats mode if no subcommand."""
        ctx.ensure_object(dict)
        has_subcommand = ctx.obj.get("_has_subcommand", False)

        if has_subcommand:
            return super().invoke(ctx)

        assert self.callback is not None, "QueryFirstGroup requires a callback"
        with ctx:
            ctx.invoke(self.callback, **ctx.params)

        self.handle_default_mode(ctx)

    def handle_default_mode(self, ctx: click.Context) -> None:
        """Dispatch no-subcommand mode for subclasses."""
        raise NotImplementedError


__all__ = ["QueryFirstGroupBase", "handle_query_mode"]
