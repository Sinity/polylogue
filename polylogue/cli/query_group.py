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
    from polylogue.cli.verb_names import VERB_NAMES

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
        # Optional `then` connector: `find Q then read` → strip `then` and
        # treat the following token as the verb.
        if arg == "then" and index + 1 < len(args) and args[index + 1] in VERB_NAMES:
            index += 1
            arg = args[index]
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
        if isinstance(param, click.Option) and not param.is_flag:
            n = param.nargs if param.nargs > 0 else 1
            for opt in param.opts + param.secondary_opts:
                arity[opt] = n
    return arity


def _iter_option_values(args: list[str], index: int, nargs: int):  # type: ignore[no-untyped-def]
    """Yield the next ``nargs`` positional args from the arg list.

    Tolerates truncated input (shell tab-completion supplies a partial
    command line where the trailing option may have no value yet); in
    that case we yield whatever values are present without raising.
    """
    for j in range(1, nargs + 1):
        if index + j >= len(args):
            return
        yield args[index + j]


def _find_root_option_after_verb(group: click.Group, verb: str, remaining: list[str]) -> str | None:
    """Find a root group option that appears after the verb."""
    root_opts: set[str] = set()
    for param in group.params:
        if isinstance(param, click.Option):
            for opt in param.opts:
                root_opts.add(opt)
    command = group.commands.get(verb)
    command_opts: set[str] = set()
    if command is not None:
        ctx = click.Context(command)
        for param in command.get_params(ctx):
            if isinstance(param, click.Option):
                command_opts.update(param.opts)
                command_opts.update(param.secondary_opts)
    for arg in remaining:
        if arg in root_opts and arg not in command_opts:
            return arg
    return None


def _detect_subcommand_and_verb(group: click.Group, args: list[str]) -> tuple[str | None, str | None]:
    """Look at ``args`` and return the (subcommand, verb) the parser will dispatch to."""
    from polylogue.cli.verb_names import VERB_NAMES

    subcommand: str | None = None
    verb: str | None = None
    for arg in args:
        if arg.startswith("-"):
            continue
        if arg in group.commands and arg not in VERB_NAMES:
            subcommand = arg
            break
        if arg in VERB_NAMES:
            verb = arg
            break
    return subcommand, verb


class QueryFirstGroupBase(click.Group):
    """Custom Click group that routes to query mode by default."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse args, preserving raw query terms."""
        # Capture the raw argv slice up-front so ``--diagnose`` can show it
        # exactly as the user typed it, before _split_query_mode_args mutates
        # the list into per-mode buckets.
        original_args = list(args)
        ctx.meta["polylogue_raw_args"] = original_args

        parse_args, query_terms, has_subcommand = _split_query_mode_args(self, args)
        ctx.meta["polylogue_has_subcommand"] = has_subcommand
        ctx.meta["polylogue_query_terms"] = query_terms if not has_subcommand else ()
        subcommand, verb = _detect_subcommand_and_verb(self, original_args)
        ctx.meta["polylogue_dispatch_subcommand"] = subcommand
        ctx.meta["polylogue_dispatch_verb"] = verb
        return list(super().parse_args(ctx, parse_args))

    def invoke(self, ctx: click.Context) -> object:
        """Invoke the group, dispatching to query or stats mode if no subcommand."""
        self._maybe_emit_diagnose(ctx)

        if ctx.meta.get("polylogue_has_subcommand", False):
            return super().invoke(ctx)

        assert self.callback is not None, "QueryFirstGroup requires a callback"
        with ctx:
            ctx.invoke(self.callback, **ctx.params)

        self.handle_default_mode(ctx)
        return None

    def _maybe_emit_diagnose(self, ctx: click.Context) -> None:
        """Emit parser-decision diagnostics on stderr when ``--diagnose`` is set."""
        if not ctx.params.get("diagnose"):
            return
        # Guard against emitting the diagnose banner twice if ``invoke`` is
        # called more than once (Click's testing harness sometimes does this).
        if ctx.meta.get("polylogue_diagnose_emitted"):
            return
        ctx.meta["polylogue_diagnose_emitted"] = True

        from polylogue.cli.parser_diagnostics import emit_parser_decision

        emit_parser_decision(
            ctx.meta.get("polylogue_raw_args") or [],
            has_subcommand=bool(ctx.meta.get("polylogue_has_subcommand", False)),
            subcommand=ctx.meta.get("polylogue_dispatch_subcommand"),
            query_terms=ctx.meta.get("polylogue_query_terms") or (),
            verb=ctx.meta.get("polylogue_dispatch_verb"),
            registered_commands=sorted(self.commands.keys()),
        )

    def handle_default_mode(self, ctx: click.Context) -> None:
        """Dispatch no-subcommand mode for subclasses."""
        raise NotImplementedError
