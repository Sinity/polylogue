"""Lightweight Click group for the query-first CLI.

This module intentionally only imports ``click`` so that ``--help``,
tab-completion, and every subcommand never pay the heavy
archive/storage/operations import cost (~2.5 s).  The full query
execution logic lives in ``query.py`` and is only imported when a
query actually runs.
"""

from __future__ import annotations

import click


def _split_query_mode_args(group: click.Group, args: list[str]) -> tuple[list[str], tuple[str, ...], bool, bool]:
    """Split args into (click_args, query_terms, has_subcommand, explicit_query).

    Lives here (not in query.py) so that parse_args can use it without
    importing the heavy query.py module.

    ``explicit_query`` is True when the user signalled query intent explicitly,
    either via the leading ``find`` keyword (#1842 canonical grammar) or the
    ``--`` positional escape. It lets ``invoke`` apply the strict-floor hint
    only to unsignalled bare roots.
    """
    from polylogue.cli.verb_names import VERB_NAMES

    option_arity = _option_arity(group)
    option_args: list[str] = []
    query_terms: list[str] = []
    explicit_query = False
    index = 0

    while index < len(args):
        arg = args[index]
        if arg == "--":
            query_terms.extend(args[index + 1 :])
            explicit_query = True
            break
        if arg.startswith("-"):
            option_args.append(arg)
            nargs = option_arity.get(arg, 0)
            option_args.extend(_iter_option_values(args, index, nargs))
            index += nargs + 1
            continue
        # Leading `find` keyword (#1842): strip it and mark the query explicit.
        # `find` is the canonical "this is a query" marker; only the FIRST
        # positional token is the keyword — a later `find` is a literal term.
        if not query_terms and not explicit_query and arg == "find":
            explicit_query = True
            index += 1
            continue
        if not query_terms and not explicit_query and arg in group.commands and arg not in VERB_NAMES:
            return args, (), True, explicit_query
        # Optional `then` connector: `find Q then read` → strip `then` and
        # treat the following token as the verb.
        via_then = False
        if arg == "then" and index + 1 < len(args) and args[index + 1] in VERB_NAMES:
            index += 1
            arg = args[index]
            via_then = True
        if arg in VERB_NAMES:
            if arg == "mark" and index + 1 < len(args) and args[index + 1] == "candidates":
                return args, (), True, explicit_query
            # After an explicit `find`, a verb word sitting in query position —
            # no `then`, nothing collected yet — is the search term itself:
            # `find read` searches for "read", it does not run the read action.
            if explicit_query and not via_then and not query_terms:
                query_terms.append(arg)
                index += 1
                continue
            misplaced = _find_root_option_after_verb(group, arg, list(args[index + 1 :]))
            if misplaced is not None:
                raise click.UsageError(
                    f"Query filters and root output flags must appear before the verb. Move {misplaced} before `{arg}`."
                )
            verb_args = option_args + [arg] + list(args[index + 1 :])
            return verb_args, tuple(query_terms), True, explicit_query
        query_terms.append(arg)
        index += 1

    return option_args, tuple(query_terms), False, explicit_query


def _looks_like_query_expression(query_terms: tuple[str, ...]) -> bool:
    """Heuristic: did the user clearly intend a query rather than mistype a command?

    Strict floor (#1842): an unquoted plain-word root prints a hint instead of
    silently searching. We still run a bare query when the user signalled intent
    structurally: a single argv token carrying internal whitespace can only have
    come from shell quoting (``'machine learning'``), and any token using field
    syntax (``repo:x``, ``since:7d``) is unambiguously a query expression.
    """
    if len(query_terms) == 1 and any(ws in query_terms[0] for ws in (" ", "\t", "\n")):
        return True
    return any(":" in term for term in query_terms)


def _bare_root_error_message(group: click.Group, query_terms: tuple[str, ...]) -> str:
    """Build the strict-floor hint for an unsignalled bare root (#1842).

    Folds in the existing did-you-mean suggestion (a single token close to a
    registered command, e.g. ``lst`` → ``list``) so the typo affordance that
    used to surface on the no-results path is preserved on the refusal path.
    """
    query = " ".join(query_terms)
    lines = [f"No such command {query!r}."]
    if len(query_terms) == 1:
        token = query_terms[0].strip()
        if token:
            from polylogue.cli.parser_diagnostics import looks_like_subcommand_typo

            suggestions = looks_like_subcommand_typo(token, sorted(group.commands.keys()))
            if suggestions:
                lines += ["", "If you meant a subcommand, try: " + ", ".join(f"`polylogue {s}`" for s in suggestions)]
    lines += [
        "",
        "To search the archive, use the `find` keyword or quote the expression:",
        f"    polylogue find {query}",
        "    polylogue 'QUERY' then read",
        "",
        "Run `polylogue --help` to list commands.",
    ]
    return "\n".join(lines)


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

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[click.shell_completion.CompletionItem]:
        from polylogue.cli.shell_completion_values import (
            complete_query_actions,
            complete_query_expression_context_fields,
            complete_query_expression_fields,
        )

        context_items = complete_query_expression_context_fields(ctx, None, incomplete)
        if context_items is not None:
            return context_items

        items = list(super().shell_complete(ctx, incomplete))
        replacement_items = [
            *complete_query_actions(ctx, None, incomplete),
            *complete_query_expression_fields(ctx, None, incomplete),
        ]
        replacements = {item.value: item for item in replacement_items}
        merged = [replacements.pop(item.value, item) for item in items]
        merged.extend(replacements.values())
        return merged

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse args, preserving raw query terms."""
        # Capture the raw argv slice up-front so ``--diagnose`` can show it
        # exactly as the user typed it, before _split_query_mode_args mutates
        # the list into per-mode buckets.
        original_args = list(args)
        ctx.meta["polylogue_raw_args"] = original_args

        parse_args, query_terms, has_subcommand, explicit_query = _split_query_mode_args(self, args)
        ctx.meta["polylogue_has_subcommand"] = has_subcommand
        ctx.meta["polylogue_query_terms"] = query_terms if not has_subcommand else ()
        ctx.meta["polylogue_explicit_query"] = explicit_query
        subcommand, verb = _detect_subcommand_and_verb(self, original_args)
        ctx.meta["polylogue_dispatch_subcommand"] = subcommand
        ctx.meta["polylogue_dispatch_verb"] = verb
        return list(super().parse_args(ctx, parse_args))

    def invoke(self, ctx: click.Context) -> object:
        """Invoke the group, dispatching to query or stats mode if no subcommand."""
        self._maybe_emit_diagnose(ctx)

        if ctx.meta.get("polylogue_has_subcommand", False):
            return super().invoke(ctx)

        # Strict command floor (#1842): an unquoted, unsignalled bare root that
        # is not a known command must not silently become a search. Print a hint
        # pointing at `find` / quoting instead. Explicit query intent (leading
        # `find`, `--` escape) and structurally-clear expressions still run.
        query_terms: tuple[str, ...] = ctx.meta.get("polylogue_query_terms", ()) or ()
        explicit_query = bool(ctx.meta.get("polylogue_explicit_query", False))
        if query_terms and not explicit_query and not _looks_like_query_expression(query_terms):
            raise click.UsageError(_bare_root_error_message(self, query_terms))

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
