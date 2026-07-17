"""CLI entrypoint (query-first design with subcommands).

The CLI uses a hybrid structure:
- Positional args without subcommand prefix → query mode
- Known subcommand prefixes (`ops`, etc.) → subcommand mode
- No args → status / stats mode
"""

from __future__ import annotations

import os
import sys
from time import perf_counter
from typing import TYPE_CHECKING

import click
from click.shell_completion import CompletionItem

from polylogue.cli.click_command_registration import _LazyCommand, _LazyGroup, register_root_commands
from polylogue.cli.click_option_groups import apply_query_mode_options
from polylogue.cli.command_inventory import ROOT_COMMAND_ROLE_SECTIONS
from polylogue.cli.help_markdown import render_help_markdown
from polylogue.cli.machine_main import extract_option as _extract_option
from polylogue.cli.machine_main import run_machine_entry
from polylogue.cli.query_group import QueryFirstGroupBase
from polylogue.cli.shared.formatting import should_use_plain
from polylogue.cli.shared.types import AppEnv
from polylogue.cli.shell_words import completion_words
from polylogue.cli.verb_names import QUERY_VERB_NAMES
from polylogue.logging import configure_logging
from polylogue.version import POLYLOGUE_VERSION

if TYPE_CHECKING:
    from polylogue.cli.select import SelectSessionRow
    from polylogue.config import Config
    from polylogue.ui import UI


_CLI_CALLBACK_STARTED_AT = perf_counter()


class QueryFirstGroup(QueryFirstGroupBase):
    """Project-specific query-first CLI group."""

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Group root commands by product role instead of one flat command drawer."""
        rows_by_section: list[tuple[str, list[tuple[str, str]], str | None]] = []
        rendered: set[str] = set()
        for section in ROOT_COMMAND_ROLE_SECTIONS:
            rows: list[tuple[str, str]] = []
            for name in section.commands:
                cmd = self.get_command(ctx, name)
                if cmd is None:
                    continue
                rendered.add(name)
                help_text = cmd.get_short_help_str()
                rows.append((name, help_text))
            if rows:
                rows_by_section.append((section.title, rows, section.footer))

        remaining: list[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            if name in rendered:
                continue
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            remaining.append((name, cmd.get_short_help_str()))
        if remaining:
            rows_by_section.append(("Other commands", remaining, None))
        if not rows_by_section:
            return

        with formatter.section("Commands"):
            for index, (title, rows, footer) in enumerate(rows_by_section):
                if index:
                    formatter.write_paragraph()
                formatter.write_text(f"{title}:")
                with formatter.indentation():
                    formatter.write_dl(rows)
                    if footer:
                        formatter.write_text(footer)

    def handle_default_mode(self, ctx: click.Context) -> None:
        _handle_query_mode(ctx)

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        """Keep query-action completion tied to action contracts after ``then``."""

        count_field = _count_operator_completion_field()
        if count_field is not None:
            from polylogue.cli.shell_completion_values import (
                query_completion_candidate_to_click_item,
                query_count_operator_candidates,
            )

            return [
                query_completion_candidate_to_click_item(candidate)
                for candidate in query_count_operator_candidates(count_field, incomplete)
            ]
        date_field = _date_operator_completion_field()
        if date_field is not None:
            from polylogue.cli.shell_completion_values import (
                query_completion_candidate_to_click_item,
                query_date_operator_candidates,
            )

            return [
                query_completion_candidate_to_click_item(candidate)
                for candidate in query_date_operator_candidates(date_field, incomplete)
            ]
        numeric_field = _numeric_operator_completion_field()
        if numeric_field is not None:
            from polylogue.cli.shell_completion_values import (
                query_completion_candidate_to_click_item,
                query_numeric_operator_candidates,
            )

            return [
                query_completion_candidate_to_click_item(candidate)
                for candidate in query_numeric_operator_candidates(numeric_field, incomplete)
            ]
        if _is_after_then_completion():
            from polylogue.cli.shell_completion_values import complete_query_result_actions

            return complete_query_result_actions(ctx, None, incomplete)
        if _should_complete_then_connector(incomplete):
            return [CompletionItem("then", help="Connect query results to a verb/action.")]
        return super().shell_complete(ctx, incomplete)


def _completion_words() -> tuple[str, ...]:
    return completion_words()


def _is_after_then_completion() -> bool:
    words = _completion_words()
    raw_words = os.environ.get("COMP_WORDS", "")
    if raw_words.endswith(" ") and words:
        return words[-1].lower() == "then"
    return len(words) >= 2 and words[-2].lower() == "then"


def _is_sessions_where_context(words: tuple[str, ...]) -> bool:
    if words and words[0].lower() == "find":
        words = words[1:]
    lowered = tuple(word.lower() for word in words)
    if any(word.startswith("sessions where") for word in lowered):
        return True
    return any(word == "sessions" and lowered[index + 1] == "where" for index, word in enumerate(lowered[:-1]))


def _field_from_operator_completion_words(words: tuple[str, ...]) -> str | None:
    raw_words = os.environ.get("COMP_WORDS", "")
    if raw_words.endswith(" ") and words:
        return words[-1].lower()
    if len(words) < 2:
        return None
    if len(words) >= 3 and words[-3].lower() == "between":
        return None
    return words[-2].lower()


def _count_operator_completion_field() -> str | None:
    from polylogue.archive.query.metadata import count_query_fields

    words = _completion_words()
    previous = _field_from_operator_completion_words(words)
    if previous is None or previous not in set(count_query_fields()):
        return None
    if previous in {"messages", "words"} or _is_sessions_where_context(words):
        return previous
    return None


def _date_operator_completion_field() -> str | None:
    from polylogue.archive.query.metadata import date_query_fields

    words = _completion_words()
    previous = _field_from_operator_completion_words(words)
    if previous is None or previous not in set(date_query_fields()):
        return None
    return previous


def _numeric_operator_completion_field() -> str | None:
    from polylogue.archive.query.metadata import numeric_query_fields

    words = _completion_words()
    previous = _field_from_operator_completion_words(words)
    if previous is None or previous not in set(numeric_query_fields()):
        return None
    if _is_sessions_where_context(words):
        return previous
    return None


def _should_complete_then_connector(incomplete: str) -> bool:
    if not "then".startswith(incomplete.lower()):
        return False
    words = _completion_words()
    prior = words[:-1]
    if not prior or "then" in prior:
        return False
    if prior[0] in QUERY_VERB_NAMES:
        return False
    if prior[0] == "find":
        return len(prior) > 1
    return any(":" in word for word in prior)


def _handle_query_mode(ctx: click.Context) -> None:
    """Handle query mode: display stats or perform search."""
    if (
        not ctx.meta.get("polylogue_query_terms")
        and sys.stdin.isatty()
        and sys.stdout.isatty()
        and _show_bare_tty_triage(ctx, ctx.obj)
    ):
        return
    from polylogue.cli.query import handle_query_mode

    handle_query_mode(ctx, show_stats=_show_stats)


def _show_bare_tty_triage(ctx: click.Context, env: AppEnv) -> bool:
    """Render the interactive no-argument landing surface.

    Returns ``False`` only when no archive exists, letting the caller render
    ordinary Click help instead of presenting an empty archive as usable.
    """

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.cli.root_request import RootModeRequest
    from polylogue.cli.select import select_session_rows
    from polylogue.cli.shared.helpers import load_effective_config
    from polylogue.paths import archive_file_set_root_for_paths

    config = load_effective_config(env)
    archive_root = archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)
    if not (archive_root / "index.db").exists():
        click.echo(ctx.get_help())
        return True

    rows = None if bool(ctx.params.get("no_daemon")) else _bare_tty_daemon_rows(config)
    source = "daemon" if rows is not None else "direct"
    if rows is None:
        try:
            rows = run_coroutine_sync(select_session_rows(env, RootModeRequest.from_params({}), limit=5))
        except Exception:
            click.echo(ctx.get_help())
            return True

    click.echo(f"Archive: ready ({source})")
    click.echo("Recent sessions:")
    if rows:
        for row in rows:
            click.echo(f"  {row.label}")
    else:
        click.echo("  No sessions yet.")
    click.echo("Next: polylogue find …  |  polylogue read <id>  |  polylogue continue <id>")
    return True


def _bare_tty_daemon_rows(config: Config) -> list[SelectSessionRow] | None:
    """Fetch the minimal recent-session page from a config-matched daemon."""

    from pathlib import Path

    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.cli.select import SelectSessionRow
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
    from polylogue.version import POLYLOGUE_VERSION

    client = DaemonClient(
        Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "polylogue" / "daemon.sock",
        auth_token=getattr(config, "api_auth_token", None),
    )
    if (
        client.probe(
            archive_root=str(config.archive_root),
            index_schema_version=INDEX_SCHEMA_VERSION,
            daemon_version=POLYLOGUE_VERSION,
        )
        is None
    ):
        return None
    payload = client.cli_query({"query": (), "limit": 5})
    items = payload.get("items") if payload is not None else None
    if not isinstance(items, list):
        return None
    rows: list[SelectSessionRow] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        session_id = item.get("id")
        origin = item.get("origin")
        title = item.get("title")
        if isinstance(session_id, str) and isinstance(origin, str) and isinstance(title, str):
            date = item.get("date")
            rows.append(SelectSessionRow(session_id, origin, title, date if isinstance(date, str) else None))
    return rows


def _show_stats(env: AppEnv, *, verbose: bool = False) -> None:
    """Show fast status when daemon is reachable, otherwise archive summary."""
    if not verbose:
        try:
            from polylogue.cli.commands.status import show_fast_status

            show_fast_status(env)
            return
        except Exception:
            pass
    from polylogue.cli.shared.helpers import print_summary

    print_summary(env, verbose=verbose)


def create_ui(plain: bool) -> UI:
    """Create the CLI UI without importing the UI stack during CLI definition."""
    from polylogue.ui import create_ui as _create_ui

    return _create_ui(plain)


def _emit_help_markdown(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(render_help_markdown(ctx.command, prog_name=ctx.info_name or "polylogue"), nl=False)
    ctx.exit(0)


# Main CLI group with query-mode options
@click.group(
    cls=QueryFirstGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option(
    "--help-markdown",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_emit_help_markdown,
    help="Emit the full --help tree (root + every subcommand) as Markdown and exit.",
)
@apply_query_mode_options
@click.version_option(version=POLYLOGUE_VERSION, prog_name="polylogue")
@click.pass_context
def cli(
    ctx: click.Context,
    # Filters
    conv_id: str | None,
    contains: tuple[str, ...],
    exclude_text: tuple[str, ...],
    retrieval_lane: str | None,
    lexical: bool,
    semantic: bool,
    origin: str | None,
    exclude_origin: str | None,
    repo: str | None,
    project: str | None,
    tag: str | None,
    exclude_tag: str | None,
    title: str | None,
    referenced_path: tuple[str, ...],
    cwd_prefix: str | None,
    action: tuple[str, ...],
    exclude_action: tuple[str, ...],
    action_sequence: str | None,
    action_text: tuple[str, ...],
    tool: tuple[str, ...],
    exclude_tool: tuple[str, ...],
    similar_text: str | None,
    has_type: tuple[str, ...],
    filter_has_tool_use: bool,
    filter_has_thinking: bool,
    filter_has_paste: bool,
    typed_only: bool,
    min_messages: int | None,
    max_messages: int | None,
    min_words: int | None,
    since_session_id: str | None,
    since: str | None,
    until: str | None,
    cursor: str | None,
    limit: int | None,
    offset: int,
    latest: bool,
    sort: str | None,
    reverse: bool,
    sample: int | None,
    # Output
    output: str | None,
    output_format: str | None,
    explain_query: bool,
    output_as_json: bool,
    # Streaming
    stream: bool,
    # Modifiers
    set_meta: tuple[tuple[str, str], ...],
    add_tag: tuple[str, ...],
    # Global
    plain: bool,
    no_daemon: bool,
    verbose: bool,
    diagnose: bool,
) -> None:
    """Polylogue - AI session archive.

    \b
    Dispatch model (query-first):
        Use `polylogue find QUERY then ACTION` for query-result workflows.
        Quoted query text is also accepted when followed by an action:
        `polylogue 'QUERY' then read`.
        Run `polylogue --help` to see the full subcommand list, or
        `polylogue --diagnose <args>` to have the parser explain how it
        routed your invocation.

    \b
    Product roles:
        Search/read/action:   find QUERY then read|select|mark|analyze|delete|continue; facets
        Setup/demo/evidence:  config, init, import, demo, tutorial
        Reader/TUI:           dashboard --status, dashboard
        Operations:           status (same as polylogue ops status), ops diagnostics, ops maintenance, ops backup

    \b
    Query mode (default):
        polylogue find "search terms"
        polylogue --origin claude-ai-export --since "last week" find "search terms"
        polylogue --latest find 'repo:polylogue' then read --to browser

    \b
    Verbs (actions on matched sessions):
        polylogue find id:abc then read          # exact ref: one selected session
        polylogue find id:abc then select --json # expose selected-session identity
        polylogue find id:abc then read --view messages
        polylogue find id:abc then read --to browser
        polylogue find id:abc then analyze --facets
        polylogue facets --query "vector store" --format json
        polylogue find "urgent" then delete --dry-run
        polylogue find 'repo:polylogue' then read --all --format ndjson

    \b
    Combined filters:
        polylogue --referenced-path README.md find 'repo:polylogue' then read
        polylogue find 'actions where tool:bash AND text:pytest' then read --view messages
        polylogue --action-sequence file_read,file_edit,shell find 'repo:polylogue' then analyze
        polylogue --action-text "pytest -q" find 'repo:polylogue' then read
        polylogue find 'pytest -q tests/unit/core/test_semantic_facts.py' --retrieval-lane actions
        polylogue --origin claude-code-session --since 2026-01-01 find 'repo:polylogue' then analyze --by repo --format json
        polylogue find 'actions where action:file_edit AND path:polylogue/cli' then read --view messages
        polylogue find 'near:"sqlite locking bug in parser"' then read

    \b
    Modifiers (write operations):
        polylogue find "urgent" then mark --tag-add review

    \b
    See also:
        polylogue --help                  # this screen
        polylogue find --help             # query workflow help
        polylogue <subcommand> --help     # per-subcommand help
        polylogue --diagnose <args>       # explain parser decisions
    """
    # #1689: --json forces plain output and defaults to JSON format.
    if output_as_json:
        plain = True
        if not output_format:
            output_format = "json"
        # Root query mode builds RootModeRequest from ctx.params, not from this
        # callback's local variables. Persist the normalized values so
        # `polylogue find QUERY --json` reaches the archive renderer.
        ctx.params["plain"] = plain
        ctx.params["output_format"] = output_format

    # Keep the ordinary command path on the stdlib-backed logger.  The
    # structlog setup is deliberately expensive and this callback also runs
    # for nested command help, where no diagnostic logging is emitted.
    if verbose:
        configure_logging(verbose=True)

    use_plain = should_use_plain(plain=plain)
    debug_timing = os.environ.get("POLYLOGUE_DEBUG_TIMING", "").lower() in {"1", "true", "yes", "on"}
    env = AppEnv(plain=use_plain, debug_timing=debug_timing)
    env.record_timing("cli-callback", _CLI_CALLBACK_STARTED_AT)
    ctx.obj = env
    if debug_timing:
        ctx.call_on_close(env.emit_debug_timings)


@click.command("find", hidden=True, context_settings={"help_option_names": ["-h", "--help"]})
def find_help() -> None:
    """Search the archive, then optionally run an action.

    \b
    Usage:
        polylogue find QUERY
        polylogue find QUERY then ACTION

    \b
    Examples:
        polylogue find "browser capture"
        polylogue find id:abc then read --view messages
        polylogue find 'repo:polylogue since:7d' then analyze --facets
        polylogue find 'repo:polylogue tag:stale' then delete --dry-run

    \b
    Notes:
        `find` is the explicit query marker, not a normal subcommand.
        Put root filters before `find`, and verb-specific options after ACTION:
        `polylogue --origin chatgpt-export find "sqlite" then read --all`.

        Exact refs (`id:...`, `session:...`) are identity filters. A miss
        returns no target instead of broadening to text search. Text queries can
        return many ranked sessions; singleton actions require an exact ref, an
        explicit selection, `--first`, or an action-specific multi flag.

        Action ownership: `mark` changes selected session overlays;
        `mark candidates` reviews assertion candidates; `analyze --facets`
        reports named aggregate families and marks deferred families honestly.
    """


cli.add_command(find_help)
register_root_commands(cli)

_QUERY_VERB_HELP: dict[str, str] = {
    "analyze": "Analyze matched sessions and named facet families.",
    "continue": "Resume a session or compile successor context as JSON.",
    "delete": "Delete matched sessions.",
    "mark": "Mark selected sessions; review candidates under mark candidates.",
    "read": "Read matched sessions (route to view/destination).",
    "select": "Select one matched session or print candidate identities.",
}

for _verb in sorted(QUERY_VERB_NAMES):
    _attr = f"{_verb.replace('-', '_')}_verb"
    _cls = _LazyGroup if _verb in {"analyze", "mark"} else _LazyCommand
    _command = _cls(
        _verb,
        "polylogue.cli.query_verbs",
        _attr,
        short_help=_QUERY_VERB_HELP.get(_verb),
    )
    if isinstance(_command, _LazyGroup):
        _command.invoke_without_command = True
        _command.no_args_is_help = False
    cli.add_command(_command)


def main() -> None:
    """CLI entrypoint with machine-error handling.

    When ``--format json`` is detected in argv, Click exceptions and
    unexpected errors are emitted as structured JSON on stdout instead of
    Click's default plain-text stderr output.
    """
    import sys

    run_machine_entry(cli, sys.argv[1:])


__all__ = [
    "QueryFirstGroup",
    "_extract_option",
    "_handle_query_mode",
    "cli",
    "main",
]
