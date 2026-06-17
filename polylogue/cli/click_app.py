"""CLI entrypoint (query-first design with subcommands).

The CLI uses a hybrid structure:
- Positional args without subcommand prefix → query mode
- Known subcommand prefixes (doctor, schema, check, etc.) → subcommand mode
- No args → stats mode
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click
from click.shell_completion import CompletionItem

from polylogue.cli.click_command_registration import _LazyCommand, register_root_commands
from polylogue.cli.click_option_groups import apply_query_mode_options
from polylogue.cli.help_markdown import render_help_markdown
from polylogue.cli.machine_main import extract_option as _extract_option
from polylogue.cli.machine_main import run_machine_entry
from polylogue.cli.query_group import QueryFirstGroupBase
from polylogue.cli.shared.formatting import should_use_plain
from polylogue.cli.shared.types import AppEnv
from polylogue.cli.verb_names import QUERY_VERB_NAMES
from polylogue.logging import configure_logging
from polylogue.version import POLYLOGUE_VERSION

if TYPE_CHECKING:
    from polylogue.ui import UI


class QueryFirstGroup(QueryFirstGroupBase):
    """Project-specific query-first CLI group."""

    def handle_default_mode(self, ctx: click.Context) -> None:
        _handle_query_mode(ctx)

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        """Keep query-action completion tied to action contracts after ``then``."""

        if _is_after_then_completion():
            from polylogue.cli.shell_completion_values import complete_query_actions

            return complete_query_actions(ctx, None, incomplete)
        if _should_complete_then_connector(incomplete):
            return [CompletionItem("then", help="Connect query results to a verb/action.")]
        return super().shell_complete(ctx, incomplete)


def _completion_words() -> tuple[str, ...]:
    raw_words = os.environ.get("COMP_WORDS", "")
    words = tuple(part for part in raw_words.split() if part)
    if words and words[0] == "polylogue":
        return words[1:]
    return words


def _is_after_then_completion() -> bool:
    words = _completion_words()
    return len(words) >= 2 and words[-2] == "then"


def _should_complete_then_connector(incomplete: str) -> bool:
    if not "then".startswith(incomplete.lower()):
        return False
    words = _completion_words()
    prior = words[:-1]
    if not prior or "then" in prior:
        return False
    if prior[0] in QUERY_VERB_NAMES:
        return False
    return prior[0] == "find" or any(":" in word for word in prior)


def _handle_query_mode(ctx: click.Context) -> None:
    """Handle query mode: display stats or perform search."""
    from polylogue.cli.query import handle_query_mode

    handle_query_mode(ctx, show_stats=_show_stats)


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
    message_type: str | None,
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
    output_as_json: bool,
    transform: str | None,
    no_code_blocks: bool,
    no_tool_calls: bool,
    no_tool_outputs: bool,
    no_file_reads: bool,
    prose_only: bool,
    # Streaming
    stream: bool,
    dialogue_only: bool,
    message_role: tuple[str, ...],
    # Modifiers
    set_meta: tuple[tuple[str, str], ...],
    add_tag: tuple[str, ...],
    # Global
    plain: bool,
    verbose: bool,
    diagnose: bool,
) -> None:
    """Polylogue - AI session archive.

    \b
    Dispatch model (query-first):
        Any positional token that is NOT a registered subcommand or verb is
        treated as a search query against the archive. So `polylogue foo`
        searches for `foo` — it does NOT raise "unknown subcommand".
        Run `polylogue --help` to see the full subcommand list, or
        `polylogue --diagnose <args>` to have the parser explain how it
        routed your invocation.

    \b
    Query mode (default):
        polylogue "search terms"
        polylogue --origin claude-ai-export --since "last week"
        polylogue --latest --output browser

    \b
    Verbs (actions on matched sessions):
        polylogue "error" --origin claude-ai-export --since 2025-01 list
        polylogue --has thinking --sort tokens list --limit 10
        polylogue --origin chatgpt-export count
        polylogue --origin codex-session stats --by origin
        polylogue --latest read
        polylogue "urgent" --tag review delete --dry-run
        polylogue list --format json
        polylogue find id:abc then read
        polylogue find id:abc then read --view messages
        polylogue find id:abc then read --to browser
        polylogue find 'repo:polylogue' then read --all --format ndjson

    \b
    Combined filters:
        polylogue --referenced-path README.md --action file_read list
        polylogue --action search --action file_edit list
        polylogue --action-sequence file_read,file_edit,shell list
        polylogue --action-text "pytest -q" list
        polylogue "pytest -q tests/unit/core/test_semantic_facts.py" --retrieval-lane actions --limit 5
        polylogue --action other stats --by tool --format json
        polylogue --origin claude-code-session --since 2026-01-01 stats --by repo --format json
        polylogue --tool bash --exclude-tool read list
        polylogue --similar "sqlite locking bug in parser" --limit 5

    \b
    Modifiers (write operations):
        polylogue "urgent" --add-tag review

    \b
    See also:
        polylogue --help                  # this screen
        polylogue <subcommand> --help     # per-subcommand help
        polylogue --diagnose <args>       # explain parser decisions
    """
    # #1689: --json forces plain output and defaults to JSON format.
    if output_as_json:
        plain = True
        if not output_format:
            output_format = "json"

    # Set up logging early so all output goes to stderr
    configure_logging(verbose=verbose)

    use_plain = should_use_plain(plain=plain)
    env = AppEnv(ui=create_ui(use_plain))
    ctx.obj = env


register_root_commands(cli)

_QUERY_VERB_HELP: dict[str, str] = {
    "count": "Print count of matched sessions.",
    "delete": "Delete matched sessions.",
    "list": "List matched sessions.",
    "read": "Read matched sessions (route to view/destination).",
    "recent": "List the most recently updated sessions.",
    "stats": "Show statistics for matched sessions.",
}

for _verb in sorted(QUERY_VERB_NAMES):
    _attr = f"{_verb.replace('-', '_')}_verb"
    cli.add_command(
        _LazyCommand(
            _verb,
            "polylogue.cli.query_verbs",
            _attr,
            short_help=_QUERY_VERB_HELP.get(_verb),
        )
    )


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
