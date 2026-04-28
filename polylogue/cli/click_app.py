"""CLI entrypoint (query-first design with subcommands).

The CLI uses a hybrid structure:
- Positional args without subcommand prefix → query mode
- Known subcommand prefixes (sync, check, mcp, etc.) → subcommand mode
- No args → stats mode
"""

from __future__ import annotations

import click

from polylogue.cli.click_command_registration import (
    completions_command,
    dashboard_command,
    mcp_command,
    register_root_commands,
)
from polylogue.cli.click_option_groups import apply_query_mode_options
from polylogue.cli.formatting import should_use_plain
from polylogue.cli.machine_main import extract_option as _extract_option
from polylogue.cli.machine_main import run_machine_entry
from polylogue.cli.query import QueryFirstGroupBase, handle_query_mode
from polylogue.cli.query_verbs import QUERY_VERBS
from polylogue.cli.shared.types import AppEnv
from polylogue.logging import configure_logging
from polylogue.ui import create_ui
from polylogue.version import POLYLOGUE_VERSION


class QueryFirstGroup(QueryFirstGroupBase):
    """Project-specific query-first CLI group."""

    def handle_default_mode(self, ctx: click.Context) -> None:
        _handle_query_mode(ctx)


def _handle_query_mode(ctx: click.Context) -> None:
    """Handle query mode: display stats or perform search."""
    handle_query_mode(ctx, show_stats=_show_stats)


def _show_stats(env: AppEnv, *, verbose: bool = False) -> None:
    """Show archive statistics."""
    from polylogue.cli.shared.helpers import print_summary

    print_summary(env, verbose=verbose)


# Main CLI group with query-mode options
@click.group(
    cls=QueryFirstGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
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
    provider: str | None,
    exclude_provider: str | None,
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
    min_messages: int | None,
    max_messages: int | None,
    min_words: int | None,
    message_type: str | None,
    since_session_id: str | None,
    since: str | None,
    until: str | None,
    limit: int | None,
    offset: int,
    latest: bool,
    sort: str | None,
    reverse: bool,
    sample: int | None,
    # Output
    output: str | None,
    output_format: str | None,
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
    tail: bool,
    plain: bool,
    verbose: bool,
) -> None:
    """Polylogue - AI conversation archive.

    \b
    Query mode (default):
        polylogue "search terms"
        polylogue -p claude-ai --since "last week"
        polylogue --latest --output browser

    \b
    Verbs (actions on matched conversations):
        polylogue "error" -p claude-ai --since 2025-01 list
        polylogue --has thinking --sort tokens list --limit 10
        polylogue -p chatgpt count
        polylogue --provider codex stats --by provider
        polylogue --latest open
        polylogue "urgent" --tag review delete --dry-run
        polylogue list --format json

    \b
    Combined filters:
        polylogue --path README.md --action file_read list
        polylogue --action search --action file_edit list
        polylogue --action-sequence file_read,file_edit,shell list
        polylogue --action-text "pytest -q" list
        polylogue "pytest -q tests/unit/core/test_semantic_facts.py" --retrieval-lane actions --limit 5
        polylogue --tail --provider claude-code --latest list
        polylogue --action other stats --by tool --format json
        polylogue --provider claude-code --since 2026-01-01 stats --by repo --format json
        polylogue --tool bash --exclude-tool read list
        polylogue --similar "sqlite locking bug in parser" --limit 5

    \b
    Modifiers (write operations):
        polylogue "urgent" --add-tag review

    Run `polylogue <command> --help` for subcommand details.
    """
    # Set up logging early so all output goes to stderr
    configure_logging(verbose=verbose)

    # Set up environment
    use_plain = should_use_plain(plain=plain)
    env = AppEnv(ui=create_ui(use_plain))
    ctx.obj = env


register_root_commands(cli)

for _verb in QUERY_VERBS:
    cli.add_command(_verb)


def main() -> None:
    """CLI entrypoint with machine-error handling.

    When ``--json`` is detected in argv, Click exceptions and unexpected
    errors are caught and emitted as structured JSON on stdout instead of
    Click's default plain-text stderr output.
    """
    import sys

    run_machine_entry(cli, sys.argv[1:])


__all__ = [
    "QueryFirstGroup",
    "_extract_option",
    "_handle_query_mode",
    "cli",
    "completions_command",
    "dashboard_command",
    "main",
    "mcp_command",
]
