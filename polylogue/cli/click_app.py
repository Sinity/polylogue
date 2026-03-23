"""CLI entrypoint (query-first design with subcommands).

The CLI uses a hybrid structure:
- Positional args without subcommand prefix → query mode
- Known subcommand prefixes (sync, check, mcp, etc.) → subcommand mode
- No args → stats mode
"""

from __future__ import annotations

import click

from polylogue.cli.commands.auth import auth_command
from polylogue.cli.commands.check import check_command
from polylogue.cli.commands.completions import completions_command
from polylogue.cli.commands.dashboard import dashboard_command
from polylogue.cli.commands.embed import embed_command
from polylogue.cli.commands.generate import generate_command
from polylogue.cli.commands.mcp import mcp_command
from polylogue.cli.commands.qa import qa_command
from polylogue.cli.commands.reset import reset_command
from polylogue.cli.commands.run import run_command, sources_command
from polylogue.cli.commands.schema import schema_command
from polylogue.cli.commands.site import site_command
from polylogue.cli.commands.tags import tags_command
from polylogue.cli.formatting import announce_plain_mode, plain_forced_by_env, should_use_plain
from polylogue.cli.machine_main import extract_option as _extract_option
from polylogue.cli.machine_main import run_machine_entry
from polylogue.cli.query_frontdoor import QueryFirstGroupBase, handle_query_mode
from polylogue.cli.types import AppEnv
from polylogue.lib.query_spec import QUERY_ACTION_TYPES
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
    from polylogue.cli.helpers import print_summary

    print_summary(env, verbose=verbose)

# Main CLI group with query-mode options
@click.group(
    cls=QueryFirstGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
# --- Filter options ---
@click.option("--id", "-i", "conv_id", help="Conversation ID (exact or prefix match)")
@click.option("--contains", "-c", multiple=True, help="FTS term (repeatable = AND)")
@click.option("--exclude-text", multiple=True, help="Exclude FTS term")
@click.option("--provider", "-p", help="Include providers (comma = OR)")
@click.option("--exclude-provider", help="Exclude providers")
@click.option("--tag", "-t", help="Include tags (comma = OR, supports key:value)")
@click.option("--exclude-tag", help="Exclude tags")
@click.option("--title", help="Title contains")
@click.option("--path", "path_terms", multiple=True, help="Touched path contains substring (repeatable = AND)")
@click.option("--action", multiple=True, type=click.Choice(QUERY_ACTION_TYPES), help="Require semantic action category (repeatable = AND)")
@click.option("--exclude-action", multiple=True, type=click.Choice(QUERY_ACTION_TYPES), help="Exclude semantic action category (repeatable = AND)")
@click.option("--action-sequence", help="Require ordered semantic action subsequence (comma-separated)")
@click.option("--action-text", multiple=True, help="Require text within normalized action evidence (repeatable = AND)")
@click.option("--tool", multiple=True, help="Require normalized tool name (repeatable = AND)")
@click.option("--exclude-tool", multiple=True, help="Exclude normalized tool name (repeatable = AND)")
@click.option("--similar", "similar_text", help="Semantic similarity query (requires embeddings)")
@click.option("--has", "has_type", multiple=True, help="Filter by content: thinking (reasoning), tools (calls), summary, attachments")
@click.option("--has-tool-use", "filter_has_tool_use", is_flag=True, help="Only conversations with tool use (SQL pushdown)")
@click.option("--has-thinking", "filter_has_thinking", is_flag=True, help="Only conversations with thinking blocks (SQL pushdown)")
@click.option("--min-messages", type=int, help="Minimum message count")
@click.option("--max-messages", type=int, help="Maximum message count")
@click.option("--min-words", type=int, help="Minimum total word count")
@click.option("--since", help="After date (ISO, 'yesterday', 'last week')")
@click.option("--until", help="Before date")
@click.option("--limit", "-n", type=int, help="Max results")
@click.option("--latest", is_flag=True, help="Most recent (= --sort date --limit 1)")
@click.option(
    "--sort",
    type=click.Choice(["date", "tokens", "messages", "words", "longest", "random"]),
    help="Sort by field",
)
@click.option("--reverse", is_flag=True, help="Reverse sort order")
@click.option("--sample", type=int, help="Random sample of N conversations")
# --- Output options ---
@click.option(
    "--output",
    "-o",
    help="Output destinations: browser, clipboard, stdout (comma-separated)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json", "html", "obsidian", "org", "yaml", "plaintext", "csv"]),
    help="Output format",
)
@click.option("--fields", help="Fields for list/json: id, title, provider, date, messages, words, tags, summary")
@click.option("--list", "list_mode", is_flag=True, help="Force list format")
@click.option("--stats", "stats_only", is_flag=True, help="Only statistics, no content")
@click.option("--count", "count_only", is_flag=True, help="Print matched count and exit")
@click.option(
    "--stats-by",
    "stats_by",
    type=click.Choice(["provider", "month", "year", "day", "action", "tool"]),
    help="Aggregate statistics by dimension",
)
@click.option("--open", "open_result", is_flag=True, help="Open result in browser/editor")
@click.option(
    "--transform",
    type=click.Choice(["strip-tools", "strip-thinking", "strip-all"]),
    help="Remove content: strip-tools (tool calls), strip-thinking (reasoning), strip-all (both)",
)
# --- Streaming options (memory-efficient for large conversations) ---
@click.option("--stream", is_flag=True, help="Stream output (low memory). Requires --latest or -i ID. Incompatible with --transform")
@click.option("--dialogue-only", "-d", is_flag=True, help="Show only user/assistant messages")
# --- Modifier options (write operations) ---
@click.option("--set", "set_meta", nargs=2, multiple=True, help="Set metadata key value")
@click.option("--add-tag", multiple=True, help="Add tags (comma-separated)")
@click.option("--delete", "delete_matched", is_flag=True, help="Delete matched (requires filter)")
@click.option("--dry-run", is_flag=True, help="Preview changes without executing")
@click.option("--force", is_flag=True, help="Skip confirmation for bulk operations")
# --- Global options ---
@click.option("--plain", is_flag=True, help="Force non-interactive plain output")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.version_option(version=POLYLOGUE_VERSION, prog_name="polylogue")
@click.pass_context
def cli(
    ctx: click.Context,
    # Filters
    conv_id: str | None,
    contains: tuple[str, ...],
    exclude_text: tuple[str, ...],
    provider: str | None,
    exclude_provider: str | None,
    tag: str | None,
    exclude_tag: str | None,
    title: str | None,
    path_terms: tuple[str, ...],
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
    since: str | None,
    until: str | None,
    limit: int | None,
    latest: bool,
    sort: str | None,
    reverse: bool,
    sample: int | None,
    # Output
    output: str | None,
    output_format: str | None,
    fields: str | None,
    list_mode: bool,
    stats_only: bool,
    count_only: bool,
    stats_by: str | None,
    open_result: bool,
    transform: str | None,
    # Streaming
    stream: bool,
    dialogue_only: bool,
    # Modifiers
    set_meta: tuple[tuple[str, str], ...],
    add_tag: tuple[str, ...],
    delete_matched: bool,
    dry_run: bool,
    force: bool,
    # Global
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
    Combined filters:
        polylogue "error" -p claude-ai --since 2025-01 --list
        polylogue --has thinking --sort tokens --limit 10
        polylogue -t important --stats-by provider
        polylogue --path /realm/project/polylogue/README.md --action file_read --list
        polylogue --action search --action file_edit --list
        polylogue --action-sequence file_read,file_edit,shell --list
        polylogue --action-text "pytest -q" --list
        polylogue --action other --stats-by tool --format json
        polylogue --tool bash --exclude-tool read --list
        polylogue --similar "sqlite locking bug in parser" --limit 5

    \b
    Modifiers (write operations):
        polylogue "urgent" --add-tag review --dry-run
        polylogue -p old --delete --dry-run

    \b
    Subcommands:
        polylogue run       Parse/render/index pipeline
        polylogue check     Health check and repair
        polylogue qa        Composable QA (audit, exercises, invariants)
        polylogue generate  Synthetic data generation
        polylogue embed     Generate vector embeddings
        polylogue tags      List tags with counts
        polylogue site      Build static HTML archive
        polylogue sources   List configured sources
        polylogue schema    Schema generation, package versioning, and evidence
        polylogue mcp       Start MCP server

    Run `polylogue <command> --help` for subcommand details.
    """
    # Set up logging early so all output goes to stderr
    configure_logging(verbose=verbose)

    # Set up environment
    use_plain = should_use_plain(plain=plain)
    env = AppEnv(ui=create_ui(use_plain))
    ctx.obj = env

    # Announce plain mode if auto-detected (not explicitly requested)
    if use_plain and not plain and not plain_forced_by_env():
        announce_plain_mode()


# Register subcommands
cli.add_command(run_command)
cli.add_command(sources_command)
cli.add_command(check_command)
cli.add_command(reset_command)
cli.add_command(mcp_command)
cli.add_command(auth_command)
cli.add_command(completions_command)
cli.add_command(dashboard_command)
cli.add_command(embed_command)
cli.add_command(site_command)
cli.add_command(tags_command)
cli.add_command(generate_command)
cli.add_command(qa_command)
cli.add_command(schema_command)


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
