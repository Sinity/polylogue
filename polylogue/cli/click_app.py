"""CLI entrypoint (query-first design with subcommands).

The CLI uses a hybrid structure:
- Positional args without subcommand prefix → query mode
- Known subcommand prefixes (sync, check, mcp, etc.) → subcommand mode
- No args → stats mode
"""

from __future__ import annotations

import os
from typing import Any

import click

from polylogue.cli.commands.auth import auth_command
from polylogue.cli.commands.check import check_command
from polylogue.cli.commands.completions import completions_command
from polylogue.cli.commands.dashboard import dashboard_command
from polylogue.cli.commands.demo import demo_command
from polylogue.cli.commands.embed import embed_command
from polylogue.cli.commands.mcp import mcp_command
from polylogue.cli.commands.reset import reset_command
from polylogue.cli.commands.run import run_command, sources_command
from polylogue.cli.commands.site import site_command
from polylogue.cli.commands.tags import tags_command
from polylogue.cli.formatting import announce_plain_mode, should_use_plain
from polylogue.cli.types import AppEnv
from polylogue.lib.log import configure_logging
from polylogue.ui import create_ui
from polylogue.version import POLYLOGUE_VERSION


class QueryFirstGroup(click.Group):
    """Custom Click group that routes to query mode by default.

    This group treats positional args as query terms unless they match
    a known subcommand.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse args, converting positional args to --query-term options.

        This allows subcommands to work normally while also supporting
        query mode with positional args.
        """
        # Check if first non-option arg is a subcommand
        first_arg_idx = None
        for i, arg in enumerate(args):
            if not arg.startswith("-"):
                first_arg_idx = i
                break

        # If first positional is a subcommand, let Click handle normally
        # Store flag for invoke() to check (ctx.args is empty after subcommand parsing)
        if first_arg_idx is not None and args[first_arg_idx] in self.commands:
            ctx.ensure_object(dict)
            ctx.obj["_has_subcommand"] = True
            return super().parse_args(ctx, args)

        # Build value-taking options dynamically from Click params
        # Maps option name → nargs (how many values it consumes)
        value_options: dict[str, int] = {}
        for param in self.params:
            if isinstance(param, click.Option) and not param.is_flag:
                nargs = param.nargs if param.nargs > 0 else 1
                for opt in param.opts + param.secondary_opts:
                    value_options[opt] = nargs

        # Convert positional args to --query-term options
        # This allows the main command to receive them as options
        new_args = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("-"):
                new_args.append(arg)
                nargs = value_options.get(arg, 0)
                for _ in range(nargs):
                    if i + 1 < len(args):
                        i += 1
                        new_args.append(args[i])
            else:
                # Positional arg = query term
                new_args.extend(["--query-term", arg])
            i += 1

        return list(super().parse_args(ctx, new_args))

    def invoke(self, ctx: click.Context) -> Any:
        """Invoke the group, dispatching to query or stats mode if no subcommand."""
        # Check if parse_args detected a subcommand (flag set during parsing)
        # We can't use ctx.args here because Click empties it after parsing subcommands
        ctx.ensure_object(dict)
        has_subcommand = ctx.obj.get("_has_subcommand", False)

        if has_subcommand:
            # Let Click handle subcommand dispatch normally
            return super().invoke(ctx)

        # No subcommand: run the callback to set up ctx.obj, then handle query mode
        # Call the callback manually
        assert self.callback is not None, "QueryFirstGroup requires a callback"
        with ctx:
            ctx.invoke(self.callback, **ctx.params)

        _handle_query_mode(ctx)


def _handle_query_mode(ctx: click.Context) -> None:
    """Handle query mode: display stats or perform search."""
    from polylogue.cli.query import execute_query

    env: AppEnv = ctx.obj
    params = ctx.params

    # Extract query-related params
    query_terms = params.get("query_term", ())
    has_filters = any(
        params.get(k)
        for k in (
            "conv_id",
            "contains",
            "exclude_text",
            "provider",
            "exclude_provider",
            "tag",
            "exclude_tag",
            "has_type",
            "since",
            "until",
            "title",
            "latest",
        )
    )

    # Output mode flags that should trigger query execution
    has_output_mode = any(
        params.get(k)
        for k in (
            "list_mode",
            "limit",
            "stats_only",
            "count_only",
            "stream",
            "dialogue_only",
        )
    )

    # Modifier flags that require query execution
    has_modifiers = any(
        params.get(k)
        for k in (
            "add_tag",
            "set_meta",
            "delete_matched",
        )
    )

    # Stats mode: no query terms, no filters, no output mode, and no modifiers
    if not query_terms and not has_filters and not has_output_mode and not has_modifiers:
        _show_stats(env, verbose=params.get("verbose", False))
        return

    # Query mode: execute search
    # Convert query_term tuple to query for execute_query
    params_copy = dict(params)
    params_copy["query"] = query_terms
    execute_query(env, params_copy)


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
# Query terms captured via --query-term (injected by parse_args override)
@click.option("--query-term", multiple=True, hidden=True, help="Query term (internal)")
# --- Filter options ---
@click.option("--id", "-i", "conv_id", help="Conversation ID (exact or prefix match)")
@click.option("--contains", "-c", multiple=True, help="FTS term (repeatable = AND)")
@click.option("--exclude-text", multiple=True, help="Exclude FTS term")
@click.option("--provider", "-p", help="Include providers (comma = OR)")
@click.option("--exclude-provider", help="Exclude providers")
@click.option("--tag", "-t", help="Include tags (comma = OR, supports key:value)")
@click.option("--exclude-tag", help="Exclude tags")
@click.option("--title", help="Title contains")
@click.option("--has", "has_type", multiple=True, help="Has: thinking, tools, summary, attachments")
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
    type=click.Choice(["provider", "month", "year", "day"]),
    help="Aggregate statistics by dimension",
)
@click.option("--open", "open_result", is_flag=True, help="Open result in browser/editor")
@click.option(
    "--transform",
    type=click.Choice(["strip-tools", "strip-thinking", "strip-all"]),
    help="Transform output: strip-tools, strip-thinking, or strip-all",
)
# --- Streaming options (memory-efficient for large conversations) ---
@click.option("--stream", is_flag=True, help="Stream output (low memory). Requires --latest or -i ID")
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
    # Query terms (hidden, injected by parse_args override)
    query_term: tuple[str, ...],
    # Filters
    conv_id: str | None,
    contains: tuple[str, ...],
    exclude_text: tuple[str, ...],
    provider: str | None,
    exclude_provider: str | None,
    tag: str | None,
    exclude_tag: str | None,
    title: str | None,
    has_type: tuple[str, ...],
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
        polylogue -p claude --since "last week"
        polylogue --latest --output browser

    \b
    Combined filters:
        polylogue "error" -p claude --since 2025-01 --list
        polylogue --has thinking --sort tokens --limit 10
        polylogue -t important --stats-by provider

    \b
    Modifiers (write operations):
        polylogue "urgent" --add-tag review --dry-run
        polylogue -p old --delete --dry-run

    \b
    Subcommands:
        polylogue run       Parse/render/index pipeline
        polylogue check     Health check and repair
        polylogue embed     Generate vector embeddings
        polylogue tags      List tags with counts
        polylogue site      Build static HTML archive
        polylogue sources   List configured sources
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
    env_force = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    forced_plain = bool(env_force and env_force.lower() not in {"0", "false", "no"})
    if use_plain and not plain and not forced_plain:
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
cli.add_command(demo_command)


def main() -> None:
    cli()


__all__ = ["cli", "main"]
