"""CLI entrypoint (query-first design with subcommands).

The CLI uses a hybrid structure:
- Positional args without subcommand prefix → query mode
- Known subcommand prefixes (sync, check, mcp, etc.) → subcommand mode
- No args → stats mode
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click

from polylogue.cli.commands.auth import auth_command
from polylogue.cli.commands.check import check_command
from polylogue.cli.commands.completions import completions_command
from polylogue.cli.commands.config import config_command
from polylogue.cli.commands.mcp import mcp_command
from polylogue.cli.commands.reset import reset_command
from polylogue.cli.commands.serve import serve_command
from polylogue.cli.commands.sync import sources_command, sync_command
from polylogue.cli.formatting import announce_plain_mode, should_use_plain
from polylogue.cli.types import AppEnv
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

        # Otherwise, convert positional args to --query-term options
        # This allows the main command to receive them as options
        new_args = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("-"):
                new_args.append(arg)
                # Check if this option takes a value
                if arg in (
                    "-c", "--contains", "-C", "--no-contains",
                    "--regex", "--no-regex",
                    "-p", "--provider", "-P", "--no-provider",
                    "-t", "--tag", "-T", "--no-tag",
                    "--title", "--has", "--no-has",
                    "--since", "--until",
                    "-i", "--id", "-n", "--limit",
                    "--sort", "--sample", "--similar",
                    "-o", "--output", "-f", "--format",
                    "--fields", "--set", "--unset",
                    "--add-tag", "--rm-tag", "--annotate",
                    "--config",
                ):
                    if i + 1 < len(args) and not args[i + 1].startswith("-"):
                        i += 1
                        new_args.append(args[i])
            else:
                # Positional arg = query term
                new_args.extend(["--query-term", arg])
            i += 1

        return super().parse_args(ctx, new_args)

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

        return _handle_query_mode(ctx)


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
            "contains",
            "no_contains",
            "regex",
            "no_regex",
            "provider",
            "no_provider",
            "tag",
            "no_tag",
            "has_type",
            "no_has",
            "since",
            "until",
            "title",
            "id_prefix",
            "latest",
            "similar",
        )
    )

    # Stats mode: no query terms and no filters
    if not query_terms and not has_filters:
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
@click.option("--contains", "-c", multiple=True, help="FTS term (repeatable = AND)")
@click.option("--no-contains", "-C", multiple=True, help="Exclude FTS term")
@click.option("--regex", multiple=True, help="Regex pattern")
@click.option("--no-regex", multiple=True, help="Exclude regex pattern")
@click.option("--provider", "-p", help="Include providers (comma = OR)")
@click.option("--no-provider", "-P", help="Exclude providers")
@click.option("--tag", "-t", help="Include tags (comma = OR, supports key:value)")
@click.option("--no-tag", "-T", help="Exclude tags")
@click.option("--title", help="Title contains")
@click.option(
    "--has", "has_type", multiple=True, help="Has: thinking, tools, summary, attachments"
)
@click.option("--no-has", multiple=True, help="Missing types")
@click.option("--since", help="After date (ISO, 'yesterday', 'last week')")
@click.option("--until", help="Before date")
@click.option("--id", "-i", "id_prefix", help="ID prefix match")
@click.option("--limit", "-n", type=int, help="Max results")
@click.option("--latest", is_flag=True, help="Most recent (= --sort date --limit 1)")
@click.option(
    "--sort",
    type=click.Choice(["date", "tokens", "messages", "words", "longest", "random"]),
    help="Sort by field",
)
@click.option("--reverse", is_flag=True, help="Reverse sort order")
@click.option("--sample", type=int, help="Random sample of N conversations")
@click.option("--similar", help="Rank by embedding similarity")
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
    type=click.Choice(["markdown", "json", "html", "obsidian", "org"]),
    help="Output format",
)
@click.option("--fields", help="Select fields for list/json output (comma-separated)")
@click.option("--list", "list_mode", is_flag=True, help="Force list format")
@click.option("--stats", "stats_only", is_flag=True, help="Only statistics, no content")
@click.option("--pick", is_flag=True, help="Interactive picker (uses fzf if available)")
@click.option("--by-month", is_flag=True, help="Aggregate by month")
@click.option("--by-provider", is_flag=True, help="Aggregate by provider")
@click.option("--by-tag", is_flag=True, help="Aggregate by tag")
@click.option("--open", "open_result", is_flag=True, help="Open result in browser/editor")
@click.option("--csv", "csv_path", type=click.Path(path_type=Path), help="Write CSV to file")
# --- Modifier options (write operations) ---
@click.option("--set", "set_meta", nargs=2, multiple=True, help="Set metadata key value")
@click.option("--unset", multiple=True, help="Remove metadata key")
@click.option("--add-tag", multiple=True, help="Add tags (comma-separated)")
@click.option("--rm-tag", multiple=True, help="Remove tags (comma-separated)")
@click.option("--annotate", help="LLM annotation prompt")
@click.option("--delete", "delete_matched", is_flag=True, help="Delete matched (requires filter)")
# --- Global options ---
@click.option("--plain", is_flag=True, help="Force non-interactive plain output")
@click.option("--interactive", is_flag=True, help="Force interactive output")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path), help="Path to config.json"
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.version_option(version=POLYLOGUE_VERSION, prog_name="polylogue")
@click.pass_context
def cli(
    ctx: click.Context,
    # Query terms (hidden, injected by parse_args override)
    query_term: tuple[str, ...],
    # Filters
    contains: tuple[str, ...],
    no_contains: tuple[str, ...],
    regex: tuple[str, ...],
    no_regex: tuple[str, ...],
    provider: str | None,
    no_provider: str | None,
    tag: str | None,
    no_tag: str | None,
    title: str | None,
    has_type: tuple[str, ...],
    no_has: tuple[str, ...],
    since: str | None,
    until: str | None,
    id_prefix: str | None,
    limit: int | None,
    latest: bool,
    sort: str | None,
    reverse: bool,
    sample: int | None,
    similar: str | None,
    # Output
    output: str | None,
    output_format: str | None,
    fields: str | None,
    list_mode: bool,
    stats_only: bool,
    pick: bool,
    by_month: bool,
    by_provider: bool,
    by_tag: bool,
    open_result: bool,
    csv_path: Path | None,
    # Modifiers
    set_meta: tuple[tuple[str, str], ...],
    unset: tuple[str, ...],
    add_tag: tuple[str, ...],
    rm_tag: tuple[str, ...],
    annotate: str | None,
    delete_matched: bool,
    # Global
    plain: bool,
    interactive: bool,
    config_path: Path | None,
    verbose: bool,
) -> None:
    """Polylogue - AI conversation archive.

    \b
    Query mode (default):
        polylogue "search terms"
        polylogue -p claude --since "last week"
        polylogue --latest --output browser

    \b
    Subcommands:
        polylogue sync    Sync sources to database
        polylogue check   Health check and repair
        polylogue mcp     Start MCP server
        polylogue reset   Reset database
        polylogue config  Configuration

    Run `polylogue <command> --help` for subcommand details.
    """
    # Set up environment
    use_plain = should_use_plain(plain=plain, interactive=interactive)
    env = AppEnv(ui=create_ui(use_plain), config_path=config_path)
    ctx.obj = env

    # Announce plain mode if auto-detected (not explicitly requested)
    env_force = os.environ.get("POLYLOGUE_FORCE_PLAIN")
    forced_plain = bool(env_force and env_force.lower() not in {"0", "false", "no"})
    if use_plain and not plain and not interactive and not forced_plain:
        announce_plain_mode()


# Register subcommands
cli.add_command(sync_command)
cli.add_command(sources_command)
cli.add_command(check_command)
cli.add_command(reset_command)
cli.add_command(mcp_command)
cli.add_command(auth_command)
cli.add_command(completions_command)
cli.add_command(config_command)
cli.add_command(serve_command)


def main() -> None:
    cli()


__all__ = ["cli", "main"]
