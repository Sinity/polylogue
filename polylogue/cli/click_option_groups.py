"""Grouped query-mode option decorators for the root CLI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

from polylogue.lib.query_spec import QUERY_ACTION_TYPES, QUERY_RETRIEVAL_LANES

ClickCallable = Callable[..., Any]

FILTER_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option("--id", "-i", "conv_id", help="Conversation ID (exact or prefix match)"),
    click.option("--contains", "-c", multiple=True, help="FTS term (repeatable = AND)"),
    click.option("--exclude-text", multiple=True, help="Exclude FTS term"),
    click.option(
        "--retrieval-lane",
        type=click.Choice(QUERY_RETRIEVAL_LANES),
        help="Query lane: dialogue FTS, action text, or hybrid",
    ),
    click.option("--provider", "-p", help="Include providers (comma = OR)"),
    click.option("--exclude-provider", help="Exclude providers"),
    click.option("--tag", "-t", help="Include tags (comma = OR, supports key:value)"),
    click.option("--exclude-tag", help="Exclude tags"),
    click.option("--title", help="Title contains"),
    click.option(
        "--path",
        "path_terms",
        multiple=True,
        help="Touched path contains substring (repeatable = AND)",
    ),
    click.option(
        "--action",
        multiple=True,
        type=click.Choice(QUERY_ACTION_TYPES),
        help="Require semantic action category (repeatable = AND)",
    ),
    click.option(
        "--exclude-action",
        multiple=True,
        type=click.Choice(QUERY_ACTION_TYPES),
        help="Exclude semantic action category (repeatable = AND)",
    ),
    click.option(
        "--action-sequence",
        help="Require ordered semantic action subsequence (comma-separated)",
    ),
    click.option(
        "--action-text",
        multiple=True,
        help="Require text within normalized action evidence (repeatable = AND)",
    ),
    click.option("--tool", multiple=True, help="Require normalized tool name (repeatable = AND)"),
    click.option("--exclude-tool", multiple=True, help="Exclude normalized tool name (repeatable = AND)"),
    click.option("--similar", "similar_text", help="Semantic similarity query (requires embeddings)"),
    click.option(
        "--has",
        "has_type",
        multiple=True,
        help="Filter by content: thinking (reasoning), tools (calls), summary, attachments",
    ),
    click.option(
        "--has-tool-use",
        "filter_has_tool_use",
        is_flag=True,
        help="Only conversations with tool use (SQL pushdown)",
    ),
    click.option(
        "--has-thinking",
        "filter_has_thinking",
        is_flag=True,
        help="Only conversations with thinking blocks (SQL pushdown)",
    ),
    click.option("--min-messages", type=int, help="Minimum message count"),
    click.option("--max-messages", type=int, help="Maximum message count"),
    click.option("--min-words", type=int, help="Minimum total word count"),
    click.option("--since", help="After date (ISO, 'yesterday', 'last week')"),
    click.option("--until", help="Before date"),
    click.option("--limit", "-n", type=int, help="Max results"),
    click.option("--latest", is_flag=True, help="Most recent (= --sort date --limit 1)"),
    click.option(
        "--sort",
        type=click.Choice(["date", "tokens", "messages", "words", "longest", "random"]),
        help="Sort by field",
    ),
    click.option("--reverse", is_flag=True, help="Reverse sort order"),
    click.option("--sample", type=int, help="Random sample of N conversations"),
)

OUTPUT_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option(
        "--output",
        "-o",
        help="Output destinations: browser, clipboard, stdout (comma-separated)",
    ),
    click.option(
        "--format",
        "-f",
        "output_format",
        type=click.Choice(
            ["markdown", "json", "html", "obsidian", "org", "yaml", "plaintext", "csv"]
        ),
        help="Output format",
    ),
    click.option(
        "--fields",
        help="Fields for list/json: id, title, provider, date, messages, words, tags, summary",
    ),
    click.option("--list", "list_mode", is_flag=True, help="Force list format"),
    click.option("--stats", "stats_only", is_flag=True, help="Only statistics, no content"),
    click.option("--count", "count_only", is_flag=True, help="Print matched count and exit"),
    click.option(
        "--stats-by",
        "stats_by",
        type=click.Choice(["provider", "month", "year", "day", "action", "tool", "project", "work-kind"]),
        help="Aggregate statistics by dimension",
    ),
    click.option("--open", "open_result", is_flag=True, help="Open result in browser/editor"),
    click.option(
        "--transform",
        type=click.Choice(["strip-tools", "strip-thinking", "strip-all"]),
        help="Remove content: strip-tools (tool calls), strip-thinking (reasoning), strip-all (both)",
    ),
)

STREAMING_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option(
        "--stream",
        is_flag=True,
        help="Stream output (low memory). Requires --latest or -i ID. Incompatible with --transform",
    ),
    click.option("--dialogue-only", "-d", is_flag=True, help="Show only user/assistant messages"),
)

MODIFIER_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option("--set", "set_meta", nargs=2, multiple=True, help="Set metadata key value"),
    click.option("--add-tag", multiple=True, help="Add tags (comma-separated)"),
    click.option("--delete", "delete_matched", is_flag=True, help="Delete matched (requires filter)"),
    click.option("--dry-run", is_flag=True, help="Preview changes without executing"),
    click.option("--force", is_flag=True, help="Skip confirmation for bulk operations"),
)

GLOBAL_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option("--plain", is_flag=True, help="Force non-interactive plain output"),
    click.option("-v", "--verbose", is_flag=True, help="Verbose output"),
)


def _apply_option_group(
    func: ClickCallable,
    decorators: tuple[Callable[[ClickCallable], ClickCallable], ...],
) -> ClickCallable:
    for decorator in reversed(decorators):
        func = decorator(func)
    return func


def apply_query_mode_options(func: ClickCallable) -> ClickCallable:
    """Apply the grouped root query-mode options in their canonical order."""
    for decorators in reversed((
        FILTER_OPTION_DECORATORS,
        OUTPUT_OPTION_DECORATORS,
        STREAMING_OPTION_DECORATORS,
        MODIFIER_OPTION_DECORATORS,
        GLOBAL_OPTION_DECORATORS,
    )):
        func = _apply_option_group(func, decorators)
    return func


__all__ = [
    "FILTER_OPTION_DECORATORS",
    "GLOBAL_OPTION_DECORATORS",
    "MODIFIER_OPTION_DECORATORS",
    "OUTPUT_OPTION_DECORATORS",
    "STREAMING_OPTION_DECORATORS",
    "apply_query_mode_options",
]
