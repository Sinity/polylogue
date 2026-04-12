"""Grouped query-mode option decorators for the root CLI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

from polylogue.cli.shell_completion_values import (
    complete_conversation_ids,
    complete_provider_values,
    complete_tag_values,
    complete_tool_values,
)
from polylogue.lib.provider_identity import CORE_SCHEMA_PROVIDERS
from polylogue.lib.query_spec import QUERY_ACTION_TYPES, QUERY_RETRIEVAL_LANES

ClickCallable = Callable[..., Any]

# Providers the user can filter by (excludes "unknown" and "drive" which are internal).
_CLI_PROVIDER_CHOICES: tuple[str, ...] = CORE_SCHEMA_PROVIDERS


def _complete_providers(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[click.shell_completion.CompletionItem]:
    return complete_provider_values(ctx, param, incomplete)


def _validate_provider_tokens(
    ctx: click.Context,
    param: click.Parameter,
    value: str | None,  # noqa: ARG001
) -> str | None:
    if not value:
        return value
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    bad = [t for t in tokens if t not in _CLI_PROVIDER_CHOICES]
    if bad:
        raise click.BadParameter(
            f"Unknown provider(s): {', '.join(bad)}. Valid: {', '.join(_CLI_PROVIDER_CHOICES)}",
            param_hint="--provider",
        )
    return value


FILTER_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option(
        "--id",
        "-i",
        "conv_id",
        help="Conversation ID (exact or prefix match)",
        shell_complete=complete_conversation_ids,
    ),
    click.option("--contains", "-c", multiple=True, help="FTS term (repeatable = AND)"),
    click.option("--exclude-text", multiple=True, help="Exclude FTS term"),
    click.option(
        "--retrieval-lane",
        type=click.Choice(QUERY_RETRIEVAL_LANES),
        help="Query lane: dialogue FTS, action text, or hybrid",
    ),
    click.option(
        "--provider",
        "-p",
        help="Include providers (comma = OR)",
        callback=_validate_provider_tokens,
        shell_complete=_complete_providers,
    ),
    click.option(
        "--exclude-provider",
        help="Exclude providers",
        callback=_validate_provider_tokens,
        shell_complete=_complete_providers,
    ),
    click.option("--tag", "-t", help="Include tags (comma = OR, supports key:value)", shell_complete=complete_tag_values),
    click.option("--exclude-tag", help="Exclude tags", shell_complete=complete_tag_values),
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
    click.option("--tool", multiple=True, help="Require normalized tool name (repeatable = AND)", shell_complete=complete_tool_values),
    click.option(
        "--exclude-tool",
        multiple=True,
        help="Exclude normalized tool name (repeatable = AND)",
        shell_complete=complete_tool_values,
    ),
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
        type=click.Choice(["markdown", "json", "html", "obsidian", "org", "yaml", "plaintext", "csv"]),
        help="Output format (for --latest, --stream, or verb output)",
    ),
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
    for decorators in reversed(
        (
            FILTER_OPTION_DECORATORS,
            OUTPUT_OPTION_DECORATORS,
            STREAMING_OPTION_DECORATORS,
            MODIFIER_OPTION_DECORATORS,
            GLOBAL_OPTION_DECORATORS,
        )
    ):
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
