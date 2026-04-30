"""Grouped query-mode option decorators for the root CLI."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import click

from polylogue.cli.query_contracts import normalize_message_role_option
from polylogue.cli.shell_completion_values import (
    complete_conversation_ids,
    complete_provider_values,
    complete_tag_values,
    complete_tool_values,
)
from polylogue.lib.provider_identity import CORE_SCHEMA_PROVIDERS
from polylogue.lib.query.spec import QUERY_ACTION_TYPES, QUERY_RETRIEVAL_LANES

ClickCallable: TypeAlias = Callable[..., object]

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
    value: str | None,
) -> str | None:
    if not value:
        return None
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    bad = [t for t in tokens if t not in _CLI_PROVIDER_CHOICES]
    if bad:
        param_name = f"--{param.name.replace(chr(95), chr(45))}" if param.name else "--provider"
        raise click.BadParameter(
            f"Unknown provider(s): {', '.join(bad)}. Valid: {', '.join(_CLI_PROVIDER_CHOICES)}",
            param_hint=param_name,
        )
    return value


def _validate_message_role_tokens(
    ctx: click.Context,
    _param: click.Parameter,
    value: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        return normalize_message_role_option(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--message-role") from exc


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
    click.option(
        "--repo",
        "-r",
        help="Filter by repository name (comma = OR)",
        shell_complete=complete_tag_values,
    ),
    click.option(
        "--tag", "-t", help="Include tags (comma = OR, supports key:value)", shell_complete=complete_tag_values
    ),
    click.option("--exclude-tag", help="Exclude tags", shell_complete=complete_tag_values),
    click.option("--title", help="Title contains"),
    click.option(
        "--referenced-path",
        "referenced_path",
        multiple=True,
        help="Referenced file path contains substring (repeatable = AND)",
    ),
    click.option(
        "--cwd-prefix",
        "cwd_prefix",
        default=None,
        help="Filter conversations whose recorded working directory starts with this prefix",
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
    click.option(
        "--tool",
        multiple=True,
        help="Require normalized tool name (repeatable = AND)",
        shell_complete=complete_tool_values,
    ),
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
    click.option(
        "--has-paste",
        "filter_has_paste",
        is_flag=True,
        help="Only conversations with pasted content (SQL pushdown)",
    ),
    click.option(
        "--typed-only",
        "typed_only",
        is_flag=True,
        help="Only conversations without pasted content (typed prose only)",
    ),
    click.option("--min-messages", type=int, help="Minimum message count"),
    click.option("--max-messages", type=int, help="Maximum message count"),
    click.option("--min-words", type=int, help="Minimum total word count"),
    click.option(
        "--message-type",
        "message_type",
        type=click.Choice(["summary", "tool_use", "tool_result", "thinking"]),
        help="Filter by message content type (summary, tool_use, tool_result, thinking)",
    ),
    click.option(
        "--since-session",
        "since_session_id",
        help="Show sessions in same cwd after this conversation ID",
    ),
    click.option("--since", help="After date (ISO, 'yesterday', 'last week')"),
    click.option("--until", help="Before date"),
    click.option("--limit", "-n", type=int, help="Max results"),
    click.option("--offset", type=int, default=0, help="Offset for paginated results"),
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
    click.option("--no-code-blocks", is_flag=True, help="Exclude fenced and structured code blocks from output"),
    click.option("--no-tool-calls", is_flag=True, help="Exclude tool invocation records from output"),
    click.option("--no-tool-outputs", is_flag=True, help="Exclude tool-result payloads from output"),
    click.option("--no-file-reads", is_flag=True, help="Exclude file-read payloads while keeping other tool output"),
    click.option("--prose-only", is_flag=True, help="Show only authored prose text"),
)

STREAMING_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option(
        "--stream",
        is_flag=True,
        help="Stream output (low memory). Requires --latest or -i ID. Incompatible with --transform",
    ),
    click.option("--dialogue-only", "-d", is_flag=True, help="Show only user/assistant messages"),
    click.option(
        "--message-role",
        "message_role",
        multiple=True,
        callback=_validate_message_role_tokens,
        help="Show only selected message roles (repeatable or comma-separated: user, assistant, system, tool, unknown)",
    ),
)

MODIFIER_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option("--set", "set_meta", nargs=2, multiple=True, help="Set metadata key value"),
    click.option("--add-tag", multiple=True, help="Add tags (comma-separated)"),
)

GLOBAL_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option("--tail", is_flag=True, help="Tail ahead-of-archive Claude Code source state during queries"),
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
