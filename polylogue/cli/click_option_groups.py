"""Grouped query-mode option decorators for the root CLI.

Heavy imports (archive.message.types, archive.query.spec, query_contracts,
shell_completion_values) are deferred inside the functions that need them
so that ``--help`` and simple subcommands never pay the storage import cost.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import click

from polylogue.core.sources import CORE_SCHEMA_ORIGINS

ClickCallable: TypeAlias = Callable[..., object]

# Origins the user can filter by (excludes the internal "unknown-export").
_CLI_ORIGIN_CHOICES: tuple[str, ...] = CORE_SCHEMA_ORIGINS


def _lazy_shell_complete(source: str) -> Callable[..., object]:
    """Return a shell-completion callback that imports the completion machinery lazily."""

    def _complete(ctx: click.Context, param: click.Parameter, incomplete: str):  # type: ignore[no-untyped-def]
        from polylogue.cli.shell_completion_values import complete_query_source

        return complete_query_source(source)(ctx, param, incomplete)  # type: ignore[arg-type]

    _complete.__name__ = f"complete_{source}"
    return _complete


_complete_action = _lazy_shell_complete("action")
_complete_action_sequence = _lazy_shell_complete("action_sequence")
_complete_session_id = _lazy_shell_complete("session_id")
_complete_cwd_prefix = _lazy_shell_complete("cwd_prefix")
_complete_message_type = _lazy_shell_complete("message_type")
_complete_origin = _lazy_shell_complete("origin")
_complete_repo = _lazy_shell_complete("repo")
_complete_retrieval_lane = _lazy_shell_complete("retrieval_lane")
_complete_tag = _lazy_shell_complete("tag")
_complete_tool = _lazy_shell_complete("tool")


class _LazyChoice(click.Choice):  # type: ignore[type-arg]
    """Click Choice that resolves its options list on first access."""

    def __init__(self, loader: Callable[[], list[str]], name: str = "") -> None:
        super().__init__([name])  # placeholder — replaced on first resolve
        self._loader = loader
        self._name = name
        self._resolved = False

    def _resolve(self) -> None:
        if not self._resolved:
            self.choices = self._loader()
            self._resolved = True

    def convert(self, value: object, param: click.Parameter | None, ctx: click.Context | None) -> object:
        self._resolve()
        return super().convert(value, param, ctx)

    def get_metavar(self, param: click.Parameter, ctx: click.Context | None = None) -> str:
        if self._name:
            return self._name.upper()
        return "TEXT"


def _load_message_types() -> list[str]:
    from polylogue.archive.message.types import MessageType

    return [m.value for m in MessageType]


def _load_retrieval_lanes() -> list[str]:
    from polylogue.archive.query.spec import QUERY_RETRIEVAL_LANES

    return list(QUERY_RETRIEVAL_LANES)


def _load_action_types() -> list[str]:
    from polylogue.archive.query.spec import QUERY_ACTION_TYPES

    return list(QUERY_ACTION_TYPES)


def _validate_origin_tokens(
    ctx: click.Context,
    param: click.Parameter,
    value: str | None,
) -> str | None:
    if not value:
        return None
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    bad = [t for t in tokens if t not in _CLI_ORIGIN_CHOICES]
    if bad:
        param_name = f"--{param.name.replace(chr(95), chr(45))}" if param.name else "--origin"
        raise click.BadParameter(
            f"Unknown origin(s): {', '.join(bad)}. Valid: {', '.join(_CLI_ORIGIN_CHOICES)}",
            param_hint=param_name,
        )
    return value


def _validate_message_role_tokens(
    ctx: click.Context,
    _param: click.Parameter,
    value: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        from polylogue.cli.query_contracts import normalize_message_role_option

        return normalize_message_role_option(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--message-role") from exc


FILTER_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option(
        "--id",
        "-i",
        "conv_id",
        help="Session ID (exact or prefix match)",
        shell_complete=_complete_session_id,
    ),
    click.option("--contains", "-c", multiple=True, help="FTS term (repeatable = AND)"),
    click.option("--exclude-text", multiple=True, help="Exclude FTS term"),
    click.option(
        "--retrieval-lane",
        type=_LazyChoice(_load_retrieval_lanes, "lane"),
        help="Query lane: dialogue FTS, action text, or hybrid",
        shell_complete=_complete_retrieval_lane,
    ),
    click.option(
        "--lexical",
        "lexical",
        is_flag=True,
        default=False,
        help="Force FTS-only search (overrides hybrid auto-elevation when embeddings are populated).",
    ),
    click.option(
        "--semantic",
        "semantic",
        is_flag=True,
        default=False,
        help="Treat the query as a similarity prompt (vector-only; requires embeddings).",
    ),
    click.option(
        "--origin",
        help="Include origins (comma = OR)",
        callback=_validate_origin_tokens,
        shell_complete=_complete_origin,
    ),
    click.option(
        "--exclude-origin",
        help="Exclude origins",
        callback=_validate_origin_tokens,
        shell_complete=_complete_origin,
    ),
    click.option(
        "--repo",
        "-r",
        help="Filter by repository name (comma = OR)",
        shell_complete=_complete_repo,
    ),
    click.option("--tag", "-t", help="Include tags (comma = OR, supports key:value)", shell_complete=_complete_tag),
    click.option("--exclude-tag", help="Exclude tags", shell_complete=_complete_tag),
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
        help="Filter sessions whose recorded working directory starts with this prefix",
        shell_complete=_complete_cwd_prefix,
    ),
    click.option(
        "--action",
        multiple=True,
        type=_LazyChoice(_load_action_types, "action"),
        help="Require semantic action category (repeatable = AND)",
        shell_complete=_complete_action,
    ),
    click.option(
        "--exclude-action",
        multiple=True,
        type=_LazyChoice(_load_action_types, "action"),
        help="Exclude semantic action category (repeatable = AND)",
        shell_complete=_complete_action,
    ),
    click.option(
        "--action-sequence",
        help="Require ordered semantic action subsequence (comma-separated)",
        shell_complete=_complete_action_sequence,
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
        shell_complete=_complete_tool,
    ),
    click.option(
        "--exclude-tool",
        multiple=True,
        help="Exclude normalized tool name (repeatable = AND)",
        shell_complete=_complete_tool,
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
        help="Only sessions with tool use (SQL pushdown)",
    ),
    click.option(
        "--has-thinking",
        "filter_has_thinking",
        is_flag=True,
        help="Only sessions with thinking blocks (SQL pushdown)",
    ),
    click.option(
        "--has-paste",
        "filter_has_paste",
        is_flag=True,
        help="Only sessions with pasted content (SQL pushdown)",
    ),
    click.option(
        "--typed-only",
        "typed_only",
        is_flag=True,
        help="Only sessions without pasted content (typed prose only)",
    ),
    click.option("--min-messages", type=int, help="Minimum message count"),
    click.option("--max-messages", type=int, help="Maximum message count"),
    click.option("--min-words", type=int, help="Minimum total word count"),
    click.option(
        "--message-type",
        "message_type",
        type=_LazyChoice(_load_message_types, "type"),
        help="Filter by message content type (message, summary, tool_use, tool_result, thinking, context, protocol)",
        shell_complete=_complete_message_type,
    ),
    click.option(
        "--since-session",
        "since_session_id",
        help="Show sessions in same cwd after this session ID",
    ),
    click.option("--since", help="After date (ISO, 'yesterday', 'last week')"),
    click.option("--until", help="Before date"),
    click.option("--limit", "-l", "-n", type=int, help="Max results"),
    click.option("--offset", type=int, default=0, help="Offset for paginated results"),
    click.option(
        "--cursor",
        "cursor",
        type=str,
        default=None,
        help=(
            "Opaque keyset cursor from a previous response's next_cursor. "
            "Stable across archive growth for ranked search (#1268)."
        ),
    ),
    click.option("--latest", is_flag=True, help="Most recent (= --sort date --limit 1)"),
    click.option(
        "--sort",
        type=click.Choice(["date", "tokens", "messages", "words", "longest", "random"]),
        help="Sort by field",
    ),
    click.option("--reverse", is_flag=True, help="Reverse sort order"),
    click.option("--sample", type=int, help="Random sample of N sessions"),
)

OUTPUT_OPTION_DECORATORS: tuple[Callable[[ClickCallable], ClickCallable], ...] = (
    click.option(
        "--output",
        "-o",
        help="Output destinations: browser, clipboard, stdout (comma-separated)",
    ),
    click.option(
        "--json",
        "output_as_json",
        is_flag=True,
        default=False,
        help="Shortcut for --format json. Disables color and progress for pipeable output. (#1689)",
    ),
    click.option(
        "--format",
        "-f",
        "output_format",
        type=click.Choice(["markdown", "json", "ndjson", "html", "obsidian", "org", "yaml", "plaintext", "csv"]),
        help=(
            "Output format (for --latest, --stream, or verb output). "
            "`ndjson` emits one JSON document per line, streaming-friendly for shell "
            "pipelines and LLM tool-use harnesses."
        ),
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
    click.option("--plain", is_flag=True, help="Force non-interactive plain output"),
    click.option("-v", "--verbose", is_flag=True, help="Verbose output"),
    click.option(
        "--diagnose",
        is_flag=True,
        default=False,
        help=(
            "Explain CLI parser decisions on stderr before running. "
            "Useful when query-first dispatch surprises you: shows whether "
            "a bare token was routed to a subcommand or interpreted as a search query."
        ),
    ),
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
