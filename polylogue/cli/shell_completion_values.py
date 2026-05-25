"""Shell-completion helpers for archive-backed CLI values.

All archive-backed completions route through ``ArchiveOperations`` —
conversation ID, tag, repo-name, cwd-prefix, and tool-name aggregates
are exposed as typed methods on the operation layer, so this surface
no longer reaches into ``session_profiles`` / ``action_events`` via
raw SQL (closes #860 inventory item).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import click
from click.shell_completion import CompletionItem

from polylogue.archive.message.types import MessageType
from polylogue.archive.query.fields import CompletionSource
from polylogue.archive.query.spec import QUERY_ACTION_TYPES, QUERY_RETRIEVAL_LANES, QUERY_SEQUENCE_ACTION_TYPES
from polylogue.operations.archive import ArchiveOperations as _ArchiveOperations
from polylogue.operations.archive import CompletionAggregate
from polylogue.paths import db_path

Callback = Callable[[_ArchiveOperations], Awaitable[list[CompletionItem]]]

_PROVIDER_DESCRIPTIONS: Final[dict[str, str]] = {
    "chatgpt": "ChatGPT web exports (lab: OpenAI)",
    "claude-ai": "Claude web exports (lab: Anthropic)",
    "claude-code": "Claude Code local sessions (source: claude-code)",
    "codex": "Codex CLI local sessions (source: codex)",
    "gemini": "Google AI Studio exports (source: aistudio, lab: Google)",
    "gemini-cli": "Gemini CLI local sessions (source: gemini-cli, lab: Google)",
    "hermes": "Hermes agent sessions (source: hermes)",
    "antigravity": "Antigravity brain artifacts (source: antigravity)",
}

_MAX_ID_COMPLETIONS = 24
_MAX_VALUE_COMPLETIONS = 32
CompletionCallback = Callable[[click.Context, click.Parameter, str], list[CompletionItem]]


def _split_csv_incomplete(incomplete: str) -> tuple[str, str]:
    if "," not in incomplete:
        return "", incomplete.strip()
    parts = incomplete.split(",")
    prefix_parts = [part.strip() for part in parts[:-1] if part.strip()]
    prefix = ",".join(prefix_parts)
    if prefix:
        prefix += ","
    return prefix, parts[-1].strip()


def _with_csv_prefix(items: list[CompletionItem], prefix: str) -> list[CompletionItem]:
    if not prefix:
        return items
    return [CompletionItem(f"{prefix}{item.value}", type=item.type, help=item.help) for item in items]


def _db_exists() -> bool:
    return db_path().exists()


async def _with_operations(
    action: Callback,
) -> list[CompletionItem]:
    """Create ArchiveOperations, run *action*, and close services."""
    from polylogue.operations.archive import ArchiveOperations
    from polylogue.services import build_runtime_services

    services = build_runtime_services()
    try:
        operations = ArchiveOperations.from_services(services)
        return await action(operations)
    finally:
        await services.close()


def _run_completion(action: Callback) -> list[CompletionItem]:
    """Bridge an async completion action into the synchronous Click callback world."""
    try:
        return list(asyncio.run(_with_operations(action)))
    except Exception:
        return []


def _aggregates_to_items(
    aggregates: list[CompletionAggregate],
    *,
    unit: str,
) -> list[CompletionItem]:
    return [CompletionItem(agg.value, help=f"{agg.count} {unit}") for agg in aggregates]


def _trim_help(value: str, *, limit: int = 72) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + chr(0x2026)


def _static_completion_items(
    values: tuple[str, ...],
    incomplete: str,
    *,
    csv: bool = False,
) -> list[CompletionItem]:
    prefix, current = _split_csv_incomplete(incomplete) if csv else ("", incomplete.strip())
    current_lower = current.lower()
    items = [CompletionItem(value) for value in values if not current_lower or value.lower().startswith(current_lower)]
    return _with_csv_prefix(items, prefix) if csv else items


def complete_provider_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    current_lower = current.lower()
    items = [
        CompletionItem(name, help=_PROVIDER_DESCRIPTIONS.get(name))
        for name in _PROVIDER_DESCRIPTIONS
        if not current_lower or name.startswith(current_lower)
    ]
    return _with_csv_prefix(items, prefix)


def complete_action_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(QUERY_ACTION_TYPES, incomplete)


def complete_action_sequence_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(QUERY_SEQUENCE_ACTION_TYPES, incomplete, csv=True)


def complete_message_type_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(tuple(message_type.value for message_type in MessageType), incomplete)


def complete_retrieval_lane_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    return _static_completion_items(QUERY_RETRIEVAL_LANES, incomplete)


def complete_conversation_ids(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    current = incomplete.strip()
    current_lower = current.lower()

    if not _db_exists():
        return []

    async def _query(ops: _ArchiveOperations) -> list[CompletionItem]:
        conversations = await ops.list_conversations(limit=100)
        items: list[CompletionItem] = []
        for conv in conversations:
            cid = str(conv.id)
            title = conv.title or ""
            source_name = str(conv.provider)
            display = title or cid
            if current and not (
                cid.startswith(current) or (":" in cid and current in cid) or (title and current_lower in title.lower())
            ):
                continue
            items.append(
                CompletionItem(
                    cid,
                    help=f"{source_name} \u00b7 {_trim_help(display)}",
                )
            )
        items.sort(key=lambda item: item.value)
        return items[:_MAX_ID_COMPLETIONS]

    return _run_completion(_query)


def complete_open_targets(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    return complete_conversation_ids(ctx, param, incomplete)


def complete_tag_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    current_lower = current.lower()

    if not _db_exists():
        return []

    async def _query(ops: _ArchiveOperations) -> list[CompletionItem]:
        tags = await ops.list_tags()
        sorted_tags = sorted(tags.items(), key=lambda x: (-x[1], x[0]))
        items: list[CompletionItem] = []
        for name, cnt in sorted_tags:
            if current_lower and not name.lower().startswith(current_lower):
                continue
            items.append(CompletionItem(name, help=f"{cnt} conversations"))
            if len(items) >= _MAX_VALUE_COMPLETIONS:
                break
        return items

    items = _run_completion(_query)
    return _with_csv_prefix(items, prefix)


def complete_repo_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    if not _db_exists():
        return []

    async def _query(ops: _ArchiveOperations) -> list[CompletionItem]:
        aggregates = await ops.list_session_repo_names(prefix=current, limit=_MAX_VALUE_COMPLETIONS)
        return _aggregates_to_items(aggregates, unit="sessions")

    items = _run_completion(_query)
    return _with_csv_prefix(items, prefix)


def complete_cwd_prefix_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    current = incomplete.strip()
    if not _db_exists():
        return []

    async def _query(ops: _ArchiveOperations) -> list[CompletionItem]:
        aggregates = await ops.list_session_cwd_prefixes(prefix=current, limit=_MAX_VALUE_COMPLETIONS)
        return _aggregates_to_items(aggregates, unit="sessions")

    return _run_completion(_query)


def complete_tool_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    current = incomplete.strip().lower()
    if not _db_exists():
        return []

    async def _query(ops: _ArchiveOperations) -> list[CompletionItem]:
        aggregates = await ops.list_action_tool_names(prefix=current, limit=_MAX_VALUE_COMPLETIONS)
        return _aggregates_to_items(aggregates, unit="actions")

    return _run_completion(_query)


COMPLETION_SOURCE_HANDLERS: Final[Mapping[CompletionSource, CompletionCallback]] = {
    "action": complete_action_values,
    "action_sequence": complete_action_sequence_values,
    "conversation_id": complete_conversation_ids,
    "cwd_prefix": complete_cwd_prefix_values,
    "message_type": complete_message_type_values,
    "provider": complete_provider_values,
    "repo": complete_repo_values,
    "retrieval_lane": complete_retrieval_lane_values,
    "tag": complete_tag_values,
    "tool": complete_tool_values,
}


def complete_query_source(source: CompletionSource) -> CompletionCallback:
    return COMPLETION_SOURCE_HANDLERS[source]


__all__ = [
    "COMPLETION_SOURCE_HANDLERS",
    "complete_action_sequence_values",
    "complete_action_values",
    "complete_conversation_ids",
    "complete_cwd_prefix_values",
    "complete_message_type_values",
    "complete_open_targets",
    "complete_provider_values",
    "complete_query_source",
    "complete_repo_values",
    "complete_retrieval_lane_values",
    "complete_tag_values",
    "complete_tool_values",
]
