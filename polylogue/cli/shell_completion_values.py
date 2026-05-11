"""Shell-completion helpers for archive-backed CLI values.

Conversation ID and tag completions route through
``ArchiveOperations`` (async, bridged via ``asyncio.run``).
Repo, CWD-prefix, and tool-name completions still use raw SQL on
``session_profiles`` / ``action_events`` \u2014 those tables lack
ArchiveOperations read methods (follow-up in #862 or a successor).
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import click
from click.shell_completion import CompletionItem

from polylogue.archive.message.types import MessageType
from polylogue.archive.query.fields import CompletionSource
from polylogue.archive.query.spec import QUERY_ACTION_TYPES, QUERY_RETRIEVAL_LANES, QUERY_SEQUENCE_ACTION_TYPES
from polylogue.operations.archive import ArchiveOperations as _ArchiveOperations
from polylogue.paths import db_path
from polylogue.storage.sqlite.connection import open_read_connection

Callback = Callable[[_ArchiveOperations], Awaitable[list[CompletionItem]]]

_PROVIDER_DESCRIPTIONS: Final[dict[str, str]] = {
    "chatgpt": "OpenAI ChatGPT exports",
    "claude-ai": "Anthropic Claude web exports",
    "claude-code": "Claude Code local sessions",
    "codex": "OpenAI Codex sessions",
    "gemini": "Google AI Studio exports",
    "gemini-cli": "Gemini CLI local sessions",
    "hermes": "Hermes agent sessions",
    "antigravity": "Antigravity local brain artifacts",
}

_MAX_ID_COMPLETIONS = 24
_MAX_VALUE_COMPLETIONS = 32
CompletionHelpBuilder = Callable[[sqlite3.Row], str]
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


def _fetch_rows(sql: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
    if not _db_exists():
        return []
    try:
        with open_read_connection() as conn:
            cursor = conn.execute(sql, params)
            return list(cursor.fetchall())
    except sqlite3.Error:
        return []


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


def _rows_to_completion_items(
    rows: list[sqlite3.Row],
    *,
    value_column: str,
    help_builder: CompletionHelpBuilder | None = None,
) -> list[CompletionItem]:
    items: list[CompletionItem] = []
    for row in rows:
        help_text = help_builder(row) if help_builder is not None else None
        items.append(CompletionItem(str(row[value_column]), help=help_text))
    return items


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
            provider_name = str(conv.provider)
            display = title or cid
            if current and not (
                cid.startswith(current) or (":" in cid and current in cid) or (title and current_lower in title.lower())
            ):
                continue
            items.append(
                CompletionItem(
                    cid,
                    help=f"{provider_name} \u00b7 {_trim_help(display)}",
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
    # NOTE: Raw SQL on session_profiles \u2014 no ArchiveOperations read
    # method for repo-name aggregation yet.
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    rows = _fetch_rows(
        """
        SELECT
            repo.value AS repo_name,
            COUNT(*) AS cnt
        FROM session_profiles AS sp,
             json_each(COALESCE(sp.repo_names_json, '[]')) AS repo
        WHERE (? = '' OR repo.value LIKE ?)
        GROUP BY repo.value
        ORDER BY cnt DESC, repo.value ASC
        LIMIT ?
        """,
        (
            current,
            f"{current}%",
            _MAX_VALUE_COMPLETIONS,
        ),
    )
    items = _rows_to_completion_items(
        rows,
        value_column="repo_name",
        help_builder=lambda row: f"{int(row['cnt'])} sessions",
    )
    return _with_csv_prefix(items, prefix)


def complete_cwd_prefix_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    # NOTE: Raw SQL on session_profiles \u2014 no ArchiveOperations read
    # method for cwd-prefix aggregation yet.
    del ctx, param
    current = incomplete.strip()
    rows = _fetch_rows(
        """
        SELECT
            cwd.value AS cwd_path,
            COUNT(*) AS cnt
        FROM session_profiles AS sp,
             json_each(COALESCE(json_extract(sp.evidence_payload_json, '$.cwd_paths'), '[]')) AS cwd
        WHERE (? = '' OR cwd.value LIKE ?)
        GROUP BY cwd.value
        ORDER BY cnt DESC, cwd.value ASC
        LIMIT ?
        """,
        (
            current,
            f"{current}%",
            _MAX_VALUE_COMPLETIONS,
        ),
    )
    return _rows_to_completion_items(
        rows,
        value_column="cwd_path",
        help_builder=lambda row: f"{int(row['cnt'])} sessions",
    )


def complete_tool_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    # NOTE: Raw SQL on action_events \u2014 no ArchiveOperations read
    # method for tool-name aggregation yet.
    del ctx, param
    current = incomplete.strip().lower()
    rows = _fetch_rows(
        """
        SELECT
            normalized_tool_name,
            COUNT(*) AS cnt
        FROM action_events
        WHERE (? = '' OR normalized_tool_name LIKE ?)
        GROUP BY normalized_tool_name
        ORDER BY cnt DESC, normalized_tool_name ASC
        LIMIT ?
        """,
        (
            current,
            f"{current}%",
            _MAX_VALUE_COMPLETIONS,
        ),
    )
    return _rows_to_completion_items(
        rows,
        value_column="normalized_tool_name",
        help_builder=lambda row: f"{int(row['cnt'])} actions",
    )


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
