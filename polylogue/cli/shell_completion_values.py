"""Shell-completion helpers for archive-backed CLI values."""

from __future__ import annotations

import sqlite3
from typing import Final

import click
from click.shell_completion import CompletionItem

from polylogue.paths import db_path
from polylogue.storage.backends.connection import open_read_connection

_PROVIDER_DESCRIPTIONS: Final[dict[str, str]] = {
    "chatgpt": "OpenAI ChatGPT exports",
    "claude-ai": "Anthropic Claude web exports",
    "claude-code": "Claude Code local sessions",
    "codex": "OpenAI Codex sessions",
    "gemini": "Google Gemini exports",
}

_MAX_ID_COMPLETIONS = 24
_MAX_VALUE_COMPLETIONS = 32


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


def _trim_help(value: str, *, limit: int = 72) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


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


def complete_conversation_ids(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    current = incomplete.strip()
    current_lower = current.lower()
    rows = _fetch_rows(
        """
        SELECT
            conversation_id,
            provider_name,
            COALESCE(
                NULLIF(json_extract(metadata, '$.title'), ''),
                NULLIF(title, ''),
                conversation_id
            ) AS display_title
        FROM conversations
        WHERE (
            ? = ''
            OR conversation_id LIKE ?
            OR conversation_id LIKE ?
            OR LOWER(COALESCE(json_extract(metadata, '$.title'), title, '')) LIKE ?
        )
        ORDER BY sort_key DESC, conversation_id ASC
        LIMIT ?
        """,
        (
            current,
            f"{current}%",
            f"%:{current}%",
            f"%{current_lower}%",
            _MAX_ID_COMPLETIONS,
        ),
    )
    items: list[CompletionItem] = []
    for row in rows:
        conv_id = str(row["conversation_id"])
        provider = str(row["provider_name"] or "unknown")
        title = _trim_help(str(row["display_title"] or conv_id))
        items.append(CompletionItem(conv_id, help=f"{provider} · {title}"))
    return items


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
    rows = _fetch_rows(
        """
        SELECT
            tag.value AS tag_name,
            COUNT(*) AS cnt
        FROM conversations,
             json_each(json_extract(metadata, '$.tags')) AS tag
        WHERE metadata IS NOT NULL
          AND json_extract(metadata, '$.tags') IS NOT NULL
          AND (? = '' OR tag.value LIKE ?)
        GROUP BY tag.value
        ORDER BY cnt DESC, tag.value ASC
        LIMIT ?
        """,
        (
            current,
            f"{current}%",
            _MAX_VALUE_COMPLETIONS,
        ),
    )
    items = [CompletionItem(str(row["tag_name"]), help=f"{int(row['cnt'])} conversations") for row in rows]
    return _with_csv_prefix(items, prefix)


def complete_tool_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
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
    return [
        CompletionItem(str(row["normalized_tool_name"]), help=f"{int(row['cnt'])} actions")
        for row in rows
    ]


__all__ = [
    "complete_conversation_ids",
    "complete_open_targets",
    "complete_provider_values",
    "complete_tag_values",
    "complete_tool_values",
]
