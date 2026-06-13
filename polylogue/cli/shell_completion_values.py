"""Shell-completion helpers for archive-backed CLI values.

All archive-backed completions read the ``index.db``
through :class:`~polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore`.
Session-id, tag, repo-name, and tool-name values come from native
session/tag/repo/action read models; cwd-prefix has no archive source
yet and degrades to an empty completion list.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final

import click
from click.shell_completion import CompletionItem

from polylogue.archive.message.types import MessageType
from polylogue.archive.query.fields import CompletionSource
from polylogue.archive.query.spec import QUERY_ACTION_TYPES, QUERY_RETRIEVAL_LANES, QUERY_SEQUENCE_ACTION_TYPES
from polylogue.paths import active_index_db_path

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

ArchiveCompletionAction = Callable[["ArchiveStore"], list[CompletionItem]]

_ORIGIN_DESCRIPTIONS: Final[dict[str, str]] = {
    "chatgpt-export": "ChatGPT web exports (lab: OpenAI)",
    "claude-ai-export": "Claude web exports (lab: Anthropic)",
    "claude-code-session": "Claude Code local sessions (lab: Anthropic)",
    "codex-session": "Codex CLI local sessions (lab: OpenAI)",
    "aistudio-drive": "Google AI Studio / Drive exports (lab: Google)",
    "gemini-cli-session": "Gemini CLI local sessions (lab: Google)",
    "hermes-session": "Hermes agent sessions",
    "antigravity-session": "Antigravity brain artifacts",
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
    return active_index_db_path().exists()


def _run_completion(action: ArchiveCompletionAction) -> list[CompletionItem]:
    """Open the archive, run *action*, and return its items.

    Any failure (missing/locked database, unexpected schema) degrades to an
    empty list so completion never raises into the shell.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    index_db = active_index_db_path()
    if not index_db.exists():
        return []
    try:
        with ArchiveStore.open_existing(index_db.parent, read_only=True) as archive:
            return list(action(archive))
    except Exception:
        return []


def _stats_by_items(group_by: str, prefix: str, *, unit: str) -> ArchiveCompletionAction:
    """Archive completion action over ``ArchiveStore.stats_by`` group counts."""

    prefix_lower = prefix.lower()

    def action(archive: ArchiveStore) -> list[CompletionItem]:
        grouped = archive.stats_by(group_by)
        ordered = sorted(grouped.items(), key=lambda pair: (-pair[1], pair[0]))
        items: list[CompletionItem] = []
        for value, count in ordered:
            if prefix_lower and not value.lower().startswith(prefix_lower):
                continue
            items.append(CompletionItem(value, help=f"{count} {unit}"))
            if len(items) >= _MAX_VALUE_COMPLETIONS:
                break
        return items

    return action


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


def complete_origin_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    prefix, current = _split_csv_incomplete(incomplete)
    current_lower = current.lower()
    items = [
        CompletionItem(name, help=_ORIGIN_DESCRIPTIONS.get(name))
        for name in _ORIGIN_DESCRIPTIONS
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


def complete_session_ids(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    current = incomplete.strip()
    current_lower = current.lower()

    if not _db_exists():
        return []

    def _query(archive: ArchiveStore) -> list[CompletionItem]:
        summaries = archive.list_summaries(limit=100)
        items: list[CompletionItem] = []
        for summary in summaries:
            cid = str(summary.session_id)
            title = summary.title or ""
            source_name = summary.provider.value
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

    def _query(archive: ArchiveStore) -> list[CompletionItem]:
        tags = archive.list_user_tags()
        sorted_tags = sorted(tags.items(), key=lambda x: (-x[1], x[0]))
        items: list[CompletionItem] = []
        for name, cnt in sorted_tags:
            if current_lower and not name.lower().startswith(current_lower):
                continue
            items.append(CompletionItem(name, help=f"{cnt} sessions"))
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

    items = _run_completion(_stats_by_items("repo", current, unit="sessions"))
    return _with_csv_prefix(items, prefix)


def complete_cwd_prefix_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param, incomplete
    # The read models do not expose a session-cwd aggregate
    # yet, so cwd-prefix completion has no source to draw from. Returning an
    # empty list keeps the completer well-behaved (no traceback) until a
    # cwd projection lands.
    return []


def complete_tool_values(
    ctx: click.Context,
    param: click.Parameter,
    incomplete: str,
) -> list[CompletionItem]:
    del ctx, param
    current = incomplete.strip().lower()
    if not _db_exists():
        return []

    return _run_completion(_stats_by_items("tool", current, unit="actions"))


COMPLETION_SOURCE_HANDLERS: Final[Mapping[CompletionSource, CompletionCallback]] = {
    # Ordered to match the deterministic (sorted) order produced by
    # ``query_completion_sources()`` so the registry-coverage contract holds.
    "action": complete_action_values,
    "action_sequence": complete_action_sequence_values,
    "cwd_prefix": complete_cwd_prefix_values,
    "message_type": complete_message_type_values,
    "origin": complete_origin_values,
    "repo": complete_repo_values,
    "retrieval_lane": complete_retrieval_lane_values,
    "session_id": complete_session_ids,
    "tag": complete_tag_values,
    "tool": complete_tool_values,
}


def complete_query_source(source: CompletionSource) -> CompletionCallback:
    return COMPLETION_SOURCE_HANDLERS[source]


__all__ = [
    "COMPLETION_SOURCE_HANDLERS",
    "complete_action_sequence_values",
    "complete_action_values",
    "complete_session_ids",
    "complete_cwd_prefix_values",
    "complete_message_type_values",
    "complete_origin_values",
    "complete_query_source",
    "complete_repo_values",
    "complete_retrieval_lane_values",
    "complete_tag_values",
    "complete_tool_values",
]
