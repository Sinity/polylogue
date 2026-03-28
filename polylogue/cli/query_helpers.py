"""Shared helpers for CLI query execution."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary


def result_id(result: Conversation | ConversationSummary) -> str:
    return str(result.id)


def result_provider(result: Conversation | ConversationSummary) -> str:
    return str(result.provider)


def result_title(result: Conversation | ConversationSummary) -> str:
    title = result.display_title
    return title if title else result_id(result)[:20]


def result_date(result: Conversation | ConversationSummary) -> datetime | None:
    display_date = getattr(result, "display_date", None)
    if isinstance(display_date, datetime):
        return display_date
    updated_at = getattr(result, "updated_at", None)
    if isinstance(updated_at, datetime):
        return updated_at
    created_at = getattr(result, "created_at", None)
    if isinstance(created_at, datetime):
        return created_at
    return None


def summary_to_dict(summary: ConversationSummary, message_count: int) -> dict[str, object]:
    return {
        "id": str(summary.id),
        "provider": str(summary.provider),
        "title": summary.display_title,
        "date": summary.display_date.isoformat() if summary.display_date else None,
        "tags": summary.tags,
        "summary": summary.summary,
        "messages": message_count,
    }
