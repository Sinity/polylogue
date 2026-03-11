"""Shared helpers for CLI query execution."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, NoReturn

import click

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.lib.query_spec import ConversationQuerySpec


def coerce_query_spec(params: dict[str, Any] | ConversationQuerySpec) -> ConversationQuerySpec:
    from polylogue.lib.query_spec import ConversationQuerySpec

    if isinstance(params, ConversationQuerySpec):
        return params
    return ConversationQuerySpec.from_params(params)


def describe_query_filters(params: dict[str, Any] | ConversationQuerySpec) -> list[str]:
    """Build a human-readable list of active filters from params or spec."""
    return coerce_query_spec(params).describe()


def no_results(
    env: AppEnv,
    params: dict[str, Any] | ConversationQuerySpec,
    *,
    exit_code: int = 2,
) -> NoReturn:
    """Print a helpful no-results message and exit."""
    filters = describe_query_filters(params)
    if filters:
        click.echo("No conversations matched filters:", err=True)
        for item in filters:
            click.echo(f"  {item}", err=True)
        click.echo("Hint: try broadening your filters or use --list to browse", err=True)
    else:
        click.echo("No conversations matched.", err=True)
    raise SystemExit(exit_code)


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
