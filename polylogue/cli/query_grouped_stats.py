"""Grouped stats output helpers for summaries and hydrated conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.cli.query_grouped_stats_semantic import output_semantic_grouped_stats
from polylogue.cli.query_grouped_stats_summary import (
    output_stats_by_grouped_conversations,
    output_stats_by_summaries,
)

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Conversation
    from polylogue.lib.query_spec import ConversationQuerySpec


def output_stats_by_conversations(
    env: AppEnv,
    results: list[Conversation],
    dimension: str,
    *,
    selection: ConversationQuerySpec | None = None,
    output_format: str = "text",
) -> None:
    if not results:
        env.ui.console.print("No conversations matched.")
        return

    if output_semantic_grouped_stats(
        env,
        results,
        dimension,
        selection=selection,
        output_format=output_format,
    ):
        return

    output_stats_by_grouped_conversations(
        env,
        results,
        dimension,
        output_format=output_format,
    )


__all__ = [
    "output_stats_by_conversations",
    "output_stats_by_summaries",
]
