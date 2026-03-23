"""Stable query summary/stats output surface."""

from __future__ import annotations

from polylogue.cli.query_grouped_stats import (
    output_stats_by_conversations,
    output_stats_by_summaries,
)
from polylogue.cli.query_list_output import (
    conversations_to_csv,
    format_summary_list,
    output_summary_list,
)
from polylogue.cli.query_profile_stats import (
    output_stats_by_profile_ids,
    output_stats_by_profile_query,
    output_stats_by_profile_summaries,
)
from polylogue.cli.query_semantic_slice import (
    SemanticStatsSlice,
    action_matches_slice,
    filtered_action_events,
    normalized_tool_name,
    path_matches_slice,
)
from polylogue.cli.query_semantic_stats import (
    output_stats_by_semantic_ids,
    output_stats_by_semantic_query,
    output_stats_by_semantic_summaries,
)
from polylogue.cli.query_sql_stats import output_stats_sql
from polylogue.cli.query_stats_structured import emit_structured_stats

__all__ = [
    "SemanticStatsSlice",
    "action_matches_slice",
    "conversations_to_csv",
    "emit_structured_stats",
    "filtered_action_events",
    "format_summary_list",
    "normalized_tool_name",
    "output_stats_by_conversations",
    "output_stats_by_profile_ids",
    "output_stats_by_profile_query",
    "output_stats_by_profile_summaries",
    "output_stats_by_semantic_ids",
    "output_stats_by_semantic_query",
    "output_stats_by_semantic_summaries",
    "output_stats_by_summaries",
    "output_stats_sql",
    "output_summary_list",
    "path_matches_slice",
]
