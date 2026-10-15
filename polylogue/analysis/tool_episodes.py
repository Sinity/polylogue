"""Call/result episodes over the canonical action projection."""

from __future__ import annotations

from polylogue.analysis.archive import PaginatedInsightQuery
from polylogue.analysis.archive_models import ArchiveInsightModel


class ToolEpisodeResult(ArchiveInsightModel):
    block_id: str
    output_text: str | None = None
    is_error: int | None = None
    exit_code: int | None = None
    outcome: str
    unknown_reason: str | None = None


class ToolEpisodeInsight(ArchiveInsightModel):
    episode_id: str
    session_id: str
    message_id: str
    origin: str
    tool_use_block_id: str
    results: tuple[ToolEpisodeResult, ...] = ()
    tool_name: str | None = None
    semantic_type: str | None = None
    call_input: str | None = None
    output_text: str | None = None
    outcome_unknown_reason: str | None = None
    result_state: str
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()
    next_action: str | None = None
    followup_class: str | None = None
    caveat: str


class ToolEpisodeQuery(PaginatedInsightQuery):
    origin: str | None = None
    tag: str | None = None
    repo: str | None = None
    since: str | None = None
    until: str | None = None
    session_id: str | None = None
    tool: str | None = None
    result_state: str | None = None


__all__ = ["ToolEpisodeInsight", "ToolEpisodeQuery", "ToolEpisodeResult"]
