"""Call/result episodes over the canonical action projection."""

from __future__ import annotations

from polylogue.analysis.archive import PaginatedInsightQuery
from polylogue.analysis.archive_models import ArchiveInsightModel


class ToolEpisodeInsight(ArchiveInsightModel):
    episode_id: str
    session_id: str
    message_id: str
    origin: str
    tool_use_block_id: str
    tool_result_block_id: str | None = None
    tool_name: str | None = None
    semantic_type: str | None = None
    call_input: str | None = None
    result_output: str | None = None
    is_error: int | None = None
    exit_code: int | None = None
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


__all__ = ["ToolEpisodeInsight", "ToolEpisodeQuery"]
