"""Rendering helpers for archive-product CLI commands."""

from __future__ import annotations

from polylogue.cli.products_rendering_aggregate import (
    render_day_session_summaries,
    render_provider_analytics,
    render_session_tag_rollups,
    render_week_session_summaries,
)
from polylogue.cli.products_rendering_session import (
    render_session_enrichments,
    render_session_phases,
    render_session_profiles,
    render_session_work_events,
    render_work_threads,
)
from polylogue.cli.products_rendering_support import summarize_archive_debt

__all__ = [
    "render_day_session_summaries",
    "render_provider_analytics",
    "render_session_enrichments",
    "render_session_phases",
    "render_session_profiles",
    "render_session_tag_rollups",
    "render_session_work_events",
    "render_week_session_summaries",
    "render_work_threads",
    "summarize_archive_debt",
]
