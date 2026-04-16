"""CLI helper functions."""

from __future__ import annotations

from pathlib import Path

from polylogue.cli.formatting import format_sources_summary
from polylogue.cli.helper_source_selection import (
    complete_configured_source_names,
    complete_run_source_names,
    maybe_prompt_sources,
    resolve_sources,
)
from polylogue.cli.helper_source_state import load_last_source, save_last_source, source_state_path
from polylogue.cli.helper_summary import print_summary_impl
from polylogue.cli.helper_support import fail, load_effective_config
from polylogue.health import cached_health_summary, get_health
from polylogue.operations import get_provider_counts, list_provider_analytics_products
from polylogue.pipeline.runner import latest_run


def latest_render_path(render_root: Path) -> Path | None:
    if not render_root.exists():
        return None
    candidates = list(render_root.rglob("conversation.md")) + list(render_root.rglob("conversation.html"))
    if not candidates:
        return None
    latest: Path | None = None
    latest_mtime = 0.0
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = path
        except OSError:
            continue
    return latest


def print_summary(env, *, verbose: bool = False) -> None:
    print_summary_impl(
        env,
        verbose=verbose,
        latest_run_fn=latest_run,
        format_sources_summary_fn=format_sources_summary,
        cached_health_summary_fn=cached_health_summary,
        get_health_fn=get_health,
        get_provider_counts_fn=get_provider_counts,
        list_provider_analytics_products_fn=list_provider_analytics_products,
    )


__all__ = [
    "cached_health_summary",
    "complete_configured_source_names",
    "complete_run_source_names",
    "fail",
    "format_sources_summary",
    "get_health",
    "get_provider_counts",
    "latest_render_path",
    "latest_run",
    "list_provider_analytics_products",
    "load_effective_config",
    "load_last_source",
    "maybe_prompt_sources",
    "print_summary",
    "resolve_sources",
    "save_last_source",
    "source_state_path",
]
