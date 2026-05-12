"""CLI helper functions.

Heavy imports used only by ``print_summary`` are deferred inside the
function so that commands that only need ``fail`` or
``load_effective_config`` don't pay the ~800 ms operations/ui import
cost at startup.
"""

from __future__ import annotations

from polylogue.cli.shared.formatting import format_sources_summary
from polylogue.cli.shared.helper_source_selection import (
    complete_configured_source_names,
    complete_run_source_names,
    maybe_prompt_sources,
    resolve_sources,
)
from polylogue.cli.shared.helper_source_state import load_last_source, save_last_source, source_state_path
from polylogue.cli.shared.helper_support import fail, load_effective_config
from polylogue.cli.shared.types import AppEnv


def print_summary(env: AppEnv, *, verbose: bool = False) -> None:
    from polylogue.cli.shared.helper_summary import print_summary_impl
    from polylogue.operations import get_provider_counts, list_provider_analytics_insights
    from polylogue.readiness import get_readiness, quick_readiness_summary

    print_summary_impl(
        env,
        verbose=verbose,
        format_sources_summary_fn=format_sources_summary,
        quick_readiness_summary_fn=quick_readiness_summary,
        get_readiness_fn=get_readiness,
        get_provider_counts_fn=get_provider_counts,
        list_provider_analytics_insights_fn=list_provider_analytics_insights,
    )


__all__ = [
    "complete_configured_source_names",
    "complete_run_source_names",
    "fail",
    "format_sources_summary",
    "load_effective_config",
    "load_last_source",
    "maybe_prompt_sources",
    "print_summary",
    "resolve_sources",
    "save_last_source",
    "source_state_path",
]
