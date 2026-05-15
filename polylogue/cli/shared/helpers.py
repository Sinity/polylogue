"""CLI helper functions.

Heavy imports used only by ``print_summary`` are deferred inside the
function so that commands that only need ``fail`` or
``load_effective_config`` don't pay the ~800 ms operations/ui import
cost at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.insights.archive import ProviderAnalyticsInsight
    from polylogue.readiness import ReadinessReport
    from polylogue.services import RuntimeServices


def quick_readiness_summary(archive_root: Path) -> str:
    from polylogue.readiness import quick_readiness_summary as _quick_readiness_summary

    return _quick_readiness_summary(archive_root)


def get_readiness(config: Config) -> ReadinessReport:
    from polylogue.readiness import get_readiness as _get_readiness

    return _get_readiness(config)


async def get_provider_counts(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[tuple[str, int]]:
    from polylogue.operations import get_provider_counts as _get_provider_counts

    return await _get_provider_counts(services=services, db_path=db_path)


async def list_provider_analytics_insights(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[ProviderAnalyticsInsight]:
    from polylogue.operations import list_provider_analytics_insights as _list_provider_analytics_insights

    return await _list_provider_analytics_insights(services=services, db_path=db_path)


def print_summary(env: AppEnv, *, verbose: bool = False) -> None:
    from polylogue.cli.shared.helper_summary import print_summary_impl

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
    "get_provider_counts",
    "get_readiness",
    "load_effective_config",
    "load_last_source",
    "list_provider_analytics_insights",
    "maybe_prompt_sources",
    "print_summary",
    "quick_readiness_summary",
    "resolve_sources",
    "save_last_source",
    "source_state_path",
]
