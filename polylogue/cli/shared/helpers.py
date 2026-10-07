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
from polylogue.cli.shared.helper_support import (
    DaemonRequiredError,
    fail,
    indeterminate_refusal,
    load_effective_config,
    mutation_refusal,
)
from polylogue.cli.shared.types import AppEnv

if TYPE_CHECKING:
    from polylogue.analysis.archive import ArchiveCoverageInsight
    from polylogue.config import Config
    from polylogue.readiness import ReadinessReport
    from polylogue.services import RuntimeServices


def quick_readiness_summary(archive_root: Path) -> str:
    from polylogue.readiness import quick_readiness_summary as _quick_readiness_summary

    return _quick_readiness_summary(archive_root)


def get_readiness(config: Config) -> ReadinessReport:
    from polylogue.readiness import get_readiness as _get_readiness

    return _get_readiness(config)


def _summary_read(
    services: RuntimeServices | None,
    db_path: Path | None,
    operation: str,
    payload: dict[str, object],
) -> dict[str, object]:
    from polylogue.cli.operation_kernel import OperationRequest, dispatch
    from polylogue.config import load_polylogue_config

    config = services.get_config() if services is not None else load_polylogue_config()
    result = dispatch(config, OperationRequest(operation, payload), archive_root=db_path.parent if db_path else None)
    if not isinstance(result.value, dict):
        raise ValueError(f"{operation} returned a non-object result")
    return result.value


async def get_origin_counts(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[tuple[str, int]]:
    result = _summary_read(services, db_path, "query.aggregate", {"mode": "stats_by", "group_by": "origin"})
    groups = result.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("query.aggregate omitted origin groups")
    return sorted(((str(name), int(count)) for name, count in groups.items()), key=lambda item: (-item[1], item[0]))


async def list_archive_coverage_insights(
    *,
    services: RuntimeServices | None = None,
    db_path: Path | None = None,
) -> list[ArchiveCoverageInsight]:
    from polylogue.analysis.archive import ArchiveCoverageInsight, ArchiveCoverageInsightQuery
    from polylogue.operations.insight_contracts import InsightListResult

    query = ArchiveCoverageInsightQuery()
    result = _summary_read(
        services,
        db_path,
        "insights.list",
        {"page": {"insight": "archive_coverage", "query": query.model_dump(mode="json", exclude_none=True)}},
    )
    page = InsightListResult.model_validate(result).page
    return [ArchiveCoverageInsight.model_validate(item) for item in page.items]


def print_summary(env: AppEnv, *, verbose: bool = False) -> None:
    from polylogue.cli.shared.helper_summary import print_summary_impl

    print_summary_impl(
        env,
        verbose=verbose,
        format_sources_summary_fn=format_sources_summary,
        quick_readiness_summary_fn=quick_readiness_summary,
        get_readiness_fn=get_readiness,
        get_origin_counts_fn=get_origin_counts,
        list_archive_coverage_insights_fn=list_archive_coverage_insights,
    )


__all__ = [
    "complete_configured_source_names",
    "complete_run_source_names",
    "DaemonRequiredError",
    "fail",
    "indeterminate_refusal",
    "mutation_refusal",
    "format_sources_summary",
    "get_origin_counts",
    "get_readiness",
    "load_effective_config",
    "load_last_source",
    "list_archive_coverage_insights",
    "maybe_prompt_sources",
    "print_summary",
    "quick_readiness_summary",
    "resolve_sources",
    "save_last_source",
    "source_state_path",
]
