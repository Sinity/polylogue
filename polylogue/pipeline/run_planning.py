"""Preview planning for pipeline runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.pipeline.run_support import run_coroutine_sync, select_sources
from polylogue.storage.backends import create_backend
from polylogue.storage.state_views import PlanResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend


def plan_sources(
    config: Config,
    *,
    stage: str = "all",
    stage_sequence: Sequence[str] | None = None,
    ui: object | None = None,
    source_names: Sequence[str] | None = None,
    backend: SQLiteBackend | None = None,
    progress_callback: ProgressCallback | None = None,
    force_reparse: bool = False,
) -> PlanResult:
    """Build a canonical preview plan without writing pipeline state."""
    from polylogue.pipeline.services.planning import PlanningService

    async def _build() -> PlanResult:
        planner = PlanningService(backend=active_backend, config=config)
        plan = await planner.build_plan(
            sources=select_sources(config, source_names),
            stage=stage,
            stage_sequence=tuple(stage_sequence) if stage_sequence is not None else None,
            ui=ui,
            progress_callback=progress_callback,
            preview=True,
            force_reparse=force_reparse,
        )
        return plan.summary

    owns_backend = backend is None
    active_backend = backend or create_backend()
    try:
        return run_coroutine_sync(_build())
    finally:
        if owns_backend:
            run_coroutine_sync(active_backend.close())


__all__ = ["plan_sources"]
