"""Execution helpers for run and sources commands."""

from __future__ import annotations

from polylogue.cli.run_observers import progress_observer as _progress_observer
from polylogue.config import Config
from polylogue.pipeline.observers import CompositeObserver, RunObserver
from polylogue.pipeline.runner import run_sources
from polylogue.protocols import ProgressCallback
from polylogue.storage.state_views import PlanResult, RunResult
from polylogue.sync_bridge import run_coroutine_sync


def execute_sync_once(
    cfg: Config,
    env,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RunResult:
    return run_coroutine_sync(
        run_sources(
            config=cfg,
            stage=stage,
            plan=plan_snapshot,
            ui=env.ui,
            source_names=selected_sources,
            progress_callback=progress_callback,
            render_format=render_format,
            backend=env.backend,
            repository=env.repository,
        )
    )


def run_with_progress(
    cfg: Config,
    env,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    observer: RunObserver | None = None,
) -> RunResult:
    with _progress_observer(env) as progress_observer:
        progress_bridge = (
            CompositeObserver([progress_observer, observer])
            if observer is not None
            else progress_observer
        )
        result = execute_sync_once(
            cfg,
            env,
            stage,
            selected_sources,
            render_format,
            plan_snapshot=plan_snapshot,
            progress_callback=progress_bridge.on_progress,
        )
        progress_observer.on_completed(result)
        return result


def run_sync_once(
    cfg: Config,
    env,
    stage: str,
    selected_sources: list[str] | None,
    render_format: str,
    plan_snapshot: PlanResult | None = None,
    observer: RunObserver | None = None,
) -> RunResult:
    try:
        result = run_with_progress(
            cfg,
            env,
            stage,
            selected_sources,
            render_format,
            plan_snapshot=plan_snapshot,
            observer=observer,
        )
    except Exception as exc:
        if observer is not None:
            observer.on_error(exc)
        raise

    if observer is not None:
        observer.on_completed(result)
    return result


__all__ = ["execute_sync_once", "run_sync_once", "run_with_progress"]
