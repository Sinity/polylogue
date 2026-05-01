"""Slow-query transparency helpers for CLI query execution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import click

from polylogue.cli.query_contracts import QueryOutputSpec
from polylogue.lib.query.retrieval_candidates import uses_action_read_model
from polylogue.lib.query.spec import ConversationQuerySpec

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from polylogue.storage.action_events.artifacts import ActionEventArtifactState

_T = TypeVar("_T")

MACHINE_OUTPUT_FORMATS = frozenset({"json", "yaml", "csv"})
SLOW_QUERY_NOTICE_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class QuerySlowNotice:
    """One human-facing slow-query notice."""

    route: str
    retrieval_lane: str
    fallback_detail: str | None = None

    def message(self, *, elapsed_seconds: float) -> str:
        parts = [
            f"Query still running after {_format_elapsed(elapsed_seconds)}",
            f"route: {self.route}",
            f"retrieval: {self.retrieval_lane}",
        ]
        if self.fallback_detail:
            parts.append(self.fallback_detail)
        return f"{parts[0]} ({'; '.join(parts[1:])})."


def slow_query_notice_threshold() -> float:
    """Return the slow-query notice threshold, allowing local override."""
    raw_value = os.environ.get("POLYLOGUE_SLOW_QUERY_NOTICE_SECONDS")
    if raw_value is None:
        return SLOW_QUERY_NOTICE_SECONDS
    try:
        value = float(raw_value)
    except ValueError:
        return SLOW_QUERY_NOTICE_SECONDS
    return max(value, 0.0)


def should_emit_slow_query_notes(output: QueryOutputSpec) -> bool:
    """Return whether progress notes are safe for this output mode."""
    return output.output_format not in MACHINE_OUTPUT_FORMATS


def _format_elapsed(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    remainder = int(seconds % 60)
    return f"{minutes}m{remainder:02d}s"


def _async_method(obj: object, method_name: str) -> Callable[..., Awaitable[object]] | None:
    candidate = getattr(obj, method_name, None)
    if not callable(candidate):
        return None
    return cast(Callable[..., Awaitable[object]], candidate)


async def _action_event_state(repository: object) -> ActionEventArtifactState | None:
    method = _async_method(repository, "get_action_event_artifact_state")
    if method is None:
        return None
    try:
        result = await method()
    except Exception:
        logger.exception("_action_event_state: get_action_event_artifact_state() failed")
        return None
    return cast("ActionEventArtifactState | None", result)


def _action_fallback_detail(state: ActionEventArtifactState | None) -> str | None:
    if state is None or state.ready:
        return None
    return state.repair_detail()


async def build_query_slow_notice(
    repository: object,
    selection: object,
    *,
    route: str,
) -> QuerySlowNotice:
    """Build a slow-query notice from facts available after the operation stalls."""
    retrieval_lane = "unknown"
    fallback_detail = None
    if isinstance(selection, ConversationQuerySpec):
        try:
            plan = selection.to_plan()
        except Exception:
            logger.exception("build_query_slow_notice: selection.to_plan() failed")
            plan = None
        if plan is not None:
            retrieval_lane = plan.retrieval_lane
            if uses_action_read_model(plan):
                fallback_detail = _action_fallback_detail(await _action_event_state(repository))
    return QuerySlowNotice(
        route=route,
        retrieval_lane=retrieval_lane,
        fallback_detail=fallback_detail,
    )


async def observe_slow_query(
    operation: Awaitable[_T],
    *,
    enabled: bool,
    notice_factory: Callable[[], Awaitable[QuerySlowNotice]],
    threshold_seconds: float | None = None,
    emit: Callable[[str], None] | None = None,
) -> _T:
    """Run an operation and emit one human notice if it exceeds the threshold."""
    if not enabled:
        return await operation

    threshold = slow_query_notice_threshold() if threshold_seconds is None else max(threshold_seconds, 0.0)
    task = asyncio.ensure_future(operation)
    notice_task: asyncio.Task[QuerySlowNotice] | None = None
    try:
        done, _pending = await asyncio.wait({task}, timeout=threshold)
        if done:
            return task.result()

        notice_task = asyncio.ensure_future(notice_factory())
        operation_task = cast("asyncio.Task[object]", task)
        notice_task_object = cast("asyncio.Task[object]", notice_task)
        watched = {operation_task, notice_task_object}
        done, _pending = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)  # type: ignore[arg-type]
        if operation_task in done:
            notice_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await notice_task
            return task.result()

        sink = emit or (lambda message: click.echo(message, err=True))
        try:
            notice = notice_task.result()
        except Exception:
            logger.exception("observe_slow_query: notice_factory() failed")
        else:
            sink(notice.message(elapsed_seconds=threshold))
        return await task
    except asyncio.CancelledError:
        task.cancel()
        pending: list[asyncio.Task[object]] = [cast("asyncio.Task[object]", task)]
        if notice_task is not None:
            notice_task.cancel()
            pending.append(cast("asyncio.Task[object]", notice_task))
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*pending, return_exceptions=True)
        raise


__all__ = [
    "MACHINE_OUTPUT_FORMATS",
    "QuerySlowNotice",
    "SLOW_QUERY_NOTICE_SECONDS",
    "build_query_slow_notice",
    "observe_slow_query",
    "should_emit_slow_query_notes",
    "slow_query_notice_threshold",
]
