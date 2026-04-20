"""Async rendering service for pipeline operations."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.lib.metrics import read_current_rss_mb
from polylogue.logging import get_logger
from polylogue.protocols import OutputRenderer, ProgressCallback
from polylogue.storage.state_views import RenderFailurePayload

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["RenderService", "RenderResult"]

# Per-conversation timeout: prevents infinite hangs on huge conversations
# or lock contention with concurrent processes.
RENDER_TIMEOUT_S = 120

# Log a warning when a single render exceeds this duration.
SLOW_RENDER_THRESHOLD_S = 10.0
_DEFAULT_RENDER_WORKER_LIMIT = 4
_HIGH_RSS_RENDER_LIMIT_MB = 1024.0
_VERY_HIGH_RSS_RENDER_LIMIT_MB = 2048.0


class RenderResult:
    """Result of a rendering operation."""

    def __init__(self) -> None:
        self.rendered_count: int = 0
        self.failures: list[RenderFailurePayload] = []
        self.worker_count: int = 0
        self.rss_start_mb: float | None = None
        self.rss_end_mb: float | None = None
        self.max_current_rss_mb: float | None = None

    def record_success(self) -> None:
        """Record a successful render."""
        self.rendered_count += 1

    def record_failure(self, conversation_id: str, error: str) -> None:
        """Record a rendering failure."""
        self.failures.append(
            {
                "conversation_id": conversation_id,
                "error": error,
            }
        )

    def observe_current_rss(self) -> None:
        current_rss_mb = read_current_rss_mb()
        if current_rss_mb is None:
            return
        if self.max_current_rss_mb is None or current_rss_mb > self.max_current_rss_mb:
            self.max_current_rss_mb = current_rss_mb


def _resolve_default_render_workers(max_workers: int | None) -> int:
    if max_workers is not None:
        return max_workers
    worker_limit = min(os.cpu_count() or 4, _DEFAULT_RENDER_WORKER_LIMIT)
    current_rss_mb = read_current_rss_mb()
    if current_rss_mb is None:
        return worker_limit
    if current_rss_mb >= _VERY_HIGH_RSS_RENDER_LIMIT_MB:
        return min(worker_limit, 1)
    if current_rss_mb >= _HIGH_RSS_RENDER_LIMIT_MB:
        return min(worker_limit, 2)
    return worker_limit


class RenderService:
    """Service for rendering conversations to Markdown and HTML (async version).

    Performance features:
    - Connection pool for concurrent DB reads (read_pool)
    - Per-conversation timeout to prevent hangs
    - Straggler detection with slow-render logging
    - Worker count scaled to CPU count
    """

    def __init__(
        self,
        renderer: OutputRenderer,
        render_root: Path,
        backend: SQLiteBackend | None = None,
    ):
        """Initialize the async rendering service.

        Args:
            renderer: OutputRenderer implementation for rendering conversations
            render_root: Root directory for rendered output
            backend: Optional shared backend for connection pooling
        """
        self.renderer = renderer
        self.render_root = render_root
        self.backend = backend

    async def render_conversations(
        self,
        conversation_ids: Iterable[str] | AsyncIterable[str],
        *,
        total: int | None = None,
        max_workers: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> RenderResult:
        """Render multiple conversations with connection pooling and timeouts.

        Architecture:
        1. Opens a read pool with N connections (matching worker count)
        2. N async workers pull conversation IDs from a queue
        3. Each render: DB query (pool conn) → CPU work (thread) → file write (thread)
        4. Per-conversation timeout prevents infinite hangs
        5. Slow renders (>10s) are logged for investigation
        """
        result = RenderResult()
        result.rss_start_mb = read_current_rss_mb()
        if total is None and isinstance(conversation_ids, Sequence):
            total = len(conversation_ids)
        if total == 0:
            result.rss_end_mb = result.rss_start_mb
            return result

        max_workers = _resolve_default_render_workers(max_workers)
        worker_limit = total if total is not None else max_workers
        worker_count = max(1, min(max_workers, worker_limit))
        result.worker_count = worker_count
        result.observe_current_rss()

        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max(worker_count * 2, 1))

        async def _render_one(convo_id: str) -> None:
            t0 = time.perf_counter()
            try:
                await asyncio.wait_for(
                    self.renderer.render(convo_id, self.render_root),
                    timeout=RENDER_TIMEOUT_S,
                )
                result.record_success()
            except asyncio.TimeoutError:
                logger.warning(
                    "Render timed out after %ds",
                    RENDER_TIMEOUT_S,
                    conversation_id=convo_id,
                )
                result.record_failure(convo_id, f"render timed out after {RENDER_TIMEOUT_S}s")
            except Exception as exc:
                logger.warning("Failed to render conversation %s: %s", convo_id, exc)
                result.record_failure(convo_id, str(exc))

            elapsed = time.perf_counter() - t0
            result.observe_current_rss()
            if elapsed > SLOW_RENDER_THRESHOLD_S:
                logger.info(
                    "Slow render: %.1fs",
                    elapsed,
                    conversation_id=convo_id,
                )

            if progress_callback is not None:
                done = result.rendered_count + len(result.failures)
                desc = f"Rendering: {done}/{total}" if total is not None else f"Rendering: {done}"
                progress_callback(1, desc=desc)

        async def _worker() -> None:
            while True:
                convo_id = await queue.get()
                if convo_id is None:
                    queue.task_done()
                    return
                try:
                    await _render_one(convo_id)
                finally:
                    queue.task_done()

        # Use read pool for connection reuse if backend is available
        async def _run_workers() -> None:
            workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]

            async for convo_id in _iter_conversation_ids(conversation_ids):
                await queue.put(convo_id)

            await queue.join()

            for _ in range(worker_count):
                await queue.put(None)

            await asyncio.gather(*workers, return_exceptions=False)

        if self.backend is not None:
            async with self.backend.read_pool(size=worker_count):
                await _run_workers()
        else:
            await _run_workers()

        result.rss_end_mb = read_current_rss_mb()
        result.observe_current_rss()
        return result


async def _iter_conversation_ids(
    conversation_ids: Iterable[str] | AsyncIterable[str],
) -> AsyncIterator[str]:
    """Normalize sync and async conversation ID sources."""
    if isinstance(conversation_ids, AsyncIterable):
        async for convo_id in conversation_ids:
            yield convo_id
        return

    for convo_id in conversation_ids:
        yield convo_id
