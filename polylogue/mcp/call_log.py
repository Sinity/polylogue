"""Best-effort MCP call telemetry transport to the archive daemon."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

from polylogue.logging import get_logger

if TYPE_CHECKING:
    from polylogue.config import PolylogueConfig

logger = get_logger(__name__)
_QUEUE_CAPACITY = 256
_POST_TIMEOUT_S = 0.5


@dataclass(frozen=True, slots=True)
class McpCallLogEvent:
    """One completed MCP invocation, ready for daemon-owned persistence."""

    call_id: str
    tool_name: str
    session_id: str | None
    started_at_ms: int
    finished_at_ms: int
    success: bool
    error_detail: str | None


@dataclass(frozen=True, slots=True)
class _Delivery:
    event: McpCallLogEvent
    daemon_url: str
    auth_token: str | None


class _McpCallLogDispatcher:
    """Bounded, non-blocking producer with one best-effort HTTP worker."""

    def __init__(self) -> None:
        self._queue: queue.Queue[_Delivery] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, delivery: _Delivery) -> None:
        self._ensure_started()
        try:
            self._queue.put_nowait(delivery)
        except queue.Full:
            logger.debug("MCP call-log queue full; dropping %s", delivery.event.tool_name)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for queued deliveries; intended for deterministic shutdown/tests."""
        settled = threading.Event()

        def wait_for_queue() -> None:
            self._queue.join()
            settled.set()

        threading.Thread(target=wait_for_queue, name="mcp-call-log-flush", daemon=True).start()
        return settled.wait(timeout)

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="mcp-call-log-sender",
                    daemon=True,
                )
                self._thread.start()

    def _run(self) -> None:
        while True:
            delivery = self._queue.get()
            try:
                _post_call_log(delivery)
            except Exception:
                logger.debug(
                    "MCP call-log delivery failed for %s",
                    delivery.event.tool_name,
                    exc_info=True,
                )
            finally:
                self._queue.task_done()


def _post_call_log(delivery: _Delivery) -> None:
    body = json.dumps(asdict(delivery.event), separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if delivery.auth_token:
        headers["Authorization"] = f"Bearer {delivery.auth_token}"
    request = Request(
        f"{delivery.daemon_url.rstrip('/')}/api/telemetry/mcp-calls",
        data=body,
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=_POST_TIMEOUT_S) as response:
        if response.status != 200:
            raise RuntimeError(f"daemon returned HTTP {response.status}")


_DISPATCHER = _McpCallLogDispatcher()


def enqueue_mcp_call_log(
    config: PolylogueConfig,
    *,
    tool_name: str,
    session_id: str | None,
    started_at_ms: int,
    finished_at_ms: int,
    success: bool,
    error_detail: str | None,
) -> None:
    """Queue one call without blocking the MCP response or writing SQLite."""
    _DISPATCHER.submit(
        _Delivery(
            event=McpCallLogEvent(
                call_id=str(uuid.uuid4()),
                tool_name=tool_name,
                session_id=session_id,
                started_at_ms=started_at_ms,
                finished_at_ms=finished_at_ms,
                success=success,
                error_detail=error_detail,
            ),
            daemon_url=config.daemon_url,
            auth_token=config.api_auth_token,
        )
    )


def flush_mcp_call_log(*, timeout: float = 5.0) -> bool:
    """Wait until all queued MCP call-log events have been attempted."""
    return _DISPATCHER.flush(timeout)


__all__ = ["McpCallLogEvent", "enqueue_mcp_call_log", "flush_mcp_call_log"]
