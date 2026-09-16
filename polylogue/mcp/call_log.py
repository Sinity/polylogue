"""Durable MCP call telemetry outbox and daemon transport."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from polylogue.core.durable_fs import atomic_replace, sync_directory
from polylogue.logging import DEBUG, WARNING, emit, get_logger

if TYPE_CHECKING:
    from polylogue.config import PolylogueConfig

logger = get_logger(__name__)
_QUEUE_CAPACITY = 256
_POST_TIMEOUT_S = 0.5
_OUTBOX_VERSION = 1
_SCAN_INTERVAL_S = 0.25
_DRAIN_BATCH_SIZE = 64
_MAX_RETRY_DELAY_S = 30.0

# Mirrors of the daemon's own validation in
# ``polylogue.daemon.http._handle_mcp_call_log``: a record that violates any of
# these is rejected with 400 forever, so the client must refuse to spool it
# rather than mint a poison record.
_MAX_PAYLOAD_BYTES = 16_384
_MAX_SESSION_IDS = 256
_MAX_SESSION_ID_CHARS = 2048
_MAX_ERROR_DETAIL_CHARS = 512

# Aggregate admission control for the durable outbox.  Telemetry is not archive
# evidence, so shedding a *new* record under pressure is proportionate -- but
# only when the drop is counted and visible (``McpCallOutboxStatus.records_dropped``
# plus a warning log).
_MAX_PENDING_RECORDS = 4096
_MAX_PENDING_BYTES = 32 * 1024 * 1024


class McpCallLogRecordRejectedError(Exception):
    """A telemetry record was refused local durable admission."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class McpCallLogEvent:
    """One completed MCP invocation, ready for daemon-owned persistence."""

    call_id: str
    tool_name: str
    session_id: str | None
    session_ids: tuple[str, ...]
    started_at_ms: int
    finished_at_ms: int
    success: bool
    error_detail: str | None


@dataclass(frozen=True, slots=True)
class _Delivery:
    event: McpCallLogEvent
    daemon_url: str
    auth_token: str | None


@dataclass(frozen=True, slots=True)
class McpCallOutboxStatus:
    """Observable pressure and durable delivery debt for one outbox."""

    pending_count: int
    pending_bytes: int
    quarantined_count: int
    quarantined_bytes: int
    oldest_started_at_ms: int | None
    wake_queue_depth: int
    wakeups_dropped: int
    delivery_failures: int
    records_dropped: int = 0


def _outbox_root(config: PolylogueConfig) -> Path:
    from polylogue.paths import state_home

    archive_identity = str(Path(config.archive_root).expanduser().resolve())
    namespace = hashlib.sha256(archive_identity.encode()).hexdigest()[:20]
    return state_home() / "mcp-call-log" / namespace / "pending"


def _fsync_directory(path: Path) -> None:
    sync_directory(path)


def _validate_event(event: McpCallLogEvent) -> None:
    """Refuse a record the daemon would permanently 400."""
    if len(event.session_ids) > _MAX_SESSION_IDS:
        raise McpCallLogRecordRejectedError(f"session_ids exceeds {_MAX_SESSION_IDS}")
    if event.session_id is not None and len(event.session_id) > _MAX_SESSION_ID_CHARS:
        raise McpCallLogRecordRejectedError(f"session_id exceeds {_MAX_SESSION_ID_CHARS} chars")
    if any(len(value) > _MAX_SESSION_ID_CHARS for value in event.session_ids):
        raise McpCallLogRecordRejectedError(f"session_ids entry exceeds {_MAX_SESSION_ID_CHARS} chars")
    if event.error_detail is not None and len(event.error_detail) > _MAX_ERROR_DETAIL_CHARS:
        raise McpCallLogRecordRejectedError(f"error_detail exceeds {_MAX_ERROR_DETAIL_CHARS} chars")


def _outbox_pressure(root: Path) -> tuple[int, int]:
    """Current (record count, byte total) of one pending directory."""
    count = 0
    total = 0
    if root.is_dir():
        for path in root.glob("*.json"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
            count += 1
    return count, total


def _persist_delivery(config: PolylogueConfig, event: McpCallLogEvent) -> Path:
    """Atomically establish the local durable-delivery boundary.

    Raises :class:`McpCallLogRecordRejectedError` instead of spooling a record the
    daemon would permanently reject, or one that would push the outbox past
    its declared aggregate admission limits.
    """
    _validate_event(event)
    root = _outbox_root(config)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        logger.debug("Could not tighten MCP call outbox permissions", exc_info=True)
    target = root / f"{event.call_id}.json"
    payload = {
        "version": _OUTBOX_VERSION,
        "event": asdict(event),
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    # The daemon's limit applies to the posted event body, not the spool
    # envelope; measure exactly what ``_post_call_log`` will send.
    wire_bytes = len(json.dumps(asdict(event), separators=(",", ":")).encode("utf-8"))
    if wire_bytes > _MAX_PAYLOAD_BYTES:
        raise McpCallLogRecordRejectedError(f"payload {wire_bytes} bytes exceeds {_MAX_PAYLOAD_BYTES}")
    pending_count, pending_bytes = _outbox_pressure(root)
    if pending_count >= _MAX_PENDING_RECORDS:
        raise McpCallLogRecordRejectedError(
            f"outbox holds {pending_count} pending records (limit {_MAX_PENDING_RECORDS})"
        )
    if pending_bytes + len(body) > _MAX_PENDING_BYTES:
        raise McpCallLogRecordRejectedError(f"outbox would exceed {_MAX_PENDING_BYTES} pending bytes")
    atomic_replace(target, body)
    return target


def _read_spooled_delivery(path: Path, config: PolylogueConfig) -> _Delivery:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != _OUTBOX_VERSION:
        raise ValueError("unsupported MCP call outbox record")
    raw_event = payload.get("event")
    if not isinstance(raw_event, dict):
        raise ValueError("invalid MCP call outbox record")
    event = McpCallLogEvent(
        call_id=str(raw_event["call_id"]),
        tool_name=str(raw_event["tool_name"]),
        session_id=None if raw_event.get("session_id") is None else str(raw_event["session_id"]),
        session_ids=tuple(str(value) for value in raw_event.get("session_ids", ())),
        started_at_ms=int(raw_event["started_at_ms"]),
        finished_at_ms=int(raw_event["finished_at_ms"]),
        success=raw_event["success"],
        error_detail=None if raw_event.get("error_detail") is None else str(raw_event["error_detail"]),
    )
    if not isinstance(event.success, bool):
        raise ValueError("invalid MCP call outbox success flag")
    from polylogue.daemon.api_auth import resolve_api_auth_token

    return _Delivery(
        event=event,
        daemon_url=config.daemon_url,
        auth_token=resolve_api_auth_token(config.api_auth_token, allow_no_auth=config.api_allow_no_auth),
    )


class _McpCallLogDispatcher:
    """Wake-hint queue over a size- and count-bounded durable filesystem outbox."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Path] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._roots: dict[Path, PolylogueConfig] = {}
        self._state_lock = threading.Lock()
        self._wakeups_dropped = 0
        self._delivery_failures = 0
        self._records_dropped = 0
        self._retry_state: dict[Path, tuple[float, float]] = {}

    def submit(self, config: PolylogueConfig, event: McpCallLogEvent) -> None:
        try:
            path = _persist_delivery(config, event)
        except McpCallLogRecordRejectedError as exc:
            with self._state_lock:
                self._records_dropped += 1
            emit(
                "mcp.call_log.record_dropped",
                WARNING,
                outcome="degraded",
                reason=exc.reason,
                call_id=event.call_id,
                tool_name=event.tool_name,
                records_dropped=self._records_dropped,
            )
            return
        with self._state_lock:
            self._roots[path.parent] = config
        self._ensure_started()
        try:
            self._queue.put_nowait(path.parent)
        except queue.Full:
            with self._state_lock:
                self._wakeups_dropped += 1
            logger.debug("MCP call-log wake queue full; event remains durably spooled")

    def register(self, config: PolylogueConfig) -> None:
        root = _outbox_root(config)
        with self._state_lock:
            self._roots[root] = config
        self._ensure_started()
        try:
            self._queue.put_nowait(root)
        except queue.Full:
            with self._state_lock:
                self._wakeups_dropped += 1

    def flush(self, config: PolylogueConfig, timeout: float = 5.0) -> bool:
        """Wait until the durable outbox is acknowledged or timeout expires."""
        self.register(config)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.status(config).pending_count == 0:
                return True
            time.sleep(0.01)
        return self.status(config).pending_count == 0

    def status(self, config: PolylogueConfig) -> McpCallOutboxStatus:
        root = _outbox_root(config)
        paths = tuple(root.glob("*.json")) if root.is_dir() else ()
        quarantine = root.parent / "quarantine"
        quarantined_paths = tuple(quarantine.glob("*.json")) if quarantine.is_dir() else ()
        pending_bytes = 0
        oldest_started_at_ms: int | None = None
        for path in paths:
            try:
                pending_bytes += path.stat().st_size
                started_at_ms = _read_spooled_delivery(path, config).event.started_at_ms
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if oldest_started_at_ms is None or started_at_ms < oldest_started_at_ms:
                oldest_started_at_ms = started_at_ms
        with self._state_lock:
            return McpCallOutboxStatus(
                pending_count=len(paths),
                pending_bytes=pending_bytes,
                quarantined_count=len(quarantined_paths),
                quarantined_bytes=sum(path.stat().st_size for path in quarantined_paths if path.is_file()),
                oldest_started_at_ms=oldest_started_at_ms,
                wake_queue_depth=self._queue.qsize(),
                wakeups_dropped=self._wakeups_dropped,
                delivery_failures=self._delivery_failures,
                records_dropped=self._records_dropped,
            )

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop this worker instance without discarding durable pending files."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def quiesce(self, timeout: float = 5.0) -> None:
        """Stop the worker, drop in-process routing state, and allow restart.

        The durable outbox is untouched -- only this process's wake queue,
        registered roots, retry backoff and counters are released. Unlike
        :meth:`shutdown`, the dispatcher is usable again afterwards.

        The worker thread resolves ``api_auth_token_path()`` (and therefore
        ``XDG_DATA_HOME``) at delivery time, so a dispatcher left scanning a
        root whose daemon never answers keeps minting token files into
        whatever environment the process happens to have later. That is
        correct for a long-lived MCP server, whose environment is fixed, and
        wrong for any process that re-points XDG between units of work.
        """
        thread = self._thread
        self._stop.set()
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError("MCP call-log dispatcher thread did not stop")
        with self._state_lock:
            self._roots.clear()
            self._retry_state.clear()
            self._wakeups_dropped = 0
            self._delivery_failures = 0
            self._records_dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        with self._start_lock:
            self._thread = None
            self._stop.clear()

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
        while not self._stop.is_set():
            wake_root: Path | None = None
            try:
                wake_root = self._queue.get(timeout=_SCAN_INTERVAL_S)
            except queue.Empty:
                pass
            else:
                self._queue.task_done()
            with self._state_lock:
                roots = list(self._roots.items())
            if wake_root is not None:
                roots.sort(key=lambda item: item[0] != wake_root)
            for root, config in roots:
                with self._state_lock:
                    retry_not_before, _retry_delay = self._retry_state.get(
                        root,
                        (0.0, _SCAN_INTERVAL_S),
                    )
                if time.monotonic() < retry_not_before:
                    continue
                try:
                    self._drain_root_once(root, config)
                except Exception:
                    with self._state_lock:
                        self._delivery_failures += 1
                    logger.exception("MCP call-log outbox scan failed for %s", root)
                self._forget_idle_root(root, config)

    def _forget_idle_root(self, root: Path, config: PolylogueConfig) -> None:
        """Release drained roots without racing a concurrent submission."""
        with self._state_lock:
            if self._roots.get(root) is not config:
                return
            if root.is_dir() and next(root.glob("*.json"), None) is not None:
                return
            self._roots.pop(root, None)
            self._retry_state.pop(root, None)

    def _drain_root_once(self, root: Path, config: PolylogueConfig) -> None:
        paths = sorted(root.glob("*.json"))[:_DRAIN_BATCH_SIZE] if root.is_dir() else ()
        for path in paths:
            try:
                delivery = _read_spooled_delivery(path, config)
            except Exception:
                # An unreadable record can never become deliverable, so leaving
                # it pending would keep the root scanned forever. Settle it
                # through the same durable quarantine as a permanent rejection.
                logger.warning("Invalid MCP call-log outbox record %s", path, exc_info=True)
                self._quarantine(path, reason="unparseable_record")
                continue
            try:
                _post_call_log(delivery)
            except HTTPError as exc:
                if _is_permanent_rejection(exc.code):
                    # A permanent 4xx (conflict, or a record this daemon will
                    # never accept) must not block every later record in the
                    # root: quarantine it and keep draining.
                    self._quarantine(path, reason="permanent_rejection", status_code=exc.code)
                    continue
                with self._state_lock:
                    self._delivery_failures += 1
                    _not_before, retry_delay = self._retry_state.get(
                        root,
                        (0.0, _SCAN_INTERVAL_S),
                    )
                    self._retry_state[root] = (
                        time.monotonic() + retry_delay,
                        min(retry_delay * 2, _MAX_RETRY_DELAY_S),
                    )
                logger.debug("MCP call-log delivery failed for %s", path.name, exc_info=True)
                break
            except Exception:
                with self._state_lock:
                    self._delivery_failures += 1
                    _not_before, retry_delay = self._retry_state.get(
                        root,
                        (0.0, _SCAN_INTERVAL_S),
                    )
                    self._retry_state[root] = (
                        time.monotonic() + retry_delay,
                        min(retry_delay * 2, _MAX_RETRY_DELAY_S),
                    )
                logger.debug("MCP call-log delivery failed for %s", path.name, exc_info=True)
                break
            with self._state_lock:
                self._retry_state.pop(root, None)
            path.unlink(missing_ok=True)

    def _quarantine(self, path: Path, *, reason: str, status_code: int | None = None) -> None:
        """Durably settle one record that can never be delivered as spooled."""
        quarantine = path.parent.parent / "quarantine"
        quarantine.mkdir(mode=0o700, parents=True, exist_ok=True)
        _fsync_directory(quarantine.parent)
        target = quarantine / path.name
        try:
            os.replace(path, target)
        except FileNotFoundError:
            if target.exists():
                return
            emit(
                "mcp.call_log.record_settled_elsewhere",
                DEBUG,
                outcome="ok",
                reason="settled_by_another_worker",
                path=str(path),
            )
            return
        _fsync_directory(path.parent)
        _fsync_directory(quarantine)
        with self._state_lock:
            self._delivery_failures += 1
        emit(
            "mcp.call_log.record_quarantined",
            WARNING,
            outcome="degraded",
            reason=reason,
            status_code=status_code,
            path=str(target),
        )


def _is_permanent_rejection(code: int) -> bool:
    """True for a 4xx the daemon will never accept on retry (429 is transient)."""
    return 400 <= code < 500 and code != 429


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
        receipt = json.loads(response.read())
    if not isinstance(receipt, dict) or receipt.get("call_id") != delivery.event.call_id:
        raise RuntimeError("daemon returned a mismatched MCP call-log receipt")


_DISPATCHER = _McpCallLogDispatcher()


def enqueue_mcp_call_log(
    config: PolylogueConfig,
    *,
    tool_name: str,
    session_id: str | None,
    session_ids: tuple[str, ...] = (),
    started_at_ms: int,
    finished_at_ms: int,
    success: bool,
    error_detail: str | None,
) -> None:
    """Durably spool one call, then wake the asynchronous daemon transport."""
    _DISPATCHER.submit(
        config,
        McpCallLogEvent(
            call_id=str(uuid.uuid4()),
            tool_name=tool_name,
            session_id=session_id,
            session_ids=tuple(dict.fromkeys(session_ids)),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            success=success,
            error_detail=error_detail,
        ),
    )


def flush_mcp_call_log(*, timeout: float = 5.0) -> bool:
    """Wait until all durably spooled calls receive daemon acknowledgement."""
    from polylogue.config import load_polylogue_config

    return _DISPATCHER.flush(load_polylogue_config(), timeout)


def start_mcp_call_log() -> None:
    """Start restart-recovery scanning before the first MCP invocation."""
    from polylogue.config import load_polylogue_config

    config = load_polylogue_config()
    if config.no_daemon or config.daemon_client_mode == "off":
        return
    _DISPATCHER.register(config)


def reset_mcp_call_log(timeout: float = 5.0) -> None:
    """Quiesce the process-global call-log dispatcher; durable spool is kept."""
    _DISPATCHER.quiesce(timeout)


def mcp_call_outbox_status() -> McpCallOutboxStatus:
    """Return current durable delivery debt and wake-queue pressure."""
    from polylogue.config import load_polylogue_config

    return _DISPATCHER.status(load_polylogue_config())


__all__ = [
    "McpCallLogEvent",
    "McpCallOutboxStatus",
    "enqueue_mcp_call_log",
    "flush_mcp_call_log",
    "mcp_call_outbox_status",
    "reset_mcp_call_log",
    "start_mcp_call_log",
]
