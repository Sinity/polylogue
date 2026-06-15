"""Bounded daemon status snapshots for request-time status surfaces."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from polylogue.core.json import JSONDocument, json_document
from polylogue.daemon.fts_status import fts_readiness_info
from polylogue.paths import active_index_db_path

_MAX_FRESH_AGE_S = 30.0
_SNAPSHOT_LOCK = threading.Lock()
_REFRESH_LOCK = threading.Lock()
_SNAPSHOT: StatusSnapshot | None = None


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """One cached daemon status payload plus refresh metadata."""

    payload: JSONDocument
    captured_monotonic: float
    captured_at: str
    refresh_error: str | None = None

    def with_metadata(self) -> JSONDocument:
        age_s = max(0.0, time.monotonic() - self.captured_monotonic)
        payload: dict[str, object] = dict(self.payload)
        payload.setdefault("component_readiness", _minimal_component_readiness(payload))
        payload["status_snapshot"] = {
            "state": "fresh" if age_s <= _MAX_FRESH_AGE_S else "stale",
            "captured_at": self.captured_at,
            "age_s": round(age_s, 3),
            "refresh_error": self.refresh_error,
        }
        return json_document(payload)


def _minimal_status_payload(*, refresh_in_progress: bool = False, refresh_error: str | None = None) -> JSONDocument:
    """Return a request-safe status envelope with no archive-scale scans."""
    from polylogue.daemon.status import _check_daemon_liveness, browser_capture_status_payload

    dbf = active_index_db_path()
    wal = dbf.with_suffix(".db-wal")
    fts_payload: dict[str, object] = {}
    if dbf.exists():
        try:
            fts_payload = fts_readiness_info(dbf)
        except Exception as exc:
            refresh_error = refresh_error or str(exc)
    now = datetime.now(UTC).isoformat()
    payload: dict[str, object] = {
        "ok": True,
        "daemon": "polylogued",
        "daemon_liveness": _check_daemon_liveness(),
        "checked_at": now,
        "component_state": {
            "watcher": "running",
            "api": "running",
            "browser_capture": "unknown",
        },
        "live": False,
        "browser_capture": browser_capture_status_payload(None),
        "db_path": str(dbf),
        "db_size_bytes": dbf.stat().st_size if dbf.exists() else 0,
        "wal_size_bytes": wal.stat().st_size if wal.exists() else 0,
        "blob_dir_size_bytes": 0,
        "disk_free_bytes": 0,
        "quick_check_result": None,
        "quick_check_age_s": None,
        "watcher_roots": [],
        "browser_capture_active": False,
        "failing_files": [],
        "live_cursor": {},
        "live_ingest_attempts": {},
        "catchup": {},
        "convergence": {},
        "operations": [],
        "last_ingestion_batch": None,
        "fts_readiness": fts_payload,
        "embedding_readiness": {},
        "memory": {},
        "health": {},
        "raw_parse_failures": 0,
        "raw_validation_failures": 0,
        "raw_quarantined": 0,
        "raw_maintenance_failures": 0,
        "raw_detection_warnings": 0,
        "raw_failure_samples": [],
        "status_snapshot": {
            "state": "refreshing" if refresh_in_progress else "minimal",
            "captured_at": now,
            "age_s": 0.0,
            "refresh_error": refresh_error,
        },
    }
    payload["component_readiness"] = _minimal_component_readiness(payload)
    return json_document(payload)


def _minimal_component_readiness(payload: Mapping[str, object]) -> dict[str, object]:
    component_state = payload.get("component_state")
    fts = payload.get("fts_readiness")
    if not isinstance(component_state, dict):
        component_state = {}
    if not isinstance(fts, dict):
        fts = {}
    api_state = str(component_state.get("api", "unknown"))
    watcher_state = str(component_state.get("watcher", "unknown"))
    browser_state = str(component_state.get("browser_capture", "unknown"))
    fts_ready = bool(fts.get("messages_ready", False))
    return {
        "daemon_api": _minimal_component("daemon_api", "daemon", _state_to_readiness(api_state), api_state),
        "daemon_watcher": _minimal_component(
            "daemon_watcher",
            "daemon",
            _state_to_readiness(watcher_state),
            watcher_state,
        ),
        "browser_capture": _minimal_component(
            "browser_capture",
            "daemon",
            _state_to_readiness(browser_state),
            browser_state,
        ),
        "archive_storage": _minimal_component("archive_storage", "archive", "unknown", "minimal snapshot"),
        "daemon_ingest": _minimal_component("daemon_ingest", "daemon", "unknown", "minimal snapshot"),
        "embeddings": _minimal_component("embeddings", "semantic", "unknown", "minimal snapshot"),
        "search": _minimal_component(
            "search",
            "lexical",
            "ready" if fts_ready else "unknown",
            "ready" if fts_ready else "unknown",
        ),
    }


def _state_to_readiness(state: str) -> str:
    return {
        "running": "ready",
        "degraded": "degraded",
        "stopped": "missing",
        "disabled": "missing",
    }.get(state, "unknown")


def _minimal_component(component: str, scope: str, state: str, summary: str) -> dict[str, object]:
    return {
        "component": component,
        "scope": scope,
        "state": state,
        "summary": summary,
        "last_success": None,
        "last_attempt": None,
        "counts": {},
        "caveats": [],
        "repair_hint": None,
        "evidence_refs": [],
    }


def get_status_snapshot_payload() -> JSONDocument:
    """Return the current cached status payload or a minimal request-safe one."""
    with _SNAPSHOT_LOCK:
        snapshot = _SNAPSHOT
    if snapshot is not None:
        return snapshot.with_metadata()
    return _minimal_status_payload(refresh_in_progress=_REFRESH_LOCK.locked())


def refresh_status_snapshot(*, payload: JSONDocument | None = None) -> StatusSnapshot:
    """Refresh the global daemon status snapshot if no refresh is in progress."""
    global _SNAPSHOT
    if not _REFRESH_LOCK.acquire(blocking=False):
        with _SNAPSHOT_LOCK:
            snapshot = _SNAPSHOT
        if snapshot is not None:
            return snapshot
        return StatusSnapshot(
            payload=_minimal_status_payload(refresh_in_progress=True),
            captured_monotonic=time.monotonic(),
            captured_at=datetime.now(UTC).isoformat(),
        )
    try:
        captured_at = datetime.now(UTC).isoformat()
        refresh_error: str | None = None
        try:
            if payload is None:
                payload = _minimal_status_payload()
        except Exception as exc:
            refresh_error = str(exc)
            payload = _minimal_status_payload(refresh_error=refresh_error)
        snapshot = StatusSnapshot(
            payload=json_document(dict(payload)),
            captured_monotonic=time.monotonic(),
            captured_at=captured_at,
            refresh_error=refresh_error,
        )
        with _SNAPSHOT_LOCK:
            _SNAPSHOT = snapshot
        return snapshot
    finally:
        _REFRESH_LOCK.release()


def reset_status_snapshot() -> None:
    """Drop the cached daemon status snapshot.

    The process-wide ``_SNAPSHOT`` singleton is intentionally long-lived in a
    running daemon. Tests that prime it via :func:`refresh_status_snapshot`
    (e.g. with a deliberately minimal payload) must clear it afterwards so the
    cached payload does not leak into unrelated status-surface tests in the same
    process. Exposed for test teardown; not used in production.
    """
    global _SNAPSHOT
    with _SNAPSHOT_LOCK:
        _SNAPSHOT = None


def snapshot_state_for_metrics() -> dict[str, Any]:
    """Return bounded snapshot metadata for metrics."""
    with _SNAPSHOT_LOCK:
        snapshot = _SNAPSHOT
    if snapshot is None:
        return {"age_s": -1.0, "state": "missing", "refresh_error": ""}
    age_s = max(0.0, time.monotonic() - snapshot.captured_monotonic)
    return {
        "age_s": round(age_s, 3),
        "state": "fresh" if age_s <= _MAX_FRESH_AGE_S else "stale",
        "refresh_error": snapshot.refresh_error or "",
    }


__all__ = [
    "get_status_snapshot_payload",
    "refresh_status_snapshot",
    "snapshot_state_for_metrics",
]
