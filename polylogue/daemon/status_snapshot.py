"""Bounded daemon status snapshots for request-time status surfaces.

Testmon fan-out note (polylogue-9e5.11): this file is a testmon dependency
"hub" -- its recorded fingerprint touches essentially every test in the
suite, so a change here gets no narrowing benefit from testmon (expect a
full-suite-equivalent selection regardless of edit size). Review changes
with that blast radius in mind; see docs/test-economics.md.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.core.json import JSONDocument, json_document
from polylogue.daemon.fts_status import fts_readiness_info
from polylogue.paths import active_index_db_path
from polylogue.readiness.capability import (
    STATUS_SNAPSHOT_FRESHNESS_MAX_AGE_S,
    normalize_raw_frontier_status_payload,
    unknown_raw_frontier_integrity_projection,
)

_SNAPSHOT_LOCK = threading.Lock()
_REFRESH_LOCK = threading.Lock()
_RUNTIME_COMPONENT_LOCK = threading.Lock()
_SNAPSHOT: StatusSnapshot | None = None


@dataclass(frozen=True, slots=True)
class RuntimeComponentState:
    """Request-safe daemon component flags captured at daemon startup."""

    api_enabled: bool | None = None
    watcher_enabled: bool | None = None
    watcher_roots: tuple[str, ...] = ()
    browser_capture_enabled: bool | None = None
    browser_capture_spool_path: Path | None = None


_RUNTIME_COMPONENT_STATE = RuntimeComponentState()


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """One cached daemon status payload plus refresh metadata."""

    payload: JSONDocument
    captured_monotonic: float
    captured_at: str
    refresh_error: str | None = None

    def with_metadata(self) -> JSONDocument:
        age_s = max(0.0, time.monotonic() - self.captured_monotonic)
        state = "fresh" if age_s <= STATUS_SNAPSHOT_FRESHNESS_MAX_AGE_S else "stale"
        base_payload: dict[str, object] = dict(self.payload)
        base_payload.setdefault("component_readiness", _minimal_component_readiness(base_payload))
        payload: dict[str, object] = normalize_raw_frontier_status_payload(
            base_payload,
            snapshot_state=state,
        )
        payload["status_snapshot"] = {
            "state": state,
            "captured_at": self.captured_at,
            "age_s": round(age_s, 3),
            "refresh_error": self.refresh_error,
        }
        payload["daemon_write_coordinator"] = _daemon_write_coordinator_payload()
        return json_document(payload)


def configure_runtime_components(
    *,
    api_enabled: bool | None = None,
    watcher_enabled: bool | None = None,
    watcher_roots: tuple[str, ...] = (),
    browser_capture_enabled: bool | None = None,
    browser_capture_spool_path: Path | None = None,
) -> None:
    """Record daemon component switches for request-safe status snapshots."""
    global _RUNTIME_COMPONENT_STATE
    with _RUNTIME_COMPONENT_LOCK:
        _RUNTIME_COMPONENT_STATE = RuntimeComponentState(
            api_enabled=api_enabled,
            watcher_enabled=watcher_enabled,
            watcher_roots=tuple(watcher_roots),
            browser_capture_enabled=browser_capture_enabled,
            browser_capture_spool_path=browser_capture_spool_path,
        )


def _runtime_component_state() -> RuntimeComponentState:
    with _RUNTIME_COMPONENT_LOCK:
        return _RUNTIME_COMPONENT_STATE


def _component_state_from_flag(flag: bool | None, *, default_when_unknown: str = "unknown") -> str:
    if flag is True:
        return "running"
    if flag is False:
        return "stopped"
    return default_when_unknown


def _disk_free_bytes(path: Path) -> int:
    """Return free bytes for the nearest existing parent of ``path``."""
    target = path if path.exists() else path.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        st = os.statvfs(target)
    except OSError:
        return 0
    return int(st.f_frsize * st.f_bavail)


def _minimal_status_payload(*, refresh_in_progress: bool = False, refresh_error: str | None = None) -> JSONDocument:
    """Return a request-safe status envelope with no archive-scale scans."""
    from polylogue.daemon.status import _check_daemon_liveness, browser_capture_status_public_payload

    dbf = active_index_db_path()
    wal = dbf.with_suffix(".db-wal")
    fts_payload: dict[str, object] = {}
    if dbf.exists():
        try:
            fts_payload = fts_readiness_info(dbf)
        except Exception as exc:
            refresh_error = refresh_error or str(exc)
    now = datetime.now(UTC).isoformat()
    runtime = _runtime_component_state()
    browser_capture = dict(browser_capture_status_public_payload(runtime.browser_capture_spool_path))
    browser_capture_enabled = runtime.browser_capture_enabled is True
    browser_capture["active"] = browser_capture_enabled
    frontier_reason = refresh_error or "rich status snapshot unavailable"
    payload: dict[str, object] = {
        "ok": False,
        "daemon": "polylogued",
        "daemon_liveness": _check_daemon_liveness(),
        "checked_at": now,
        "component_state": {
            "watcher": _component_state_from_flag(runtime.watcher_enabled),
            "api": _component_state_from_flag(runtime.api_enabled, default_when_unknown="running"),
            "browser_capture": _component_state_from_flag(runtime.browser_capture_enabled),
        },
        "live": False,
        "browser_capture": json_document(browser_capture),
        "db_path": str(dbf),
        "db_size_bytes": dbf.stat().st_size if dbf.exists() else 0,
        "wal_size_bytes": wal.stat().st_size if wal.exists() else 0,
        "blob_dir_size_bytes": 0,
        "disk_free_bytes": _disk_free_bytes(dbf),
        "quick_check_result": None,
        "quick_check_age_s": None,
        "watcher_roots": list(runtime.watcher_roots),
        "browser_capture_active": browser_capture_enabled,
        "failing_files": [],
        "live_cursor": {},
        "live_ingest_attempts": {},
        "catchup": {},
        "convergence": {},
        "operations": [],
        "last_ingestion_batch": None,
        "fts_readiness": fts_payload,
        "raw_materialization_readiness": {
            "available": False,
            "total": 0,
            "critical": 0,
            "warning": 0,
            "actionable": 0,
            "blocked": 0,
            "affected_total": 0,
            "affected_actionable": 0,
            "affected_open": 0,
            "category_counts": {},
            "source_family_counts": {},
            "sampled_rows": [],
        },
        "raw_frontier_integrity": _minimal_raw_frontier_integrity(frontier_reason),
        "embedding_readiness": {},
        "memory": {},
        "health": {},
        "daemon_write_coordinator": _daemon_write_coordinator_payload(),
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


def _minimal_raw_frontier_integrity(reason: str) -> dict[str, object]:
    """Return an explicit unknown projection when the rich tier scan has not run."""

    return unknown_raw_frontier_integrity_projection(reason).to_dict()


def _daemon_write_coordinator_payload() -> dict[str, object]:
    """Project current in-process writer state into every status snapshot."""
    from polylogue.daemon.write_coordinator import daemon_write_telemetry_payload

    return daemon_write_telemetry_payload()


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
        "raw_materialization": _minimal_component(
            "raw_materialization",
            "archive",
            "unknown",
            "minimal snapshot",
        ),
        "raw_frontier_integrity": _minimal_component(
            "raw_frontier_integrity",
            "archive",
            "unknown",
            "minimal snapshot",
        ),
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


def refresh_status_snapshot(*, payload: JSONDocument | None = None, rich: bool = True) -> StatusSnapshot:
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
                if rich:
                    from polylogue.daemon.status import daemon_status_payload

                    # Raw replay selection expands authority cohorts and is a
                    # diagnostic operation, not a bounded health projection.
                    # Running it in the periodic snapshot can strand a
                    # default-executor worker through process shutdown.
                    payload = daemon_status_payload(
                        include_raw_replay_backlog=False,
                        include_exact_raw_materialization_readiness=False,
                        include_archive_debt=False,
                    )
                else:
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
    global _SNAPSHOT, _RUNTIME_COMPONENT_STATE
    with _SNAPSHOT_LOCK:
        _SNAPSHOT = None
    with _RUNTIME_COMPONENT_LOCK:
        _RUNTIME_COMPONENT_STATE = RuntimeComponentState()


def snapshot_state_for_metrics() -> dict[str, Any]:
    """Return bounded snapshot metadata for metrics."""
    with _SNAPSHOT_LOCK:
        snapshot = _SNAPSHOT
    if snapshot is None:
        return {"age_s": -1.0, "state": "missing", "refresh_error": ""}
    age_s = max(0.0, time.monotonic() - snapshot.captured_monotonic)
    return {
        "age_s": round(age_s, 3),
        "state": "fresh" if age_s <= STATUS_SNAPSHOT_FRESHNESS_MAX_AGE_S else "stale",
        "refresh_error": snapshot.refresh_error or "",
    }


__all__ = [
    "configure_runtime_components",
    "get_status_snapshot_payload",
    "refresh_status_snapshot",
    "reset_status_snapshot",
    "snapshot_state_for_metrics",
]
