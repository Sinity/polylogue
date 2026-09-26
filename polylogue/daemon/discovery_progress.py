"""Cheap, process-local progress for the currently active source discovery."""

from __future__ import annotations

import threading
import time
import weakref
from dataclasses import dataclass

from polylogue.logging import emit

_LOG_INTERVAL_S = 15.0


@dataclass(slots=True)
class _ActiveDiscovery:
    token: object
    source: str
    started: float
    last_advanced: float
    last_logged: float
    owner_ref: weakref.ReferenceType[object] | None = None
    inspected: int = 0
    accepted: int = 0
    rejected: int = 0
    running: bool = True


_lock = threading.Lock()
_running: dict[object, _ActiveDiscovery] = {}
_pending: dict[object, _ActiveDiscovery] = {}
_last_begin_log: dict[str, float] = {}
_last_end_state: dict[str, tuple[int, int, int, bool, bool]] = {}


def _prune_dead_pending() -> None:
    for token, pending in tuple(_pending.items()):
        if pending.owner_ref is not None and pending.owner_ref() is None:
            del _pending[token]


def begin_discovery(source: str, *, owner: object | None = None) -> object:
    """Begin before entering a worker's first filesystem probe."""
    token = object()
    now = time.monotonic()
    with _lock:
        active = None
        for prior_token, pending in tuple(_pending.items()):
            if pending.source == source and (
                (pending.owner_ref is None and owner is None)
                or (pending.owner_ref is not None and pending.owner_ref() is owner)
            ):
                active = _pending.pop(prior_token)
                break
        if active is None:
            active = _ActiveDiscovery(token, source, now, now, now, weakref.ref(owner) if owner is not None else None)
        active.token = token
        active.running = True
        _running[token] = active
        considered, accepted, rejected = active.inspected, active.accepted, active.rejected
        duration_ms = (now - active.started) * 1000
        age_ms = (now - active.last_advanced) * 1000
        should_log = source not in _last_begin_log
        if should_log:
            _last_begin_log[source] = now
    if should_log:
        emit(
            "daemon.intake.discovery",
            outcome="ok",
            phase="discovering",
            source_name=source,
            considered=considered,
            accepted=accepted,
            rejected=rejected,
            duration_ms=duration_ms,
            age_ms=age_ms,
        )
    return token


def advance_discovery(token: object, *, inspected: int = 0, disposition: str | None = None) -> None:
    """Count one cheap walk observation; emit only on a timed interval."""
    with _lock:
        active = _running.get(token)
        if active is None:
            return
        now = time.monotonic()
        active.inspected += inspected
        active.accepted += disposition == "accepted"
        active.rejected += disposition in {"excluded", "fault", "alias"}
        active.last_advanced = now
        if now - active.last_logged < _LOG_INTERVAL_S:
            return
        active.last_logged = now
        source, considered, accepted, rejected = (active.source, active.inspected, active.accepted, active.rejected)
        duration_ms = (now - active.started) * 1000
    emit(
        "daemon.intake.discovery",
        outcome="ok",
        phase="discovering",
        source_name=source,
        considered=considered,
        accepted=accepted,
        rejected=rejected,
        duration_ms=duration_ms,
        age_ms=0.0,
    )


def end_discovery(token: object, *, failed: bool = False, pending: bool = False) -> None:
    with _lock:
        active = _running.pop(token, None)
        if active is None:
            return
        active.running = False
        if pending and not failed:
            _pending[token] = active
        now = time.monotonic()
        source, considered, accepted, rejected = (active.source, active.inspected, active.accepted, active.rejected)
        duration_ms = (now - active.started) * 1000
        age_ms = (now - active.last_advanced) * 1000
        state = (considered, accepted, rejected, pending, failed)
        should_log = failed or (considered > 0 and _last_end_state.get(source) != state)
        _last_end_state[source] = state
    if should_log:
        emit(
            "daemon.intake.discovery",
            outcome="error" if failed else "ok",
            phase="discovery_pending" if pending and not failed else "discovered",
            source_name=source,
            considered=considered,
            accepted=accepted,
            rejected=rejected,
            duration_ms=duration_ms,
            age_ms=age_ms,
        )


def active_discovery_payload() -> dict[str, object] | None:
    """Read counters without filesystem or database I/O."""
    with _lock:
        _prune_dead_pending()
        running = tuple(_running.values())
        pending = tuple(_pending.values())
        candidates = running + pending
        if not candidates:
            return None
        current = min(running or pending, key=lambda state: state.started)
        now = time.monotonic()
        payload: dict[str, object] = {
            "mode": "discovering" if running else "discovery_pending",
            "current_phase": "discovering" if running else "discovery_pending",
            "current_source": current.source,
            "current_path": None,
            "discovery_pending": True,
            "discovery_age_s": round(max(0.0, now - min(state.started for state in candidates)), 3),
            "discovery_last_advanced_age_s": round(max(0.0, now - min(state.last_advanced for state in candidates)), 3),
            "discovery_inspected_count": sum(state.inspected for state in candidates),
            "discovery_accepted_count": sum(state.accepted for state in candidates),
            "discovery_rejected_count": sum(state.rejected for state in candidates),
            "discovery_active_walk_count": len(running),
            "discovery_pending_walk_count": len(pending),
            "discovery_counter_scope": "all_active_and_pending_walks",
            "planned_file_count": None,
            "eta_s": None,
        }
    return payload


def log_discovery_progress_if_due() -> None:
    """Sample active work from the daemon's background status cadence."""
    with _lock:
        _prune_dead_pending()
        now = time.monotonic()
        due = []
        for active in _running.values():
            if now - active.last_logged < _LOG_INTERVAL_S:
                continue
            active.last_logged = now
            due.append(
                (
                    active.source,
                    active.running,
                    active.inspected,
                    active.accepted,
                    active.rejected,
                    (now - active.started) * 1000,
                    (now - active.last_advanced) * 1000,
                )
            )
    for source, running, considered, accepted, rejected, duration_ms, age_ms in due:
        emit(
            "daemon.intake.discovery",
            outcome="ok",
            phase="discovering" if running else "discovery_pending",
            source_name=source,
            considered=considered,
            accepted=accepted,
            rejected=rejected,
            duration_ms=duration_ms,
            age_ms=age_ms,
        )


def reset_discovery_progress() -> None:
    """Clear process-local state when a daemon/test status lifecycle resets."""
    with _lock:
        _running.clear()
        _pending.clear()
        _last_begin_log.clear()
        _last_end_state.clear()


def overlay_active_discovery(payload: dict[str, object]) -> dict[str, object]:
    active = active_discovery_payload()
    if active is None:
        return payload
    result = dict(payload)
    catchup = result.get("catchup")
    result["catchup"] = {**(catchup if isinstance(catchup, dict) else {}), **active}
    return result
