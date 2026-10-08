"""Cheap, process-local progress for the currently active source discovery."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from polylogue.logging import ERROR, INFO, emit

if TYPE_CHECKING:
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

_T = TypeVar("_T")
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


@dataclass(slots=True)
class _ColdBuildPreparation:
    """Work a cold build does before its dispatcher exists.

    Baseline enumeration and revision hashing, capacity projection, the
    source snapshot and generation creation all run inside one writer call
    before the first intake page. Without this state, status reports idle for
    that whole interval.
    """

    phase: str
    started: float
    phase_started: float
    last_advanced: float
    last_logged: float
    inspected: int = 0
    revisions: int = 0
    hashed_bytes: int = 0


_preparation: _ColdBuildPreparation | None = None


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


def abandon_discovery(owner: object) -> None:
    """Forget parked progress for an owner that can no longer be scheduled."""
    with _lock:
        for token, pending in tuple(_pending.items()):
            if pending.owner_ref is not None and pending.owner_ref() is owner:
                del _pending[token]


def begin_cold_build_preparation(phase: str = "baseline_walk") -> None:
    """Mark the start of a cold build's pre-dispatcher preparation."""
    global _preparation
    now = time.monotonic()
    with _lock:
        _preparation = _ColdBuildPreparation(phase, now, now, now, now)
    _emit_preparation(phase, 0, 0, 0, 0.0, 0.0, outcome="ok")


def advance_cold_build_preparation(
    phase: str, *, inspected: int = 0, revisions: int = 0, hashed_bytes: int = 0
) -> None:
    """Count preparation work; a phase change or a timed interval emits INFO."""
    with _lock:
        state = _preparation
        if state is None:
            return
        now = time.monotonic()
        changed = phase != state.phase
        if changed:
            state.phase = phase
            state.phase_started = now
        state.inspected += max(0, inspected)
        state.revisions += max(0, revisions)
        state.hashed_bytes += max(0, hashed_bytes)
        state.last_advanced = now
        if not changed and now - state.last_logged < _LOG_INTERVAL_S:
            return
        state.last_logged = now
        snapshot = (state.phase, state.inspected, state.revisions, state.hashed_bytes, (now - state.started) * 1000)
    _emit_preparation(*snapshot, 0.0, outcome="ok")


def end_cold_build_preparation(*, failed: bool = False, cancelled: bool = False) -> None:
    """Clear the preparation phase.

    ``cancelled`` is a shutdown or task cancellation: the phase stops without
    having failed, so it is reported at INFO as skipped rather than as an error.
    """
    global _preparation
    with _lock:
        state = _preparation
        _preparation = None
        if state is None:
            return
        now = time.monotonic()
        snapshot = (
            state.phase if failed or cancelled else "prepared",
            state.inspected,
            state.revisions,
            state.hashed_bytes,
            (now - state.started) * 1000,
            (now - state.last_advanced) * 1000,
        )
    if cancelled:
        _emit_preparation(*snapshot, outcome="skipped", reason="cancelled")
    else:
        _emit_preparation(*snapshot, outcome="error" if failed else "ok")


def settle_cold_build_preparation(done: asyncio.Task[object]) -> None:
    """End preparation from the writer execution's own completion."""
    if done.cancelled():
        end_cold_build_preparation(cancelled=True)
        return
    exception = done.exception()
    if exception is None:
        end_cold_build_preparation()
    elif isinstance(exception, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        end_cold_build_preparation(cancelled=True)
    else:
        end_cold_build_preparation(failed=True)


async def run_source_observation(observe: Callable[[threading.Event], _T]) -> _T:
    """Run a read-only source scan on its own thread, off the archive writer.

    ``observe`` receives a cancellation event that is set once this caller
    stops waiting. The thread is a daemon thread, so loop shutdown never
    joins a scan that is still reading a large source.
    """
    loop = asyncio.get_running_loop()
    completed: asyncio.Future[_T] = loop.create_future()
    cancel = threading.Event()

    def deliver(result: _T | None = None, error: BaseException | None = None) -> None:
        if completed.done():
            return
        if error is not None:
            completed.set_exception(error)
        else:
            completed.set_result(cast(_T, result))

    def run() -> None:
        try:
            result = observe(cancel)
        except BaseException as exc:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(deliver, None, exc)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(deliver, result)

    threading.Thread(target=run, name="cold-source-observation", daemon=True).start()
    try:
        return await completed
    finally:
        cancel.set()


async def run_cold_build_preparation(
    coordinator: DaemonWriteCoordinator,
    actor: str,
    observe: Callable[..., Any],
    function: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Observe the cold build's sources off the writer, then bind them in it.

    ``observe`` receives ``progress=`` and ``cancelled=`` and runs on its own
    thread: the source walk, classification and hashing only read source
    files, so they never hold the archive writer. ``function`` then receives
    ``observed=`` (the observation) and ``progress=`` as the writer call.

    The coordinator shields an admitted execution from caller cancellation,
    so a shutdown that cancels this caller does not stop that writer call;
    preparation then ends from the execution's completion, not here. A
    request cancelled before admission, which never ran, ends as cancelled
    at the caller, as does a cancelled observation.
    """
    begin_cold_build_preparation()
    try:
        observed = await run_source_observation(
            lambda cancel: observe(progress=advance_cold_build_preparation, cancelled=cancel.is_set)
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        end_cold_build_preparation(cancelled=True)
        raise
    except BaseException:
        end_cold_build_preparation(failed=True)
        raise
    admitted: list[bool] = []
    try:
        result = await coordinator.run_sync_with_completion(
            actor,
            function,
            settle_cold_build_preparation,
            lambda: admitted.append(True),
            *args,
            observed=observed,
            progress=advance_cold_build_preparation,
            **kwargs,
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        if not admitted:
            end_cold_build_preparation(cancelled=True)
        raise
    except BaseException:
        end_cold_build_preparation(failed=True)
        raise
    # The completion callback may not have run yet; ending is idempotent.
    end_cold_build_preparation()
    return result


def _emit_preparation(
    phase: str,
    inspected: int,
    revisions: int,
    hashed_bytes: int,
    duration_ms: float,
    age_ms: float,
    *,
    outcome: str,
    reason: str | None = None,
) -> None:
    fields: dict[str, object] = {} if reason is None else {"reason": reason}
    emit(
        "daemon.cold_build.preparation",
        level=ERROR if outcome == "error" else INFO,
        outcome=outcome,
        phase=phase,
        considered=inspected,
        files=revisions,
        bytes=hashed_bytes,
        duration_ms=duration_ms,
        age_ms=age_ms,
        **fields,
    )


def _preparation_payload(now: float) -> dict[str, object] | None:
    state = _preparation
    if state is None:
        return None
    return {
        "mode": "cold_build_preparing",
        "current_phase": state.phase,
        "current_source": None,
        "current_path": None,
        "last_advanced_age_s": round(max(0.0, now - state.last_advanced), 3),
        "preparation_age_s": round(max(0.0, now - state.started), 3),
        "preparation_phase_age_s": round(max(0.0, now - state.phase_started), 3),
        "preparation_inspected_count": state.inspected,
        "preparation_revision_count": state.revisions,
        "preparation_hashed_bytes": state.hashed_bytes,
        # The baseline being enumerated IS the denominator; until it is
        # sealed there is nothing to project completion against.
        "planned_file_count": None,
        "eta_s": None,
    }


def active_discovery_payload() -> dict[str, object] | None:
    """Read counters without filesystem or database I/O."""
    with _lock:
        _prune_dead_pending()
        running = tuple(_running.values())
        pending = tuple(_pending.values())
        candidates = running + pending
        if not candidates:
            return _preparation_payload(time.monotonic())
        current = min(running or pending, key=lambda state: state.started)
        now = time.monotonic()
        payload: dict[str, object] = {
            "mode": "discovering" if running else "discovery_pending",
            "current_phase": "discovering" if running else "discovery_pending",
            "current_source": current.source,
            "current_path": None,
            "discovery_pending": True,
            "discovery_age_s": round(max(0.0, now - min(state.started for state in candidates)), 3),
            # This field describes the aggregate walk population (as do the
            # counters below), so any advancing walk proves recent progress.
            # Using the oldest timestamp made one stale pending walk mask
            # progress from every active walk.
            "discovery_last_advanced_age_s": round(max(0.0, now - max(state.last_advanced for state in candidates)), 3),
            "discovery_inspected_count": sum(state.inspected for state in candidates),
            "discovery_accepted_count": sum(state.accepted for state in candidates),
            "discovery_rejected_count": sum(state.rejected for state in candidates),
            "discovery_active_walk_count": len(running),
            "discovery_pending_walk_count": len(pending),
            "discovery_counter_scope": "all_active_and_pending_walks",
        }
        preparation = _preparation_payload(now)
    # Preparation runs before the dispatcher walks anything, so a running
    # preparation describes the current phase even if an earlier walk left
    # pending counters behind.
    return {**payload, **preparation} if preparation is not None else payload


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
        preparation_due = None
        state = _preparation
        if state is not None and now - state.last_logged >= _LOG_INTERVAL_S:
            state.last_logged = now
            preparation_due = (
                state.phase,
                state.inspected,
                state.revisions,
                state.hashed_bytes,
                (now - state.started) * 1000,
                (now - state.last_advanced) * 1000,
            )
    if preparation_due is not None:
        _emit_preparation(*preparation_due, outcome="ok")
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
    global _preparation
    with _lock:
        _preparation = None
        _running.clear()
        _pending.clear()
        _last_begin_log.clear()
        _last_end_state.clear()


#: Keys that describe the whole build rather than the discovery walk.
_BUILD_PROGRESS_KEYS = frozenset({"mode", "current_phase", "current_source", "current_path"})


def overlay_active_discovery(payload: dict[str, object]) -> dict[str, object]:
    """Merge live discovery counters into the catch-up status.

    A cold build pages discovery while it admits and writes. Once the build
    has a planned denominator its mode, phase and ETA describe the build, and
    the walk only adds its ``discovery_*`` counters; overwriting them made the
    status read ``discovery_pending`` with no ETA for the whole paged walk.
    """
    active = active_discovery_payload()
    if active is None:
        return payload
    result = dict(payload)
    raw_catchup = result.get("catchup")
    catchup: dict[str, object] = dict(raw_catchup) if isinstance(raw_catchup, dict) else {}
    build_in_progress = catchup.get("planned_raw_revision_count") is not None and active.get("mode") != (
        "cold_build_preparing"
    )
    if build_in_progress:
        active = {key: value for key, value in active.items() if key not in _BUILD_PROGRESS_KEYS}
    result["catchup"] = {**catchup, **active}
    return result
