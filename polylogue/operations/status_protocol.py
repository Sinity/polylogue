"""Budgeted component-snapshot protocol shared by status surfaces (polylogue-20d.17).

Daemon/archive status (``polylogued status``, ``/api/status``) and
agent-coordination status previously combined millisecond facts with
multi-second raw/debt/embedding/Beads/process/archive/handoff probes in one
synchronous call, so one slow probe stalled every answer. This module gives
both consumers one ``StatusComponentSpec``/``ComponentSnapshot`` protocol:
each declared component collects independently under its own deadline and
reports an explicit ``fresh``/``stale``/``refreshing``/``timed_out``/
``unavailable``/``degraded`` state instead of hanging (or silently
mislabeling) the whole response.

Collectors run on plain daemon threads, not ``concurrent.futures``: that
executor registers a process-exit join that would hang the interpreter
forever behind one permanently-stuck collector (a real risk here — some
collectors do blocking SQLite/subprocess work). At most one attempt is ever
in flight per component: a component whose collector is still running from a
prior deadline miss is reported ``refreshing``/``timed_out`` rather than
resubmitted, so the thread count stays bounded by the number of declared
components no matter how many ``collect()`` calls happen while it hangs.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Literal

ComponentState = Literal["fresh", "stale", "refreshing", "timed_out", "unavailable", "degraded"]

_POLL_STEP_S = 0.02


class ComponentUnavailableError(Exception):
    """Raise from a collector for a structurally-absent source (not a transient error)."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class StatusComponentSpec:
    """Declares one independently-scheduled status fact.

    ``deadline_s`` bounds how long a fresh collection attempt is waited on
    before this component reports ``timed_out`` (the collector keeps running
    in the background; nothing else is blocked). ``ttl_s`` bounds how long a
    successful value is served as ``fresh`` before a background refresh is
    kicked and the (still-valid) value is served as ``stale`` in the
    meantime. ``fingerprint`` (if given) forces a refresh whenever its return
    value changes, even inside the TTL window, so a changed source cannot be
    hidden by an unexpired TTL. ``detail_only`` components are excluded from
    the default/compact ``collect()`` selection — they are the explicit,
    opt-in "exact diagnostics" the design reserves for a resumable detail
    query, never run inline on the default path.
    """

    name: str
    scope: str
    collector: Callable[[], Any]
    deadline_s: float = 1.0
    cost_class: Literal["cheap", "moderate", "expensive"] = "cheap"
    fingerprint: Callable[[], str] | None = None
    ttl_s: float = 10.0
    detail_only: bool = False


@dataclass(frozen=True, slots=True)
class ComponentSnapshot:
    """One component's most recent collection outcome."""

    name: str
    scope: str
    state: ComponentState
    value: Any
    captured_at: str
    age_s: float
    deadline_s: float
    fingerprint: str | None = None
    error: str | None = None
    last_good_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.name,
            "scope": self.scope,
            "state": self.state,
            "value": self.value,
            "captured_at": self.captured_at,
            "age_s": round(self.age_s, 3),
            "deadline_s": self.deadline_s,
            "fingerprint": self.fingerprint,
            "error": self.error,
            "last_good_at": self.last_good_at,
        }


@dataclass
class _Attempt:
    """One in-flight or completed collector invocation, run on a daemon thread."""

    thread: threading.Thread
    done: threading.Event
    submitted_monotonic: float
    reported_timeout: bool = False
    outcome: list[tuple[str, Any]] = field(default_factory=list)


@dataclass
class _Good:
    """The most recent successful collection, kept for last-good evidence."""

    snapshot: ComponentSnapshot
    captured_monotonic: float
    fingerprint: str | None


class StatusComponentRegistry:
    """Collects declared components independently, retaining last-good evidence."""

    def __init__(self, specs: Sequence[StatusComponentSpec]) -> None:
        self._specs: dict[str, StatusComponentSpec] = {spec.name: spec for spec in specs}
        self._lock = threading.Lock()
        self._pending: dict[str, _Attempt] = {}
        self._good: dict[str, _Good] = {}

    @property
    def specs(self) -> tuple[StatusComponentSpec, ...]:
        return tuple(self._specs.values())

    def collect(self, *, names: Sequence[str] | None = None) -> dict[str, ComponentSnapshot]:
        """Return current snapshots for the named (default: non-detail) components.

        Wall time is bounded by the slowest component that has never
        completed a successful collection, not by the sum of every
        component's deadline: components with a cached value (fresh or
        stale) resolve immediately, and only genuinely-new collections are
        waited on, concurrently, each capped at its own ``deadline_s``.
        """
        selected = self._select(names)
        results: dict[str, ComponentSnapshot] = {}
        waiting: dict[str, tuple[StatusComponentSpec, _Attempt, float]] = {}

        with self._lock:
            for spec in selected:
                attempt = self._pending.get(spec.name)
                good = self._good.get(spec.name)
                fp = self._safe_fingerprint(spec)
                fp_changed = fp is not None and good is not None and good.fingerprint not in (None, fp)

                if attempt is not None and not attempt.done.is_set():
                    if attempt.reported_timeout:
                        results[spec.name] = self._refreshing_snapshot(spec, good)
                    else:
                        waiting[spec.name] = (spec, attempt, attempt.submitted_monotonic + spec.deadline_s)
                    continue

                if attempt is not None and attempt.done.is_set():
                    results[spec.name] = self._finalize_locked(spec, attempt, fp)
                    continue

                if good is not None and not fp_changed and (monotonic() - good.captured_monotonic) < spec.ttl_s:
                    results[spec.name] = self._fresh_snapshot(good)
                    continue

                new_attempt = self._start_attempt_locked(spec)
                if good is not None:
                    results[spec.name] = self._stale_snapshot(spec, good)
                else:
                    waiting[spec.name] = (spec, new_attempt, new_attempt.submitted_monotonic + spec.deadline_s)

        while waiting:
            for name, (spec, attempt, deadline_mono) in list(waiting.items()):
                remaining = max(0.0, min(_POLL_STEP_S, deadline_mono - monotonic()))
                if attempt.done.wait(timeout=remaining):
                    with self._lock:
                        results[name] = self._finalize_locked(spec, attempt, self._safe_fingerprint(spec))
                    del waiting[name]
                elif monotonic() >= deadline_mono:
                    with self._lock:
                        attempt.reported_timeout = True
                        results[name] = self._timed_out_snapshot(spec, self._good.get(name))
                    del waiting[name]
        return results

    def request_refresh(self, name: str) -> None:
        """Kick a background collection for ``name`` without waiting on it."""
        spec = self._specs[name]
        with self._lock:
            attempt = self._pending.get(name)
            if attempt is None or attempt.done.is_set():
                self._start_attempt_locked(spec)

    def last_good(self, name: str) -> ComponentSnapshot | None:
        with self._lock:
            good = self._good.get(name)
        return good.snapshot if good is not None else None

    def reset(self) -> None:
        """Drop all cached/pending state. Test-only; a running registry never resets itself."""
        with self._lock:
            self._pending.clear()
            self._good.clear()

    # -- internals -----------------------------------------------------

    def _select(self, names: Sequence[str] | None) -> list[StatusComponentSpec]:
        if names is not None:
            return [self._specs[n] for n in names]
        return [s for s in self._specs.values() if not s.detail_only]

    def _safe_fingerprint(self, spec: StatusComponentSpec) -> str | None:
        if spec.fingerprint is None:
            return None
        try:
            return spec.fingerprint()
        except Exception:
            return None

    def _start_attempt_locked(self, spec: StatusComponentSpec) -> _Attempt:
        attempt = _Attempt(thread=None, done=threading.Event(), submitted_monotonic=monotonic())  # type: ignore[arg-type]
        thread = threading.Thread(
            target=_run_collector,
            args=(spec, attempt),
            name=f"status-component:{spec.name}",
            daemon=True,
        )
        attempt.thread = thread
        self._pending[spec.name] = attempt
        thread.start()
        return attempt

    def _finalize_locked(
        self, spec: StatusComponentSpec, attempt: _Attempt, fingerprint: str | None
    ) -> ComponentSnapshot:
        del self._pending[spec.name]
        kind, payload = attempt.outcome[0]
        now_iso, now_mono = _now_iso(), monotonic()
        good = self._good.get(spec.name)
        if kind == "ok":
            snapshot = ComponentSnapshot(
                name=spec.name,
                scope=spec.scope,
                state="fresh",
                value=payload,
                captured_at=now_iso,
                age_s=0.0,
                deadline_s=spec.deadline_s,
                fingerprint=fingerprint,
                last_good_at=now_iso,
            )
            self._good[spec.name] = _Good(snapshot=snapshot, captured_monotonic=now_mono, fingerprint=fingerprint)
            return snapshot
        state: ComponentState = "unavailable" if isinstance(payload, ComponentUnavailableError) else "degraded"
        return ComponentSnapshot(
            name=spec.name,
            scope=spec.scope,
            state=state,
            value=good.snapshot.value if good is not None else None,
            captured_at=now_iso,
            age_s=0.0,
            deadline_s=spec.deadline_s,
            fingerprint=fingerprint,
            error=str(payload),
            last_good_at=good.snapshot.captured_at if good is not None else None,
        )

    def _fresh_snapshot(self, good: _Good) -> ComponentSnapshot:
        age_s = max(0.0, monotonic() - good.captured_monotonic)
        return ComponentSnapshot(
            name=good.snapshot.name,
            scope=good.snapshot.scope,
            state="fresh",
            value=good.snapshot.value,
            captured_at=good.snapshot.captured_at,
            age_s=age_s,
            deadline_s=good.snapshot.deadline_s,
            fingerprint=good.fingerprint,
            last_good_at=good.snapshot.captured_at,
        )

    def _stale_snapshot(self, spec: StatusComponentSpec, good: _Good) -> ComponentSnapshot:
        age_s = max(0.0, monotonic() - good.captured_monotonic)
        return ComponentSnapshot(
            name=spec.name,
            scope=spec.scope,
            state="stale",
            value=good.snapshot.value,
            captured_at=good.snapshot.captured_at,
            age_s=age_s,
            deadline_s=spec.deadline_s,
            fingerprint=good.fingerprint,
            last_good_at=good.snapshot.captured_at,
        )

    def _refreshing_snapshot(self, spec: StatusComponentSpec, good: _Good | None) -> ComponentSnapshot:
        now_iso = _now_iso()
        return ComponentSnapshot(
            name=spec.name,
            scope=spec.scope,
            state="refreshing",
            value=good.snapshot.value if good is not None else None,
            captured_at=now_iso,
            age_s=max(0.0, monotonic() - good.captured_monotonic) if good is not None else 0.0,
            deadline_s=spec.deadline_s,
            last_good_at=good.snapshot.captured_at if good is not None else None,
        )

    def _timed_out_snapshot(self, spec: StatusComponentSpec, good: _Good | None) -> ComponentSnapshot:
        now_iso = _now_iso()
        return ComponentSnapshot(
            name=spec.name,
            scope=spec.scope,
            state="timed_out",
            value=good.snapshot.value if good is not None else None,
            captured_at=now_iso,
            age_s=max(0.0, monotonic() - good.captured_monotonic) if good is not None else 0.0,
            deadline_s=spec.deadline_s,
            error=f"collector exceeded deadline_s={spec.deadline_s:g}",
            last_good_at=good.snapshot.captured_at if good is not None else None,
        )


def _run_collector(spec: StatusComponentSpec, attempt: _Attempt) -> None:
    try:
        value = spec.collector()
        attempt.outcome.append(("ok", value))
    except Exception as exc:
        attempt.outcome.append(("error", exc))
    finally:
        attempt.done.set()


__all__ = [
    "ComponentSnapshot",
    "ComponentState",
    "ComponentUnavailableError",
    "StatusComponentRegistry",
    "StatusComponentSpec",
]
