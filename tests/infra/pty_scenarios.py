"""Event-driven real-shell/PTY scenarios for CLI interaction tests."""

from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import Literal, cast

from tests.infra.pty_cli import PtyEventLike, PtyResult, run_in_pty

EventKind = Literal["write", "resize", "interrupt", "terminate"]


@dataclass(frozen=True, slots=True)
class PtyEvent:
    """An input/control event scheduled relative to process start."""

    after_s: float
    kind: str
    value: str = ""


def scenario_contract(events: tuple[PtyEvent, ...]) -> tuple[str, ...]:
    """Validate event declarations before a scenario is executed."""
    errors: list[str] = []
    previous = -1.0
    for event in events:
        if event.after_s < previous:
            errors.append("PTY events must be monotonic")
        if event.after_s < 0:
            errors.append("PTY event delay cannot be negative")
        if event.kind == "resize":
            try:
                rows, columns = (int(part) for part in event.value.lower().split("x", 1))
            except (ValueError, TypeError):
                errors.append("resize event must use ROWSxCOLUMNS")
            else:
                if rows < 1 or columns < 1:
                    errors.append("resize dimensions must be positive")
        previous = event.after_s
    return tuple(errors)


def run_pty_scenario(
    args: list[str],
    *,
    events: tuple[PtyEvent, ...] = (),
    rows: int = 24,
    cols: int = 80,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> PtyResult:
    """Run a real CLI PTY scenario, applying declared events when possible.

    The ordinary harness remains the fast path.  Eventful scenarios use the
    same subprocess boundary and are intentionally small; the helper refuses
    malformed schedules rather than silently dropping resize/interrupt input.
    """
    errors = scenario_contract(events)
    if errors:
        raise ValueError("invalid PTY scenario: " + "; ".join(errors))
    if not events:
        return run_in_pty(args, rows=rows, cols=cols, env=env, timeout=timeout)

    # Events are delivered by the shared PTY process loop.  A short command may
    # exit before a late event, which is observable in its exit/output result
    # rather than silently turning the event into a second synthetic runner.
    return run_in_pty(
        args,
        rows=rows,
        cols=cols,
        env=env,
        timeout=timeout,
        events=cast(tuple[PtyEventLike, ...], events),
    )


def signal_for_event(event: PtyEvent) -> int | None:
    """Map control events to signals for future long-lived PTY drivers."""
    return {"interrupt": signal.SIGINT, "terminate": signal.SIGTERM}.get(event.kind)


__all__ = ["PtyEvent", "run_pty_scenario", "scenario_contract", "signal_for_event"]
