"""Tests for watcher registration gate observability + timeout.

Covers two production mutations that would each defeat the fix independently:

1. ``_await_watcher_registration`` (``daemon/cli.py``) is the single helper every
   ``watcher_registered``-gated periodic maintenance loop now waits through
   instead of a bare ``await watcher_registered.wait()``. Before this bead a
   watcher that never registered (crash, hang, or the
   schema-preflight-blocked startup path) parked every gated loop on that
   ``Event.wait()`` forever, with zero journal signal.
2. ``run_daemon_services`` now emits a loud ``ERROR`` event and a
   ``maintenance_loops_parked`` daemon event, naming exactly which
   maintenance loops are withheld, whenever the watcher is schema-blocked
   at startup -- previously the only signal was the one-time schema
   preflight ERROR with no enumeration of what that decision froze.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.logging import capture
from polylogue.sources.live import WatchSource


def test_gate_noop_when_no_gate_given() -> None:
    """Callers without a watcher (``watcher_registered=None``) proceed as before.

    Anti-vacuity: removing the ``watcher_registered is None`` short-circuit in
    ``_await_watcher_registration`` makes this raise ``AttributeError`` on
    ``None.is_set()`` instead of returning.
    """
    from polylogue.daemon import cli as daemon_cli

    with capture() as records:
        asyncio.run(daemon_cli._await_watcher_registration(None, loop_name="unit-test-loop", timeout_s=0.01))

    assert [r for r in records if r["event"] == "daemon.watcher_registered.timeout"] == []


def test_gate_returns_immediately_when_event_preset() -> None:
    """A gate already released must not produce a timeout warning.

    Anti-vacuity: a mutation that logs the timeout warning unconditionally
    (instead of only on ``asyncio.TimeoutError``) makes this fail.
    """
    from polylogue.daemon import cli as daemon_cli

    event = asyncio.Event()
    event.set()

    with capture() as records:
        asyncio.run(daemon_cli._await_watcher_registration(event, loop_name="unit-test-loop", timeout_s=5.0))

    assert [r for r in records if r["event"] == "daemon.watcher_registered.timeout"] == []


def test_gate_times_out_and_warns_then_proceeds() -> None:
    """A gate that never releases must still let the loop proceed, once, loudly.

    Anti-vacuity: reverting to the old bare ``await watcher_registered.wait()``
    (no ``asyncio.wait_for``/timeout) makes this test hang until the pytest
    stall timeout instead of returning; removing the
    ``daemon.watcher_registered.timeout`` event makes the assertions below fail.
    """
    from polylogue.daemon import cli as daemon_cli

    event = asyncio.Event()  # never set

    with capture() as records:
        asyncio.run(daemon_cli._await_watcher_registration(event, loop_name="unit-test-loop", timeout_s=0.02))

    timeouts = [r for r in records if r["event"] == "daemon.watcher_registered.timeout"]
    assert len(timeouts) == 1
    assert timeouts[0]["level"] == "warning"
    assert timeouts[0]["outcome"] == "unmeasured"
    assert timeouts[0]["loop"] == "unit-test-loop"
    assert timeouts[0]["timeout_ms"] == 20.0


def test_run_daemon_services_schema_block_logs_parked_loops_and_emits_event() -> None:
    """Schema-blocked startup must name every withheld loop, loudly and durably.

    Anti-vacuity: deleting the ``emit("daemon.maintenance_loops.parked", ...)``
    call (or the ``emit_daemon_event("maintenance_loops_parked", ...)``
    call) from the ``if watcher_blocked:`` branch in
    ``run_daemon_services`` makes the corresponding assertion below fail; the
    prior behavior only recorded the single schema-preflight ERROR with no
    enumeration of what was withheld.
    """
    from polylogue.daemon import cli as daemon_cli
    from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier

    class FakeServer:
        shutdown_called = False
        close_called = False

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            raise RuntimeError("server stopped")

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.close_called = True

    async def lifecycle_heartbeat() -> None:
        await asyncio.Event().wait()

    async def fake_health_check() -> None:
        await asyncio.Event().wait()

    def fail_background_work(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema-blocked daemon must not start DB background work")

    server = FakeServer()
    critical = HealthAlert(
        check_name="schema_version",
        tier=HealthTier.FAST,
        severity=HealthSeverity.CRITICAL,
        message="archive2 is not runtime v8",
        checked_at="2026-08-03T00:00:00+00:00",
    )
    recorded_events: list[tuple[str, dict[str, object] | None]] = []

    def fake_emit_daemon_event(kind: str, *, payload: dict[str, object] | None = None, **_kwargs: object) -> None:
        recorded_events.append((kind, payload))

    with (
        patch.object(daemon_cli, "_check_schema_version_fast", return_value=critical),
        patch.object(daemon_cli, "_periodic_wal_checkpoint", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_heartbeat", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_lifecycle_heartbeat", lifecycle_heartbeat),
        patch.object(daemon_cli, "_periodic_convergence_check", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_health_check", fake_health_check),
        patch.object(daemon_cli, "_periodic_db_optimize", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_status_snapshot_refresh", side_effect=fail_background_work),
        patch.object(daemon_cli, "_periodic_drive_source_catchup", side_effect=fail_background_work),
        patch("polylogue.daemon.convergence.DaemonConverger", side_effect=fail_background_work),
        patch.object(daemon_cli, "make_server", return_value=server),
        patch("polylogue.daemon.events.emit_daemon_event", side_effect=fake_emit_daemon_event),
        capture() as records,
        pytest.raises(RuntimeError, match="server stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                enable_watch=True,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    parked_records = [r for r in records if r["event"] == "daemon.maintenance_loops.parked"]
    assert len(parked_records) == 1
    assert parked_records[0]["level"] == "error"
    assert parked_records[0]["outcome"] == "refused"
    assert parked_records[0]["reason"] == "schema_version_mismatch"
    loop_count_arg = parked_records[0]["loops"]
    assert (
        loop_count_arg == len(daemon_cli._SCHEMA_BLOCKED_MAINTENANCE_LOOP_NAMES) + 1
    )  # +1: drive catchup (default on)
    # Each parked loop is named by its own event rather than a joined string:
    # ``error_detail`` truncates at 300 characters and would drop the tail.
    parked_loop_names = [str(r["loop"]) for r in records if r["event"] == "daemon.maintenance_loop.parked"]
    assert len(parked_loop_names) == loop_count_arg
    for name in daemon_cli._SCHEMA_BLOCKED_MAINTENANCE_LOOP_NAMES:
        assert name in parked_loop_names
    assert daemon_cli._SCHEMA_BLOCKED_OPTIONAL_DRIVE_CATCHUP_LOOP_NAME in parked_loop_names
    loop_names_arg = ", ".join(parked_loop_names)

    parked_events = [payload for kind, payload in recorded_events if kind == "maintenance_loops_parked"]
    assert len(parked_events) == 1
    payload = parked_events[0]
    assert payload is not None
    assert payload["reason"] == "schema_version_mismatch"
    assert payload["loop_count"] == loop_count_arg
    recorded_loop_names = payload["loop_names"]
    assert isinstance(recorded_loop_names, list)
    assert set(recorded_loop_names) == set(loop_names_arg.split(", "))
