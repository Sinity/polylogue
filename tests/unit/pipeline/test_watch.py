"""Tests for the watch loop abstraction."""

from __future__ import annotations

from unittest.mock import MagicMock

from polylogue.pipeline.events import SyncEvent
from polylogue.pipeline.watch import WatchRunner


def _make_result(conversations: int = 0) -> MagicMock:
    """Create a mock RunResult."""
    result = MagicMock()
    result.counts = {"conversations": conversations}
    return result


class TestWatchRunner:
    def test_calls_sync_fn_and_handler(self):
        """WatchRunner calls sync_fn and dispatches event to handler."""
        result = _make_result(3)
        call_count = 0

        def sync_fn():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                runner.stop()
            return result

        handler = MagicMock()
        runner = WatchRunner(sync_fn=sync_fn, handler=handler, interval=0)
        runner.run()

        assert call_count >= 1
        assert handler.on_sync.called
        event = handler.on_sync.call_args[0][0]
        assert isinstance(event, SyncEvent)
        assert event.new_conversations == 3

    def test_on_idle_called_when_no_new(self):
        """on_idle callback is called when no new conversations."""
        result = _make_result(0)
        on_idle = MagicMock()

        def sync_fn():
            runner.stop()
            return result

        runner = WatchRunner(sync_fn=sync_fn, handler=MagicMock(), interval=0, on_idle=on_idle)
        runner.run()

        on_idle.assert_called_once_with(result)

    def test_on_idle_not_called_when_new(self):
        """on_idle is NOT called when there are new conversations."""
        result = _make_result(5)
        on_idle = MagicMock()

        def sync_fn():
            runner.stop()
            return result

        runner = WatchRunner(sync_fn=sync_fn, handler=MagicMock(), interval=0, on_idle=on_idle)
        runner.run()

        on_idle.assert_not_called()

    def test_on_error_called_on_exception(self):
        """on_error callback receives exceptions from sync_fn."""
        call_count = 0
        on_error = MagicMock()

        def sync_fn():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                runner.stop()
                return _make_result(0)
            raise RuntimeError("sync failed")

        runner = WatchRunner(sync_fn=sync_fn, handler=MagicMock(), interval=0, on_error=on_error)
        runner.run()

        on_error.assert_called_once()
        assert isinstance(on_error.call_args[0][0], RuntimeError)

    def test_stop_terminates_loop(self):
        """stop() causes the loop to exit."""
        call_count = 0

        def sync_fn():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                runner.stop()
            return _make_result(0)

        runner = WatchRunner(sync_fn=sync_fn, handler=MagicMock(), interval=0)
        runner.run()

        assert call_count == 3

    def test_keyboard_interrupt_stops_loop(self):
        """KeyboardInterrupt stops the watch loop."""
        call_count = 0

        def sync_fn():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt
            return _make_result(0)

        runner = WatchRunner(sync_fn=sync_fn, handler=MagicMock(), interval=0)
        runner.run()  # Should not raise

        assert call_count == 2

    def test_handler_receives_correct_event(self):
        """Handler receives SyncEvent with correct data."""
        result = _make_result(42)
        received_events: list[SyncEvent] = []

        class TestHandler:
            def on_sync(self, event: SyncEvent) -> None:
                received_events.append(event)

        def sync_fn():
            runner.stop()
            return result

        runner = WatchRunner(sync_fn=sync_fn, handler=TestHandler(), interval=0)
        runner.run()

        assert len(received_events) == 1
        assert received_events[0].new_conversations == 42
        assert received_events[0].run_result is result
