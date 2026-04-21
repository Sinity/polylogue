"""Tests for the watch loop abstraction."""

from __future__ import annotations

from unittest.mock import MagicMock

from polylogue.pipeline.observers import RunObserver
from polylogue.pipeline.watch import WatchRunner
from polylogue.storage.run_state import RunCounts, RunResult


def _make_result(conversations: int = 0) -> RunResult:
    """Create a minimal RunResult."""
    return RunResult(
        run_id="watch-test",
        counts=RunCounts(conversations=conversations),
        indexed=True,
        index_error=None,
        duration_ms=0,
    )


class TestWatchRunner:
    def test_calls_sync_fn_and_observer(self) -> None:
        """WatchRunner calls sync_fn and forwards completion to the observer."""
        result = _make_result(3)
        call_count = 0

        def sync_fn() -> RunResult:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                runner.stop()
            return result

        observer = MagicMock(spec=RunObserver)
        runner = WatchRunner(sync_fn=sync_fn, observer=observer, interval=0)
        runner.run()

        assert call_count >= 1
        observer.on_completed.assert_called()
        assert observer.on_completed.call_args[0][0] is result

    def test_on_idle_called_when_no_new(self) -> None:
        """Observer idle hook is called when no new conversations are found."""
        result = _make_result(0)
        observer = MagicMock(spec=RunObserver)

        def sync_fn() -> RunResult:
            runner.stop()
            return result

        runner = WatchRunner(sync_fn=sync_fn, observer=observer, interval=0)
        runner.run()

        observer.on_idle.assert_called_once_with(result)

    def test_on_idle_not_called_when_new(self) -> None:
        """Observer idle hook is not called when new conversations are found."""
        result = _make_result(5)
        observer = MagicMock(spec=RunObserver)

        def sync_fn() -> RunResult:
            runner.stop()
            return result

        runner = WatchRunner(sync_fn=sync_fn, observer=observer, interval=0)
        runner.run()

        observer.on_idle.assert_not_called()

    def test_on_error_called_on_exception(self) -> None:
        """Observer error hook receives exceptions from sync_fn."""
        call_count = 0
        observer = MagicMock(spec=RunObserver)

        def sync_fn() -> RunResult:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                runner.stop()
                return _make_result(0)
            raise RuntimeError("sync failed")

        runner = WatchRunner(sync_fn=sync_fn, observer=observer, interval=0)
        runner.run()

        observer.on_error.assert_called_once()
        assert isinstance(observer.on_error.call_args[0][0], RuntimeError)

    def test_stop_terminates_loop(self) -> None:
        """stop() causes the loop to exit."""
        call_count = 0

        def sync_fn() -> RunResult:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                runner.stop()
            return _make_result(0)

        runner = WatchRunner(sync_fn=sync_fn, observer=MagicMock(spec=RunObserver), interval=0)
        runner.run()

        assert call_count == 3

    def test_keyboard_interrupt_stops_loop(self) -> None:
        """KeyboardInterrupt stops the watch loop."""
        call_count = 0

        def sync_fn() -> RunResult:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt
            return _make_result(0)

        runner = WatchRunner(sync_fn=sync_fn, observer=MagicMock(spec=RunObserver), interval=0)
        runner.run()  # Should not raise

        assert call_count == 2

    def test_observer_receives_result(self) -> None:
        """Observer receives the completed run result."""
        result = _make_result(42)
        received: list[RunResult] = []

        class TestObserver(RunObserver):
            def on_completed(self, completed: RunResult) -> None:
                received.append(completed)

        def sync_fn() -> RunResult:
            runner.stop()
            return result

        runner = WatchRunner(sync_fn=sync_fn, observer=TestObserver(), interval=0)
        runner.run()

        assert received == [result]
