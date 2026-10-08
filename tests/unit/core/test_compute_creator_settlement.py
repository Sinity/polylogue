from __future__ import annotations

import threading

from polylogue.core.compute import BoundedComputeAdapter, retain_current_creator_settlement


def _parked_cleanup(adapter: BoundedComputeAdapter) -> tuple[threading.Event, threading.Event]:
    """Submit a task that parks on its creator thread until asked to settle."""
    registered = threading.Event()
    requested = threading.Event()

    def task() -> None:
        release = retain_current_creator_settlement(requested.set, owner_count=1, failure_types=("OSError",))
        assert release is not None
        registered.set()
        requested.wait()
        release()

    adapter.submit(task)
    assert registered.wait(10)
    return registered, requested


def test_adapter_retry_reaches_a_cleanup_owner_parked_on_its_creator() -> None:
    """A creator-parked owner is visible to, and woken by, its adapter.

    Anti-vacuity: without the registration the adapter reports nothing
    retained, its retry wakes nobody, and joining the adapter waits forever
    on the parked thread.
    """
    adapter = BoundedComputeAdapter(max_workers=1)
    try:
        _registered, requested = _parked_cleanup(adapter)
        retained = adapter.retained_sql_settlements()
        assert len(retained) == 1
        assert retained[0].owner_count == 1 and retained[0].failure_types == ("OSError",)
        adapter.retry_sql_settlement()
        assert requested.wait(10)
    finally:
        adapter.shutdown(wait=True)
    assert adapter.retained_sql_settlements() == ()


def test_adapter_shutdown_wakes_a_parked_cleanup_owner() -> None:
    adapter = BoundedComputeAdapter(max_workers=1)
    _registered, requested = _parked_cleanup(adapter)
    adapter.shutdown(wait=True)
    assert requested.is_set()
    assert adapter.retained_sql_settlements() == ()


def test_outside_a_compute_task_nothing_is_registered() -> None:
    assert retain_current_creator_settlement(lambda: None, owner_count=1, failure_types=()) is None
