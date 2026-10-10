"""The shared foreground quantum survives physically owned background preparation."""

from __future__ import annotations

import threading
from functools import partial

import pytest

from polylogue.core.compute import BoundedComputeAdapter, CancellationHandle, DaemonBackpressureError

pytestmark = pytest.mark.uses_real_clock("physical workers settle on explicit OS events")
_QUANTUM = 1024 * 1024


def _block(started: threading.Event, release: threading.Event) -> None:
    started.set()
    release.wait()


def _signal_result(started: threading.Event) -> str:
    started.set()
    return "large"


@pytest.mark.parametrize("exclusive", [True, False])
def test_exclusive_background_allows_one_shared_foreground_quantum(exclusive: bool) -> None:
    adapter = BoundedComputeAdapter(max_workers=2, queue_units=8, queue_bytes=8 * _QUANTUM)
    background_started = threading.Event()
    foreground_started = threading.Event()
    release_background = threading.Event()
    release_foreground = threading.Event()

    def background() -> None:
        background_started.set()
        release_background.wait()

    def foreground() -> None:
        foreground_started.set()
        release_foreground.wait()

    try:
        bg = adapter.submit(
            background,
            admission_class="incremental-background",
            exclusive_bytes=exclusive,
            estimated_bytes=0 if exclusive else 9 * _QUANTUM,
        )
        assert background_started.wait(2)
        fg = adapter.submit(foreground, admission_class="interactive-read", estimated_bytes=_QUANTUM)
        assert foreground_started.wait(2)
        snapshot = adapter.snapshot()
        assert snapshot.used_bytes == 9 * _QUANTUM
        assert snapshot.to_dict()["foreground_allowance_used_bytes"] == _QUANTUM
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(lambda: None, admission_class="control", estimated_bytes=0)
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(lambda: None, admission_class="bulk-candidate", estimated_bytes=0)
        release_foreground.set()
        fg.future.result(timeout=2)
        assert adapter.snapshot().used_bytes == 8 * _QUANTUM
        next_fg = adapter.submit(lambda: "control", admission_class="control", estimated_bytes=_QUANTUM)
        assert next_fg.future.result(timeout=2) == "control"
        release_background.set()
        bg.future.result(timeout=2)
        assert adapter.snapshot().used_bytes == 0
    finally:
        release_background.set()
        release_foreground.set()
        adapter.shutdown(wait=True)


def test_background_exclusive_can_follow_the_existing_foreground_quantum() -> None:
    adapter = BoundedComputeAdapter(max_workers=2, queue_units=8, queue_bytes=8 * _QUANTUM)
    release = threading.Event()
    fg_started = threading.Event()
    bg_started = threading.Event()
    try:
        fg = adapter.submit(partial(_block, fg_started, release), admission_class="control", estimated_bytes=_QUANTUM)
        assert fg_started.wait(2)
        bg = adapter.submit(
            partial(_block, bg_started, release), admission_class="incremental-background", exclusive_bytes=True
        )
        assert bg_started.wait(2)
        assert adapter.snapshot().used_bytes == 9 * _QUANTUM
        release.set()
        fg.future.result(timeout=2)
        bg.future.result(timeout=2)
        assert adapter.snapshot().used_bytes == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("workers", [1, 2])
def test_larger_foreground_waits_for_exclusive_physical_settlement(workers: int) -> None:
    adapter = BoundedComputeAdapter(max_workers=workers, queue_units=8, queue_bytes=8 * _QUANTUM)
    started = threading.Event()
    release = threading.Event()
    handle = CancellationHandle()
    fg_started = threading.Event()
    try:
        bg = adapter.submit(
            partial(_block, started, release),
            admission_class="incremental-background",
            exclusive_bytes=True,
            cancellation=handle,
        )
        assert started.wait(2)
        fg = adapter.submit(
            partial(_signal_result, fg_started), admission_class="interactive-read", estimated_bytes=20 * _QUANTUM
        )
        assert not fg_started.wait(0.05)
        snapshot = adapter.snapshot()
        assert snapshot.used_bytes == 8 * _QUANTUM
        assert snapshot.queued_input_bytes == 20 * _QUANTUM
        assert snapshot.to_dict()["foreground_pending_bytes"] == 8 * _QUANTUM
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(lambda: None, admission_class="control", estimated_bytes=_QUANTUM)
        handle.cancel()
        assert not fg_started.wait(0.05)
        release.set()
        bg.future.result(timeout=2)
        assert fg.future.result(timeout=2) == "large"
        assert fg_started.is_set()
        assert adapter.snapshot().used_bytes == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


def test_pending_foreground_cancellation_releases_its_position_without_byte_underflow() -> None:
    adapter = BoundedComputeAdapter(max_workers=2, queue_units=8, queue_bytes=8 * _QUANTUM)
    started = threading.Event()
    release = threading.Event()
    try:
        bg = adapter.submit(partial(_block, started, release), admission_class="bulk-candidate", exclusive_bytes=True)
        assert started.wait(2)
        fg = adapter.submit(
            lambda: pytest.fail("cancelled pending work ran"), admission_class="control", estimated_bytes=2 * _QUANTUM
        )
        assert fg.future.cancel()
        assert adapter.snapshot().used_bytes == 8 * _QUANTUM
        assert adapter.snapshot().queued_input_bytes == 0
        assert adapter.snapshot().to_dict()["foreground_pending_bytes"] == 0
        replacement = adapter.submit(lambda: "read", admission_class="interactive-read", estimated_bytes=_QUANTUM)
        assert replacement.future.result(timeout=2) == "read"
        release.set()
        bg.future.result(timeout=2)
        assert adapter.snapshot().used_bytes == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


def test_active_foreground_cancellation_retains_its_allowance_until_physical_settlement() -> None:
    adapter = BoundedComputeAdapter(max_workers=2, queue_units=8, queue_bytes=8 * _QUANTUM)
    bg_started = threading.Event()
    fg_started = threading.Event()
    release_bg = threading.Event()
    release_fg = threading.Event()
    handle = CancellationHandle()
    try:
        bg = adapter.submit(
            partial(_block, bg_started, release_bg),
            admission_class="incremental-background",
            exclusive_bytes=True,
        )
        assert bg_started.wait(2)
        fg = adapter.submit(
            partial(_block, fg_started, release_fg),
            admission_class="interactive-read",
            estimated_bytes=_QUANTUM,
            cancellation=handle,
        )
        assert fg_started.wait(2)
        handle.cancel()
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(lambda: None, admission_class="control", estimated_bytes=_QUANTUM)
        assert adapter.snapshot().foreground_allowance_used_bytes == _QUANTUM
        release_fg.set()
        fg.future.result(timeout=2)
        replacement = adapter.submit(lambda: None, admission_class="control", estimated_bytes=_QUANTUM)
        replacement.future.result(timeout=2)
        release_bg.set()
        bg.future.result(timeout=2)
        assert adapter.snapshot().used_bytes == 0
    finally:
        release_bg.set()
        release_fg.set()
        adapter.shutdown(wait=True)


@pytest.mark.parametrize(("workers", "units"), [(4, 3), (1, 1)])
def test_multi_slot_foreground_waits_and_single_worker_small_foreground_remains_runnable(
    workers: int,
    units: int,
) -> None:
    adapter = BoundedComputeAdapter(max_workers=workers, queue_units=8, queue_bytes=8 * _QUANTUM)
    bg_started = threading.Event()
    fg_started = threading.Event()
    release = threading.Event()
    try:
        bg = adapter.submit(
            partial(_block, bg_started, release), admission_class="bulk-candidate", exclusive_bytes=True
        )
        assert bg_started.wait(2)
        fg = adapter.submit(lambda: fg_started.set(), admission_class="control", units=units, estimated_bytes=_QUANTUM)
        assert not fg_started.wait(0.05)
        release.set()
        bg.future.result(timeout=2)
        fg.future.result(timeout=2)
        assert fg_started.is_set()
        assert adapter.snapshot().used_bytes == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)
