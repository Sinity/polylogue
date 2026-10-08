"""Real bounded compute adapters for tests that drive production owners.

Production stage, derivation and drain constructors require the daemon's
``BoundedComputeAdapter``. Tests use a real one (never a stub) and settle it
with ``shutdown(wait=True)`` so no worker thread outlives the test.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from polylogue.core.compute import BoundedComputeAdapter


@contextmanager
def owned_compute_adapter(*, max_workers: int = 1, queue_units: int = 1) -> Iterator[BoundedComputeAdapter]:
    """Yield a real adapter and join its workers on exit."""
    adapter = BoundedComputeAdapter(max_workers=max_workers, queue_units=queue_units)
    try:
        yield adapter
    finally:
        adapter.shutdown(wait=True)


@pytest.fixture
def bounded_compute_adapter() -> Iterator[BoundedComputeAdapter]:
    """One real adapter for the whole test, joined at teardown."""
    with owned_compute_adapter() as adapter:
        yield adapter
