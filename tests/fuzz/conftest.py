"""Fuzz test fixtures.

Auto-applied autouse fixture seeds the ``random`` module to a fixed value
for every fuzz test, making the per-test byte streams deterministic and
reproducible. Without this, an iteration N failure inside a 1000-iteration
loop has no recipe to reproduce.

We deliberately do NOT seed ``random`` at module import time: pytest-randomly
sets a session-wide seed at collection time, and re-seeding inside the test
function is what guarantees per-test determinism regardless of test order
or session seed.
"""

from __future__ import annotations

import hashlib
import random

import pytest

# A stable, distinct-per-test base seed. Per-test offsets come from
# pytest's request fixture (test name hashed) so every fuzz test gets a
# unique deterministic stream.
_BASE_SEED = 0xF0_22_F0_22


@pytest.fixture(autouse=True)
def _seed_fuzz_random(request: pytest.FixtureRequest) -> None:
    """Seed ``random`` with a stable, per-test-deterministic value."""
    test_seed = _BASE_SEED ^ (int.from_bytes(hashlib.sha256(request.node.nodeid.encode()).digest()[:4]) & 0xFFFFFFFF)
    random.seed(test_seed)
