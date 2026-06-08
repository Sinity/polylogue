"""Tiered scale fixtures for performance/regression tests (issue #1183).

Provides three explicit tiers with documented session/message counts
and pytest markers so each tier runs in the right gate:

  ``tier_small_db``  — ~100 sessions / ~1,000 messages
                       Marker: ``@pytest.mark.scale_small``
                       Default ``devtools verify`` includes these.
  ``tier_medium_db`` — ~1,000 sessions / ~10,000 messages
                       Marker: ``@pytest.mark.scale_medium``
                       ``devtools verify --lab`` includes these.
  ``tier_large_db``  — ~10,000 sessions / ~100,000 messages
                       Marker: ``@pytest.mark.scale_large``
                       Nightly CI / explicit campaigns only.

The factories return cached SQLite paths per test session — generating
the large tier is expensive (~minutes), so tests within a session share
one fixture instance. Each tier uses the same realistic distribution
helpers used by ``tests/benchmarks/conftest.py``.

Growth-shape rule: tests that assert latency across tiers must compare
ratios (large/medium, medium/small) rather than absolute milliseconds.
Absolute budgets bake in host-machine assumptions; ratio budgets stay
portable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class ScaleTier:
    """Declarative scale-tier definition.

    Attributes:
        name: tier label (``small``/``medium``/``large``)
        target_messages: approximate number of messages to seed; the seeder
            stops once the running total reaches this floor.
        target_sessions: approximate number of sessions expected
            for this tier (informational; the seeder is driven by message
            count and the realistic-distribution profile).
        marker: pytest marker name registered in ``pyproject.toml``.
    """

    name: str
    target_messages: int
    target_sessions: int
    marker: str


SCALE_SMALL = ScaleTier(
    name="small",
    target_messages=1_000,
    target_sessions=100,
    marker="scale_small",
)

SCALE_MEDIUM = ScaleTier(
    name="medium",
    target_messages=10_000,
    target_sessions=1_000,
    marker="scale_medium",
)

SCALE_LARGE = ScaleTier(
    name="large",
    target_messages=100_000,
    target_sessions=10_000,
    marker="scale_large",
)

ALL_TIERS: tuple[ScaleTier, ...] = (SCALE_SMALL, SCALE_MEDIUM, SCALE_LARGE)
ALL_MARKERS: tuple[str, ...] = tuple(t.marker for t in ALL_TIERS)


def _seed_tier_db(db_path: Path, tier: ScaleTier, *, seed: int = 1183) -> dict[str, int]:
    """Seed ``db_path`` with realistic data for the requested tier.

    Defers the heavy import to fixture invocation so unit collection stays
    fast — ``tests.benchmarks.conftest`` pulls in storage backends, the
    rebuild_index path, and the synthetic corpus generator.
    """
    from tests.benchmarks.conftest import _seed_realistic_db  # local import: heavy

    return _seed_realistic_db(db_path, target_messages=tier.target_messages, seed=seed)


# ---------------------------------------------------------------------------
# Session-scoped tier factories
#
# Each fixture is session-scoped so the (expensive) seeding only happens
# once per test session. The factory returns a ``Path`` to the seeded
# SQLite database; tests open it read-only via the standard
# ``open_bench_store`` / repository helpers.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tier_small_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small-tier SQLite DB: ~100 convs / ~1k messages. Default verify gate."""
    db_path = tmp_path_factory.mktemp("scale_small") / "small.db"
    _seed_tier_db(db_path, SCALE_SMALL)
    return db_path


@pytest.fixture(scope="session")
def tier_medium_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Medium-tier SQLite DB: ~1k convs / ~10k messages. Lab gate."""
    db_path = tmp_path_factory.mktemp("scale_medium") / "medium.db"
    _seed_tier_db(db_path, SCALE_MEDIUM)
    return db_path


@pytest.fixture(scope="session")
def tier_large_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Large-tier SQLite DB: ~10k convs / ~100k messages. Nightly gate."""
    db_path = tmp_path_factory.mktemp("scale_large") / "large.db"
    _seed_tier_db(db_path, SCALE_LARGE)
    return db_path


# Public re-exports for documentation/registry consumers.
__all__ = [
    "ALL_MARKERS",
    "ALL_TIERS",
    "SCALE_LARGE",
    "SCALE_MEDIUM",
    "SCALE_SMALL",
    "ScaleTier",
    "tier_large_db",
    "tier_medium_db",
    "tier_small_db",
]


# Factory protocol re-exports for `Callable[..., Path]`-typed call sites.
_FactoryReExports: tuple[Callable[..., Path], ...] = ()
