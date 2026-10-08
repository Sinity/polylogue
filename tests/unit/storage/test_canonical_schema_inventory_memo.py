"""The canonical fresh-DDL inventory is built once per exact baseline and ordered declared steps.

``initialize_active_archive_root`` runs once per ingest batch -- once per
catch-up chunk of a rebuild -- and its startup reconciliation rebuilds the
canonical schema image of every durable tier from an in-memory database. That
image is a pure function of the tier, the target version and the registered
baseline and ordered declared migration SQL: it never reads the archive.
Recomputing it per chunk is the per-chunk fixed cost this memo removes.

Anti-vacuity:

* drop the ``lru_cache`` and ``test_canonical_inventory_is_built_once`` sees a
  second in-memory build for a repeated call;
* drop ``archive_ddl`` from the memo key and
  ``test_substituted_ddl_is_not_served_from_the_memo`` gets the first tier's
  stale inventory for a changed DDL.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Any, cast

import pytest

from polylogue.storage.sqlite import archive_tiers, migration_runner
from polylogue.storage.sqlite import durable_change_train as durable
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_TIER = ArchiveTier.SOURCE


def _target_version() -> int:
    return durable.DURABLE_MIGRATION_ADOPTION_FLOORS[_TIER] + 1


_STEP_SQL = "-- migration-safety: additive-no-backup\nCREATE TABLE memo_step_items (id INTEGER PRIMARY KEY) STRICT;\n"


@pytest.fixture(autouse=True)
def _one_declared_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare one synthetic step above the floor, as a shipped migration would.

    The archive currently ships no numbered Source migration, so the image of
    a post-floor version needs a declared chain to exist at all.
    """
    step = migration_runner.MigrationStep(
        tier=_TIER, version=_target_version(), name="002_memo_step_items.sql", sql=_STEP_SQL, requires_backup=False
    )
    real_load = migration_runner._load_migrations
    monkeypatch.setattr(
        migration_runner, "_load_migrations", lambda tier: (step,) if tier is _TIER else real_load(tier)
    )
    versions = dict(migration_runner.ARCHIVE_VERSION_BY_TIER)
    versions[_TIER] = _target_version()
    monkeypatch.setattr(migration_runner, "ARCHIVE_VERSION_BY_TIER", versions)


class _MemoryBuildSpy:
    """Count the in-memory databases the canonical image builder creates."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.builds = 0
        real_connect = sqlite3.connect

        def counting_connect(target: Any = None, *args: Any, **kwargs: Any) -> sqlite3.Connection:
            if target == ":memory:":
                self.builds += 1
            return cast(sqlite3.Connection, real_connect(target, *args, **kwargs))

        monkeypatch.setattr(sqlite3, "connect", counting_connect)


@pytest.fixture(autouse=True)
def _cold_memo() -> Any:
    durable._canonical_schema_inventory_for_ddl.cache_clear()
    yield
    durable._canonical_schema_inventory_for_ddl.cache_clear()


def test_canonical_inventory_is_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _MemoryBuildSpy(monkeypatch)
    version = _target_version()

    first = durable._canonical_schema_inventory(_TIER, version)
    assert spy.builds == 1
    second = durable._canonical_schema_inventory(_TIER, version)

    assert spy.builds == 1, "a repeated canonical image must not rebuild an in-memory database"
    assert second is first
    assert len(first.sha256) == 64


def test_substituted_ddl_is_not_served_from_the_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    version = _target_version()
    original = durable._canonical_schema_inventory(_TIER, version)

    registry = dict(ARCHIVE_BASELINE_DDL_BY_TIER)
    registry[_TIER] = f"{registry[_TIER]}\nCREATE TABLE memo_probe_table (probe_id TEXT PRIMARY KEY) STRICT;\n"
    monkeypatch.setattr(archive_tiers, "ARCHIVE_BASELINE_DDL_BY_TIER", registry)

    substituted = durable._canonical_schema_inventory(_TIER, version)

    assert substituted.sha256 != original.sha256
    assert any(item.name == "memo_probe_table" for item in substituted.objects)


def test_a_different_target_version_is_its_own_entry() -> None:
    version = _target_version()
    first = durable._canonical_schema_inventory(_TIER, version)
    second = durable._canonical_schema_inventory(_TIER, durable.DURABLE_MIGRATION_ADOPTION_FLOORS[_TIER])

    assert first is not second


def test_normalized_schema_sql_memo_preserves_the_transform() -> None:
    quoted = 'CREATE TABLE "memo_probe" ("probe_id" TEXT PRIMARY KEY)'
    bare = "CREATE TABLE memo_probe (probe_id TEXT PRIMARY KEY)"

    assert migration_runner._normalize_schema_sql(quoted) == migration_runner._normalize_schema_sql(bare)
    assert migration_runner._normalize_schema_sql(None) == ""
    # A literal is part of the inventory and must survive normalization.
    literal = "CREATE TABLE memo_probe (probe_id TEXT DEFAULT 'kept')"
    assert "kept" in migration_runner._normalize_schema_sql(literal)


def test_baseline_inventory_is_the_baseline_ddl_without_declared_steps() -> None:
    floor = durable.DURABLE_MIGRATION_ADOPTION_FLOORS[_TIER]
    baseline = durable._canonical_schema_inventory(_TIER, floor)
    runtime = durable._canonical_schema_inventory(_TIER, _target_version())
    assert baseline.sha256 != runtime.sha256
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(ARCHIVE_BASELINE_DDL_BY_TIER[_TIER])
        assert baseline.sha256 == migration_runner.capture_durable_schema_inventory(connection).sha256


def test_undeclared_future_inventory_is_refused() -> None:
    # One past the runtime's declared chain: no installed step reaches it.
    with pytest.raises(migration_runner.DurableChangeTrainError):
        durable._canonical_schema_inventory(_TIER, durable._runtime_durable_version(_TIER) + 1)
