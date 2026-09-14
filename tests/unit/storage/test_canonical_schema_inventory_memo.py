"""The canonical fresh-DDL inventory is built once per (tier, version, DDL).

``initialize_active_archive_root`` runs once per ingest batch -- once per
catch-up chunk of a rebuild -- and its startup reconciliation rebuilds the
canonical schema image of every durable tier from an in-memory database. That
image is a pure function of the tier, the target version and the registered
DDL: it never reads the archive. Recomputing it per chunk is the per-chunk
fixed cost this memo removes.

Anti-vacuity:

* drop the ``lru_cache`` and ``test_canonical_inventory_is_built_once`` sees a
  second in-memory build for a repeated call;
* drop ``archive_ddl`` from the memo key and
  ``test_substituted_ddl_is_not_served_from_the_memo`` gets the first tier's
  stale inventory for a changed DDL.
"""

from __future__ import annotations

import sqlite3
from typing import Any, cast

import pytest

from polylogue.storage.sqlite import durable_change_train as durable
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_TIER = ArchiveTier.SOURCE


def _target_version() -> int:
    return durable.DURABLE_MIGRATION_ADOPTION_FLOORS[_TIER] + 1


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

    registry = dict(ARCHIVE_DDL_BY_TIER)
    registry[_TIER] = f"{registry[_TIER]}\nCREATE TABLE memo_probe_table (probe_id TEXT PRIMARY KEY) STRICT;\n"
    monkeypatch.setattr(migration_runner, "ARCHIVE_DDL_BY_TIER", registry)

    substituted = durable._canonical_schema_inventory(_TIER, version)

    assert substituted.sha256 != original.sha256
    assert any(item.name == "memo_probe_table" for item in substituted.objects)


def test_a_different_target_version_is_its_own_entry() -> None:
    version = _target_version()
    first = durable._canonical_schema_inventory(_TIER, version)
    second = durable._canonical_schema_inventory(_TIER, version + 1)

    assert first is not second


def test_normalized_schema_sql_memo_preserves_the_transform() -> None:
    quoted = 'CREATE TABLE "memo_probe" ("probe_id" TEXT PRIMARY KEY)'
    bare = "CREATE TABLE memo_probe (probe_id TEXT PRIMARY KEY)"

    assert migration_runner._normalize_schema_sql(quoted) == migration_runner._normalize_schema_sql(bare)
    assert migration_runner._normalize_schema_sql(None) == ""
    # A literal is part of the inventory and must survive normalization.
    literal = "CREATE TABLE memo_probe (probe_id TEXT DEFAULT 'kept')"
    assert "kept" in migration_runner._normalize_schema_sql(literal)
