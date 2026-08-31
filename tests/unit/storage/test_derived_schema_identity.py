"""Derived-tier identity stamps refuse stale rebuildable state."""

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.schema_identity import (
    DerivedTier,
    derived_schema_identity,
    read_schema_identity,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.schema_bootstrap import SchemaSkew


def test_index_identity_changes_when_a_fingerprint_input_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    before = derived_schema_identity(DerivedTier.INDEX)
    monkeypatch.setattr(
        "polylogue.sources.origin_specs.lowering_fingerprint",
        lambda: "mutated-lowering-input",
    )
    after = derived_schema_identity(DerivedTier.INDEX)
    assert after != before


def test_stamped_wrong_identity_is_refused_before_index_use(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE schema_identity SET identity = 'wrong' WHERE tier = 'index'")

    with pytest.raises(SchemaSkew, match="stale derived tier"):
        initialize_archive_database(path, ArchiveTier.INDEX)


def test_fresh_index_carries_current_identity(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as conn:
        assert read_schema_identity(conn, DerivedTier.INDEX) == derived_schema_identity(DerivedTier.INDEX)
