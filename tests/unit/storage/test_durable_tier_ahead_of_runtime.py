"""polylogue-o7d39: a durable tier ahead of the runtime is a stale runtime, not data to move aside."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_source_tier_newer_than_runtime_tells_operator_to_update_the_runtime(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    newer = archive_tier_spec(ArchiveTier.SOURCE).version + 1
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {newer}")
    with pytest.raises(RuntimeError, match="Update the installed Polylogue runtime") as info:
        initialize_archive_database(path, ArchiveTier.SOURCE)
    assert ".stale" not in str(info.value)
