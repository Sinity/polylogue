"""Regression canaries for the retired campaign archive-path boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.archive_tuple_location import ArchiveTupleAllocator, ArchiveTupleError
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_connection


def test_open_connection_on_raw_sentinel_reproduces_phantom_benchmark_db(tmp_path: Path) -> None:
    """The historical raw sentinel remains an executable anti-vacuity mutant."""

    initialize_active_archive_root(tmp_path)
    active_index = ArchiveLocation.resolve(tmp_path).active_index_path

    with open_connection(tmp_path / "benchmark.db") as connection:
        connection.execute("CREATE TABLE phantom (value TEXT)")
        connection.commit()

    assert (tmp_path / "benchmark.db").is_file()
    assert active_index.resolve() != (tmp_path / "benchmark.db").resolve()
    with sqlite3.connect(active_index) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'phantom'"
        ).fetchone() == (0,)


def test_inactive_campaign_destination_is_typed_and_not_a_benchmark_sentinel(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    candidate = ArchiveTupleAllocator(ArchiveLocation.resolve(tmp_path)).allocate(owner_id="campaign")

    assert candidate.destination(ArchiveTier.INDEX).path.name == "index.db"
    assert candidate.destination(ArchiveTier.INDEX).path.name != "benchmark.db"
    with pytest.raises(ArchiveTupleError):
        # A campaign cannot reinterpret its tuple root as an active archive
        # or silently reopen a caller-provided filename.
        initialize_active_archive_root(candidate.candidate_root)
