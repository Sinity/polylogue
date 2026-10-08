"""The Source authorizer verifies physical custody once per statement compile."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite import connection_profile


def _counting_stats(monkeypatch: pytest.MonkeyPatch, target: Path) -> list[Path]:
    observed: list[Path] = []
    real_stat = Path.stat

    def stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self == target:
            observed.append(self)
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", stat)
    return observed


def test_each_statement_reverifies_and_one_compile_verifies_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: caching across statements makes the second count zero;
    dropping the epoch makes a multi-column read stat once per column."""
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root

    with write_lease("test.authorizer-epoch", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        source = tmp_path / "source.db"
        connection = connection_profile.open_source_tier_write_connection(source, archive_root=tmp_path)
        try:
            stats = _counting_stats(monkeypatch, source.resolve())
            sql = "SELECT raw_id, source_path, blob_hash, acquired_at_ms FROM raw_sessions"
            with closing(connection.execute(sql)) as cursor:
                cursor.fetchall()
            assert len(stats) == 1
            with closing(connection.execute(sql)) as cursor:
                cursor.fetchall()
            assert len(stats) == 2
        finally:
            connection.close()


def test_replaced_source_file_refuses_the_next_statement(tmp_path: Path) -> None:
    """A statement compiled after the file incarnation changed is denied."""
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root

    with write_lease("test.authorizer-incarnation", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        source = tmp_path / "source.db"
        connection = connection_profile.open_source_tier_write_connection(source, archive_root=tmp_path)
        try:
            with closing(connection.execute("SELECT count(*) FROM raw_sessions")) as cursor:
                cursor.fetchone()
            replacement = tmp_path / "replacement.db"
            replacement.write_bytes(source.read_bytes())
            original = tmp_path / "original.db"
            source.rename(original)
            replacement.rename(source)
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                connection.execute("SELECT count(*) FROM raw_sessions")
        finally:
            connection.close()
