"""Archive diagnostics read stale tiers through the SQLite read boundary."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.maintenance import archive_verification
from polylogue.readiness import _open_readiness_probe_connection
from polylogue.storage.sqlite.archive_tiers import archive_plan, schema_inventory
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def _stale_source(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE specimen (value INTEGER)")
        connection.execute("INSERT INTO specimen VALUES (7)")
        connection.execute("PRAGMA user_version = 999999")
        connection.commit()


def test_diagnostic_routes_keep_stale_tiers_readable_through_named_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing a diagnostic factory call with a raw connect loses this proof."""
    source = tmp_path / "source.db"
    _stale_source(source)
    opened: list[Path] = []

    def audited_open(path: Path, **kwargs: object) -> sqlite3.Connection:
        assert kwargs == {"validate_schema": False}
        opened.append(Path(path))
        return open_readonly_connection(path, validate_schema=False)

    monkeypatch.setattr(archive_plan, "open_readonly_connection", audited_open)
    monkeypatch.setattr(schema_inventory, "open_readonly_connection", audited_open)
    monkeypatch.setattr(archive_verification, "open_readonly_connection", audited_open)
    monkeypatch.setattr("polylogue.readiness.open_readonly_connection", audited_open)

    assert archive_plan._tier_schema_fingerprint(source)
    with closing(schema_inventory._open_read_only(source, tier=ArchiveTier.SOURCE)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 999999
        assert connection.execute("SELECT value FROM specimen").fetchone()[0] == 7
    check = archive_verification._check_tier_schema(tmp_path, 1)
    assert check.evidence["tiers"]["source"]["actual_version"] == 999999
    with _open_readiness_probe_connection(source) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    assert opened == [source] * 4


def test_schema_census_reader_rejects_writes_and_writable_attach(tmp_path: Path) -> None:
    """A raw mode=ro open allows writable PRAGMAs and writable ATTACH."""
    source = tmp_path / "source.db"
    _stale_source(source)
    attached = tmp_path / "writable.db"

    with closing(schema_inventory._open_read_only(source, tier=ArchiveTier.SOURCE)) as connection:
        assert connection.execute("SELECT value FROM specimen").fetchone()[0] == 7
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        for statement in (
            "INSERT INTO specimen VALUES (8)",
            "UPDATE specimen SET value = 8",
            "DELETE FROM specimen",
            "CREATE TABLE extra (value INTEGER)",
            "PRAGMA cache_size = 1",
        ):
            with pytest.raises(sqlite3.Error):
                connection.execute(statement)
        with pytest.raises(sqlite3.Error):
            connection.execute("ATTACH DATABASE ? AS writable", (str(attached),))

    assert not attached.exists()
    with closing(sqlite3.connect(source)) as connection:
        assert connection.execute("SELECT value FROM specimen").fetchall() == [(7,)]


def test_diagnostic_missing_tier_remains_a_missing_tier(tmp_path: Path) -> None:
    missing = tmp_path / "source.db"
    with pytest.raises(schema_inventory.SchemaCensusError, match="tier file is missing"):
        schema_inventory._open_read_only(missing, tier=ArchiveTier.SOURCE)
    with pytest.raises(sqlite3.Error):
        with _open_readiness_probe_connection(missing):
            pass
    assert not missing.exists()
