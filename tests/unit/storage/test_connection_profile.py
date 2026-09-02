from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.core.errors import SchemaSkew
from polylogue.storage.sqlite import connection_profile
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_open_readonly_connection_uses_descriptor_bound_database(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('selected')")

    descriptor_handle = db_path.open("rb")
    try:
        reader = connection_profile.open_readonly_connection(db_path, opened_main_fd=descriptor_handle.fileno())
        try:
            assert reader.execute("SELECT value FROM evidence").fetchone() == ("selected",)
        finally:
            reader.close()
    finally:
        descriptor_handle.close()


def test_open_readonly_connection_refuses_without_descriptor_bound_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "evidence.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

    descriptor_handle = db_path.open("rb")
    try:
        monkeypatch.setattr(connection_profile, "_descriptor_database_uri", lambda _fd, _suffix: None)
        with pytest.raises(RuntimeError, match="descriptor-bound path"):
            connection_profile.open_readonly_connection(db_path, opened_main_fd=descriptor_handle.fileno())
    finally:
        descriptor_handle.close()


def test_open_readonly_connection_rejects_immutable_with_descriptor(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")

    descriptor_handle = db_path.open("rb")
    try:
        with pytest.raises(ValueError, match="immutable mode"):
            connection_profile.open_readonly_connection(
                db_path,
                immutable=True,
                opened_main_fd=descriptor_handle.fileno(),
            )
    finally:
        descriptor_handle.close()


@pytest.mark.parametrize("factory", [connection_profile.open_connection, connection_profile.open_daemon_connection])
def test_schema_skew_write_profiles_refuse_stale_archive_tier_before_returning_connection(
    tmp_path: Path, factory: Callable[..., sqlite3.Connection]
) -> None:
    db_path = tmp_path / "user.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER] - 1}")

    with pytest.raises(SchemaSkew) as excinfo:
        factory(db_path)

    assert excinfo.value.tier == ArchiveTier.USER.value
    assert excinfo.value.expected == ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER]
    assert excinfo.value.found == ARCHIVE_VERSION_BY_TIER[ArchiveTier.USER] - 1
    assert "rebuild or migrate" in excinfo.value.remedy


def test_schema_skew_read_profile_refuses_stale_archive_tier(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] - 1}")

    with pytest.raises(SchemaSkew, match="index schema skew"):
        connection_profile.open_readonly_connection(db_path)


def test_schema_skew_explicit_tier_checks_noncanonical_generation_path(tmp_path: Path) -> None:
    db_path = tmp_path / "generation.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] - 1}")

    with pytest.raises(SchemaSkew) as excinfo:
        connection_profile.open_readonly_connection(db_path, tier=ArchiveTier.SOURCE)

    assert excinfo.value.tier == ArchiveTier.SOURCE.value


@pytest.mark.parametrize("tier", [ArchiveTier.INDEX, ArchiveTier.OPS])
def test_derived_schema_identity_is_checked_at_profile_seam(tmp_path: Path, tier: ArchiveTier) -> None:
    db_path = tmp_path / f"{tier.value}.db"
    initialize_archive_database(db_path, tier)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE schema_identity SET identity = 'wrong' WHERE tier = ?", (tier.value,))

    with pytest.raises(SchemaSkew) as excinfo:
        connection_profile.open_readonly_connection(db_path)

    assert excinfo.value.tier == tier.value
    assert excinfo.value.found == "wrong"
    assert excinfo.value.expected != "wrong"
    assert "rebuild the derived tier" in excinfo.value.remedy


def test_derived_schema_identity_missing_is_refused_at_profile_seam(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    initialize_archive_database(db_path, ArchiveTier.INDEX)
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE schema_identity")

    with pytest.raises(SchemaSkew) as excinfo:
        connection_profile.open_readonly_connection(db_path, tier=ArchiveTier.INDEX)

    assert excinfo.value.tier == DerivedTier.INDEX.value
    assert excinfo.value.found is None


def test_connection_context_passes_explicit_tier_to_profile_seam(tmp_path: Path) -> None:
    db_path = tmp_path / "generation.sqlite"
    initialize_archive_database(db_path, ArchiveTier.SOURCE)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA user_version = 0")

    with pytest.raises(SchemaSkew) as excinfo:
        with connection_profile.connection_context(db_path, tier=ArchiveTier.SOURCE):
            pass

    assert excinfo.value.tier == ArchiveTier.SOURCE.value
