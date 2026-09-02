from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.core.errors import SchemaSkew
from polylogue.storage.sqlite import connection_profile
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
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
@pytest.mark.parametrize("tier", [ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT])
def test_schema_skew_write_profiles_route_durable_tiers_to_safe_remedy(
    tmp_path: Path, factory: Callable[..., sqlite3.Connection], tier: ArchiveTier
) -> None:
    db_path = tmp_path / f"{tier.value}.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[tier] - 1}")

    with pytest.raises(SchemaSkew) as excinfo:
        factory(db_path)

    assert excinfo.value.tier == tier.value
    assert excinfo.value.expected == ARCHIVE_VERSION_BY_TIER[tier]
    assert excinfo.value.found == ARCHIVE_VERSION_BY_TIER[tier] - 1
    assert "numbered durable-tier migration path" in excinfo.value.remedy
    assert "verified backup" in excinfo.value.remedy
    assert "rebuild" not in excinfo.value.remedy


@pytest.mark.parametrize("factory", [connection_profile.open_connection, connection_profile.open_daemon_connection])
@pytest.mark.parametrize("tier", [ArchiveTier.INDEX, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS])
def test_schema_skew_write_profiles_keep_rebuild_remedy_for_non_durable_tiers(
    tmp_path: Path, factory: Callable[..., sqlite3.Connection], tier: ArchiveTier
) -> None:
    db_path = tmp_path / f"{tier.value}.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[tier] - 1}")

    with pytest.raises(SchemaSkew) as excinfo:
        factory(db_path, tier=tier)

    assert "rebuild" in excinfo.value.remedy
    assert "derived/disposable tier" in excinfo.value.remedy


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
