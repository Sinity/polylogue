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
    # A tier-named file makes the factory assert that tier's declared version;
    # this test's subject is descriptor binding.
    db_path = tmp_path / "descriptor-bound.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
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
    db_path = tmp_path / "index.db"
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
    db_path = tmp_path / "index.db"
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
    assert "durable state" in excinfo.value.remedy
    assert "do not rebuild" in excinfo.value.remedy
    assert "migrate-tier user" in excinfo.value.remedy


@pytest.mark.parametrize(
    ("tier", "expected_terms"),
    [
        (ArchiveTier.SOURCE, ("durable state", "do not rebuild", "migrate-tier source")),
        (ArchiveTier.AUDIT, ("durable state", "do not rebuild", "migrate-tier audit")),
        (ArchiveTier.INDEX, ("rebuildable derived state", "rebuild or recreate")),
        (ArchiveTier.EMBEDDINGS, ("expensive_rebuild derived state", "rebuild or recreate")),
        (ArchiveTier.OPS, ("disposable derived state", "rebuild or recreate")),
    ],
)
def test_schema_skew_remedy_matches_tier_durability(tier: ArchiveTier, expected_terms: tuple[str, ...]) -> None:
    remedy = connection_profile._schema_skew_remedy(tier)

    for term in expected_terms:
        assert term in remedy


def test_schema_skew_read_profile_refuses_stale_archive_tier(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] - 1}")

    with pytest.raises(SchemaSkew, match="index schema skew"):
        connection_profile.open_readonly_connection(db_path)


def test_schema_skew_diagnostic_read_profile_opens_stale_archive_tier(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    stale_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX] - 1
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {stale_version}")

    reader = connection_profile.open_readonly_connection(db_path, validate_schema=False)
    try:
        assert reader.execute("PRAGMA user_version").fetchone() == (stale_version,)
    finally:
        reader.close()


def test_schema_skew_explicit_tier_checks_noncanonical_generation_path(tmp_path: Path) -> None:
    db_path = tmp_path / "generation.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] - 1}")

    with pytest.raises(SchemaSkew) as excinfo:
        connection_profile.open_readonly_connection(db_path, tier=ArchiveTier.SOURCE)

    assert excinfo.value.tier == ArchiveTier.SOURCE.value


def test_scratch_synchronous_override_only_honours_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: production profiles keep NORMAL unless the harness asks for OFF."""
    profile = connection_profile.WRITE_CONNECTION_PROFILE
    monkeypatch.delenv(connection_profile.SCRATCH_SYNCHRONOUS_ENV, raising=False)
    assert "PRAGMA synchronous = NORMAL" in profile.pragma_statements
    monkeypatch.setenv(connection_profile.SCRATCH_SYNCHRONOUS_ENV, "FULL")
    assert "PRAGMA synchronous = NORMAL" in profile.pragma_statements
    monkeypatch.setenv(connection_profile.SCRATCH_SYNCHRONOUS_ENV, "off")
    assert "PRAGMA synchronous = OFF" in profile.pragma_statements
    assert "PRAGMA synchronous = NORMAL" not in profile.pragma_statements


@pytest.mark.parametrize("factory", [connection_profile.open_connection, connection_profile.open_daemon_connection])
@pytest.mark.parametrize(
    "sibling_tier", [ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS]
)
def test_index_write_profiles_refuse_stale_sibling_before_attach(
    tmp_path: Path,
    factory: Callable[..., sqlite3.Connection],
    sibling_tier: ArchiveTier,
) -> None:
    root = tmp_path
    index_path = root / "index.db"
    with sqlite3.connect(index_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
    sibling_path = root / f"{sibling_tier.value}.db"
    with sqlite3.connect(sibling_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[sibling_tier] - 1}")

    with pytest.raises(SchemaSkew) as excinfo:
        factory(index_path)

    assert excinfo.value.tier == sibling_tier.value
    assert excinfo.value.expected == ARCHIVE_VERSION_BY_TIER[sibling_tier]
    assert excinfo.value.found == ARCHIVE_VERSION_BY_TIER[sibling_tier] - 1
