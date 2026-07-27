from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.storage.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityConflictError,
    ArchiveLocation,
    ArchiveOwnershipError,
    OwnedArchiveLocation,
    assert_owns_archive_location,
    assert_writable_archive_identity,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _touch_tiers(root: Path) -> None:
    root.mkdir()
    for name in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        (root / name).touch()


def test_path_aliases_resolve_to_equal_archive_identity(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _touch_tiers(root)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    direct = ArchiveIdentity.resolve(root)
    through_alias = ArchiveIdentity.resolve(alias)

    assert direct.durable_id == through_alias.durable_id
    assert direct.active_generation == through_alias.active_generation
    assert not direct.conflicts_with(through_alias)


def test_location_keeps_durable_tiers_at_configured_root_and_follows_active_pointer(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    canonical = tmp_path / "canonical"
    _touch_tiers(configured)
    _touch_tiers(canonical)
    for name in ("source.db", "user.db", "ops.db", "embeddings.db"):
        (configured / name).unlink()
        (configured / name).symlink_to(canonical / name)
    (configured / ".index-active-pointer").write_text(str(canonical / "index.db"), encoding="utf-8")

    location = ArchiveLocation.resolve(configured)

    assert location.active_index_path == canonical / "index.db"
    assert location.active_tier("source").resolved_path == (canonical / "source.db").resolve()
    assert location.active_tier("ops").resolved_path == (canonical / "ops.db").resolve()
    assert location.shadow_index is not None
    assert location.shadow_index.resolved_path == configured / "index.db"


def test_split_roots_sharing_durable_tiers_reject_distinct_indexes_before_mutation(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    active = tmp_path / "active"
    _touch_tiers(configured)
    _touch_tiers(active)
    for name in ("source.db", "user.db", "ops.db", "embeddings.db"):
        (active / name).unlink()
        (active / name).symlink_to(configured / name)
    configured_index_before = (configured / "index.db").stat()
    active_index_before = (active / "index.db").stat()

    with pytest.raises(ArchiveIdentityConflictError, match="writable index generations differ"):
        assert_writable_archive_identity(configured_root=configured, active_root=active)

    assert (configured / "index.db").stat().st_ino == configured_index_before.st_ino
    assert (active / "index.db").stat().st_ino == active_index_before.st_ino
    assert (configured / "index.db").stat().st_size == 0
    assert (active / "index.db").stat().st_size == 0


def test_distinct_archives_do_not_conflict(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    active = tmp_path / "active"
    _touch_tiers(configured)
    _touch_tiers(active)
    identity = assert_writable_archive_identity(configured_root=configured, active_root=active)
    assert identity.tier("index").resolved_path == active / "index.db"


@pytest.mark.parametrize("shared_tier", ["source.db", "user.db"])
def test_sharing_any_irreplaceable_tier_rejects_divergent_index(
    tmp_path: Path,
    shared_tier: str,
) -> None:
    configured = tmp_path / "configured"
    active = tmp_path / "active"
    _touch_tiers(configured)
    _touch_tiers(active)
    (active / shared_tier).unlink()
    (active / shared_tier).symlink_to(configured / shared_tier)

    with pytest.raises(ArchiveIdentityConflictError):
        assert_writable_archive_identity(configured_root=configured, active_root=active)


def test_archive_store_writer_route_enforces_identity_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured"
    active = tmp_path / "active"
    _touch_tiers(configured)
    _touch_tiers(active)
    for name in ("source.db", "user.db"):
        (active / name).unlink()
        (active / name).symlink_to(configured / name)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(configured))

    with pytest.raises(ArchiveIdentityConflictError):
        ArchiveStore.open_existing(active, read_only=False)

    assert (active / "index.db").stat().st_size == 0


@pytest.mark.parametrize("missing_side", ["active", "configured"])
def test_missing_index_cannot_bypass_split_root_preflight(tmp_path: Path, missing_side: str) -> None:
    configured = tmp_path / "configured"
    active = tmp_path / "active"
    _touch_tiers(configured)
    _touch_tiers(active)
    for name in ("source.db", "user.db"):
        (active / name).unlink()
        (active / name).symlink_to(configured / name)
    missing_root = active if missing_side == "active" else configured
    (missing_root / "index.db").unlink()

    with pytest.raises(ArchiveIdentityConflictError):
        assert_writable_archive_identity(configured_root=configured, active_root=active)

    assert not (missing_root / "index.db").exists()


def test_owned_location_rejects_concurrent_acquire_before_any_sqlite_file_exists(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    # No tier files exist yet -- the fixture proves the failure below happens
    # strictly before any SQLite tier file is created, not merely before an
    # already-open connection would have used it.
    location = ArchiveLocation.resolve(root)

    first = OwnedArchiveLocation.acquire(location)
    try:
        with pytest.raises(ArchiveOwnershipError, match="already owned"):
            OwnedArchiveLocation.acquire(location)
    finally:
        first.release()

    for name in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        assert not (root / name).exists(), f"{name} must not exist: ownership must fail before SQLite opens"


def test_owned_location_reclaims_lock_left_by_dead_process(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    location = ArchiveLocation.resolve(root)
    lock_path = root / ".archive-ownership.lock"
    # A pid that cannot plausibly be alive: simulate a crashed prior owner
    # without depending on any real process's exit timing.
    dead_pid = 2**30
    while _pid_is_alive_for_test(dead_pid):
        dead_pid += 1
    lock_path.write_text(f"pid={dead_pid} host=stale token=dead\n", encoding="utf-8")

    owned = OwnedArchiveLocation.acquire(location)
    try:
        assert owned.owner_id is not None
        # Reclaim must retry the SAME inode, never create a competing one: a
        # prior implementation swapped in a fresh file via os.replace, which
        # could let two racing reclaimers each believe they hold exclusive
        # ownership of a *different* inode. No such artifact should exist.
        ownership_paths = [p.name for p in root.iterdir() if "ownership" in p.name]
        assert ownership_paths == [lock_path.name]
    finally:
        owned.release()


def _pid_is_alive_for_test(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_assert_owns_archive_location_rejects_foreign_root(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _touch_tiers(root_a)
    _touch_tiers(root_b)
    location_a = ArchiveLocation.resolve(root_a)
    location_b = ArchiveLocation.resolve(root_b)

    owned = OwnedArchiveLocation.acquire(location_a)
    try:
        with pytest.raises(ArchiveOwnershipError, match="does not cover this location"):
            assert_owns_archive_location(owned, location_b)
        # A matching location is accepted without raising.
        assert_owns_archive_location(owned, location_a)
    finally:
        owned.release()


def test_assert_owns_archive_location_rejects_stale_generation_after_pointer_rotation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    generation_a = tmp_path / "gen-a"
    generation_b = tmp_path / "gen-b"
    _touch_tiers(root)
    generation_a.mkdir()
    generation_b.mkdir()
    (generation_a / "index.db").touch()
    (generation_b / "index.db").touch()
    (root / ".index-active-pointer").write_text(str(generation_a / "index.db"), encoding="utf-8")

    owned = OwnedArchiveLocation.acquire(ArchiveLocation.resolve(root))
    try:
        # Simulate a concurrent promotion rotating the active generation
        # after ownership was acquired: the proof must not silently cover it.
        (root / ".index-active-pointer").unlink()
        (root / ".index-active-pointer").write_text(str(generation_b / "index.db"), encoding="utf-8")
        rotated = ArchiveLocation.resolve(root)

        with pytest.raises(ArchiveOwnershipError, match="stale for the current active generation"):
            assert_owns_archive_location(owned, rotated)
    finally:
        owned.release()
