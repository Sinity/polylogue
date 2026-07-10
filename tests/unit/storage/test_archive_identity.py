from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityConflictError,
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


def test_sharing_only_one_durable_tier_does_not_conflate_archives(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    active = tmp_path / "active"
    _touch_tiers(configured)
    _touch_tiers(active)
    (active / "source.db").unlink()
    (active / "source.db").symlink_to(configured / "source.db")

    identity = assert_writable_archive_identity(configured_root=configured, active_root=active)

    assert identity.tier("user").stable_id != ArchiveIdentity.resolve(configured).tier("user").stable_id


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
