"""Facade wiring for the durable ``user_settings`` liveness slice (polylogue-at44).

``user_settings`` had DDL + a migration but no runtime caller before this
module -- these tests exercise the write-capable facade methods
(``set_setting``/``get_setting``/``list_settings``) end-to-end against a real
archive, proving the async ``Polylogue`` facade and the sync storage helpers
in ``user_settings_write.py`` agree (the "STORAGE TWINS" wiring the bead
calls out).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.user_settings_write import ArchiveUserSettingEnvelope


def _init_tiers(archive_root: Path, *, with_user: bool = True) -> None:
    """Bootstrap through the production owner, not a hand-rolled tier set.

    polylogue-r29bv routed ``set_setting`` onto the actuator/executor cycle,
    so the write now opens the archive the way every other facade mutation
    does -- which means the root has to be a real archive (format marker,
    audit tier) rather than three tier files in a directory.
    """

    initialize_active_archive_root(archive_root)
    if not with_user:
        (archive_root / "user.db").unlink()


async def test_get_setting_returns_none_when_unset(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        assert await poly.get_setting("subscription_tier") is None
        assert await poly.list_settings() == []


async def test_set_and_get_setting_round_trip(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        written = await poly.set_setting("subscription_tier", "max_5x")
        assert isinstance(written, ArchiveUserSettingEnvelope)
        assert written.value == "max_5x"

        fetched = await poly.get_setting("subscription_tier")
        assert fetched == written

        updated = await poly.set_setting("subscription_tier", "pro")
        assert updated.value == "pro"

        listed = await poly.list_settings()
        assert [row.setting_key for row in listed] == ["subscription_tier"]
        assert listed[0].value == "pro"


async def test_set_setting_rejects_unknown_key(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        with pytest.raises(ValueError, match="unknown setting key"):
            await poly.set_setting("not_a_real_setting", "anything")


async def test_set_setting_rejects_invalid_subscription_tier(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        with pytest.raises(ValueError, match="subscription_tier must be one of"):
            await poly.set_setting("subscription_tier", "not-a-real-tier")


async def test_set_setting_raises_when_user_tier_missing(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root, with_user=False)

    # The refusal moved with the route (polylogue-r29bv): the archive open
    # that every executor-routed mutation performs refuses first, naming the
    # missing durable tier, instead of the write discovering it later.
    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        with pytest.raises(RuntimeError, match="names a missing durable tier"):
            await poly.set_setting("subscription_tier", "pro")


async def test_get_setting_returns_none_when_user_tier_missing(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root, with_user=False)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        assert await poly.get_setting("subscription_tier") is None
        assert await poly.list_settings() == []
