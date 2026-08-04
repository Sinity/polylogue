"""Facade wiring for the durable ``user_settings`` liveness slice (polylogue-at44).

``user_settings`` had DDL + a migration but no runtime caller before this
module -- these tests exercise the write-capable facade methods
(``set_setting``/``get_setting``/``list_settings``) end-to-end against a real
archive, proving the async ``Polylogue`` facade and the sync storage helpers
in ``user_settings_write.py`` agree (the "STORAGE TWINS" wiring the bead
calls out).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_settings_write import ArchiveUserSettingEnvelope


def _init_tiers(archive_root: Path, *, with_user: bool = True) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(archive_root / "source.db") as conn:
        initialize_archive_tier(conn, ArchiveTier.SOURCE)
    with sqlite3.connect(archive_root / "index.db") as conn:
        initialize_archive_tier(conn, ArchiveTier.INDEX)
    if with_user:
        with sqlite3.connect(archive_root / "user.db") as conn:
            initialize_archive_tier(conn, ArchiveTier.USER)


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

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        with pytest.raises(ValueError, match="user settings tier is not initialized"):
            await poly.set_setting("subscription_tier", "pro")


async def test_get_setting_returns_none_when_user_tier_missing(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root, with_user=False)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        assert await poly.get_setting("subscription_tier") is None
        assert await poly.list_settings() == []
