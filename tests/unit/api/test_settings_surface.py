"""Facade wiring for the durable ``user_settings`` liveness slice (polylogue-at44).

``user_settings`` had DDL + a migration but no runtime caller before this
module. The facade keeps the read pair (``get_setting``/``list_settings``);
the write left it entirely (polylogue-gjwto / polylogue-r29bv) because
``user.db`` is durable and the daemon is its sole writer, so these tests seed
rows through the storage owner and prove the async facade and the sync helpers
in ``user_settings_write.py`` agree (the "STORAGE TWINS" wiring the bead calls
out). The write route's own evidence is
``tests/unit/operations/test_user_setting_write_authority.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue import Polylogue
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.user_settings_write import set_user_setting
from polylogue.storage.sqlite.connection_profile import open_connection


def _init_tiers(archive_root: Path, *, with_user: bool = True) -> None:
    """Bootstrap through the production owner, not a hand-rolled tier set."""

    initialize_active_archive_root(archive_root)
    if not with_user:
        (archive_root / "user.db").unlink()


def _seed_setting(archive_root: Path, setting_key: str, value: object) -> None:
    """Write one row through the storage owner the daemon's actuator drives."""

    conn = open_connection(archive_root / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        set_user_setting(conn, setting_key, value, author_ref="user:local")  # type: ignore[arg-type]
        conn.commit()
    finally:
        conn.close()


async def test_get_setting_returns_none_when_unset(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        assert await poly.get_setting("subscription_tier") is None
        assert await poly.list_settings() == []


async def test_get_and_list_settings_read_the_durable_row(tmp_path: Path) -> None:
    """The facade reads exactly what the storage owner wrote.

    Anti-vacuity: have ``get_setting`` answer from a cache or a default table
    instead of ``user.db`` and the seeded value below stops coming back.
    """
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root)
    _seed_setting(archive_root, "subscription_tier", "max_5x")

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        fetched = await poly.get_setting("subscription_tier")
        listed = await poly.list_settings()

    assert fetched is not None
    assert fetched.value == "max_5x"
    assert [row.setting_key for row in listed] == ["subscription_tier"]
    assert listed[0] == fetched


async def test_the_facade_exposes_no_setting_writer(tmp_path: Path) -> None:
    """The in-process write route is gone, not renamed (polylogue-gjwto AC1).

    Anti-vacuity: restore ``Polylogue.set_setting`` -- under any name that
    reaches ``_execute_facade_mutation`` -- and this goes red, which is the
    point: the acceptance pairs "no facade writer" with the declared
    ``mutation.user.setting.set`` operation, so a rename cannot satisfy it.
    """
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        assert not hasattr(poly, "set_setting")


async def test_get_setting_returns_none_when_user_tier_missing(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root, with_user=False)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        assert await poly.get_setting("subscription_tier") is None
        assert await poly.list_settings() == []
