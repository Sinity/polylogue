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

import pytest

from polylogue import Polylogue
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.user_settings_write import set_user_setting
from polylogue.storage.sqlite.connection_profile import open_connection
from tests.infra.archive_templates import run_off_event_loop


def _init_tiers_on_writer(archive_root: Path, *, with_user: bool = True) -> None:
    """Bootstrap through the production owner, not a hand-rolled tier set."""

    initialize_active_archive_root(archive_root)
    if not with_user:
        (archive_root / "user.db").unlink()


def _init_tiers(archive_root: Path, *, with_user: bool = True) -> None:
    """Run the synchronous seed off any running event loop."""
    return run_off_event_loop(lambda: _init_tiers_on_writer(archive_root, with_user=with_user))


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


async def test_settings_refuse_missing_user_authority(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _init_tiers(archive_root, with_user=False)

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        with pytest.raises(ArchiveTierUnavailableError):
            await poly.get_setting("subscription_tier")
        with pytest.raises(ArchiveTierUnavailableError):
            await poly.list_settings()


@pytest.mark.parametrize(
    "route,tier",
    [
        ("get_setting", "user"),
        ("list_settings", "user"),
        ("get_context_delivery", "user"),
        ("list_context_deliveries", "user"),
        ("list_context_injection_ledger", "ops"),
        ("correlate_hermes_context_deliveries", "source"),
        ("correlate_hermes_context_deliveries", "user"),
    ],
)
@pytest.mark.parametrize("fault", ["missing", "corrupt", "missing_table"])
async def test_facade_required_authority_faults_are_typed(
    tmp_path: Path,
    route: str,
    tier: str,
    fault: str,
) -> None:
    """Mutation: return the route's empty value on a read fault and this fails."""
    root = tmp_path / "archive"
    _init_tiers(root)
    path = root / f"{tier}.db"
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"invalid sqlite")
    else:
        table = {
            "get_setting": "user_settings",
            "list_settings": "user_settings",
            "get_context_delivery": "context_deliveries",
            "list_context_deliveries": "context_deliveries",
            "list_context_injection_ledger": "context_injection_ledger",
            "correlate_hermes_context_deliveries": "raw_hook_events" if tier == "source" else "context_deliveries",
        }[route]
        with sqlite3.connect(path) as conn:
            conn.execute(f'DROP TABLE "{table}"')
    async with Polylogue(archive_root=root, db_path=root / "index.db") as poly:
        kwargs = {
            "get_setting": {"setting_key": "subscription_tier"},
            "get_context_delivery": {"snapshot_ref": "context-snapshot:missing", "recipient_ref": "agent:neutral"},
            "correlate_hermes_context_deliveries": {"hermes_session_native_id": "neutral-session"},
        }.get(route, {})
        with pytest.raises(ArchiveTierUnavailableError) as refusal:
            await getattr(poly, route)(**kwargs)
        assert refusal.value.tier == tier
    assert path.exists() is (fault != "missing")


async def test_empty_context_authority_remains_measured_absence(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_tiers(root)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as poly:
        assert await poly.get_context_delivery("context-snapshot:missing", recipient_ref="agent:neutral") is None
        assert (await poly.list_context_deliveries()).items == ()
        assert await poly.list_context_injection_ledger() == []
        assert await poly.correlate_hermes_context_deliveries("neutral-session") == ()


@pytest.mark.parametrize("route", ["get_setting", "list_settings"])
async def test_settings_refuse_uninspectable_stored_value(tmp_path: Path, route: str) -> None:
    root = tmp_path / "archive"
    _init_tiers(root)
    _seed_setting(root, "subscription_tier", "max_5x")
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute("UPDATE user_settings SET value_json = 'invalid json'")
    async with Polylogue(archive_root=root, db_path=root / "index.db") as poly:
        kwargs = {"setting_key": "subscription_tier"} if route == "get_setting" else {}
        with pytest.raises(ArchiveTierUnavailableError):
            await getattr(poly, route)(**kwargs)
