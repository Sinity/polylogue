"""``setting set`` is a durable user.db write and must carry its three records.

polylogue-r29bv criterion 1.  ``Polylogue.set_setting`` opened ``user.db`` and
committed a row directly -- no preview, no authorization record, no audit row.
``user.db`` is the archive's one irreplaceable tier, so that is a wrong-owner
write: the only durable trace of it was the row itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _audit_counts(archive_root: Path, operation: str) -> dict[str, int]:
    with sqlite3.connect(archive_root / "audit.db") as conn:
        previews = conn.execute(
            "SELECT COUNT(*) FROM operation_previews WHERE operation_name = ?", (operation,)
        ).fetchone()[0]
        authorizations = conn.execute(
            """
            SELECT COUNT(*) FROM operation_authorizations AS a
            JOIN operation_previews AS p ON p.preview_id = a.preview_id
            WHERE p.operation_name = ?
            """,
            (operation,),
        ).fetchone()[0]
        attempts = conn.execute(
            """
            SELECT COUNT(*) FROM operation_attempts AS t
            JOIN operation_authorizations AS a ON a.authorization_id = t.authorization_id
            JOIN operation_previews AS p ON p.preview_id = a.preview_id
            WHERE p.operation_name = ?
            """,
            (operation,),
        ).fetchone()[0]
    return {"previews": int(previews), "authorizations": int(authorizations), "attempts": int(attempts)}


@pytest.mark.asyncio
async def test_setting_set_records_a_preview_an_authorization_and_an_attempt(tmp_path: Path) -> None:
    """The three records the executor cycle exists to produce.

    Anti-vacuity: restore the direct ``open_connection(user.db)`` write in
    ``Polylogue.set_setting`` and all three counts stay 0 while the value
    still lands -- which is exactly the state this bead names, and why
    asserting only that the setting was written proves nothing.
    """

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    assert _audit_counts(archive_root, "mutate-set-user-setting") == {
        "previews": 0,
        "authorizations": 0,
        "attempts": 0,
    }

    async with Polylogue(archive_root=archive_root) as poly:
        envelope = await poly.set_setting("subscription_tier", "pro")

    assert envelope.setting_key == "subscription_tier"
    assert envelope.value == "pro"
    assert _audit_counts(archive_root, "mutate-set-user-setting") == {
        "previews": 1,
        "authorizations": 1,
        "attempts": 1,
    }


@pytest.mark.asyncio
async def test_a_refused_setting_never_issues_an_authorization(tmp_path: Path) -> None:
    """Validation runs in PREPARE, so a rejected write is never authorized.

    Anti-vacuity: move ``validate_user_setting`` out of
    ``SetUserSettingActuator.prepare`` and back into the apply-time
    ``set_user_setting`` call, and an authorization is issued for a write that
    cannot happen -- the authorization count below becomes 1.
    """

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    async with Polylogue(archive_root=archive_root) as poly:
        with pytest.raises(ValueError, match="unknown setting key"):
            await poly.set_setting("not_a_registered_key", "whatever")
        with pytest.raises(ValueError, match="subscription_tier must be one of"):
            await poly.set_setting("subscription_tier", "not-a-tier")

    assert _audit_counts(archive_root, "mutate-set-user-setting")["authorizations"] == 0

    with sqlite3.connect(archive_root / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM user_settings").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_the_setting_round_trips_through_the_public_read(tmp_path: Path) -> None:
    """The route change must not alter what the surface answers.

    Anti-vacuity: drop the ``domain_receipt['envelope']`` return from
    ``SetUserSettingActuator.apply`` and the facade returns something the
    reader cannot match.
    """

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    async with Polylogue(archive_root=archive_root) as poly:
        written = await poly.set_setting("subscription_tier", "pro")
        read_back = await poly.get_setting("subscription_tier")

    assert read_back is not None
    assert (read_back.setting_key, read_back.value) == (written.setting_key, written.value)
