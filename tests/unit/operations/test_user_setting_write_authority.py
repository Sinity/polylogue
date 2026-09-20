"""``setting set`` is a durable user.db write and must carry its three records.

polylogue-r29bv criterion 1 and polylogue-gjwto. ``Polylogue.set_setting``
opened ``user.db`` and committed a row in whatever process held the surface.
``user.db`` is the archive's one irreplaceable tier, so that is a wrong-owner
write: the only durable trace of it was the row itself, and the CLI performed
it while ``polylogued run`` held the write lease. The write is now the declared
``mutation.user.setting.set`` operation, executed by the daemon.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.daemon_operations import DaemonOperationStack, running_daemon_operations

_OPERATION = "mutation.user.setting.set"
_ACTUATOR = "mutate-set-user-setting"


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


@pytest.fixture
def daemon_archive(tmp_path: Path) -> Iterator[tuple[DaemonOperationStack, Path]]:
    archive_root = (tmp_path / "archive").resolve()
    with running_daemon_operations(archive_root, seed_archive=initialize_active_archive_root) as stack:
        yield stack, archive_root


def _run(stack: DaemonOperationStack, archive_root: Path, payload: dict[str, object]) -> dict[str, Any]:
    envelope = stack.client.operation(_OPERATION, payload, archive_root=str(archive_root))
    assert envelope is not None, f"{_OPERATION} returned no envelope"
    return envelope


def _row(envelope: dict[str, Any]) -> dict[str, Any]:
    """The setting row the handler lowered onto the wire.

    The transport envelope carries the declared ``MutationResult``; the
    operation's own product value is that result's ``result`` field.
    """
    row = envelope["result"]["result"]
    assert isinstance(row, dict), envelope
    return row


def test_setting_set_records_a_preview_an_authorization_and_an_attempt(
    daemon_archive: tuple[DaemonOperationStack, Path],
) -> None:
    """The three records the executor cycle exists to produce, written by the daemon.

    Anti-vacuity: have ``mutation_user_setting_set`` call
    ``set_user_setting`` directly instead of driving the actuator through
    ``OperationExecutor``, and all three counts stay 0 while the value still
    lands -- exactly the state this bead names, and why asserting only that
    the setting was written proves nothing.
    """
    stack, archive_root = daemon_archive

    assert _audit_counts(archive_root, _ACTUATOR) == {"previews": 0, "authorizations": 0, "attempts": 0}

    envelope = _run(stack, archive_root, {"setting_key": "subscription_tier", "value": "pro"})

    assert envelope["outcome"] == "completed", envelope.get("error")
    row = _row(envelope)
    assert row["setting_key"] == "subscription_tier"
    assert row["value"] == "pro"
    assert row["author_ref"] == "user:local"
    assert _audit_counts(archive_root, _ACTUATOR) == {"previews": 1, "authorizations": 1, "attempts": 1}


def test_a_refused_setting_never_issues_an_authorization(
    daemon_archive: tuple[DaemonOperationStack, Path],
) -> None:
    """Validation runs in PREPARE, so a rejected write is never authorized.

    Anti-vacuity: move ``validate_user_setting`` out of
    ``SetUserSettingActuator.prepare`` and back into the apply-time
    ``set_user_setting`` call, and an authorization is issued for a write that
    cannot happen -- the authorization count below becomes 1.
    """
    stack, archive_root = daemon_archive

    unknown_key = _run(stack, archive_root, {"setting_key": "not_a_registered_key", "value": "whatever"})
    assert unknown_key["outcome"] != "completed", unknown_key
    assert "unknown setting key" in str(unknown_key.get("error"))

    bad_value = _run(stack, archive_root, {"setting_key": "subscription_tier", "value": "not-a-tier"})
    assert bad_value["outcome"] != "completed", bad_value
    assert "subscription_tier must be one of" in str(bad_value.get("error"))

    assert _audit_counts(archive_root, _ACTUATOR)["authorizations"] == 0
    with sqlite3.connect(archive_root / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM user_settings").fetchone()[0] == 0


def test_the_setting_round_trips_through_the_public_read(
    daemon_archive: tuple[DaemonOperationStack, Path],
) -> None:
    """The route change must not alter what the surface answers.

    Anti-vacuity: drop the ``domain_receipt['envelope']`` return from
    ``SetUserSettingActuator.apply`` and the handler has nothing to lower onto
    the wire; return a hard-coded row instead and the durable read below
    disagrees with it.
    """
    from polylogue.storage.sqlite.archive_tiers.user_settings_write import get_user_setting
    from polylogue.storage.sqlite.connection_profile import open_connection

    stack, archive_root = daemon_archive

    written = _row(_run(stack, archive_root, {"setting_key": "subscription_tier", "value": "pro"}))

    conn = open_connection(archive_root / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        stored = get_user_setting(conn, "subscription_tier")
    finally:
        conn.close()

    assert stored is not None
    assert (stored.setting_key, stored.value, stored.author_ref) == (
        written["setting_key"],
        written["value"],
        written["author_ref"],
    )


def test_updating_a_setting_replaces_the_row_rather_than_forking_one(
    daemon_archive: tuple[DaemonOperationStack, Path],
) -> None:
    """The operation is declared idempotent: a second write updates in place.

    Anti-vacuity: make ``set_user_setting`` INSERT without the upsert conflict
    clause and the row count below becomes 2.
    """
    stack, archive_root = daemon_archive

    _run(stack, archive_root, {"setting_key": "subscription_tier", "value": "max_5x"})
    updated = _run(stack, archive_root, {"setting_key": "subscription_tier", "value": "pro"})

    assert _row(updated)["value"] == "pro"
    with sqlite3.connect(archive_root / "user.db") as conn:
        rows = conn.execute("SELECT setting_key, value_json FROM user_settings").fetchall()
    assert len(rows) == 1, rows
