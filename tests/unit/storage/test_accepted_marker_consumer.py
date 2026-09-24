"""Accepted marker history is consumed through one durable user cursor."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import asdict
from pathlib import Path

import aiosqlite
import pytest

from polylogue.markers import candidates_for_block
from polylogue.storage.accepted_marker_inputs import (
    append_accepted_marker_input,
    excise_marker_input_targets_sync,
    marker_input_excision_targets_sync,
    prepare_accepted_marker_input,
)
from polylogue.storage.derived.session.marker_domain import SessionMarkerDerivation
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL
from polylogue.storage.sqlite.durable_change_train import validate_durable_migration_sidecars


def _candidate_record(text: str, *, message_id: str) -> dict[str, object]:
    candidate = candidates_for_block(message_id, f"{message_id}:0", text)[0]
    record = asdict(candidate)
    record["assertion_kind"] = candidate.assertion_kind.value if candidate.assertion_kind is not None else None
    return record


def _append_source_batch(source_db: Path, *, raw_id: str, candidate: dict[str, object]) -> None:
    async def append() -> None:
        async with aiosqlite.connect(source_db) as conn:
            await conn.executescript(SOURCE_DDL)
            batch = prepare_accepted_marker_input(
                raw_id,
                [{"session_id": f"source:{raw_id}", "candidates": [candidate]}],
            )
            await append_accepted_marker_input(conn, batch)
            await conn.commit()

    asyncio.run(append())


def _adapter(source_db: Path, user_db: Path) -> SessionMarkerDerivation:
    return SessionMarkerDerivation(
        lambda: sqlite3.connect(f"file:{source_db}?mode=ro", uri=True),
        lambda: sqlite3.connect(f"file:{user_db}?mode=ro", uri=True),
        lambda: sqlite3.connect(user_db),
    )


def _new_user_tier(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(USER_DDL)


def test_consumer_uses_retained_history_and_keeps_a_human_deletion(tmp_path: Path) -> None:
    """A later current index projection cannot erase an accepted earlier marker.

    Anti-vacuity: recover candidates from the current index instead of the
    source carrier and this test has no index marker to lower. Remove the
    durable cursor check and replay recreates the explicitly deleted assertion.
    """
    source_db = tmp_path / "source.db"
    user_db = tmp_path / "user.db"
    _new_user_tier(user_db)
    _append_source_batch(
        source_db, raw_id="earlier", candidate=_candidate_record("::note: retained lesson", message_id="m1")
    )
    adapter = _adapter(source_db, user_db)
    frame = object()

    keys, next_cursor = adapter.required_page(frame, cursor=None, limit=20)
    assert len(keys) == 1 and next_cursor is None
    replacement = adapter.compute(frame, keys[0])
    assert adapter.publish(frame, replacement) is True

    with sqlite3.connect(user_db) as user:
        assertion_id = str(user.execute("SELECT assertion_id FROM assertions").fetchone()[0])
        cursor = user.execute("SELECT stream_id, applied_sequence FROM accepted_marker_delivery_cursor").fetchone()
        assert cursor is not None and cursor[1] == 1
        user.execute(
            "UPDATE assertions SET status = 'deleted', author_kind = 'user' WHERE assertion_id = ?", (assertion_id,)
        )
        user.commit()

    restarted = _adapter(source_db, user_db)
    assert restarted.required_page(frame, cursor=None, limit=20) == ((), None)
    with sqlite3.connect(user_db) as user:
        assert user.execute(
            "SELECT status, author_kind FROM assertions WHERE assertion_id = ?", (assertion_id,)
        ).fetchone() == (
            "deleted",
            "user",
        )


def test_consumer_rolls_back_assertions_when_cursor_advance_fails(tmp_path: Path) -> None:
    """The cursor and lowered assertion belong to one real user-db transaction.

    Anti-vacuity: commit lowering before cursor advancement and the injected
    cursor trigger leaves a durable assertion even though restart retries the
    same source sequence.
    """
    source_db = tmp_path / "source.db"
    user_db = tmp_path / "user.db"
    _new_user_tier(user_db)
    _append_source_batch(
        source_db, raw_id="atomic", candidate=_candidate_record("::note: atomic lesson", message_id="m2")
    )
    adapter = _adapter(source_db, user_db)
    key = adapter.required_page(object(), cursor=None, limit=1)[0][0]
    replacement = adapter.compute(object(), key)

    with sqlite3.connect(user_db) as user:
        user.execute(
            "CREATE TRIGGER fail_cursor BEFORE INSERT ON accepted_marker_delivery_cursor "
            "BEGIN SELECT RAISE(ABORT, 'cursor failure'); END"
        )
        user.commit()
    with pytest.raises(sqlite3.IntegrityError, match="cursor failure"):
        adapter.publish(object(), replacement)
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone() == (0,)
        assert user.execute("SELECT COUNT(*) FROM accepted_marker_delivery_cursor").fetchone() == (0,)


def test_consumer_restarts_in_order_and_bad_retained_batch_blocks(tmp_path: Path) -> None:
    """Committed source positions resume once, while malformed next bytes cannot skip.

    Anti-vacuity: advance a per-session timestamp cursor or ignore a malformed
    source payload and the second accepted input is either replayed or skipped.
    """
    source_db = tmp_path / "source.db"
    user_db = tmp_path / "user.db"
    _new_user_tier(user_db)
    _append_source_batch(source_db, raw_id="first", candidate=_candidate_record("::note: first", message_id="m3"))
    _append_source_batch(source_db, raw_id="bad", candidate={"match": {}})
    adapter = _adapter(source_db, user_db)

    first_key = adapter.required_page(object(), cursor=None, limit=10)[0][0]
    assert adapter.publish(object(), adapter.compute(object(), first_key)) is True
    with sqlite3.connect(user_db) as user:
        before = user.execute("SELECT COUNT(*) FROM assertions").fetchone()
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (1,)

    bad_key = adapter.required_page(object(), cursor=None, limit=10)[0][0]
    with pytest.raises(ValueError, match="accepted marker carrier has an invalid candidate payload"):
        adapter.compute(object(), bad_key)
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone() == before
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (1,)


def test_excised_unconsumed_carrier_cannot_be_delivered(tmp_path: Path) -> None:
    """Source excision removes an unconsumed carrier before any user effect.

    Anti-vacuity: deriving markers from a current index session, or retaining a
    private consumer copy of the payload, would lower the marker after source
    excision despite the tombstone and erased accepted input.
    """
    source_db = tmp_path / "source.db"
    user_db = tmp_path / "user.db"
    _new_user_tier(user_db)
    _append_source_batch(source_db, raw_id="purged", candidate=_candidate_record("::note: remove me", message_id="m4"))

    with sqlite3.connect(source_db) as source:
        targets = marker_input_excision_targets_sync(
            source,
            target_session_ids=frozenset({"source:purged"}),
            target_raw_ids=frozenset(),
        )
        assert len(targets) == 1
        assert excise_marker_input_targets_sync(source, targets, excised_at_ms=1) == {"pending": 0, "accepted": 1}
        source.commit()

    adapter = _adapter(source_db, user_db)
    assert adapter.required_page(object(), cursor=None, limit=1) == ((), None)
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone() == (0,)
        assert user.execute("SELECT COUNT(*) FROM accepted_marker_delivery_cursor").fetchone() == (0,)


def test_delivery_cursor_migration_is_additive_and_matches_fresh_user_ddl(tmp_path: Path) -> None:
    """The numbered durable route adds the cursor without rewriting user state.

    Anti-vacuity: omitting the migration leaves the historical user tier
    without the cursor; adding a destructive rebuild loses ``user_settings``.
    """
    migration = Path("polylogue/storage/sqlite/migrations/user/004_accepted_marker_delivery_cursor.sql").read_text()
    steps = migration_runner._load_migrations(ArchiveTier.USER)
    sidecars = validate_durable_migration_sidecars(ArchiveTier.USER, tuple((step.name, step.sql) for step in steps))
    sidecar = next(item for item in sidecars if item.slot == 4)
    assert sidecar.train.target_version == 4
    assert sidecar.train.migration.requires_backup is True
    migrated_path = tmp_path / "migrated-user.db"
    with sqlite3.connect(migrated_path) as user:
        # The v3 shape is current DDL less this slot's additive table.
        user.executescript(USER_DDL)
        user.execute("DROP TABLE accepted_marker_delivery_cursor")
        user.execute("INSERT INTO user_settings(setting_key, value_json, updated_at_ms) VALUES ('retained', '{}', 1)")
        user.executescript(migration)
        migrated_cursor_ddl = user.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'accepted_marker_delivery_cursor'"
        ).fetchone()
        assert user.execute("SELECT value_json FROM user_settings WHERE setting_key = 'retained'").fetchone() == ("{}",)

    fresh_path = tmp_path / "fresh-user.db"
    with sqlite3.connect(fresh_path) as fresh:
        fresh.executescript(USER_DDL)
        fresh_cursor_ddl = fresh.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'accepted_marker_delivery_cursor'"
        ).fetchone()
    assert migrated_cursor_ddl == fresh_cursor_ddl
