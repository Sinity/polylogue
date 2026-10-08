"""Accepted marker history is consumed through one durable user cursor."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from polylogue.markers import candidates_for_block
from polylogue.storage import accepted_marker_inputs as marker_input_store
from polylogue.storage.accepted_marker_inputs import (
    MixedAcceptedMarkerInputError,
    append_accepted_marker_input,
    excise_marker_input_targets_sync,
    marker_input_excision_targets_sync,
    prepare_accepted_marker_input,
)
from polylogue.storage.derived.session.marker_domain import SessionMarkerDerivation
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL


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
    assert len(keys) == 1 and next_cursor == keys[0]
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
    adapter = _adapter(source_db, user_db)
    key = adapter.required_page(object(), cursor=None, limit=1)[0][0]
    replacement = adapter.compute(object(), key)

    with sqlite3.connect(source_db) as source:
        targets = marker_input_excision_targets_sync(
            source,
            target_session_ids=frozenset({"source:purged"}),
            target_raw_ids=frozenset(),
        )
        assert len(targets) == 1
        assert excise_marker_input_targets_sync(source, targets, excised_at_ms=1) == {"pending": 0, "accepted": 1}
        source.commit()

    assert adapter.publish(object(), replacement) is True
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone() == (0,)
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (1,)


def test_consumer_advances_across_excised_sequence_and_delivers_later_batch(tmp_path: Path) -> None:
    """A durable excision tombstone preserves progress through the source stream.

    Anti-vacuity: restore the contiguous-sequence error in ``required_page``
    and the consumer remains stuck at sequence 1 instead of reaching 3.
    """
    source_db = tmp_path / "source.db"
    user_db = tmp_path / "user.db"
    _new_user_tier(user_db)
    _append_source_batch(source_db, raw_id="first", candidate=_candidate_record("::note: first", message_id="m1"))
    _append_source_batch(source_db, raw_id="removed", candidate=_candidate_record("::note: removed", message_id="m2"))
    _append_source_batch(source_db, raw_id="third", candidate=_candidate_record("::note: third", message_id="m3"))
    with sqlite3.connect(source_db) as source:
        targets = marker_input_excision_targets_sync(
            source,
            target_session_ids=frozenset({"source:removed"}),
            target_raw_ids=frozenset(),
        )
        assert len(targets) == 1
        excise_marker_input_targets_sync(source, targets, excised_at_ms=2)
        source.commit()

    adapter = _adapter(source_db, user_db)
    first_key = adapter.required_page(object(), cursor=None, limit=1)[0][0]
    assert adapter.publish(object(), adapter.compute(object(), first_key)) is True
    tombstone_key = adapter.required_page(object(), cursor=None, limit=1)[0][0]
    assert tombstone_key.endswith(":2")
    assert adapter.publish(object(), adapter.compute(object(), tombstone_key)) is True
    final_key = adapter.required_page(object(), cursor=None, limit=1)[0][0]
    assert final_key.endswith(":3")
    assert adapter.publish(object(), adapter.compute(object(), final_key)) is True
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (3,)
        bodies = user.execute("SELECT body_text FROM assertions ORDER BY assertion_id").fetchall()
        assert all("removed" not in str(body) for (body,) in bodies)


def test_primary_barrier_holds_a_carrier_whose_session_awaits_publication(tmp_path: Path) -> None:
    """Marker lowering waits for the carrier's sessions like every session-derived domain.

    Anti-vacuity (polylogue-wtfyv review): without ``barrier_sessions`` the kernel
    ignores the barrier for this domain and lowers the unpublished session's
    markers into user.db.
    """
    from polylogue.daemon.derivation import DerivationFrame, DerivationRegistry, Outcome, PendingReason, converge

    source_db = tmp_path / "source.db"
    user_db = tmp_path / "user.db"
    _new_user_tier(user_db)
    _append_source_batch(source_db, raw_id="held", candidate=_candidate_record("::note: waits", message_id="m1"))
    adapter = _adapter(source_db, user_db)
    frame = DerivationFrame(archive_root=str(tmp_path), source_revision="r1")

    held = converge(DerivationRegistry([adapter]), frame, barrier=lambda sessions: {"source:held"} & set(sessions))

    assert held.done == 0
    assert [(outcome.outcome, outcome.reason) for outcome in held.outcomes] == [
        (Outcome.PENDING, PendingReason.BLOCKED)
    ]
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone()[0] == 0

    released = converge(DerivationRegistry([adapter]), frame, barrier=lambda sessions: set())
    assert released.done == 1


def test_one_convergence_sweep_delivers_every_accepted_marker_batch(tmp_path: Path) -> None:
    """A one-key page with no continuation used to end the sweep after the first batch."""
    from polylogue.daemon.derivation import DerivationFrame, DerivationRegistry, Outcome, converge

    source_db, user_db = tmp_path / "source.db", tmp_path / "user.db"
    _new_user_tier(user_db)
    for ordinal in range(3):
        _append_source_batch(
            source_db,
            raw_id=f"w5-{ordinal}",
            candidate=_candidate_record(f"::note: batch {ordinal}", message_id=f"w5-m{ordinal}"),
        )
    adapter = _adapter(source_db, user_db)
    frame = DerivationFrame(archive_root=str(tmp_path), source_revision="w5")
    registry = DerivationRegistry([adapter])
    held = converge(registry, frame, barrier=lambda sessions: {"source:w5-0"} & set(sessions))
    assert held.done == 0
    assert all(outcome.outcome is Outcome.PENDING for outcome in held.outcomes)
    assert len(held.outcomes) == 1, "a held stream head must not loop or overtake its sequence"
    delivered = converge(registry, frame, barrier=lambda sessions: set())
    assert delivered.done == 3
    assert delivered.count(Outcome.FAILED) == 0
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone() == (3,)
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (3,)
    assert converge(registry, frame).done == 0


def test_consumer_streams_a_large_candidate_array_without_json_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large accepted carrier is decoded and lowered one candidate at a time."""
    source_db, user_db = tmp_path / "source.db", tmp_path / "user.db"
    _new_user_tier(user_db)
    candidates = [
        _candidate_record(f"::note: retained item {index}", message_id=f"bulk-message-{index}") for index in range(256)
    ]

    async def append() -> None:
        async with aiosqlite.connect(source_db) as conn:
            await conn.executescript(SOURCE_DDL)
            batch = prepare_accepted_marker_input("bulk", [{"session_id": "bulk-session", "candidates": candidates}])
            await append_accepted_marker_input(conn, batch)
            await conn.commit()

    asyncio.run(append())

    def forbid_json_loads(*args: object, **kwargs: object) -> object:
        pytest.fail("production marker consumption must stream the carrier JSON")

    monkeypatch.setattr(
        marker_input_store,
        "json",
        SimpleNamespace(dumps=json.dumps, loads=forbid_json_loads),
    )
    adapter = _adapter(source_db, user_db)
    key = adapter.required_page(object(), cursor=None, limit=1)[0][0]
    replacement = adapter.compute(object(), key)
    assert adapter.publish(object(), replacement) is True
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone() == (256,)
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (1,)


def test_excision_streams_complete_session_membership_without_json_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Excision sees a retained sibling after its matching session in the carrier."""
    source_db = tmp_path / "source.db"
    with sqlite3.connect(source_db) as source:
        source.executescript(SOURCE_DDL)
    request_sessions: tuple[dict[str, object], ...] = tuple({"session_id": f"source:{index}"} for index in range(300))

    async def append() -> None:
        async with aiosqlite.connect(source_db) as conn:
            batch = prepare_accepted_marker_input(
                "mixed", [{"session_id": "source:0", "candidates": []}], request_sessions=request_sessions
            )
            await append_accepted_marker_input(conn, batch)
            await conn.commit()

    asyncio.run(append())

    def forbid_json_loads(*args: object, **kwargs: object) -> object:
        pytest.fail("excision membership must stream the carrier JSON")

    monkeypatch.setattr(
        marker_input_store,
        "json",
        SimpleNamespace(dumps=json.dumps, loads=forbid_json_loads),
    )
    with sqlite3.connect(source_db) as source, pytest.raises(MixedAcceptedMarkerInputError):
        marker_input_excision_targets_sync(
            source,
            target_session_ids=frozenset({"source:0"}),
            target_raw_ids=frozenset(),
        )


def test_native_marker_consumer_runs_inside_an_active_event_loop(tmp_path: Path) -> None:
    """The native source connection stays on its opening thread through delivery."""
    source_db = tmp_path / "source.db"
    user_db = tmp_path / "user.db"
    _new_user_tier(user_db)
    _append_source_batch(
        source_db, raw_id="loop", candidate=_candidate_record("::note: loop delivery", message_id="m-loop")
    )

    async def deliver() -> None:
        adapter = _adapter(source_db, user_db)
        keys, cursor = adapter.required_page(object(), cursor=None, limit=20)
        assert len(keys) == 1 and cursor == keys[0]
        replacement = adapter.compute(object(), keys[0])
        assert adapter.publish(object(), replacement) is True
        assert adapter.required_page(object(), cursor=None, limit=20) == ((), None)

    asyncio.run(deliver())
    with sqlite3.connect(user_db) as user:
        assert user.execute("SELECT COUNT(*) FROM assertions").fetchone() == (1,)
        assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (1,)
