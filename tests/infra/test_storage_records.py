"""Focused tests for shared storage-record test infrastructure."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pytest

from tests.infra.storage_records import make_message, make_session, store_records


def test_store_records_commits_within_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.storage_records as storage_helpers

    class TrackingLock:
        def __init__(self) -> None:
            self.held = False

        def __enter__(self) -> TrackingLock:
            self.held = True
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
            self.held = False
            return False

    class TrackingConnection(sqlite3.Connection):
        _lock: TrackingLock
        commit_states: list[bool]

        def commit(self) -> None:
            self.commit_states.append(self._lock.held)
            super().commit()

    lock = TrackingLock()
    conn = sqlite3.connect(":memory:", factory=TrackingConnection)
    conn._lock = lock
    conn.commit_states = []

    @contextmanager
    def fake_connection_context(passed_conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        yield passed_conn

    conn.execute("CREATE TABLE sessions (origin TEXT, native_id TEXT, content_hash BLOB)")

    monkeypatch.setattr(storage_helpers, "_WRITE_LOCK", lock)
    monkeypatch.setattr(storage_helpers, "connection_context", fake_connection_context)
    # The single seeding route is the production writer; this test is about
    # the commit/lock ordering around it, not about what it stores.
    monkeypatch.setattr(storage_helpers, "write_parsed_session_to_archive", lambda *_, **__: None)

    result = storage_helpers.store_records(
        session=make_session("test:1", title="Test", content_hash="abc123"),
        messages=[make_message("test:1:msg1", "test:1", text="Hello")],
        attachments=[],
        conn=conn,
    )

    assert result["sessions"] == 1
    assert result["messages"] == 1
    assert conn.commit_states == [True]
    conn.close()


def test_concurrent_store_records_no_deadlock(workspace_env: Mapping[str, Path]) -> None:
    import concurrent.futures

    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(None):
        pass

    def store_one(idx: int) -> dict[str, int]:
        return store_records(
            session=make_session(f"test:{idx}", title=f"Test {idx}", content_hash=f"hash{idx}"),
            messages=[make_message(f"test:{idx}:msg1", f"test:{idx}", text=f"Hello {idx}")],
            attachments=[],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(store_one, i) for i in range(10)]
        results = [future.result(timeout=30) for future in futures]

    assert len(results) == 10
    assert all(result["sessions"] == 1 for result in results)


def test_make_message_rejects_non_json_provider_meta() -> None:
    with pytest.raises(TypeError, match="message provider_meta"):
        make_message(provider_meta={"path": Path("not-json")})


def test_make_message_rejects_non_json_content_block_input() -> None:
    with pytest.raises(TypeError, match="content block tool input"):
        make_message(
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "shell",
                    "tool_input": {"path": Path("not-json")},
                }
            ]
        )


def test_make_message_preserves_json_content_block_input() -> None:
    message = make_message(
        blocks=[
            {
                "type": "tool_use",
                "tool_name": "shell",
                "tool_input": {"command": "pytest -q"},
            }
        ]
    )

    assert len(message.blocks) == 1
    assert message.blocks[0].tool_input == '{"command":"pytest -q"}'


# ---------------------------------------------------------------------------
# polylogue-ugkho: a seeded row must be indistinguishable from a produced one
# in ``identity_source`` and ``material_origin``.
#
# Anti-vacuity for this group: drop the ``CONTENT_DERIVED_IDENTITY`` branch
# from ``_record_to_parsed_session`` (so the fixture always synthesizes a
# provider id again, as it did before polylogue-ugkho) and
# ``test_seeded_rows_can_take_the_content_derived_identity_branch`` and
# ``test_seeded_identity_source_agrees_with_the_stored_native_id`` both go red
# on the ``native_id IS NULL`` / ``identity_source = 'content'`` assertions.
# Asserting the ``message_id`` string alone would NOT be anti-vacuous: that
# column is generated from ``native_id``/``content_identity``, which the
# fixture already set before this change.
# ---------------------------------------------------------------------------

_SEED_TS = "2026-01-02T03:04:05+00:00"


def _seed_both_identity_shapes(db_path: Path) -> None:
    from tests.infra.storage_records import CONTENT_DERIVED_IDENTITY, SessionBuilder

    builder = SessionBuilder(db_path, "identity-shapes")
    builder.add_message(message_id="native-1", role="user", text="native one", timestamp=_SEED_TS)
    builder.add_message(
        role="assistant",
        text="content derived one",
        timestamp=_SEED_TS,
        identity_source=CONTENT_DERIVED_IDENTITY,
    )
    builder.save()


def test_seeded_rows_can_take_the_content_derived_identity_branch(db_path: Path) -> None:
    """A seeded corpus can contain the 'c:' shape, not only the 'n:' shape."""
    from polylogue.storage.sqlite.connection import open_connection

    _seed_both_identity_shapes(db_path)

    with open_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT native_id, identity_source, content_identity, message_id FROM messages ORDER BY position"
        ).fetchall()

    assert [str(row["identity_source"]) for row in rows] == ["native", "content"]
    native_row, content_row = rows
    assert str(native_row["native_id"]) == "native-1"
    assert content_row["native_id"] is None
    assert content_row["content_identity"]
    assert ":c:" in str(content_row["message_id"])


def test_seeded_identity_source_agrees_with_the_stored_native_id(db_path: Path) -> None:
    """``identity_source`` records the branch the row's id actually took.

    This is the invariant a production row always satisfies and a hand-seeded
    row could violate: ``identity_source = 'native'`` exactly when a
    ``native_id`` is stored, and the generated ``message_id`` carries the
    matching marker.
    """
    from polylogue.storage.sqlite.connection import open_connection

    _seed_both_identity_shapes(db_path)

    with open_connection(db_path) as conn:
        rows = conn.execute("SELECT native_id, identity_source, message_id FROM messages").fetchall()

    sources = {str(row["identity_source"]) for row in rows}
    # The law is only a law if the corpus actually contains both shapes;
    # a native-only corpus satisfies it vacuously.
    assert sources == {"native", "content"}, sources
    for row in rows:
        has_native = row["native_id"] is not None
        assert str(row["identity_source"]) == ("native" if has_native else "content")
        marker = ":n:" if has_native else ":c:"
        assert marker in str(row["message_id"])


def test_content_derived_seeded_ids_are_stable_across_reseeding(db_path: Path) -> None:
    """The content-derived id is a digest of declared semantics, not an ordinal.

    Seeding the same message after an extra message is inserted *ahead* of it
    must leave its id untouched -- the property the 'c:' branch exists for.
    """
    from polylogue.storage.sqlite.connection import open_connection
    from tests.infra.storage_records import CONTENT_DERIVED_IDENTITY, SessionBuilder

    def seed(db_path: Path, key: str, *, lead: bool) -> str:
        builder = SessionBuilder(db_path, key)
        if lead:
            builder.add_message(
                role="user",
                text="an unrelated earlier turn",
                timestamp=_SEED_TS,
                identity_source=CONTENT_DERIVED_IDENTITY,
            )
        builder.add_message(
            role="assistant",
            text="the message under test",
            timestamp=_SEED_TS,
            identity_source=CONTENT_DERIVED_IDENTITY,
        )
        builder.save()
        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT message_id, session_id FROM messages WHERE session_id = ? ORDER BY position DESC LIMIT 1",
                (builder.native_session_id(),),
            ).fetchone()
        # The session-local part of the id: the 'c:<digest>.<occurrence>'
        # anchor the content branch produces.
        return str(row["message_id"]).removeprefix(f"{row['session_id']}:")

    plain = seed(db_path, "stable-a", lead=False)
    with_lead = seed(db_path, "stable-b", lead=True)

    assert plain.startswith("c:"), plain
    assert plain == with_lead


def test_seeded_session_counters_come_from_the_stored_projection(db_path: Path) -> None:
    """``authored_user_*`` is the projection's output, not the record's input.

    The fixture deliberately declares a wrong in-memory ``word_count``. A
    seeding route that summed the record's own fields (the retired
    ``_upsert_session_stats_sync`` mirror did exactly that) would publish
    ``999``; the production projection recounts from the stored message text.
    """
    from polylogue.core.enums import MaterialOrigin
    from polylogue.storage.derived.session.summary import inspect_session_summary
    from polylogue.storage.sqlite.connection import open_connection
    from tests.infra.storage_records import SessionBuilder

    builder = SessionBuilder(db_path, "counters")
    builder.add_message(
        role="user",
        text="one two three",
        timestamp=_SEED_TS,
        word_count=999,
        material_origin=MaterialOrigin.HUMAN_AUTHORED,
    )
    builder.add_message(
        role="user",
        text="four five",
        timestamp=_SEED_TS,
        word_count=999,
        material_origin=MaterialOrigin.GENERATED_CONTEXT_PACK,
    )
    builder.save()
    session_id = builder.native_session_id()

    with open_connection(db_path) as conn:
        stored = conn.execute(
            "SELECT word_count, user_word_count, authored_user_word_count, authored_user_message_count "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        origins = [
            str(row["material_origin"])
            for row in conn.execute("SELECT material_origin FROM messages ORDER BY position")
        ]
        # Production's own drift census: every stored counter compared with
        # the value recomputed from the persisted message rows.
        inspection = inspect_session_summary(conn, deadline_s=None)

    assert origins == ["human_authored", "generated_context_pack"]
    assert int(stored["word_count"]) == 5
    assert int(stored["user_word_count"]) == 5
    assert int(stored["authored_user_word_count"]) == 3
    assert int(stored["authored_user_message_count"]) == 1
    assert inspection.state == "ready", inspection
    assert inspection.stale_sessions == 0


def test_store_records_has_no_second_index_tier_writer() -> None:
    """The fixture exposes exactly one seeding route: the production writer.

    A record-level SQL route is what let seeded rows carry columns the real
    writer could not produce (polylogue-ugkho).
    """
    import tests.infra.storage_records as storage_helpers

    for retired in ("upsert_session", "upsert_message", "upsert_attachment", "_upsert_session_stats_sync"):
        assert not hasattr(storage_helpers, retired), f"{retired} is a second index-tier writer"
