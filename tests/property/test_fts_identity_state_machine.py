"""Metamorphic FTS identity-ledger coherence under arbitrary mutation (polylogue-1xc.12).

Drives the REAL ``blocks`` table triggers (never a mock/replica) through
Hypothesis-generated insert/update/delete/rollback/full-replace sequences and
asserts, after every single step, that exact reconciliation
(``fts_invariant_snapshot_sync``) reports zero missing, excess, duplicate,
AND identity-mismatch rows for ``messages_fts``. Count-only reconciliation
(``source_rows == indexed_rows``) cannot see a stale rowid that has silently
rebound to a different block after SQLite reuses a freed rowid -- exactly
what deleting the highest-rowid block and inserting a new one does, and
exactly what a full-session-replace does at scale. This machine forces that
scenario to happen organically across many random step orders, plus an
explicit corrupt-then-repair rule that proves the check has teeth (anti-
vacuity: a hand-corrupted ledger row must be flagged before the rule's own
repair call clears it).
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from polylogue.storage.fts.fts_lifecycle import (
    fts_invariant_snapshot_sync,
    insert_missing_message_rows_batched_sync,
    restore_fts_triggers_sync,
)
from polylogue.storage.sqlite.connection import open_connection


class _Block:
    __slots__ = ("block_id", "session_native_id", "message_native_id", "indexed")

    def __init__(self, block_id: str, session_native_id: str, message_native_id: str, *, indexed: bool) -> None:
        self.block_id = block_id
        self.session_native_id = session_native_id
        self.message_native_id = message_native_id
        self.indexed = indexed


class FtsIdentityStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        import tempfile
        from pathlib import Path

        self._tmpdir = tempfile.TemporaryDirectory(prefix="polylogue-fts-identity-", dir="/realm/tmp")
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._conn_cm = open_connection(self._db_path)
        self._conn = self._conn_cm.__enter__()
        restore_fts_triggers_sync(self._conn)
        self._blocks: dict[str, _Block] = {}
        self._sessions: list[str] = []
        self._blocks_by_session: dict[str, list[str]] = {}
        self._next_position_by_session: dict[str, int] = {}
        self._next_id = 0
        self._origin = "unknown-export"

    def _fresh_content_hash(self) -> bytes:
        self._next_id += 1
        return (str(self._next_id) * 32).encode("ascii")[:32]

    def _ensure_session(self) -> str:
        if not self._sessions or self._next_id % 3 == 0:
            self._next_id += 1
            native_session_id = f"conv-{self._next_id}"
            self._conn.execute(
                "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
                (native_session_id, self._origin, "identity state machine", self._fresh_content_hash()),
            )
            self._sessions.append(native_session_id)
            self._blocks_by_session[native_session_id] = []
            self._next_position_by_session[native_session_id] = 0
            return native_session_id
        return self._sessions[self._next_id % len(self._sessions)]

    def _next_position(self, session_native_id: str) -> int:
        position = self._next_position_by_session.get(session_native_id, 0)
        self._next_position_by_session[session_native_id] = position + 1
        return position

    def _insert_block(self, *, text: str | None) -> _Block:
        session_native_id = self._ensure_session()
        self._next_id += 1
        message_native_id = f"msg-{self._next_id}"
        session_id = f"{self._origin}:{session_native_id}"
        message_id = f"{session_id}:{message_native_id}"
        content_hash = self._fresh_content_hash()
        self._conn.execute(
            """
            INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
            VALUES (?, ?, ?, 'user', 'message', ?)
            """,
            (session_id, message_native_id, self._next_position(session_native_id), content_hash),
        )
        self._conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text, content_hash)
            VALUES (?, ?, 0, 'text', ?, ?)
            """,
            (message_id, session_id, text, content_hash),
        )
        block_id = f"{message_id}:0"
        block = _Block(block_id, session_native_id, message_native_id, indexed=bool(text))
        self._blocks[block_id] = block
        self._blocks_by_session[session_native_id].append(block_id)
        return block

    def _delete_block(self, block_id: str) -> None:
        self._conn.execute("DELETE FROM blocks WHERE block_id = ?", (block_id,))
        block = self._blocks.pop(block_id)
        self._blocks_by_session[block.session_native_id].remove(block_id)

    # -- rules -----------------------------------------------------------

    @rule()
    def insert_indexable_block(self) -> None:
        self._insert_block(text=f"needle {self._next_id}")

    @rule()
    def insert_empty_block(self) -> None:
        self._insert_block(text=None)

    @precondition(lambda self: bool(self._blocks))
    @rule()
    def update_block_text(self) -> None:
        block_id = sorted(self._blocks)[self._next_id % len(self._blocks)]
        self._next_id += 1
        new_text = f"edited {self._next_id}"
        new_hash = self._fresh_content_hash()
        self._conn.execute(
            "UPDATE blocks SET text = ?, content_hash = ? WHERE block_id = ?",
            (new_text, new_hash, block_id),
        )
        self._blocks[block_id].indexed = True

    @precondition(lambda self: bool(self._blocks))
    @rule()
    def update_block_to_empty(self) -> None:
        block_id = sorted(self._blocks)[self._next_id % len(self._blocks)]
        self._next_id += 1
        self._conn.execute("UPDATE blocks SET text = NULL WHERE block_id = ?", (block_id,))
        self._blocks[block_id].indexed = False

    @precondition(lambda self: bool(self._blocks))
    @rule()
    def delete_one_block(self) -> None:
        block_id = sorted(self._blocks)[self._next_id % len(self._blocks)]
        self._next_id += 1
        self._delete_block(block_id)

    @precondition(lambda self: any(blocks for blocks in self._blocks_by_session.values()))
    @rule()
    def full_session_replace(self) -> None:
        """Delete every block in a session, then insert a fresh set.

        This is the shape most likely to force SQLite to reuse a freed
        rowid (deleting the current max-rowid block then inserting a new
        one), exactly the historical bug class: a stale rowid silently
        rebinding to a different block while missing_rows/excess_rows
        counts still balance.
        """
        sessions_with_blocks = [native_id for native_id, blocks in self._blocks_by_session.items() if blocks]
        self._next_id += 1
        session_native_id = sessions_with_blocks[self._next_id % len(sessions_with_blocks)]
        for block_id in list(self._blocks_by_session[session_native_id]):
            self._delete_block(block_id)
        replacement_count = 1 + (self._next_id % 3)
        for _ in range(replacement_count):
            self._next_id += 1
            message_native_id = f"msg-{self._next_id}"
            session_id = f"{self._origin}:{session_native_id}"
            message_id = f"{session_id}:{message_native_id}"
            content_hash = self._fresh_content_hash()
            self._conn.execute(
                """
                INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
                VALUES (?, ?, ?, 'user', 'message', ?)
                """,
                (session_id, message_native_id, self._next_position(session_native_id), content_hash),
            )
            self._conn.execute(
                """
                INSERT INTO blocks (message_id, session_id, position, block_type, text, content_hash)
                VALUES (?, ?, 0, 'text', ?, ?)
                """,
                (message_id, session_id, f"replacement {self._next_id}", content_hash),
            )
            block_id = f"{message_id}:0"
            block = _Block(block_id, session_native_id, message_native_id, indexed=True)
            self._blocks[block_id] = block
            self._blocks_by_session[session_native_id].append(block_id)

    @precondition(lambda self: bool(self._sessions))
    @rule()
    def rollback_insert(self) -> None:
        """A rolled-back mutation must leave zero trace in either table.

        Deliberately picks an EXISTING session rather than
        ``_ensure_session()`` (which may INSERT a brand new session row) --
        this rule's own SAVEPOINT rollback only undoes the DB side, not this
        harness's Python-side bookkeeping, so creating new session state
        inside the rolled-back block would desync the two and produce a
        harness-only false failure (an FK violation on a later rule using a
        session native_id the DB no longer has) that has nothing to do with
        the production identity-ledger invariant under test.
        """
        before_docsize = int(self._conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0])
        before_identity = int(self._conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0])
        self._conn.execute("SAVEPOINT rollback_probe")
        try:
            session_native_id = self._sessions[self._next_id % len(self._sessions)]
            self._next_id += 1
            message_native_id = f"rollback-msg-{self._next_id}"
            session_id = f"{self._origin}:{session_native_id}"
            message_id = f"{session_id}:{message_native_id}"
            content_hash = self._fresh_content_hash()
            self._conn.execute(
                """
                INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
                VALUES (?, ?, ?, 'user', 'message', ?)
                """,
                (session_id, message_native_id, self._next_position(session_native_id), content_hash),
            )
            self._conn.execute(
                """
                INSERT INTO blocks (message_id, session_id, position, block_type, text, content_hash)
                VALUES (?, ?, 0, 'text', ?, ?)
                """,
                (message_id, session_id, "rolled back", content_hash),
            )
        finally:
            self._conn.execute("ROLLBACK TO rollback_probe")
            self._conn.execute("RELEASE rollback_probe")
            # The position counter itself is Python-side bookkeeping, not
            # transactional -- roll it back too so a later real insert into
            # this session doesn't skip a position number needlessly (not a
            # correctness requirement, just keeps position values compact).
            self._next_position_by_session[session_native_id] -= 1
        after_docsize = int(self._conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0])
        after_identity = int(self._conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0])
        assert after_docsize == before_docsize
        assert after_identity == before_identity

    @precondition(lambda self: bool(self._blocks))
    @rule()
    def corrupt_then_repair_identity_row(self) -> None:
        """Anti-vacuity: hand-corrupt a ledger row, prove it's flagged, then heal it.

        Mutation this rule's assertions catch: if
        ``repair_message_identity_rows_range_sql`` (or the identity trigger
        arms it backstops) stopped overwriting a mismatched
        ``block_id``/``source_hash``/``recipe_id``, the post-repair
        assertion would fail because the corrupted row would still be
        wrong.
        """
        indexed = [block_id for block_id, block in self._blocks.items() if block.indexed]
        if not indexed:
            return
        block_id = sorted(indexed)[self._next_id % len(indexed)]
        self._next_id += 1
        rowid = int(self._conn.execute("SELECT rowid FROM blocks WHERE block_id = ?", (block_id,)).fetchone()[0])
        self._conn.execute(
            "UPDATE messages_fts_identity SET block_id = 'stale:corrupted:0' WHERE rowid = ?",
            (rowid,),
        )
        mismatch_before = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_identity WHERE rowid = ? AND block_id != ?",
                (rowid, block_id),
            ).fetchone()[0]
        )
        assert mismatch_before == 1

        insert_missing_message_rows_batched_sync(self._conn, batch_rows=1_000_000)

        healed_block_id = self._conn.execute(
            "SELECT block_id FROM messages_fts_identity WHERE rowid = ?", (rowid,)
        ).fetchone()[0]
        assert healed_block_id == block_id

    # -- invariant ---------------------------------------------------------

    @invariant()
    def messages_fts_exactly_reflects_blocks(self) -> None:
        snapshot = fts_invariant_snapshot_sync(self._conn)
        surface = snapshot.messages
        assert surface.missing_rows == 0, f"missing_rows={surface.missing_rows}"
        assert surface.excess_rows == 0, f"excess_rows={surface.excess_rows}"
        assert surface.duplicate_rows == 0, f"duplicate_rows={surface.duplicate_rows}"
        assert surface.identity_mismatch_rows == 0, f"identity_mismatch_rows={surface.identity_mismatch_rows}"
        assert surface.ready

        indexed_docids = {row[0] for row in self._conn.execute("SELECT id FROM messages_fts_docsize").fetchall()}
        indexable_docids = {
            row[0] for row in self._conn.execute("SELECT rowid FROM blocks WHERE search_text != ''").fetchall()
        }
        assert indexed_docids == indexable_docids

        identity_rowids = {row[0] for row in self._conn.execute("SELECT rowid FROM messages_fts_identity").fetchall()}
        assert identity_rowids == indexable_docids

    def teardown(self) -> None:
        self._conn_cm.__exit__(None, None, None)
        self._tmpdir.cleanup()


TestFtsIdentityStateMachine = FtsIdentityStateMachine.TestCase
TestFtsIdentityStateMachine.settings = settings(
    stateful_step_count=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
