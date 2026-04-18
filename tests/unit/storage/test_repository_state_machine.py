"""State-transition tests for archive lifecycle operations.

Uses Hypothesis stateful testing to exercise operation sequences that
unit tests miss: save, re-save, delete, and query must remain consistent
after arbitrary interleaving.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

from hypothesis import settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

from tests.infra.storage_records import (
    make_conversation,
    make_message,
    store_records,
)


class ArchiveLifecycleStateMachine(RuleBasedStateMachine):
    """Model-based test for conversation save/delete/query consistency.

    Invariants checked after every operation:
    - COUNT(*) matches the model's expected set
    - No dangling messages (every message.conversation_id exists in conversations)
    - List result set matches expected IDs
    """

    def __init__(self) -> None:
        super().__init__()

        from polylogue.storage.backends.schema import _ensure_schema

        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "state_machine.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        _ensure_schema(self._conn)

        self._expected_ids: set[str] = set()
        self._expected_messages: dict[str, int] = {}
        self._next_id = 0

    saved_conversations = Bundle("saved_conversations")

    @rule(target=saved_conversations)
    def save_new_conversation(self) -> str:
        cid = f"sm-conv-{self._next_id}"
        self._next_id += 1

        conv = make_conversation(conversation_id=cid, provider_name="chatgpt", title=f"Conv {cid}")
        msgs = [
            make_message(message_id=f"{cid}-m1", conversation_id=cid, role="user", text="Hello"),
            make_message(message_id=f"{cid}-m2", conversation_id=cid, role="assistant", text="Hi"),
        ]

        store_records(conversation=conv, messages=msgs, attachments=[], conn=self._conn)

        self._expected_ids.add(cid)
        self._expected_messages[cid] = 2

        self._check_invariants()
        return cid

    @rule(cid=saved_conversations)
    def re_save_same_conversation(self, cid: str) -> None:
        """Re-import should be idempotent if present, or resurrect if deleted."""
        conv = make_conversation(conversation_id=cid, provider_name="chatgpt", title=f"Conv {cid}")
        msgs = [
            make_message(message_id=f"{cid}-m1", conversation_id=cid, role="user", text="Hello"),
            make_message(message_id=f"{cid}-m2", conversation_id=cid, role="assistant", text="Hi"),
        ]

        store_records(conversation=conv, messages=msgs, attachments=[], conn=self._conn)

        self._expected_ids.add(cid)
        self._expected_messages[cid] = 2

        self._check_invariants()

    @rule(cid=saved_conversations)
    def delete_conversation(self, cid: str) -> None:
        if cid not in self._expected_ids:
            return

        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
        self._conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (cid,))
        self._conn.commit()

        self._expected_ids.discard(cid)
        self._expected_messages.pop(cid, None)

        self._check_invariants()

    def _check_invariants(self) -> None:
        actual_count = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        assert actual_count == len(self._expected_ids), (
            f"Conversation count: expected {len(self._expected_ids)}, got {actual_count}"
        )

        actual_ids = {
            r["conversation_id"] for r in self._conn.execute("SELECT conversation_id FROM conversations").fetchall()
        }
        assert actual_ids == self._expected_ids, f"Conversation IDs: expected {self._expected_ids}, got {actual_ids}"

        orphan_count = self._conn.execute(
            "SELECT COUNT(*) FROM messages m "
            "WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)"
        ).fetchone()[0]
        assert orphan_count == 0, f"Found {orphan_count} orphaned messages"

        for cid, expected_msg_count in self._expected_messages.items():
            actual_msg_count = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (cid,)
            ).fetchone()[0]
            assert actual_msg_count == expected_msg_count, (
                f"Messages for {cid}: expected {expected_msg_count}, got {actual_msg_count}"
            )

    def teardown(self) -> None:
        self._conn.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


TestArchiveLifecycle = ArchiveLifecycleStateMachine.TestCase
TestArchiveLifecycle.settings = settings(
    max_examples=50,
    stateful_step_count=20,
    deadline=None,
)
