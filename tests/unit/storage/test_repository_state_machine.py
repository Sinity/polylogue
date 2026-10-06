"""State-transition tests for archive lifecycle operations.

Uses Hypothesis stateful testing to exercise operation sequences that
unit tests miss: save, re-save, delete, and query must remain consistent
after arbitrary interleaving.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from hypothesis import settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.storage_records import (
    make_message,
    make_session,
    store_records,
)

_ARCHIVE_ROOT: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _shared_archive(tmp_path_factory: pytest.TempPathFactory) -> Generator[None]:
    """Bootstrap one archive for every state-machine run.

    The production writer seals its references against a named archive Index
    and its measured creator, so an unnamed in-memory handle cannot be
    written. Bootstrapping an archive per Hypothesis example costs seconds;
    instead each run owns a disjoint session namespace in one archive.
    """
    global _ARCHIVE_ROOT
    _ARCHIVE_ROOT = bootstrap_archive_root(tmp_path_factory.mktemp("archive-lifecycle") / "archive")
    try:
        yield
    finally:
        _ARCHIVE_ROOT = None


class ArchiveLifecycleStateMachine(RuleBasedStateMachine):
    """Model-based test for session save/delete/query consistency.

    Invariants checked after every operation, within this run's namespace:
    - COUNT(*) matches the model's expected set
    - No dangling messages (every message.session_id exists in sessions)
    - List result set matches expected IDs
    """

    def __init__(self) -> None:
        super().__init__()

        from polylogue.storage.io_phase_metrics import connect_measured

        assert _ARCHIVE_ROOT is not None
        self._conn = connect_measured(_ARCHIVE_ROOT / "index.db")
        self._conn.row_factory = sqlite3.Row
        self._namespace = f"sm-{uuid.uuid4().hex[:12]}"
        self._pattern = f"chatgpt-export:{self._namespace}-%"

        self._expected_ids: set[str] = set()
        self._expected_messages: dict[str, int] = {}
        self._next_id = 0

    saved_sessions = Bundle("saved_sessions")

    @rule(target=saved_sessions)
    def save_new_session(self) -> str:
        # ``cid`` is the canonical generated ``origin:native_id`` session id so
        # it matches the stored row; the native part is the provider id.
        native = f"{self._namespace}-conv-{self._next_id}"
        cid = f"chatgpt-export:{native}"
        self._next_id += 1

        conv = make_session(session_id=cid, source_name="chatgpt", provider_session_id=native, title=f"Conv {cid}")
        msgs = [
            make_message(message_id=f"{cid}-m1", session_id=cid, role="user", text="Hello"),
            make_message(message_id=f"{cid}-m2", session_id=cid, role="assistant", text="Hi"),
        ]

        store_records(session=conv, messages=msgs, attachments=[], conn=self._conn)

        self._expected_ids.add(cid)
        self._expected_messages[cid] = 2

        self._check_invariants()
        return cid

    @rule(cid=saved_sessions)
    def re_save_same_session(self, cid: str) -> None:
        """Re-import should be idempotent if present, or resurrect if deleted."""
        native = cid.removeprefix("chatgpt-export:")
        conv = make_session(session_id=cid, source_name="chatgpt", provider_session_id=native, title=f"Conv {cid}")
        msgs = [
            make_message(message_id=f"{cid}-m1", session_id=cid, role="user", text="Hello"),
            make_message(message_id=f"{cid}-m2", session_id=cid, role="assistant", text="Hi"),
        ]

        store_records(session=conv, messages=msgs, attachments=[], conn=self._conn)

        self._expected_ids.add(cid)
        self._expected_messages[cid] = 2

        self._check_invariants()

    @rule(cid=saved_sessions)
    def delete_session(self, cid: str) -> None:
        if cid not in self._expected_ids:
            return

        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (cid,))
        self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (cid,))
        self._conn.commit()

        self._expected_ids.discard(cid)
        self._expected_messages.pop(cid, None)

        self._check_invariants()

    def _check_invariants(self) -> None:
        actual_count = self._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id LIKE ?", (self._pattern,)
        ).fetchone()[0]
        assert actual_count == len(self._expected_ids), (
            f"Session count: expected {len(self._expected_ids)}, got {actual_count}"
        )

        actual_ids = {
            r["session_id"]
            for r in self._conn.execute(
                "SELECT session_id FROM sessions WHERE session_id LIKE ?", (self._pattern,)
            ).fetchall()
        }
        assert actual_ids == self._expected_ids, f"Session IDs: expected {self._expected_ids}, got {actual_ids}"

        orphan_count = self._conn.execute(
            "SELECT COUNT(*) FROM messages m "
            "WHERE NOT EXISTS (SELECT 1 FROM sessions c WHERE c.session_id = m.session_id)"
        ).fetchone()[0]
        assert orphan_count == 0, f"Found {orphan_count} orphaned messages"

        for cid, expected_msg_count in self._expected_messages.items():
            actual_msg_count = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (cid,)
            ).fetchone()[0]
            assert actual_msg_count == expected_msg_count, (
                f"Messages for {cid}: expected {expected_msg_count}, got {actual_msg_count}"
            )

    def teardown(self) -> None:
        from tests.infra.index_writer import close_fixture_index_connection

        close_fixture_index_connection(self._conn)


TestArchiveLifecycle = ArchiveLifecycleStateMachine.TestCase
TestArchiveLifecycle.settings = settings(
    max_examples=50,
    stateful_step_count=20,
    deadline=None,
)
