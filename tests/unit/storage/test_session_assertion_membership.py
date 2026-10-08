"""Session claims use canonical composed-message membership before page selection."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.errors import DatabaseError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, list_assertion_claims, upsert_assertion
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.storage_records import SessionBuilder


def test_session_claim_membership_respects_inherited_cut_and_active_page(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with ArchiveStore(root):
        pass
    index = root / "index.db"
    parent = SessionBuilder(index, "parent")
    parent.add_message(message_id="prefix", text="inherited prefix")
    parent.add_message(message_id="later", text="outside inherited cut")
    parent.save()
    child = SessionBuilder(index, "child")
    child.add_message(message_id="own", text="child tail")
    child.save()
    foreign = SessionBuilder(index, "foreign")
    foreign.add_message(message_id="foreign", text="foreign session")
    foreign.save()
    with (
        write_lease("test.session-claims", archive_root=root),
        ArchiveStore.open_existing(root, read_only=False) as archive,
    ):
        conn = archive._conn
        message_ids = {
            str(row["session_id"]): [
                str(item[0])
                for item in conn.execute(
                    "SELECT message_id FROM messages WHERE session_id=? ORDER BY position", (row["session_id"],)
                )
            ]
            for row in conn.execute("SELECT session_id FROM sessions")
        }
        parent_id, child_id = parent.native_session_id(), child.native_session_id()
        conn.execute(
            "INSERT INTO session_links(src_session_id,dst_origin,dst_native_id,link_type,resolved_dst_session_id,"
            "branch_point_message_id,inheritance,status,confidence,evidence_json,observed_at_ms) "
            "VALUES (?, 'codex-session', ?, 'fork', ?, ?, 'prefix-sharing', NULL, 1.0, '[]', 0)",
            (child_id, parent.conv.native_id, parent_id, message_ids[parent_id][0]),
        )
        targets = {
            "session": f"session:{child_id}",
            "own": f"message:{message_ids[child_id][0]}",
            "prefix": f"message:{message_ids[parent_id][0]}",
            "post-cut": f"message:{message_ids[parent_id][1]}",
            "foreign": f"message:{message_ids[foreign.native_session_id()][0]}",
            "candidate": f"message:{message_ids[child_id][0]}",
        }
        conn.commit()
        with closing(open_connection(root / "user.db")) as user_conn:
            user_conn.row_factory = sqlite3.Row
            for time, (name, target) in enumerate(targets.items()):
                upsert_assertion(
                    user_conn,
                    assertion_id=name,
                    target_ref=target,
                    kind=AssertionKind.LESSON,
                    body_text=name,
                    status="candidate" if name == "candidate" else "active",
                    author_kind="user",
                    author_ref="user:synthetic",
                    now_ms=time,
                    # Foreign scope metadata must not grant membership.
                    scope_ref=f"session:{child_id}",
                )
            user_conn.commit()
        claims = list_assertion_claims(conn, schema="user_tier", session_id=child_id, statuses=("active",), limit=2)
        assert [claim.assertion_id for claim in claims] == ["prefix", "own"]
        all_claims = list_assertion_claims(conn, schema="user_tier", session_id=child_id, statuses=("active",))
        assert [claim.assertion_id for claim in all_claims] == ["prefix", "own", "session"]
        conn.execute("UPDATE session_links SET branch_point_message_id='missing' WHERE src_session_id=?", (child_id,))
        conn.commit()
        with pytest.raises(DatabaseError, match="incomplete lineage"):
            list_assertion_claims(conn, schema="user_tier", session_id=child_id, statuses=("active",), limit=2)

    from polylogue.browser_capture.server import mission_control_archive_facts

    assert mission_control_archive_facts(root, child_id) is None
