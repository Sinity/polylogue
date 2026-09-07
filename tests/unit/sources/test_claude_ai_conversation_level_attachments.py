"""Conversation-level attachment records require an unambiguous message owner.

A claude.ai conversation export may repeat a message's attachment record at
the conversation level, in ``attachments``/``files``, without repeating the
message id. Descriptor matching adopts the message identity only when exactly
one message owns that descriptor. Metadata-only records remain evidence, but
inline bytes without a unique owner are dropped before persistence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from polylogue.sources.parsers.claude.ai_parser import parse_ai
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.live_ingest import write_index_session

_TIMESTAMP = "2026-01-01T00:00:00Z"


def _conversation(*, message_files: list[dict[str, Any]], conversation_files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "uuid": "conv-1",
        "name": "conversation",
        "created_at": _TIMESTAMP,
        "chat_messages": [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "here it is",
                "created_at": _TIMESTAMP,
                "attachments": message_files,
            }
        ],
        "files": conversation_files,
    }


def _write(root: Path, payload: dict[str, Any]) -> tuple[int, int]:
    """Write the parsed conversation and return (attachment rows, ref rows)."""
    with ArchiveStore(root) as archive:
        write_index_session(archive, parse_ai(payload, "fallback"))
    conn = sqlite3.connect(root / "index.db")
    try:
        return (
            int(conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]),
            int(conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0]),
        )
    finally:
        conn.close()


def _unreferenced(root: Path) -> int:
    conn = sqlite3.connect(root / "index.db")
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM attachments a WHERE NOT EXISTS"
                " (SELECT 1 FROM attachment_refs r WHERE r.attachment_id = a.attachment_id)"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def test_conversation_level_repeat_of_an_idless_file_keeps_one_referenced_attachment(tmp_path: Path) -> None:
    """Anti-vacuity: descriptor matching preserves the conversation-level bytes."""
    root = tmp_path / "archive"
    descriptor = {"file_name": "report.txt", "file_type": "text/plain"}
    attachments, refs = _write(
        root,
        _conversation(
            message_files=[dict(descriptor)],
            conversation_files=[{**descriptor, "extracted_content": "the report body"}],
        ),
    )

    assert (attachments, refs) == (1, 1)
    assert _unreferenced(root) == 0

    conn = sqlite3.connect(root / "index.db")
    try:
        row = conn.execute(
            "SELECT a.display_name, a.acquisition_status, a.byte_count, r.message_id FROM attachments a"
            " JOIN attachment_refs r ON r.attachment_id = a.attachment_id"
        ).fetchone()
    finally:
        conn.close()
    # The bytes the conversation-level record carried and the owner only the
    # message-level record knew both survive on the single stored attachment.
    assert row[0] == "report.txt"
    assert row[1] == "acquired"
    assert row[2] == len("the report body")
    assert str(row[3]).endswith(":n:msg-1")


def test_conversation_level_file_with_no_message_record_stays_unreferenced(tmp_path: Path) -> None:
    """Anti-vacuity: make ``_conversation_level_identity`` adopt any message's
    identity and this gains a ref the export never asserted."""
    root = tmp_path / "archive"
    attachments, refs = _write(
        root,
        _conversation(
            message_files=[],
            conversation_files=[{"file_name": "loose.txt", "file_type": "text/plain"}],
        ),
    )

    # No message record claims this file, so no owner may be guessed: the row
    # is retained as evidence with no reference rather than attributed.
    assert (attachments, refs) == (1, 0)
    assert _unreferenced(root) == 1


def test_two_message_records_sharing_a_descriptor_leave_the_repeat_unattributed(tmp_path: Path) -> None:
    """Anti-vacuity: drop the uniqueness guard on ``owned_by_descriptor`` and the
    conversation-level record silently attaches to whichever message came
    last."""
    root = tmp_path / "archive"
    descriptor = {"file_name": "shared.txt", "file_type": "text/plain"}
    payload = _conversation(
        message_files=[dict(descriptor)],
        conversation_files=[dict(descriptor)],
    )
    payload["chat_messages"].append(
        {
            "uuid": "msg-2",
            "sender": "human",
            "text": "again",
            "created_at": "2026-01-01T00:00:01Z",
            "attachments": [dict(descriptor)],
        }
    )

    attachments, refs = _write(root, payload)

    assert (attachments, refs) == (3, 2)
    assert _unreferenced(root) == 1


def test_unowned_inline_bytes_are_dropped_without_a_message_owner(tmp_path: Path) -> None:
    root = tmp_path / "archive"

    attachments, refs = _write(
        root,
        _conversation(
            message_files=[],
            conversation_files=[{"file_name": "loose.txt", "file_type": "text/plain", "extracted_content": "body"}],
        ),
    )

    assert (attachments, refs) == (0, 0)
    assert _unreferenced(root) == 0


def test_ambiguous_inline_bytes_are_dropped_without_a_unique_owner(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    descriptor = {"file_name": "shared.txt", "file_type": "text/plain"}
    payload = _conversation(
        message_files=[dict(descriptor)],
        conversation_files=[{**descriptor, "extracted_content": "body"}],
    )
    payload["chat_messages"].append(
        {
            "uuid": "msg-2",
            "sender": "human",
            "text": "again",
            "created_at": _TIMESTAMP,
            "attachments": [dict(descriptor)],
        }
    )

    attachments, refs = _write(root, payload)

    assert (attachments, refs) == (2, 2)
    assert _unreferenced(root) == 0


def test_conversation_level_repeat_carrying_its_own_id_is_unaffected(tmp_path: Path) -> None:
    """Anti-vacuity: make ``_conversation_level_identity`` ignore
    ``meta_carries_provider_attachment_id`` and a provider-identified record
    would be rekeyed by descriptor instead of by its own id."""
    root = tmp_path / "archive"
    descriptor = {"file_uuid": "file-1", "file_name": "report.txt", "file_type": "text/plain"}
    attachments, refs = _write(
        root,
        _conversation(
            message_files=[dict(descriptor)],
            conversation_files=[{**descriptor, "extracted_content": "the report body"}],
        ),
    )

    assert (attachments, refs) == (1, 1)
    assert _unreferenced(root) == 0
