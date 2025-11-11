from __future__ import annotations

from pathlib import Path

from polylogue.importers.normalizer import build_message_record


def test_build_message_record_assigns_unique_ids_and_metadata():
    chunk = {"role": "user", "text": "Hello", "tokenCount": 4}
    links = [("attachment.txt", Path("attachments/attachment.txt"))]
    seen: set[str] = set()

    record = build_message_record(
        provider="test",
        conversation_id="conv-1",
        chunk_index=0,
        chunk=dict(chunk),
        raw_metadata={"id": "msg-1", "parent_id": None},
        attachments=links,
        tool_calls=[{"type": "tool_use", "name": "echo"}],
        seen_ids=seen,
        fallback_prefix="conv-1",
    )

    assert record.message_id == "msg-1"
    assert record.metadata["attachments"][0]["name"] == "attachment.txt"
    assert "tool_calls" in record.metadata

    duplicate = build_message_record(
        provider="test",
        conversation_id="conv-1",
        chunk_index=1,
        chunk=dict(chunk),
        raw_metadata={"id": "msg-1", "parent_id": record.message_id},
        attachments=[],
        seen_ids=seen,
        fallback_prefix="conv-1",
    )

    assert duplicate.message_id != "msg-1"
    assert duplicate.message_id.startswith("msg-1-dup")
