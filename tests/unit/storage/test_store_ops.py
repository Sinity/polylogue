"""Focused roundtrip and validation contracts for storage record helpers."""

from __future__ import annotations

import importlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from pydantic import ValidationError

from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import (
    MAX_ATTACHMENT_SIZE,
    AttachmentRecord,
    ConversationRecord,
    _json_or_none,
)
from tests.infra.storage_records import (
    _make_ref_id,
    _prune_attachment_refs,
    make_attachment,
    make_conversation,
    make_message,
    store_records,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)


def _conversation_row(conn, conversation_id: str):
    return conn.execute(
        "SELECT * FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()


def _message_count(conn, conversation_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()[0]


def _attachment_row(conn, attachment_id: str):
    return conn.execute(
        "SELECT * FROM attachments WHERE attachment_id = ?",
        (attachment_id,),
    ).fetchone()


def test_store_records_roundtrip_contract(test_conn) -> None:
    """store_records() must insert, skip, update, and handle sparse payloads coherently."""
    initial = make_conversation("conv-create", content_hash="hash-create")
    created = store_records(
        conversation=initial,
        messages=[make_message("msg-create", "conv-create", text="Hello")],
        attachments=[],
        conn=test_conn,
    )
    assert created == {
        "conversations": 1,
        "messages": 1,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }
    assert _conversation_row(test_conn, "conv-create")["title"] == "Test Conversation"
    assert _message_count(test_conn, "conv-create") == 1

    duplicate = store_records(
        conversation=initial,
        messages=[],
        attachments=[],
        conn=test_conn,
    )
    assert duplicate["conversations"] == 0
    assert duplicate["skipped_conversations"] == 1

    updated = store_records(
        conversation=make_conversation("conv-create", title="Updated Title", content_hash="hash-updated"),
        messages=[],
        attachments=[],
        conn=test_conn,
    )
    assert updated["conversations"] == 1
    assert _conversation_row(test_conn, "conv-create")["title"] == "Updated Title"
    assert _conversation_row(test_conn, "conv-create")["content_hash"] == "hash-updated"

    multi = store_records(
        conversation=make_conversation("conv-multi", title="Multi Message"),
        messages=[
            make_message(f"msg-multi-{idx}", "conv-multi", role="user" if idx % 2 == 0 else "assistant", text=f"Message {idx}")
            for idx in range(5)
        ],
        attachments=[],
        conn=test_conn,
    )
    assert multi["messages"] == 5
    assert _message_count(test_conn, "conv-multi") == 5

    sparse = store_records(
        conversation=make_conversation("conv-empty", title="Empty Conversation"),
        messages=[],
        attachments=[
            make_attachment(
                "att-empty",
                "conv-empty",
                message_id=None,
                mime_type="application/pdf",
                size_bytes=5000,
            )
        ],
        conn=test_conn,
    )
    assert sparse["conversations"] == 1
    assert sparse["messages"] == 0
    assert sparse["attachments"] == 1
    assert _attachment_row(test_conn, "att-empty")["ref_count"] == 1


def test_prune_attachment_refs_contract(test_conn) -> None:
    """Pruning refs must keep requested refs, recalculate counts, and delete zero-ref attachments."""
    conv = make_conversation("conv-prune", title="Prune Test")
    msg1 = make_message("msg-prune-1", "conv-prune", provider_message_id="ext-1", text="First")
    msg2 = make_message("msg-prune-2", "conv-prune", provider_message_id="ext-2", text="Second")
    att1 = make_attachment("att-prune-1", "conv-prune", "msg-prune-1", mime_type="image/png")
    att2 = make_attachment("att-prune-2", "conv-prune", "msg-prune-2", mime_type="image/jpeg", size_bytes=2048)
    shared_att_1 = make_attachment("att-shared", "conv-prune", "msg-prune-1", mime_type="image/png")
    shared_att_2 = make_attachment("att-shared", "conv-prune", "msg-prune-2", mime_type="image/png")
    store_records(
        conversation=conv,
        messages=[msg1, msg2],
        attachments=[att1, att2, shared_att_1, shared_att_2],
        conn=test_conn,
    )

    keep_ref = _make_ref_id("att-prune-1", "conv-prune", "msg-prune-1")
    keep_shared = _make_ref_id("att-shared", "conv-prune", "msg-prune-1")
    _prune_attachment_refs(test_conn, "conv-prune", {keep_ref, keep_shared})

    remaining_refs = test_conn.execute(
        "SELECT ref_id FROM attachment_refs WHERE conversation_id = ? ORDER BY ref_id",
        ("conv-prune",),
    ).fetchall()
    assert [row["ref_id"] for row in remaining_refs] == sorted([keep_ref, keep_shared])
    assert _attachment_row(test_conn, "att-prune-1")["ref_count"] == 1
    assert _attachment_row(test_conn, "att-shared")["ref_count"] == 1
    assert _attachment_row(test_conn, "att-prune-2") is None


def test_upsert_optional_and_attachment_contracts(test_conn) -> None:
    """Optional-field upserts and attachment metadata updates must round-trip cleanly."""
    conversation = ConversationRecord(
        conversation_id="conv-optional",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title=None,
        created_at=None,
        updated_at=None,
        content_hash="hash1",
        provider_meta=None,
    )
    assert upsert_conversation(test_conn, conversation) is True
    conv_row = _conversation_row(test_conn, "conv-optional")
    assert conv_row["title"] is None
    assert conv_row["created_at"] is None
    assert conv_row["provider_meta"] is None

    message = make_message(
        "msg-optional",
        "conv-optional",
        role=None,
        text=None,
        timestamp=None,
        provider_message_id=None,
        provider_meta=None,
    )
    assert upsert_message(test_conn, message) is True
    msg_row = test_conn.execute(
        "SELECT * FROM messages WHERE message_id = ?",
        ("msg-optional",),
    ).fetchone()
    assert msg_row["role"] is None
    assert msg_row["text"] is None
    assert msg_row["provider_message_id"] is None

    msg2 = make_message("msg-attachment-2", "conv-optional", provider_message_id="ext-msg-2", text="Second")
    assert upsert_message(test_conn, msg2) is True
    first = make_attachment("att-meta", "conv-optional", "msg-optional", mime_type="image/png")
    second = make_attachment(
        "att-meta",
        "conv-optional",
        "msg-attachment-2",
        mime_type="image/jpeg",
        size_bytes=2048,
        path="/new/path.jpg",
    )
    assert upsert_attachment(test_conn, first) is True
    assert upsert_attachment(test_conn, first) is False
    assert upsert_attachment(test_conn, second) is True
    att_row = _attachment_row(test_conn, "att-meta")
    assert att_row["mime_type"] == "image/jpeg"
    assert att_row["size_bytes"] == 2048
    assert att_row["path"] == "/new/path.jpg"
    assert att_row["ref_count"] == 2


def test_json_or_none_contract() -> None:
    """JSON serialization helper must preserve mappings and None."""
    import json

    payloads = [
        ({"key": "value"}, {"key": "value"}),
        ({"nested": {"key": "value"}, "list": [1, 2, 3]}, {"nested": {"key": "value"}, "list": [1, 2, 3]}),
        (None, None),
    ]
    for input_val, expected in payloads:
        result = _json_or_none(input_val)
        if expected is None:
            assert result is None
        else:
            assert json.loads(result) == expected


def test_make_ref_id_contract() -> None:
    """Attachment ref IDs must be deterministic and sensitive to attachment, conversation, and message."""
    same_1 = _make_ref_id("att1", "conv1", "msg1")
    same_2 = _make_ref_id("att1", "conv1", "msg1")
    different_attachment = _make_ref_id("att2", "conv1", "msg1")
    different_conversation = _make_ref_id("att1", "conv2", "msg1")
    none_message_1 = _make_ref_id("att1", "conv1", None)
    none_message_2 = _make_ref_id("att1", "conv1", None)

    assert same_1 == same_2
    assert same_1 != different_attachment
    assert same_1 != different_conversation
    assert none_message_1 == none_message_2
    assert none_message_1 != same_1
    assert same_1.startswith("ref-")
    assert len(same_1) == len("ref-") + 16


@pytest.mark.slow
def test_write_lock_prevents_concurrent_writes(test_db) -> None:
    """Threaded store_records() calls must complete without corrupting conversation or message counts."""
    results = []
    errors = []

    def write_conversation(conv_id: int) -> None:
        try:
            conv = make_conversation(f"conv{conv_id}", title=f"Conversation {conv_id}")
            messages = [make_message(f"msg{conv_id}-{i}", f"conv{conv_id}", text=f"Message {i}") for i in range(3)]
            with open_connection(test_db) as conn:
                results.append(store_records(conversation=conv, messages=messages, attachments=[], conn=conn))
        except Exception as exc:  # pragma: no cover - failure path assertion target
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write_conversation, idx) for idx in range(10)]
        for future in as_completed(futures):
            future.result()

    assert errors == []
    assert len(results) == 10
    with open_connection(test_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 10
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 30


def test_store_records_without_connection_creates_own(test_db, tmp_path, monkeypatch) -> None:
    """store_records() must honor the default DB path when no connection is supplied."""
    import polylogue.paths
    import polylogue.storage.backends.connection as connection_module
    from polylogue.storage.backends.connection import _clear_connection_cache

    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    _clear_connection_cache()
    importlib.reload(polylogue.paths)
    importlib.reload(connection_module)

    default_path = connection_module.default_db_path()
    default_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(test_db), str(default_path))

    counts = store_records(
        conversation=make_conversation("conv-default", title="No Conn Test"),
        messages=[],
        attachments=[],
    )
    assert counts["conversations"] == 1

    with open_connection(default_path) as conn:
        assert _conversation_row(conn, "conv-default") is not None


@pytest.mark.slow
def test_concurrent_upsert_same_attachment_ref_count_correct(test_db) -> None:
    """Concurrent upserts of the same attachment must keep ref_count equal to actual refs."""
    shared_attachment_id = "shared-attachment-race-test"

    def create_conversation(index: int) -> None:
        conv = make_conversation(
            f"race-conv-{index}",
            title=f"Race Test {index}",
            created_at=None,
            updated_at=None,
            content_hash=f"hash-{index}",
        )
        msg = make_message(
            f"race-msg-{index}",
            f"race-conv-{index}",
            text="test",
            timestamp=None,
            provider_meta=None,
        )
        attachment = make_attachment(
            shared_attachment_id,
            f"race-conv-{index}",
            f"race-msg-{index}",
            mime_type="text/plain",
            size_bytes=100,
            provider_meta=None,
        )
        with open_connection(test_db) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[attachment], conn=conn)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(create_conversation, range(10)))

    with open_connection(test_db) as conn:
        stored_ref_count = conn.execute(
            "SELECT ref_count FROM attachments WHERE attachment_id = ?",
            (shared_attachment_id,),
        ).fetchone()[0]
        actual_refs = conn.execute(
            "SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?",
            (shared_attachment_id,),
        ).fetchone()[0]

    assert stored_ref_count == 10
    assert actual_refs == 10
    assert stored_ref_count == actual_refs


@pytest.mark.parametrize(
    ("size_bytes", "valid"),
    [
        (0, True),
        (MAX_ATTACHMENT_SIZE, True),
        (None, True),
        (-100, False),
        (MAX_ATTACHMENT_SIZE + 1, False),
    ],
    ids=["zero", "max", "unknown", "negative", "over-max"],
)
def test_attachment_size_bytes_contract(size_bytes, valid) -> None:
    """Attachment size validation must accept supported bounds and reject invalid sizes."""
    if valid:
        record = AttachmentRecord(
            attachment_id="test",
            conversation_id="conv1",
            message_id="msg1",
            mime_type="text/plain",
            size_bytes=size_bytes,
            provider_meta=None,
        )
        assert record.size_bytes == size_bytes
    else:
        with pytest.raises(ValidationError):
            AttachmentRecord(
                attachment_id="test",
                conversation_id="conv1",
                message_id="msg1",
                mime_type="text/plain",
                size_bytes=size_bytes,
                provider_meta=None,
            )


@pytest.mark.parametrize("name", ["claude", "claude-code", "Provider123"])
def test_provider_name_accepts_valid(name) -> None:
    """Representative provider-name formats should validate."""
    record = ConversationRecord(
        conversation_id="test",
        provider_name=name,
        provider_conversation_id="ext1",
        title="Test",
        content_hash="hash123",
    )
    assert record.provider_name == name
