"""Focused tests for shared storage-record test infrastructure."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pytest

from tests.infra.storage_records import make_conversation, make_message, store_records


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

    monkeypatch.setattr(storage_helpers, "_WRITE_LOCK", lock)
    monkeypatch.setattr(storage_helpers, "connection_context", fake_connection_context)
    monkeypatch.setattr(storage_helpers, "upsert_conversation", lambda *_: True)
    monkeypatch.setattr(storage_helpers, "upsert_message", lambda *_: True)
    monkeypatch.setattr(storage_helpers, "upsert_attachment", lambda *_: True)
    monkeypatch.setattr(storage_helpers, "_prune_attachment_refs", lambda *_: None)

    result = storage_helpers.store_records(
        conversation=make_conversation("test:1", title="Test", content_hash="abc123"),
        messages=[make_message("test:1:msg1", "test:1", text="Hello")],
        attachments=[],
        conn=conn,
    )

    assert result["conversations"] == 1
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
            conversation=make_conversation(f"test:{idx}", title=f"Test {idx}", content_hash=f"hash{idx}"),
            messages=[make_message(f"test:{idx}:msg1", f"test:{idx}", text=f"Hello {idx}")],
            attachments=[],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(store_one, i) for i in range(10)]
        results = [future.result(timeout=30) for future in futures]

    assert len(results) == 10
    assert all(result["conversations"] == 1 for result in results)


def test_make_message_rejects_non_json_provider_meta() -> None:
    with pytest.raises(TypeError, match="message provider_meta"):
        make_message(provider_meta={"path": Path("not-json")})


def test_make_message_rejects_non_json_content_block_input() -> None:
    with pytest.raises(TypeError, match="content block tool input"):
        make_message(
            content_blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "shell",
                    "tool_input": {"path": Path("not-json")},
                }
            ]
        )


def test_make_message_preserves_json_content_block_input() -> None:
    message = make_message(
        content_blocks=[
            {
                "type": "tool_use",
                "tool_name": "shell",
                "tool_input": {"command": "pytest -q"},
            }
        ]
    )

    assert len(message.content_blocks) == 1
    assert message.content_blocks[0].tool_input == '{"command":"pytest -q"}'
