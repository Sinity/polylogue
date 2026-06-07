"""Integration test: search result determinism under concurrent writes (#1734).

Asserts that FTS5 queries during live ingest return stable results for
already-ingested sessions — monotonic growth, no regression, no
OperationalError or corrupt index errors.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import open_connection


def _content_hash(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _insert_session_with_message(conn: sqlite3.Connection, native_id: str, position: int, text: str) -> str:
    session_id = f"codex-session:{native_id}"
    message_native_id = f"m{position}"
    message_id = f"{session_id}:{message_native_id}"
    conn.execute(
        "INSERT INTO sessions (native_id, origin, title, content_hash, updated_at_ms) VALUES (?, ?, ?, ?, ?)",
        (native_id, "codex-session", native_id, _content_hash(f"session:{native_id}"), position),
    )
    conn.execute(
        "INSERT INTO messages (session_id, native_id, position, role, content_hash) VALUES (?, ?, ?, ?, ?)",
        (session_id, message_native_id, position, "user", _content_hash(f"message:{message_id}")),
    )
    conn.execute(
        "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, ?, ?, ?)",
        (message_id, session_id, 0, "text", text),
    )
    return message_id


@pytest.mark.slow
@pytest.mark.integration
def test_search_results_monotonic_under_concurrent_writes(workspace_env: dict[str, Path], tmp_path: Path) -> None:
    """Concurrent writer + reader: already-indexed results must be stable."""

    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root):
        pass
    db_path = archive_root / "index.db"

    # Seed initial data
    conn = open_connection(db_path)
    first_message_id = _insert_session_with_message(conn, "c1", 1, "hello world")
    _insert_session_with_message(conn, "c2", 2, "another test message")
    conn.commit()
    conn.close()

    results_lock = threading.Lock()
    search_results: list[list[str]] = []
    errors: list[Exception] = []
    stop_flag = threading.Event()

    def reader() -> None:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as rconn:
            while not stop_flag.is_set():
                try:
                    rows = rconn.execute(
                        "SELECT message_id FROM messages_fts WHERE messages_fts MATCH ? ORDER BY rank",
                        ("hello",),
                    ).fetchall()
                    msg_ids = [r[0] for r in rows]
                    with results_lock:
                        search_results.append(msg_ids)
                except Exception as exc:
                    with results_lock:
                        errors.append(exc)
                time.sleep(0.01)

    def writer() -> None:
        wconn = open_connection(db_path)
        try:
            for i in range(50):
                cid = f"c{i + 3}"
                _insert_session_with_message(wconn, cid, i + 3, f"hello concurrent message {i}")
                wconn.commit()
                time.sleep(0.005)
        finally:
            wconn.close()

    reader_thread = threading.Thread(target=reader, daemon=True)
    writer_thread = threading.Thread(target=writer, daemon=True)

    reader_thread.start()
    writer_thread.start()
    writer_thread.join()
    stop_flag.set()
    reader_thread.join()

    assert len(errors) == 0, f"Reader encountered errors: {errors}"
    assert len(search_results) > 0, "Reader never produced any search results"

    for i, result_set in enumerate(search_results):
        assert first_message_id in result_set, (
            f"Result set {i} lost pre-existing message {first_message_id!r}: {result_set}"
        )

    for i in range(1, len(search_results)):
        prev = set(search_results[i - 1])
        curr = set(search_results[i])
        missing = prev - curr
        assert not missing, f"Result set {i} lost messages from set {i - 1}: {missing}"
