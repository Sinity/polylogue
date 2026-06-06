"""Integration test: search result determinism under concurrent writes (#1734).

Asserts that FTS5 queries during live ingest return stable results for
already-ingested sessions — monotonic growth, no regression, no
OperationalError or corrupt index errors.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from polylogue.storage.sqlite.connection import open_read_connection
from polylogue.storage.sqlite.connection_profile import open_connection


@pytest.mark.slow
@pytest.mark.integration
def test_search_results_monotonic_under_concurrent_writes(workspace_env: None, tmp_path: Path) -> None:
    """Concurrent writer + reader: already-indexed results must be stable."""

    db_path = tmp_path / "test_search_determinism.db"

    # Seed initial data
    conn = open_connection(db_path)
    conn.execute("INSERT INTO sessions (id, source_name, sort_key) VALUES ('c1', 'test', 1.0)")
    conn.execute("INSERT INTO sessions (id, source_name, sort_key) VALUES ('c2', 'test', 2.0)")
    conn.execute(
        "INSERT INTO messages (id, session_id, source_name, role, text, sort_key) "
        "VALUES ('m1', 'c1', 'test', 'user', 'hello world', 1.0)"
    )
    conn.execute(
        "INSERT INTO messages (id, session_id, source_name, role, text, sort_key) "
        "VALUES ('m2', 'c2', 'test', 'user', 'another test message', 2.0)"
    )
    conn.execute(
        "INSERT INTO messages_fts(messages_fts, rowid, message_id, session_id, text) "
        "VALUES ('rebuild', NULL, NULL, NULL, NULL)"
    )
    conn.commit()
    conn.close()

    results_lock = threading.Lock()
    search_results: list[list[str]] = []
    errors: list[Exception] = []
    stop_flag = threading.Event()

    def reader() -> None:
        with open_read_connection(db_path) as rconn:
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
                mid = f"m{i + 3}"
                cid = f"c{i + 3}"
                wconn.execute(
                    "INSERT INTO sessions (id, source_name, sort_key) VALUES (?, 'test', ?)",
                    (cid, float(i + 3)),
                )
                wconn.execute(
                    "INSERT INTO messages (id, session_id, source_name, role, text, sort_key) "
                    "VALUES (?, ?, 'test', 'user', ?, ?)",
                    (mid, cid, f"hello concurrent message {i}", float(i + 3)),
                )
                wconn.execute(
                    "INSERT INTO messages_fts(messages_fts, rowid, message_id, session_id, text) "
                    "VALUES ('rebuild', NULL, NULL, NULL, NULL)"
                )
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
        assert "m1" in result_set, f"Result set {i} lost pre-existing message 'm1': {result_set}"

    for i in range(1, len(search_results)):
        prev = set(search_results[i - 1])
        curr = set(search_results[i])
        missing = prev - curr
        assert not missing, f"Result set {i} lost messages from set {i - 1}: {missing}"
