"""Random access to a sealed shard's message identities reads pages, not rows."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.sqlite import session_shard
from polylogue.storage.sqlite.session_shard import ShardIdentitySequence


def _identity_shard(path: Path, rows: int) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE messages (content_identity TEXT NOT NULL, content_occurrence INTEGER NOT NULL)")
        conn.executemany(
            "INSERT INTO messages(rowid, content_identity, content_occurrence) VALUES (?, ?, ?)",
            [(rowid, f"identity-{rowid}", rowid % 3) for rowid in range(1, rows + 1)],
        )
        conn.commit()


def test_identity_lookups_open_one_connection_per_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The writer resolves every message's identity by index; that costs one read per page.

    Anti-vacuity: open a connection per lookup (the predecessor) and a
    session of N messages opens N connections instead of ceil(N / page).
    """
    path = tmp_path / "shard.db"
    _identity_shard(path, 40)
    monkeypatch.setattr(session_shard, "_IDENTITY_PAGE_ROWS", 8)
    opened = 0
    real_connect = sqlite3.connect

    def counting_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal opened
        opened += 1
        connection: sqlite3.Connection = real_connect(*args, **kwargs)
        return connection

    monkeypatch.setattr(sqlite3, "connect", counting_connect)
    # Rows 11..35 of the shard are this session's messages.
    sequence = ShardIdentitySequence(path, 11, 35)

    assert len(sequence) == 25
    assert [sequence[index] for index in range(len(sequence))] == [
        (f"identity-{rowid}", rowid % 3) for rowid in range(11, 36)
    ]
    assert opened == 4
    assert sequence[-1] == ("identity-35", 2)
    assert sequence[0] == ("identity-11", 2)
    assert sequence[3:5] == [("identity-14", 2), ("identity-15", 0)]
    with pytest.raises(IndexError):
        sequence[25]


def test_identity_lookup_refuses_a_page_with_missing_rows(tmp_path: Path) -> None:
    path = tmp_path / "shard.db"
    _identity_shard(path, 10)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DELETE FROM messages WHERE rowid = 5")
        conn.commit()
    sequence = ShardIdentitySequence(path, 1, 10)

    with pytest.raises(session_shard.ShardRefusedError):
        sequence[0]
