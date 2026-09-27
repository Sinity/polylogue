"""Disposable SQLite scratch databases.

A scratch database spools one bounded computation to disk -- a per-session
identity index, a duplicate-id census, an ownership lookup -- and is deleted
with its temporary directory when the computation ends. Nothing reads it after
a crash, so it needs no durability: under SQLite's defaults every autocommit
statement against it is its own transaction with a journal and an fsync, which
made spooling a session's rows cost milliseconds per row on a real filesystem.

``journal_mode = MEMORY`` keeps statement and transaction rollback working
(constraint failures still abort cleanly) without a journal file, and
``synchronous = OFF`` removes the fsyncs. Neither setting may be used for an
archive tier or for an artifact another process consumes after a restart.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_scratch_database(path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a disposable, single-owner scratch database without durability costs."""
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    return conn


__all__ = ["connect_scratch_database"]
