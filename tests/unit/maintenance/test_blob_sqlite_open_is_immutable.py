"""Maintenance never opens a blob-store file as a writable SQLite database.

polylogue-ww6mj: a sweep called ``sqlite3.connect(<blob path>)`` without
``mode=ro&immutable=1``, so SQLite created ``-wal``/``-shm`` siblings inside the
content-addressed shard directories. Those siblings are not blobs: they fail
``BlobNamespaceEntryKind`` and block plan acceptance on their own.

Anti-vacuity: the fixture blob is a WAL-journal database, so an open that drops
``immutable=1`` either materializes a sibling in the shard (first assertion) or
fails outright and yields no logical-export digest (second assertion). Both
outcomes are red.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.maintenance.blob_disposition import codex_state_logical_export_hashes


def _write_codex_thread_state_blob(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("CREATE TABLE thread_spawn_edges (parent TEXT, child TEXT)")
        conn.execute("INSERT INTO threads VALUES ('t1', 'first')")
        conn.execute("INSERT INTO thread_spawn_edges VALUES ('t1', 't2')")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for sibling in (path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        sibling.unlink(missing_ok=True)


def test_maintenance_blob_open_leaves_no_sqlite_siblings_in_the_shard(tmp_path: Path) -> None:
    shard = tmp_path / "blobs" / "ab" / "cd"
    blob = shard / "abcdef0123456789"
    _write_codex_thread_state_blob(blob)
    before = {entry.name for entry in shard.iterdir()}

    digests = codex_state_logical_export_hashes(blob)

    assert digests, "the maintenance route must still read the blob's logical export"
    assert {entry.name for entry in shard.iterdir()} == before
    assert not list(shard.glob("*-wal"))
    assert not list(shard.glob("*-shm"))
