"""Neutral source-linked sessions for resident identity-reset laws."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.archive_identity import ArchiveLocation


def seed_identity_reset_sources(root: Path, count: int, *, start: int = 0) -> None:
    native_ids = tuple(f"neutral-reset-{index:04d}" for index in range(start, start + count))
    with sqlite3.connect(root / "source.db") as source:
        source.executemany(
            "INSERT INTO raw_sessions(raw_id,origin,native_id,source_path,blob_hash,blob_size,acquired_at_ms) "
            "VALUES (?, 'codex-session', ?, ?, zeroblob(32), 0, 1000)",
            [(f"raw-{native}", native, f"/neutral/selected/{native}.jsonl") for native in native_ids],
        )
    with sqlite3.connect(ArchiveLocation.resolve(root).active_index_path) as index:
        index.executemany(
            "INSERT INTO sessions(native_id,origin,raw_id,title,content_hash,created_at_ms,updated_at_ms) "
            "VALUES (?, 'codex-session', ?, 'Neutral session', zeroblob(32), 1000, 2000)",
            [(native, f"raw-{native}") for native in native_ids],
        )
