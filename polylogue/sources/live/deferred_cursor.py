"""Cursor updates for append tails that are not newline-complete yet."""

from __future__ import annotations

from pathlib import Path

from polylogue.sources.live.batch_support import (
    cursor_prefix_hash,
    encode_cursor_hash_authority,
    tail_hash_from_path,
)
from polylogue.sources.live.cursor import CursorRecord, CursorStore


def record_deferred_append_cursor(
    cursor_store: CursorStore,
    path: Path,
    *,
    cursor: CursorRecord | None,
    parser_fingerprint: str,
    source_name: str,
) -> int:
    if cursor is None:
        return 0
    try:
        stat = path.stat()
        tail_hash, tail_bytes = tail_hash_from_path(path, stat.st_size)
    except FileNotFoundError:
        return 0
    prefix_hash = cursor_prefix_hash(cursor.tail_hash)
    cursor_store.set(
        path,
        stat.st_size,
        byte_offset=cursor.byte_offset,
        last_complete_newline=cursor.last_complete_newline,
        record_count=cursor.record_count,
        last_record_ts=cursor.last_record_ts,
        parser_fingerprint=parser_fingerprint,
        content_fingerprint=cursor.content_fingerprint,
        tail_hash=(
            encode_cursor_hash_authority(prefix_hash, tail_hash, ctime_ns=stat.st_ctime_ns)
            if prefix_hash is not None
            else tail_hash
        ),
        source_name=source_name,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=cursor.failure_count,
        next_retry_at=cursor.next_retry_at,
        excluded=bool(cursor.excluded),
        allow_backward=True,
    )
    return tail_bytes
