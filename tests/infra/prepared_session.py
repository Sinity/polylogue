"""Synthetic fixtures use the canonical prepare-then-publish writer contract.

This helper preserves a fixture's caller-owned SQL snapshot. Production
admission and executor ownership are exercised by the ingest/runtime tests.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from polylogue.sources.parsers.base_models import ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedSessionRows,
    PreparedSessionShardRows,
    prepare_session_write,
    prepared_session_rows_from_shard,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.archive_tiers.write_shard import ShardIdentitySequence


def write_prepared_session(conn: sqlite3.Connection, session: ParsedSession, **options: Any) -> str:
    if "prepared_write" in options:
        return write_parsed_session_to_archive(conn, session, **options)
    rows = options.pop("prepared_rows", None)
    if isinstance(rows, PreparedSessionShardRows):
        identities = rows.entry.content_identities
        if not isinstance(identities, ShardIdentitySequence):
            raise TypeError("fixture shard must carry its sealed identity sequence")
        rows = prepared_session_rows_from_shard(identities.path, rows.session_id)
    if rows is not None and not isinstance(rows, PreparedSessionRows):
        raise TypeError("fixture rows must be the canonical prepared carrier")
    prepared = prepare_session_write(
        conn,
        session,
        merge_append=bool(options.get("merge_append", False)),
        fallback_timestamp=options.get("fallback_timestamp"),
        source_conn=options.get("source_conn"),
        raw_id=options.get("raw_id"),
        force_replace=bool(options.get("force_replace", False)),
        child_source_path=options.get("child_source_path"),
        prepared_rows=rows,
    )
    if options.get("content_hash") is None:
        options["content_hash"] = prepared.input_content_hash.hex()
    if options.get("pending_input_content_hash") is None:
        options["pending_input_content_hash"] = prepared.input_content_hash.hex()
    try:
        result = write_parsed_session_to_archive(conn, session, prepared_write=prepared, **options)
    except BaseException as primary:
        try:
            prepared.close()
        except BaseException as cleanup:
            primary.add_note(f"prepared fixture cleanup failed: {cleanup!r}")
        raise
    else:
        prepared.close()
        return result
