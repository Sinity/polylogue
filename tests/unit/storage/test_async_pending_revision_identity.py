"""The async acquisition writer stores a replaceable pending identity.

An acquisition that persists bytes before parsing carries no revision
envelope, so the writer assigns a provisional one. That provisional key is not
free-form: ``bind_source_raw_revision`` replaces it only when it matches
``pending-raw:%``, and parser-census canonicalisation excludes exactly that
prefix before partitioning the rest on an ``Origin``/``Provider`` segment. A
key spelled any other way is neither replaceable nor excludable, so an
otherwise valid import stays quarantined with an incomplete census entry --
durably blocking materialization readiness and authoritative rebuilds.

Anti-vacuity: restore the local f-string
``f"pending:{origin.value}:{source_path}:{source_index}:{raw_id}"`` in
``_write_raw_session_record_async`` and both assertions below fail -- the
stored key no longer carries the canonical prefix, and the later bind raises
a revision-identity conflict instead of succeeding.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
    durable_authority_logical_keys,
)
from polylogue.storage.sqlite.archive_tiers.source_write import (
    PENDING_RAW_LOGICAL_SOURCE_PREFIX,
    bind_source_raw_revision,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import make_raw_session


async def test_a_pending_async_acquisition_can_later_bind_its_parsed_identity(
    tmp_path: Path, frozen_clock: object
) -> None:
    del frozen_clock
    db_path = tmp_path / "source.db"
    backend = SQLiteBackend(db_path=db_path)
    record = make_raw_session(
        raw_id="raw-pending-1",
        source_name="chatgpt-export",
        source_path="/imports/conversations.json",
        source_index=0,
        blob_size=16,
        acquired_at="2026-02-02T12:00:00+00:00",
        file_mtime=None,
    )
    assert await backend.save_raw_session(record) is True

    with sqlite3.connect(db_path) as conn:
        stored = conn.execute(
            "SELECT logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (record.raw_id,),
        ).fetchone()
        assert stored is not None
        assert stored[0].startswith(PENDING_RAW_LOGICAL_SOURCE_PREFIX), stored[0]
        assert stored[1] == RawRevisionAuthority.QUARANTINED.value

        # A pending key is excluded from the durable identity set rather than
        # rejected as an unknown prefix, so the census can still answer.
        assert (
            durable_authority_logical_keys(
                raw_logical_key=stored[0],
                revision_kind=RawRevisionKind.FULL.value,
                membership_logical_keys=(),
            )
            == ()
        )

        # The wrong observable outcome the malformed prefix produced: the
        # parser has proved a session identity and the bind is refused.
        bind_source_raw_revision(
            conn,
            record.raw_id,
            RawRevisionEnvelope(
                logical_source_key="chatgpt:conversations.json",
                kind=RawRevisionKind.FULL,
                source_revision="a" * 64,
                acquisition_generation=1,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        bound = conn.execute(
            "SELECT logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (record.raw_id,),
        ).fetchone()
    assert bound == ("chatgpt:conversations.json", RawRevisionAuthority.BYTE_PROVEN.value)
