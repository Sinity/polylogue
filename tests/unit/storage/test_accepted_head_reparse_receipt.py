"""Ledger consistency when a reparse rewrites an already-accepted raw.

polylogue-2tfug: a parser fix can change a session's ``content_hash`` while
the raw evidence it was derived from is unchanged and still the accepted head
for its logical source key. If the ordinary session-write path updates
``sessions.content_hash`` without moving ``raw_revision_heads``, the replay
ledger goes stale in a way ``validate_raw_replay_application_receipt``
(``storage/raw_authority.py``) explicitly rejects: it requires every accepted
head's ``accepted_content_hash`` to equal the materialized session's
``content_hash``, so a legitimate content correction would be reported as
ledger inconsistency by a later formal replay audit.

Absorbed into polylogue-1fijp. This is the surviving red check for the shape,
driving the real head-establishing replay path and then the real ordinary
write path -- not ``record_revision_application_sync`` in isolation.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

LOGICAL_KEY = "codex:reparse-session"
SESSION_ID = "codex-session:reparse-session"
SOURCE_PATH = "reparse.jsonl"
PAYLOAD = b"a" * 32


def _session(text: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="reparse-session",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text=text)],
    )


def _hashes(root: Path) -> tuple[bytes | None, bytes | None]:
    with sqlite3.connect(root / "index.db") as conn:
        head = conn.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE logical_source_key = ?",
            (LOGICAL_KEY,),
        ).fetchone()
        session = conn.execute(
            "SELECT content_hash FROM sessions WHERE session_id = ?",
            (SESSION_ID,),
        ).fetchone()
    return (bytes(head[0]) if head and head[0] is not None else None, bytes(session[0]) if session else None)


def test_reparse_of_accepted_head_keeps_head_and_session_content_hash_in_sync(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=PAYLOAD,
            source_path=SOURCE_PATH,
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                LOGICAL_KEY,
                RawRevisionKind.FULL,
                hashlib.sha256(PAYLOAD).hexdigest(),
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        plan = archive.classify_raw_revision_cohort_for_live_watch(LOGICAL_KEY)
        archive.apply_raw_revision_replay(plan, {raw_id: _session("original parse")}, acquired_at_ms=1)

    head_hash, session_hash = _hashes(tmp_path)
    assert head_hash is not None, "the replay must establish an accepted head"
    assert head_hash == session_hash

    # A parser fix re-derives DIFFERENT content from the SAME accepted raw,
    # through the ordinary session-write path rather than a replay plan.
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_parsed_for_retained_raw_result(
            _session("corrected parse"),
            raw_id=raw_id,
            source_path=SOURCE_PATH,
            acquired_at_ms=2,
        )

    reparsed_head, reparsed_session = _hashes(tmp_path)
    assert reparsed_session != session_hash, "the reparse must actually change session content"
    assert reparsed_head == reparsed_session, (
        "raw_revision_heads.accepted_content_hash went stale against sessions.content_hash; "
        "validate_raw_replay_application_receipt rejects exactly this mismatch"
    )


def test_unchanged_reparse_of_accepted_head_issues_no_new_receipt(tmp_path: Path) -> None:
    """polylogue-2tfug AC4: idempotent re-ingest stays receipt-stable."""
    initialize_active_archive_root(tmp_path)

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=PAYLOAD,
            source_path=SOURCE_PATH,
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                LOGICAL_KEY,
                RawRevisionKind.FULL,
                hashlib.sha256(PAYLOAD).hexdigest(),
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        plan = archive.classify_raw_revision_cohort_for_live_watch(LOGICAL_KEY)
        archive.apply_raw_revision_replay(plan, {raw_id: _session("stable parse")}, acquired_at_ms=1)

    def receipt_count() -> int:
        with sqlite3.connect(tmp_path / "index.db") as conn:
            return int(conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0])

    before = receipt_count()

    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_parsed_for_retained_raw_result(
            _session("stable parse"),
            raw_id=raw_id,
            source_path=SOURCE_PATH,
            acquired_at_ms=2,
        )

    assert receipt_count() == before
