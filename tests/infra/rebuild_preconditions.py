"""Satisfy the preconditions a rebuild imposes on a hand-seeded corpus.

A fixture that seeds raws with ``write_raw_payload`` records bytes only. Two
gates then refuse the corpus before any rebuild behaviour runs:

* the schema-inference preflight wants a fresh receipt, and
* inactive-candidate construction wants a complete current-parser census, which
  compares each raw's parsed identity against its durable ``logical_source_key``,
  and frozen source authority, which refuses while a raw is still quarantined.

Both are real production requirements, so the fixtures satisfy them rather than
bypassing them. Keeping the recipe here means a new gate is taught once instead
of in every rebuild fixture.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.sources.parsers import codex as codex_parser
from polylogue.storage.sqlite.archive_tiers.revision_governance import record_current_parser_source_census

__all__ = ["decide_raw_revision_authority", "record_codex_parser_census"]


def decide_raw_revision_authority(root: Path) -> None:
    """Derive revision authority for every seeded logical source key.

    ``write_raw_payload`` records bytes; it does not run admission, so a raw
    seeded that way keeps the default ``quarantined`` authority forever. The
    inactive-candidate gate then refuses the corpus with "N raw(s) remain
    quarantined or undecided" -- correctly, because nothing ever decided them.

    This runs the same classifier live ingest runs, so authority is derived
    from the bytes rather than fabricated with an UPDATE. That distinction is
    load-bearing: a rebuild re-derives byte authority for every frozen raw and
    rejects a patched value with "re-derived different byte authority".
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with sqlite3.connect(root / "source.db") as source:
        # The classifier only considers raws carrying acquisition evidence
        # (``source_revision IS NOT NULL``); real admission sets it, seeding
        # does not. Supplying a content-derived token is fixture *evidence*,
        # not a fabricated verdict -- authority itself is still derived below.
        source.execute(
            """
            UPDATE raw_sessions
            SET source_revision = lower(hex(blob_hash))
            WHERE logical_source_key IS NOT NULL AND source_revision IS NULL
            """
        )
        source.commit()
        keys = [
            str(row[0])
            for row in source.execute(
                "SELECT DISTINCT logical_source_key FROM raw_sessions WHERE logical_source_key IS NOT NULL"
            ).fetchall()
        ]
    if not keys:
        return
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for key in keys:
            archive.classify_raw_revision_cohort_for_live_watch(key)


def _records(payload: bytes) -> list[object]:
    """Decode a JSONL codex payload into the record list its parser expects."""
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


def record_codex_parser_census(root: Path, seeded: dict[str, bytes]) -> None:
    """Record a complete current-parser census for ``{raw_id: payload}``.

    Sets each raw's durable ``logical_source_key`` first: the census compares
    the parsed identity against that key and records ``status='failed'`` when it
    is absent, which reads downstream as a stale census rather than a missing
    one.
    """
    with sqlite3.connect(root / "source.db") as source:
        for raw_id, payload in seeded.items():
            parsed = codex_parser.parse(_records(payload), raw_id)
            source.execute(
                # Deliberately NOT setting revision_authority: the rebuild
                # re-derives byte authority for every frozen raw and rejects a
                # fabricated value with "re-derived different byte authority".
                # A corpus that must be accepted has to go through real
                # admission, not a patched column.
                "UPDATE raw_sessions SET logical_source_key = ?, revision_kind = 'full' WHERE raw_id = ?",
                (f"codex-session:{parsed.provider_session_id}", raw_id),
            )
        source.commit()
        for raw_id, payload in seeded.items():
            parsed = codex_parser.parse(_records(payload), raw_id)
            record_current_parser_source_census(source, raw_id, parser_sessions=[parsed])
        source.commit()
