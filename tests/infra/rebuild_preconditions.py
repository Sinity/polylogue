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

__all__ = ["record_codex_parser_census"]


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
