"""The receipts inspector finds a Codex session's source raw under either binding.

A native rollout is bound by its revision key on ``raw_sessions``; a grouped
export names the session through ``raw_session_memberships``. Both are the
session's source material.

Anti-vacuity: reading only memberships misses the byte-bound rollout (the
native case now governed by its own revision chain); reading only the
revision key misses the grouped export.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.demo.receipts import codex_session_source_rows
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _codex(session_id: str) -> bytes:
    return (
        b'{"type":"session_meta","payload":{"id":"' + session_id.encode() + b'"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m","role":"user",'
        b'"content":[{"type":"input_text","text":"hi"}]}}\n'
    )


def _rows(root: Path, provider_session_id: str) -> list[str]:
    with closing(sqlite3.connect(root / "source.db")) as conn:
        conn.row_factory = sqlite3.Row
        return [str(row["raw_id"]) for row in codex_session_source_rows(conn, provider_session_id)]


def test_receipts_source_lookup_covers_revision_and_membership_bindings(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        native = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex("native"),
            source_path="native.jsonl",
            canonical_source_path="native.jsonl",
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            native,
            RawRevisionEnvelope(
                "codex-session:native",
                RawRevisionKind.FULL,
                native,
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        grouped = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex("grouped"),
            source_path="bundle.jsonl",
            canonical_source_path="bundle.jsonl",
            acquired_at_ms=2,
        )
        archive.commit()
    with closing(sqlite3.connect(tmp_path / "source.db")) as conn, conn:
        conn.execute(
            "INSERT INTO raw_session_memberships (raw_id, logical_source_key, provider_session_id, "
            "source_revision, normalized_content_hash, message_count) VALUES (?, ?, ?, ?, ?, ?)",
            (grouped, "codex-session:grouped", "grouped", grouped, b"\0" * 32, 1),
        )

    assert _rows(tmp_path, "native") == [native]
    assert _rows(tmp_path, "grouped") == [grouped]
    assert _rows(tmp_path, "absent") == []
