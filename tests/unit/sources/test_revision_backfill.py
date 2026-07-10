from __future__ import annotations

import sqlite3
from io import BytesIO
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def test_historical_backfill_selects_prefix_newest_independent_of_acquisition_order(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    baseline = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user","content":'
        b'[{"type":"input_text","text":"old"}]}}\n'
    )
    newest = baseline + (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant","content":'
        b'[{"type":"output_text","text":"new"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        newest_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=newest,
            source_path="session.jsonl",
            acquired_at_ms=1,
        )
        baseline_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path="session.jsonl",
            acquired_at_ms=2,
        )
        legacy_append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"response_item","payload":{"type":"message","id":"legacy-suffix"}}\n',
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=3,
        )

    result = backfill_historical_revision_evidence(tmp_path)

    assert result.scanned == 3
    assert result.classified_full == 2
    assert result.replayed_logical_sources == 1
    assert result.quarantined == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT message_count, raw_id FROM sessions").fetchone() == (2, newest_raw_id)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        row = conn.execute(
            "SELECT rowid, block_id, message_id, session_id, block_type FROM blocks ORDER BY rowid LIMIT 1"
        ).fetchone()
        assert row is not None
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (row[0],))
        conn.execute(
            """
            INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
            VALUES (?, ?, ?, ?, ?, 'stale-only-token')
            """,
            row,
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'stale' ").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = NULL WHERE logical_source_key IS NOT NULL")
        conn.commit()

    backfill_historical_revision_evidence(tmp_path)

    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'stale'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'old'").fetchone()[0] == 1
        assert set(conn.execute("SELECT raw_id, decision FROM raw_revision_applications")) == {
            (baseline_raw_id, "superseded"),
            (newest_raw_id, "selected_baseline"),
        }
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NOT NULL").fetchone()[0] == 2
        assert conn.execute(
            "SELECT revision_kind, revision_authority, parsed_at_ms FROM raw_sessions WHERE raw_id = ?",
            (legacy_append_raw_id,),
        ).fetchone() == ("unknown", "quarantined", None)

    parsed_baseline = parse_payload(
        Provider.CODEX,
        list(_iter_json_stream(BytesIO(baseline), "session.jsonl")),
        "session",
        source_path="session.jsonl",
    )[0]
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_parsed_for_retained_raw(
            parsed_baseline,
            raw_id=baseline_raw_id,
            source_path="session.jsonl",
            acquired_at_ms=3,
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT message_count, raw_id FROM sessions").fetchone() == (2, newest_raw_id)


def test_backfill_resumes_after_index_receipt_commits_before_source_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    payload = (
        b'{"type":"session_meta","payload":{"id":"session-1","timestamp":"2026-06-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"one","role":"user","content":'
        b'[{"type":"input_text","text":"one"}]}}\n'
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path="session.jsonl",
            acquired_at_ms=1,
        )

    original_mark = ArchiveStore.mark_raw_parse_succeeded

    def crash_after_index_commit(self: ArchiveStore, raw_id: str, *, provider: Provider) -> None:
        raise RuntimeError("crash after index receipt")

    monkeypatch.setattr(ArchiveStore, "mark_raw_parse_succeeded", crash_after_index_commit)
    with pytest.raises(RuntimeError, match="crash after index receipt"):
        backfill_historical_revision_evidence(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)

    monkeypatch.setattr(ArchiveStore, "mark_raw_parse_succeeded", original_mark)
    resumed = backfill_historical_revision_evidence(tmp_path)
    assert resumed.replayed_logical_sources == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == (1,)
