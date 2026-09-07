from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.sources.live import hook_paste_enrichment
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_HOOK_TIME_MS = int(datetime(2026, 5, 7, 12, 0, tzinfo=UTC).timestamp() * 1000)


def _source_tier(archive_root: Path) -> Path:
    source_db = archive_root / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    return source_db


def _seed_hook_event(
    source_db: Path,
    *,
    origin: str,
    session_native_id: str,
    record: dict[str, object],
    event_id: str,
    observed_at_ms: int = _HOOK_TIME_MS,
) -> None:
    """Write one durable hook event the way the spool drain commits it."""
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id,
                source_path, event_type, payload_json, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"hook:{event_id}",
                origin,
                f"{session_native_id}:{record['event_type']}:{event_id}",
                session_native_id,
                f"/spool/pending/{event_id}.json",
                str(record["event_type"]),
                json.dumps(record, ensure_ascii=False, sort_keys=True),
                observed_at_ms,
            ),
        )


def _paste_record(session_key: str, session_value: str, **envelope: object) -> dict[str, object]:
    return {
        "event_type": "UserPromptSubmit",
        "timestamp": "2026-05-07T12:00:00Z",
        **envelope,
        "payload": {session_key: session_value, "prompt": "Inspect [Pasted text #1]"},
    }


def _seed_paste_candidate(index_db: Path, native_id: str, hook_time_ms: int = _HOOK_TIME_MS) -> None:
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, 'codex-session', ?, ?, ?)
            """,
            (native_id, native_id.encode().ljust(32, b"s")[:32], hook_time_ms, hook_time_ms),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, content_hash, occurred_at_ms
            ) VALUES (?, 'm1', 0, 'user', ?, ?)
            """,
            (f"codex-session:{native_id}", native_id.encode().ljust(32, b"m")[:32], hook_time_ms + 100),
        )


def test_hook_paste_enrichment_updates_archive_messages(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, content_hash, created_at_ms, updated_at_ms
            ) VALUES ('codex-native-1', 'codex-session', ?, ?, ?)
            """,
            (b"s" * 32, _HOOK_TIME_MS, _HOOK_TIME_MS),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, content_hash, occurred_at_ms
            ) VALUES ('codex-session:codex-native-1', 'm1', 0, 'user', ?, ?)
            """,
            (b"m" * 32, _HOOK_TIME_MS + 100),
        )
    _seed_hook_event(
        _source_tier(tmp_path),
        origin="codex-session",
        session_native_id="codex-native-1",
        record=_paste_record("session_id", "codex-native-1"),
        event_id="e1",
    )

    updated = hook_paste_enrichment.enrich_paste_from_hooks(tmp_path / "ops.db")

    assert updated == 1
    with sqlite3.connect(index_db) as conn:
        message = conn.execute(
            """
            SELECT has_paste, paste_boundary
            FROM messages
            WHERE session_id = 'codex-session:codex-native-1'
            """
        ).fetchone()
        assert message == (1, "hash_only")
        session = conn.execute(
            "SELECT paste_count FROM sessions WHERE session_id = 'codex-session:codex-native-1'"
        ).fetchone()
        assert session == (1,)
        span = conn.execute(
            """
            SELECT start_offset, end_offset, boundary_state
            FROM paste_spans
            WHERE session_id = 'codex-session:codex-native-1'
            """
        ).fetchone()
        assert span == (0, 0, "hash_only")


def test_hook_paste_enrichment_never_reads_a_sibling_archives_source_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(polylogue-o7hx) Both tiers are derived from ``db_path.parent``, never
    from an ambient global -- a different archive's hook events (including one
    that happens to sit at the configured archive root) must never leak into
    this archive's paste enrichment."""

    real_archive = tmp_path / "real-archive"
    real_archive.mkdir()
    _seed_hook_event(
        _source_tier(real_archive),
        origin="codex-session",
        session_native_id="codex-native-1",
        record=_paste_record("session_id", "codex-native-1"),
        event_id="decoy",
    )
    # An ambient env override, if the implementation still read one, would
    # point at ``real_archive`` here -- it must have no effect.
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(real_archive))

    scratch_ops_db = tmp_path / "scratch-archive" / "ops.db"
    scratch_ops_db.parent.mkdir()

    updated = hook_paste_enrichment.enrich_paste_from_hooks(scratch_ops_db)

    assert updated == 0


def test_hook_paste_enrichment_reads_only_the_batch_sessions_events(tmp_path: Path) -> None:
    """Anti-vacuity: an unscoped read would enrich the untouched session too.

    ``raw_hook_events`` is keyed by ``(origin, session_native_id)``; the batch
    passes its archive session ids and only those sessions' events are read, so
    the read is bounded by the batch instead of the archive's whole hook
    history.
    """
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    _seed_paste_candidate(index_db, "batch-native")
    _seed_paste_candidate(index_db, "other-native")
    source_db = _source_tier(tmp_path)
    for native_id in ("batch-native", "other-native"):
        _seed_hook_event(
            source_db,
            origin="codex-session",
            session_native_id=native_id,
            record=_paste_record("session_id", native_id),
            event_id=f"e-{native_id}",
        )

    assert hook_paste_enrichment._scoped_hook_keys(("codex-session:batch-native",)) == [
        ("codex-session", "batch-native")
    ]
    assert len(hook_paste_enrichment._iter_hook_paste_events(source_db, ("codex-session:batch-native",))) == 1
    assert len(hook_paste_enrichment._iter_hook_paste_events(source_db, None)) == 2

    updated = hook_paste_enrichment.enrich_paste_from_hooks(
        tmp_path / "ops.db", session_ids=("codex-session:batch-native",)
    )

    assert updated == 1
    with sqlite3.connect(index_db) as conn:
        rows = conn.execute("SELECT session_id, has_paste FROM messages ORDER BY session_id").fetchall()
    assert rows == [("codex-session:batch-native", 1), ("codex-session:other-native", 0)]


def test_camelcase_hook_payload_sets_has_paste(tmp_path: Path) -> None:
    """bd polylogue-cp806: the camelCase payload generation carries the same
    paste ground truth as the snake_case one.

    Anti-vacuity: keyed on ``session_id`` alone, the enrichment resolves no
    session for this record and updates nothing.
    """
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    _seed_paste_candidate(index_db, "camel-native")
    _seed_hook_event(
        _source_tier(tmp_path),
        origin="codex-session",
        session_native_id="camel-native",
        record={
            "event_type": "UserPromptSubmit",
            "payload": {
                "sessionId": "camel-native",
                "promptId": "p-1",
                "permissionMode": "auto",
                "hookEventName": "UserPromptSubmit",
                "timestamp": "2026-05-07T12:00:00Z",
                "prompt": "Inspect [Pasted text #1]",
            },
        },
        event_id="camel",
    )

    updated = hook_paste_enrichment.enrich_paste_from_hooks(tmp_path / "ops.db")

    assert updated == 1
    with sqlite3.connect(index_db) as conn:
        assert conn.execute(
            "SELECT has_paste, paste_boundary FROM messages WHERE session_id = 'codex-session:camel-native'"
        ).fetchone() == (1, "hash_only")


def test_an_unkeyable_paste_record_is_reported_not_dropped_silently(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """bd polylogue-cp806: paste evidence the readers cannot key is a signal.

    A key-name mismatch reads exactly like a field that was never sent, which
    is how this class survives to the archive silently. The pass still skips
    the record -- it cannot key it -- but names which reader keys resolved, so
    an undescribed generation is distinguishable from an absent field.

    Anti-vacuity: the camelCase record in the test above takes this branch
    when ``session_id`` is read under one spelling only.
    """
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    _seed_paste_candidate(index_db, "unknown-native")
    # The record has to reach the enrichment loop to be reported, so its paste
    # marker sits in a field the detector reads while its session key does not.
    _seed_hook_event(
        _source_tier(tmp_path),
        origin="codex-session",
        session_native_id="unknown-native",
        record={
            "event_type": "UserPromptSubmit",
            "payload": {"session.id": "unknown-native", "prompt": "Inspect [Pasted text #1]"},
        },
        event_id="unkeyable",
    )

    with caplog.at_level(logging.WARNING, logger="polylogue.sources.live.hook_paste_enrichment"):
        updated = hook_paste_enrichment.enrich_paste_from_hooks(tmp_path / "ops.db")

    assert updated == 0
    message = next(record.getMessage() for record in caplog.records if "no readable session key" in record.getMessage())
    assert "reader keys matched=['prompt']" in message
    assert "payload keys=['prompt', 'session.id']" in message


def test_a_journal_file_is_no_longer_a_paste_evidence_carrier(tmp_path: Path) -> None:
    """The retired ``<provider>-<native_id>.jsonl`` journal carries nothing.

    Anti-vacuity: with the journal reader still wired, this record enriches the
    seeded message and ``updated`` is 1.
    """
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    _seed_paste_candidate(index_db, "journal-native")
    _source_tier(tmp_path)
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "codex-journal-native.jsonl").write_text(
        json.dumps(_paste_record("session_id", "journal-native")) + "\n",
        encoding="utf-8",
    )

    assert hook_paste_enrichment.enrich_paste_from_hooks(tmp_path / "ops.db") == 0
