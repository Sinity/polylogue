"""polylogue-aex0: append planning must resynthesize a lost ops.db cursor.

``ingest_cursor`` (``CursorStore``) lives in the disposable ``ops.db`` tier --
every reset (index rebuild, schema mismatch, ``polylogue ops reset``) wipes it,
forcing the next observation of a still-growing file back onto the
full-capture path even though the file itself hasn't changed shape. These
scenarios exercise ``LiveBatchProcessor._append_plan``'s secondary lookup
against ``source.db``'s durable revision-chain evidence
(``_resynthesize_cursor_from_source``): unchanged behavior when the ops.db
cursor is present, successful append recovery when it's gone but a durable
byte-proven 'full' head exists, and unchanged fallback-to-full-capture when
neither exists.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_support import _AppendPlan, encode_cursor_hash_authority
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _session_meta(session_id: str) -> bytes:
    return f'{{"type":"session_meta","payload":{{"id":"{session_id}"}}}}\n'.encode()


def _codex_message(text: str) -> bytes:
    return (
        f'{{"type":"response_item","payload":{{"type":"message","role":"user",'
        f'"content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()


def _seed_native_session(tmp_path: Path, *, session_id: str) -> None:
    """Satisfy the Codex append path's identity-lock (``_existing_provider_session_id``)."""
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, "codex-session", "unrelated-raw-id", 1, b"a" * 32, 1, 1),
        )
        conn.commit()


def _processor(tmp_path: Path, cursor: CursorStore) -> LiveBatchProcessor:
    return LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=tmp_path),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )


def test_append_plan_resynthesizes_lost_cursor_from_durable_full_head(tmp_path: Path) -> None:
    """No ops.db cursor, but source.db has a byte-proven 'full' head: append is attempted."""
    session_id = "resynth-proof"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-resynth-proof.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key=f"codex:{session_id}",
                kind=RawRevisionKind.FULL,
                source_revision="full-0",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
    appended = _codex_message("grown-after-reset")
    source.write_bytes(baseline + appended)
    _seed_native_session(tmp_path, session_id=session_id)

    cursor = CursorStore(tmp_path / "ops.db")
    assert cursor.get_record(source) is None  # the ops.db cursor is genuinely gone
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(baseline)
    assert plan.cursor_fingerprint == "full-0"
    assert appended in plan.payload


def test_append_plan_uses_ops_db_cursor_without_consulting_source_db(tmp_path: Path) -> None:
    """An existing ops.db cursor is used as before; the fallback is never invoked."""
    session_id = "existing-cursor-proof"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-existing-cursor.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    source.write_bytes(baseline)
    baseline_digest = hashlib.sha256(baseline).hexdigest()
    appended = _codex_message("grown")
    with source.open("ab") as handle:
        handle.write(appended)
    stat = source.stat()
    _seed_native_session(tmp_path, session_id=session_id)

    cursor = CursorStore(tmp_path / "ops.db")
    cursor.set(
        source,
        len(baseline),
        byte_offset=len(baseline),
        last_complete_newline=len(baseline),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        tail_hash=encode_cursor_hash_authority(baseline_digest, baseline_digest, ctime_ns=stat.st_ctime_ns),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    processor = _processor(tmp_path, cursor)

    def _fail_resynthesis(self: LiveBatchProcessor, path: Path) -> None:
        raise AssertionError("resynthesis must not be consulted when an ops.db cursor is already usable")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(LiveBatchProcessor, "_resynthesize_cursor_from_source", _fail_resynthesis)
        plan = processor._append_plan(source)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == len(baseline)
    assert appended in plan.payload


def test_append_plan_declines_when_neither_cursor_nor_durable_head_exists(tmp_path: Path) -> None:
    """A genuinely new file falls back to full capture exactly as before."""
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-brand-new.jsonl"
    source.write_bytes(_session_meta("brand-new") + _codex_message("hello"))

    cursor = CursorStore(tmp_path / "ops.db")
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)

    assert plan is None


def test_append_plan_declines_resynthesis_for_an_append_kind_head(tmp_path: Path) -> None:
    """An already-accepted append-kind head must not be treated as resynthesizable.

    Resynthesizing from a stale 'full' baseline behind an already-accepted
    append chain would create a second sibling append candidate at the same
    offset and make ``plan_revision_replay`` mark the whole chain ambiguous.
    """
    session_id = "append-head-proof"
    initialize_active_archive_root(tmp_path)
    source = tmp_path / "rollout-append-head.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    first_append_delta = _codex_message("first-append")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=baseline,
            source_path=str(source),
            acquired_at_ms=1,
            revision=RawRevisionEnvelope(
                logical_source_key=f"codex:{session_id}",
                kind=RawRevisionKind.FULL,
                source_revision="full-0",
                acquisition_generation=0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_session_meta(session_id) + first_append_delta,
            source_path=str(source),
            acquired_at_ms=2,
            revision=RawRevisionEnvelope(
                logical_source_key=f"codex:{session_id}",
                kind=RawRevisionKind.APPEND,
                source_revision="append-1",
                acquisition_generation=1,
                predecessor_source_revision="full-0",
                predecessor_raw_id=None,
                baseline_raw_id=None,
                append_start_offset=len(baseline),
                append_end_offset=len(baseline) + len(first_append_delta),
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )
        # Promote the append into the accepted chain the same way the real
        # append-ingest path does, via the durable classifier.
        archive.classify_raw_revision_cohort(f"codex:{session_id}")
    second_append = _codex_message("second-append")
    source.write_bytes(baseline + first_append_delta + second_append)
    _seed_native_session(tmp_path, session_id=session_id)

    cursor = CursorStore(tmp_path / "ops.db")
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)

    assert plan is None
