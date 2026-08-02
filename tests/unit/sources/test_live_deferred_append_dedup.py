"""polylogue-hat0: a deferred (quarantined-authority) append must not re-mint
a duplicate raw row on every subsequent, unchanged watcher pass.

Mechanism under test (see ``polylogue/sources/live/append_ingest.py``): an
append plan's bytes are durably written and revision-bound in ``source.db``
*before* the classifier decides whether its authority is accepted or
quarantined/ambiguous. When quarantined, ``record_deferred_append_cursor``
used to keep the cursor's ``byte_offset`` completely unchanged, so the next
watcher pass over the exact same, unchanged file saw ``size > byte_offset``
again, rebuilt the identical append plan, and minted a second raw_id for
byte-identical content -- forever, on every tick, with zero failure
telemetry (``mark_failed`` is never called for a deferral).

``deferred_end_offset`` closes this: it records the end of the byte range
already durably captured and pending authority resolution, distinct from
``byte_offset`` (which only advances once a plan is applied). A subsequent
observation of a file that hasn't grown past that marker, at the same mtime,
now recognizes there is nothing new to acquire.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_support import _DEFER_APPEND, _AppendPlan, encode_cursor_hash_authority
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.deferred_cursor import record_deferred_append_cursor
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


def _raw_session_count(tmp_path: Path) -> int:
    with sqlite3.connect(tmp_path / "source.db") as conn:
        return int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])


def _seed_quarantine_prone_append(tmp_path: Path, *, session_id: str) -> Path:
    """Build a cursor whose predecessor fingerprint has no matching durable raw.

    ``raw_append_revision_parent`` (revision_governance.py) looks up a row in
    ``raw_sessions`` matching ``(logical_source_key, source_revision=predecessor,
    ...)``. Seeding the ops.db cursor with a ``content_fingerprint`` that was
    never actually written as a raw revision (e.g. after a crash-interrupted
    reconciliation, or ops.db/source.db drift) reproduces the real-world
    QUARANTINED-authority path deterministically, without needing to first
    build then discard a genuine full baseline.
    """
    initialize_active_archive_root(tmp_path)
    source = tmp_path / f"rollout-{session_id}.jsonl"
    baseline = _session_meta(session_id) + _codex_message("baseline")
    source.write_bytes(baseline)
    baseline_digest = hashlib.sha256(baseline).hexdigest()
    appended = _codex_message("appended-once")
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
        content_fingerprint="orphaned-baseline-fingerprint",
        tail_hash=encode_cursor_hash_authority(baseline_digest, baseline_digest, ctime_ns=stat.st_ctime_ns),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    return source


def test_deferred_append_without_marker_replans_every_tick_forever(tmp_path: Path) -> None:
    """Confirms the mechanism precisely, without the fix's cursor marker.

    ``raw_sessions.raw_id`` is a deterministic hash of
    ``(origin, source_path, source_index, blob_hash, native_id)``
    (``deterministic_raw_session_id``), and the blob store itself skips
    writing content it already has -- so replanning a *byte-identical*
    range does not literally duplicate a SQL row or blob (content-hash
    idempotency, by design). The actual "forever" bug is that
    ``_append_plan`` keeps producing a fresh, real ``_AppendPlan`` on every
    single watcher tick -- re-running the full write/classify/defer path
    (lock contention, wasted parse work, a repeated no-op INSERT attempt)
    for zero progress -- purely because nothing marks the byte range as
    already-captured-and-pending. Without ``deferred_end_offset`` (i.e. a
    cursor left exactly as the pre-fix ``record_deferred_append_cursor``
    left it: ``byte_offset`` never advanced, no pending-range marker at
    all), a second, third, Nth observation of the identical file each
    produces another real plan forever.
    """
    source = _seed_quarantine_prone_append(tmp_path, session_id="stuck-replan-forever")
    cursor = CursorStore(tmp_path / "ops.db")
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)
    assert isinstance(plan, _AppendPlan)
    first_result = processor._ingest_append_plans([plan])
    assert first_result.deferred == [plan]
    assert _raw_session_count(tmp_path) == 1

    # Cursor is untouched (byte_offset never advanced, deferred_end_offset
    # was never set) -- exactly the pre-fix state after a deferral.
    for _tick in range(3):
        replayed_plan = processor._append_plan(source)
        assert isinstance(replayed_plan, _AppendPlan), (
            "an unmarked cursor keeps producing a fresh append plan for the identical, "
            "already-deferred byte range on every single watcher tick"
        )
        assert replayed_plan.start_offset == plan.start_offset
        assert replayed_plan.payload == plan.payload
        processor._ingest_append_plans([replayed_plan])

    # Content-hash idempotency means no literal duplicate row accumulates
    # for byte-identical replans -- but see the next test for the case
    # where the file keeps growing while still deferred, where distinct
    # overlapping raw rows genuinely do accumulate every tick.
    assert _raw_session_count(tmp_path) == 1


def test_deferred_append_without_marker_accumulates_a_new_raw_id_per_growth_tick(tmp_path: Path) -> None:
    """The literal "new raw_id every tick" claim, for a file that keeps
    growing while its authority chain never resolves (the common real-world
    shape: an actively-writing session whose baseline was lost).

    Each tick's payload spans ``[start_offset, new_end)`` -- a strictly
    larger, differently-hashed range than the previous tick, because
    ``start_offset`` never advances without the fix. That is a genuinely
    distinct ``blob_hash`` each time, so ``deterministic_raw_session_id``
    mints a genuinely new, distinct raw_id every tick, each one a
    redundant superset of the last -- durable storage grows without bound
    for a session that never converges.
    """
    source = _seed_quarantine_prone_append(tmp_path, session_id="stuck-growing")
    cursor = CursorStore(tmp_path / "ops.db")
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)
    assert isinstance(plan, _AppendPlan)
    processor._ingest_append_plans([plan])
    assert _raw_session_count(tmp_path) == 1

    for tick in range(3):
        with source.open("ab") as handle:
            handle.write(_codex_message(f"still-growing-{tick}"))
        grown_plan = processor._append_plan(source)
        assert isinstance(grown_plan, _AppendPlan)
        assert grown_plan.start_offset == plan.start_offset, "start_offset never advances without the fix"
        processor._ingest_append_plans([grown_plan])
        assert _raw_session_count(tmp_path) == tick + 2, (
            "each growth tick against an unadvanced start_offset mints a new, distinct, "
            "overlapping raw_id -- unbounded durable storage growth for a session that never converges"
        )


def test_append_plan_skips_replanning_an_unchanged_already_deferred_range(tmp_path: Path) -> None:
    """The actual fix: once ``deferred_end_offset`` is recorded, a second,
    unchanged observation of the same file must not produce a fresh append
    plan at all, so ``_ingest_append_plans`` is never called a second time
    and no duplicate raw_id is minted.
    """
    source = _seed_quarantine_prone_append(tmp_path, session_id="stuck-fixed")
    cursor = CursorStore(tmp_path / "ops.db")
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)
    assert isinstance(plan, _AppendPlan)
    result = processor._ingest_append_plans([plan])
    assert result.deferred == [plan]
    assert _raw_session_count(tmp_path) == 1

    # Exactly what batch.py's flush_append_plans loop does for a plan that
    # came back in ``append_result.deferred``.
    record_deferred_append_cursor(
        cursor,
        source,
        cursor=cursor.get_record(source),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        source_name="codex",
        deferred_end_offset=plan.last_complete_newline,
    )

    stored = cursor.get_record(source)
    assert stored is not None
    assert stored.deferred_end_offset == plan.last_complete_newline
    # byte_offset never advances for a deferred plan -- this is the exact
    # pre-fix state that made the file look like it had unconsumed new bytes
    # on every subsequent tick.
    assert stored.byte_offset < stored.deferred_end_offset

    second_plan = processor._append_plan(source)
    assert second_plan is _DEFER_APPEND, (
        "an unchanged file whose entire growth is already captured-and-deferred must not produce a fresh append plan"
    )
    assert _raw_session_count(tmp_path) == 1, "no duplicate raw_id may be minted for the same pending byte range"


def test_append_plan_still_captures_genuine_growth_past_a_deferred_range(tmp_path: Path) -> None:
    """Constraint 3: a file that keeps growing while still authority-deferred
    must not be permanently frozen -- new bytes past the already-captured
    window are still planned normally.
    """
    source = _seed_quarantine_prone_append(tmp_path, session_id="stuck-then-grows")
    cursor = CursorStore(tmp_path / "ops.db")
    processor = _processor(tmp_path, cursor)

    plan = processor._append_plan(source)
    assert isinstance(plan, _AppendPlan)
    processor._ingest_append_plans([plan])
    record_deferred_append_cursor(
        cursor,
        source,
        cursor=cursor.get_record(source),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        source_name="codex",
        deferred_end_offset=plan.last_complete_newline,
    )

    # Confirm the file is recognized as fully captured-and-pending first.
    assert processor._append_plan(source) is _DEFER_APPEND

    # Now the file genuinely grows further while still authority-deferred.
    with source.open("ab") as handle:
        handle.write(_codex_message("grew-while-deferred"))

    grown_plan = processor._append_plan(source)
    assert isinstance(grown_plan, _AppendPlan), (
        "growth past the already-captured-and-deferred window must still be planned, not frozen forever"
    )
    assert grown_plan.last_complete_newline > plan.last_complete_newline
