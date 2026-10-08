"""Typed live ingest-attempt dispositions and current producer boundaries.

The removed ``ingest_record`` worker's per-parser outcome mapping is covered
at the resident routes: live no-session/corrupt outcomes in
``test_live_intake_no_session_outcome.py``, retained strict drift refusal in
``test_retained_schema_drift_route.py``, and parser refusal in
``test_live_batch_support.py``. This module pins the disposition classifier
that remains at the live archive-write boundary and the durable attempt row.
Empty Code/Claude captures still need a current typed decode test and producer
fix; no zero-byte behavior is claimed by this classifier-only module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.compute import DaemonBackpressureError
from polylogue.core.enums import IngestOutcome
from polylogue.pipeline.ingest_outcomes import (
    classify_archive_write_exception,
    classify_decode_exception,
    success_disposition,
)


def test_a_streamed_decoder_refusal_classifies_corrupt_input() -> None:
    """A truncated document refused by the streamed (ijson) decoder is CORRUPT_INPUT, not a parser defect.

    Anti-vacuity: ``ijson.JSONError`` is not a ``ValueError``, so the decode
    classifier reported such a document as ``parser_defect``
    (polylogue-6r7wv sibling).
    """
    import io

    import ijson

    try:
        list(ijson.items(io.BytesIO(b'{"title": "cut", "mapping": {"n": {"id": "n", '), ""))
    except ijson.JSONError as exc:
        disposition = classify_decode_exception(exc)
    else:
        pytest.fail("expected a real ijson refusal")

    assert disposition.outcome is IngestOutcome.CORRUPT_INPUT
    assert disposition.retryable is False


def test_undecodable_bytes_classify_corrupt_input_via_real_decode_failure() -> None:
    """A genuine decode-stage exception (invalid UTF-8) classifies as CORRUPT_INPUT.

    Exercises ``classify_decode_exception`` against a real ``UnicodeDecodeError``
    raised by the standard library, proving the classification is type-based
    (never text-matched against the resulting error string).
    """
    try:
        b"\xff\xfe\x00\x01".decode("utf-8")
    except UnicodeDecodeError as exc:
        disposition = classify_decode_exception(exc)
    else:
        pytest.fail("expected a real UnicodeDecodeError")

    assert disposition.outcome is IngestOutcome.CORRUPT_INPUT
    assert disposition.retryable is False
    assert disposition.evidence_ref == "decode:UnicodeDecodeError"


def test_real_sqlite_lock_classifies_transient_error(tmp_path: Path) -> None:
    """A genuine ``sqlite3.OperationalError: database is locked`` is TRANSIENT_ERROR.

    Reproduces real lock contention with two live connections rather than
    constructing a fake exception, proving ``classify_archive_write_exception``
    keys off ``is_transient_sqlite_lock``'s structural check.
    """
    db_path = tmp_path / "contended.db"
    holder = sqlite3.connect(str(db_path), timeout=0)
    holder.execute("CREATE TABLE t (x INTEGER)")
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("INSERT INTO t VALUES (1)")

    contender = sqlite3.connect(str(db_path), timeout=0)
    try:
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            contender.execute("INSERT INTO t VALUES (2)")
    finally:
        holder.rollback()
        holder.close()
        contender.close()

    disposition = classify_archive_write_exception(excinfo.value)
    assert disposition.outcome is IngestOutcome.TRANSIENT_ERROR
    assert disposition.retryable is True
    assert disposition.evidence_ref == "archive_write:OperationalError"


def test_daemon_compute_backpressure_classifies_as_retryable() -> None:
    """The shared typed compute refusal is infrastructure pressure, not bad input."""
    exc = DaemonBackpressureError("daemon compute admission is saturated; retry shortly")
    disposition = classify_archive_write_exception(exc)

    assert disposition.outcome is IngestOutcome.TRANSIENT_ERROR
    assert disposition.retryable is True
    assert disposition.evidence_ref == "archive_write:DaemonBackpressureError"


def test_non_transient_database_error_classifies_parser_defect_not_swallowed() -> None:
    """A non-lock ``sqlite3.OperationalError`` (e.g. schema mismatch) is NOT retried silently."""
    exc = sqlite3.OperationalError("no such table: sessions")
    disposition = classify_archive_write_exception(exc)
    assert disposition.outcome is IngestOutcome.PARSER_DEFECT
    assert disposition.retryable is False


def test_ingest_attempt_round_trip_persists_typed_disposition(tmp_path: Path) -> None:
    """Every field AC1 names round-trips through a real ``ingest_attempts`` row."""
    from polylogue.sources.live.cursor import CursorStore

    cursor_store = CursorStore(tmp_path / "index.sqlite", ops_db_path=tmp_path / "ops.db")
    attempt_id = cursor_store.begin_ingest_attempt(paths=[Path("/x/a.jsonl")], input_bytes=10, queued_file_count=1)
    ok = cursor_store.finish_ingest_attempt(
        attempt_id, status="completed", phase="completed", disposition=success_disposition(evidence_ref="ok")
    )
    assert ok

    conn = sqlite3.connect(tmp_path / "ops.db")
    try:
        row = conn.execute(
            "SELECT outcome_code, retryable, evidence_ref FROM ingest_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row == (IngestOutcome.SUCCESS.value, 0, "ok")


def test_legacy_ingest_attempt_row_defaults_to_legacy_unknown(tmp_path: Path) -> None:
    """A row written without an explicit disposition (a legacy caller) stays legacy_unknown (AC4)."""
    from polylogue.sources.live.cursor import CursorStore

    cursor_store = CursorStore(tmp_path / "index.sqlite", ops_db_path=tmp_path / "ops.db")
    attempt_id = cursor_store.begin_ingest_attempt(paths=[Path("/x/legacy.jsonl")], input_bytes=10, queued_file_count=1)

    conn = sqlite3.connect(tmp_path / "ops.db")
    try:
        row = conn.execute(
            "SELECT outcome_code, retryable FROM ingest_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row == (IngestOutcome.LEGACY_UNKNOWN.value, None)


def test_a_decode_value_bound_refusal_is_not_a_parser_defect() -> None:
    """An unstorable decoded value is a permanent refusal, not a parser bug."""
    from polylogue.sources.value_bounds import ValueBoundRefusedError

    exc = ValueBoundRefusedError("string", 10, 5)
    disposition = classify_decode_exception(exc)

    assert disposition.outcome is IngestOutcome.VALIDATION_REJECTED
    assert disposition.evidence_ref == "decode:value_bound_refused"
