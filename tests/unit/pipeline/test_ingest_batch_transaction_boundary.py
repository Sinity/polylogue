"""The bulk ingest batch is one transaction, and FTS suspension is crash-safe.

Anti-vacuity (verified by reverting the fix, not asserted):

* ``test_batch_transaction_is_not_committed_per_session`` — with
  ``manage_transaction`` left at its default ``True`` in ``_write_session``,
  the per-session ``with conn:`` commits the batch's ``BEGIN IMMEDIATE``, so a
  *second, independent* connection observes the first session's rows while the
  batch is still running. The assertion is that durable visibility, not
  ``conn.in_transaction``.
* ``test_interrupt_during_suspended_fts_restores_triggers`` — a
  ``KeyboardInterrupt`` raised mid-batch while FTS triggers are suspended. With
  the ``except Exception`` restore handler, ``BaseException`` escapes
  uncaught and ``index.db`` is left with the FTS triggers dropped.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload
from polylogue.storage.fts.fts_lifecycle import FTS_TRIGGER_NAMES
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection
from tests.unit.pipeline.test_ingest_batch import _message_tuple, _session_data

_FTS_TRIGGERS = set(FTS_TRIGGER_NAMES)


def _raw(tmp_path: Path, name: str) -> RawSessionRecord:
    source_path = tmp_path / f"{name}.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    return RawSessionRecord(
        raw_id=f"raw-{name}",
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )


def _session(name: str) -> tuple[str, SessionWritePayload]:
    session_id = f"codex-session:{name}"
    return session_id, _session_data(
        session_id,
        content_hash=f"hash-{name}",
        message_tuples=[
            _message_tuple(
                f"msg-{name}",
                session_id,
                role="user",
                text=f"batch transaction boundary {name}",
                content_hash=f"hash-message-{name}",
                sort_key=0.0,
            )
        ],
    )


def _install_fake_ingest(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, SessionWritePayload]) -> None:
    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        return IngestRecordResult(raw_id=record.raw_id, sessions=[mapping[record.raw_id]])

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)


def _bootstrap_archive(db_path: Path, *, seed: str | None = None) -> None:
    """Create the index schema (and optionally one pre-existing session)."""
    from polylogue.pipeline.services.ingest_batch._core import _write_session

    with open_connection(db_path) as conn:
        if seed is not None:
            _, seeded = _session(seed)
            _write_session(conn, seeded)
        conn.commit()


def _session_ids_from_second_connection(db_path: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        return [row[0] for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id")]
    finally:
        conn.close()


def _existing_fts_triggers(db_path: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            if row[0] in _FTS_TRIGGERS
        }
    finally:
        conn.close()


def test_batch_transaction_is_not_committed_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    _bootstrap_archive(db_path)
    first_id, first = _session("boundary-one")
    second_id, second = _session("boundary-two")
    raws = [_raw(tmp_path, "boundary-one"), _raw(tmp_path, "boundary-two")]
    _install_fake_ingest(monkeypatch, {raws[0].raw_id: first, raws[1].raw_id: second})

    observed: list[list[str]] = []
    real_write = write_parsed_session_to_archive

    def observing_write(conn: sqlite3.Connection, *args: Any, **kwargs: Any) -> Any:
        result = real_write(conn, *args, **kwargs)
        # What a concurrent reader can durably see while the batch is mid-flight.
        observed.append(_session_ids_from_second_connection(db_path))
        return result

    monkeypatch.setattr(ingest_batch_core, "write_parsed_session_to_archive", observing_write)

    summary = _process_ingest_batch_sync(
        raws,
        db_path=db_path,
        archive_root_str=str(tmp_path / "archive"),
        blob_root_str=str(tmp_path / "blob"),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert len(observed) == 2, observed
    # No session may be durably visible to another connection before the batch
    # transaction commits.
    assert observed[0] == [], f"batch transaction already committed after first session: {observed[0]}"
    assert observed[1] == [], f"batch transaction already committed after second session: {observed[1]}"
    assert sorted(summary.changed_session_ids) == sorted([first_id, second_id])
    assert sorted(_session_ids_from_second_connection(db_path)) == sorted([first_id, second_id])


def test_interrupt_during_suspended_fts_restores_triggers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    _bootstrap_archive(db_path, seed="interrupt-seed")
    baseline = _existing_fts_triggers(db_path)
    assert baseline, "fixture precondition: the fresh index carries FTS triggers"

    first_id, first = _session("interrupt-one")
    _second_id, second = _session("interrupt-two")
    raws = [_raw(tmp_path, "interrupt-one"), _raw(tmp_path, "interrupt-two")]
    _install_fake_ingest(monkeypatch, {raws[0].raw_id: first, raws[1].raw_id: second})

    real_write = write_parsed_session_to_archive
    calls = {"n": 0}

    def interrupting_write(conn: sqlite3.Connection, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            # Operator Ctrl-C in the middle of a bulk re-ingest.
            raise KeyboardInterrupt
        return real_write(conn, *args, **kwargs)

    monkeypatch.setattr(ingest_batch_core, "write_parsed_session_to_archive", interrupting_write)

    with pytest.raises(KeyboardInterrupt):
        _process_ingest_batch_sync(
            raws,
            db_path=db_path,
            archive_root_str=str(tmp_path / "archive"),
            blob_root_str=str(tmp_path / "blob"),
            validation_mode="off",
            ingest_workers=1,
            measure_ingest_result_size=False,
            suspend_fts_triggers=True,
        )

    assert _existing_fts_triggers(db_path) == baseline, "FTS triggers left dropped after an interrupt"
    # And nothing from the aborted batch is durable.
    assert first_id not in _session_ids_from_second_connection(db_path)
