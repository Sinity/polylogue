"""``rebase_authoritative_observations`` really is one ops-tier transaction.

polylogue-5pv1p: the batch opened ``BEGIN IMMEDIATE`` but each row's write
delegated to ``upsert_ingest_cursor``, which committed unconditionally. The
first rebase therefore ended the explicit transaction and every later one ran
in autocommit, so a crash mid-batch left a partially rebased device identity.

Anti-vacuity: restoring the unconditional ``conn.commit()`` in
``upsert_ingest_cursor`` (or dropping ``manage_transaction=False`` at the call
site) makes the first rebase survive the mid-batch failure and the final
assertion go red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.sources.live.cursor import CursorObservationRebase, CursorRecord, CursorStore


def _seed(store: CursorStore, path: Path) -> CursorRecord:
    path.write_text("x" * 10, encoding="utf-8")
    store.set(path, 10, record_count=1, st_dev=1, st_ino=1, mtime_ns=1)
    record = store.get_record(path)
    assert record is not None
    return record


def _rebase(record: CursorRecord, *, ino: int) -> CursorObservationRebase:
    return CursorObservationRebase(
        path=Path(record.source_path),
        expected=record,
        st_dev=2,
        st_ino=ino,
        mtime_ns=2,
        tail_hash=f"tail-{ino}",
    )


def test_rebase_batch_rolls_back_entirely_when_a_later_row_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CursorStore(tmp_path / "index.db", ops_db_path=tmp_path / "ops.db")
    first = _seed(store, tmp_path / "a.jsonl")
    second = _seed(store, tmp_path / "b.jsonl")

    rebases = (_rebase(first, ino=11), _rebase(second, ino=12))

    original = CursorStore._write_cursor_record_on_conn
    calls = {"n": 0}

    def failing(conn, record, *, manage_transaction: bool = True):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("batch interrupted")
        return original(conn, record, manage_transaction=manage_transaction)

    monkeypatch.setattr(CursorStore, "_write_cursor_record_on_conn", staticmethod(failing))

    with pytest.raises(RuntimeError, match="batch interrupted"):
        store.rebase_authoritative_observations(rebases)

    after = store.get_record(Path(first.source_path))
    assert after is not None
    assert after.st_ino == 1, "a failed batch must not leave the first rebase committed"
    assert after.tail_hash is None
