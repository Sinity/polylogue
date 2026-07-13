"""Evidence harness for the live watcher append/cohort memory incident.

Static trace from the live daemon:

| Candidate | Production site | Trigger | Measured signal |
| --- | --- | --- | --- |
| H1 | ``_ingest_append_plans_archive`` | watcher append batch | batch/plan counts and process phases |
| H2 | ``classify_raw_revision_cohort`` | accepted append | classifier calls and accepted replay raws |
| H3 | ``ArchiveBlobPublisher.read_all`` | every historical full snapshot | exact full-blob bytes loaded |

The real incident was H1 -> H2, not historical backfill.  This scenario seeds
a byte-prefix full-snapshot cohort, then executes the production watcher append
entrypoint.  It reports anon-PSS, cgroup anon/file, process I/O deltas, and
batch counts at phase boundaries.  Host-dependent numbers have no CI budget;
route and byte-count assertions keep the harness non-vacuous.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.live.append_ingest import ingest_append_plans
from polylogue.sources.live.batch_support import _AppendPlan
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.append_cohort_memory_counter import append_cohort_memory_counter


def _codex_record(session_id: str, message_id: str, text: str) -> bytes:
    return (
        f'{{"type":"response_item","payload":{{"type":"message","id":"{message_id}",'
        f'"role":"user","content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()


def _full_snapshots(session_id: str) -> list[bytes]:
    prefix = f'{{"type":"session_meta","payload":{{"id":"{session_id}"}}}}\n'.encode()
    snapshots = [prefix]
    for index in range(3):
        snapshots.append(snapshots[-1] + _codex_record(session_id, f"history-{index}", "h" * 16_384))
    return snapshots[1:]


def _owner(archive_root: Path) -> object:
    cursor = CursorStore(archive_root / "append-cursor.sqlite")
    return SimpleNamespace(
        _cursor=cursor,
        _polylogue=SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=cursor._db_path)),
    )


def _seed_cohort_and_append_plan(archive_root: Path) -> _AppendPlan:
    initialize_active_archive_root(archive_root)
    session_id = "append-memory-proof"
    snapshots = _full_snapshots(session_id)
    source_path = archive_root / "captures" / "append-memory-proof.jsonl"
    source_path.parent.mkdir()
    append_payload = f'{{"type":"session_meta","payload":{{"id":"{session_id}"}}}}\n'.encode() + _codex_record(
        session_id, "append", "a" * 16_384
    )
    source_path.write_bytes(snapshots[-1] + append_payload)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for index, payload in enumerate(snapshots):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=str(source_path),
                acquired_at_ms=index + 1,
                revision=RawRevisionEnvelope(
                    f"codex:{session_id}",
                    RawRevisionKind.FULL,
                    f"full-{index}",
                    index,
                    # The watcher can establish the immediate append parent
                    # from an asserted full revision; classifier promotion is
                    # the boundary this harness then measures.
                    authority=RawRevisionAuthority.ASSERTED,
                ),
            )
    stat = source_path.stat()
    return _AppendPlan(
        path=source_path,
        source_name="codex",
        start_offset=len(snapshots[-1]),
        last_complete_newline=stat.st_size,
        stat_size=stat.st_size,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        payload=append_payload,
        payload_hash=hashlib.sha256(append_payload).hexdigest(),
        cursor_fingerprint="full-2",
        bytes_read=len(append_payload),
    )


def test_watcher_append_reports_cohort_memory_io_and_batch_evidence(tmp_path: Path) -> None:
    """Real watcher append invokes cohort classification and reads every full blob."""
    plan = _seed_cohort_and_append_plan(tmp_path)
    with append_cohort_memory_counter() as counter:
        result = ingest_append_plans(cast(Any, _owner(tmp_path)), [plan])

    assert result.succeeded == [plan]
    assert counter.batch_count == 1
    assert counter.plan_count == 1
    assert counter.calls_by_site["watcher_append_payload"] == 1
    assert counter.bytes_by_site["watcher_append_payload"] == len(plan.payload)
    assert counter.calls_by_site["classify_raw_revision_cohort"] == 1
    assert counter.calls_by_site["historical_full_blob.read_all"] == 3
    assert counter.bytes_by_site["historical_full_blob.read_all"] == sum(
        len(item) for item in _full_snapshots("append-memory-proof")
    )
    assert counter.calls_by_site["accepted_raw_ids"] == 1
    assert counter.bytes_by_site["accepted_raw_ids"] >= 1
    phase_names = [phase.name for phase in counter.phases]
    assert phase_names == [
        "watcher_append:before",
        "classify_raw_revision_cohort:before",
        "classify_raw_revision_cohort:after",
        "watcher_append:after",
    ], counter.summary()
    for phase in counter.phases:
        assert phase.batch_count == 1
        assert phase.plan_count == 1
    summary = counter.summary()
    for field in ("anon_pss=", "cgroup_anon=", "cgroup_file=", "io_read=", "io_write="):
        assert field in summary


def test_watcher_append_harness_fails_when_cohort_classification_is_removed(tmp_path: Path) -> None:
    """Anti-vacuity: the real append route must call its cohort classifier."""
    plan = _seed_cohort_and_append_plan(tmp_path)

    with patch.object(ArchiveStore, "classify_raw_revision_cohort", side_effect=AssertionError("cohort route removed")):
        result = ingest_append_plans(cast(Any, _owner(tmp_path)), [plan])

    assert result.succeeded == []
    assert result.failed == [plan]
