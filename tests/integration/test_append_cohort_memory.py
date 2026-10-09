"""Evidence harness for the live watcher append/cohort memory incident.

Static trace from the live daemon:

| Candidate | Production site | Trigger | Measured signal |
| --- | --- | --- | --- |
| H1 | ``ingest_append_plans`` | watcher append batch | batch/plan counts and process phases |
| H2 | ``raw_append_revision_parent`` | append authority decision | metadata calls and historical full-blob bytes |
| H3 | ``raw_revision_replay_plan`` | accepted append replay | metadata-plan calls and replayed raw bytes |

The incident was a watcher append that reread every retained full snapshot
through a cohort classifier.  The append route now decides authority from
durable metadata alone and classifies no cohort.  This scenario seeds an
already-proven full snapshot plus a live append, then executes the production
watcher entrypoint.  It reports anon-PSS, cgroup anon/file, process I/O
deltas, and batch counts at phase boundaries.  Host-dependent numbers have no
CI budget; route and byte-count assertions keep the harness non-vacuous.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.daemon.derivation import DerivationReport, Outcome
from polylogue.sources.live.batch_support import _AppendPlan
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.revision_backfill import RetainedPreparationNoProgressError
from polylogue.storage.sqlite.archive_tiers import revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.append_cohort_memory_counter import append_cohort_memory_counter
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner, run_owned_append_plans


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


def _admit_full_snapshots(
    archive_root: Path,
    source_path: Path,
    session_id: str,
    snapshots: list[bytes],
    authorities: list[RawRevisionAuthority],
) -> None:
    """Admit each full revision through the canonical raw writer on the admitted writer."""

    def acquire() -> None:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            for index, (payload, authority) in enumerate(zip(snapshots, authorities, strict=True)):
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    capture_mode=Provider.CODEX,
                    payload=payload,
                    source_path=str(source_path),
                    canonical_source_path=str(source_path),
                    source_index=0,
                    acquired_at_ms=index + 1,
                    revision=RawRevisionEnvelope(
                        f"codex-session:{session_id}",
                        RawRevisionKind.FULL,
                        f"full-{index}",
                        index,
                        authority=authority,
                    ),
                )

    asyncio.run(run_archive_fixture_write(archive_root, acquire))


def _seed_cohort_and_append_plan(
    archive_root: Path,
    *,
    session_id: str = "append-memory-proof",
    full_authority: RawRevisionAuthority = RawRevisionAuthority.BYTE_PROVEN,
) -> _AppendPlan:
    initialize_active_archive_root(archive_root)
    snapshots = _full_snapshots(session_id)
    source_path = archive_root / "captures" / f"{session_id}.jsonl"
    source_path.parent.mkdir(exist_ok=True)
    append_payload = f'{{"type":"session_meta","payload":{{"id":"{session_id}"}}}}\n'.encode() + _codex_record(
        session_id, "append", "a" * 16_384
    )
    source_path.write_bytes(snapshots[-1] + append_payload)
    _admit_full_snapshots(archive_root, source_path, session_id, snapshots, [full_authority] * len(snapshots))
    stat = source_path.stat()
    return _AppendPlan(
        path=source_path,
        canonical_source_path=str(source_path),
        captured_profile_key=None,
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
        # The watcher plans an append against the session it already bound;
        # append acquisition refuses a plan without that identity.
        native_id_hint=session_id,
    )


def _seed_partially_classified_cohort_and_append_plan(archive_root: Path) -> _AppendPlan:
    """Seed a newer asserted full that temporarily hides the new append."""
    initialize_active_archive_root(archive_root)
    session_id = "append-partial-classification-proof"
    snapshots = _full_snapshots(session_id)
    source_path = archive_root / "captures" / "append-partial-classification-proof.jsonl"
    source_path.parent.mkdir()
    append_payload = f'{{"type":"session_meta","payload":{{"id":"{session_id}"}}}}\n'.encode() + _codex_record(
        session_id, "append", "a" * 16_384
    )
    source_path.write_bytes(snapshots[-1] + append_payload)
    _admit_full_snapshots(
        archive_root,
        source_path,
        session_id,
        snapshots,
        [
            RawRevisionAuthority.BYTE_PROVEN if index < len(snapshots) - 1 else RawRevisionAuthority.ASSERTED
            for index in range(len(snapshots))
        ],
    )
    stat = source_path.stat()
    return _AppendPlan(
        path=source_path,
        canonical_source_path=str(source_path),
        captured_profile_key=None,
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
        # The watcher plans an append against the session it already bound;
        # append acquisition refuses a plan without that identity.
        native_id_hint=session_id,
    )


def test_watcher_append_uses_durable_replay_metadata_without_historical_full_reads(tmp_path: Path) -> None:
    """An established append cohort decides authority from metadata and replays only its tail."""
    plan = _seed_cohort_and_append_plan(tmp_path)

    # The owned writer runs the whole append route on its worker thread. A
    # parallel reader would record from a second thread.
    recording_threads: set[int] = set()
    with append_cohort_memory_counter() as counter:
        record = counter.record

        def record_on_one_thread(site: str, byte_count: int = 0) -> None:
            recording_threads.add(threading.get_ident())
            record(site, byte_count)

        with patch.object(counter, "record", side_effect=record_on_one_thread):
            result = run_owned_append_plans(tmp_path, _owner(tmp_path), [plan])
        counter.snapshot("quiescent")

    receipt = counter.workload_receipt(
        profile_id="workload-profile:append-cohort-canary",
        archive_id="archive:test:append-cohort",
        build_id="git:test",
        runtime_id="python:test",
        generation_id="synthetic:append-cohort",
    )

    assert result.succeeded == [plan]
    assert len(recording_threads) == 1
    assert counter.batch_count == 1
    assert counter.plan_count == 1
    assert counter.calls_by_site["watcher_append_payload"] == 1
    assert counter.bytes_by_site["watcher_append_payload"] == len(plan.payload)
    # The authority decision (bind) and the terminal receipt each look up the
    # append's durable parent; neither reads a retained full snapshot.
    assert counter.calls_by_site["raw_append_revision_parent"] == 2
    assert counter.calls_by_site["historical_full_blob.read_all"] == 0
    # Replay reuses the append plan's in-memory payload, so it reads no blob at
    # all. The bounded invariant is that it never touches every historical
    # full snapshot: three retained snapshots would exceed this bound.
    assert (
        counter.bytes_by_site["replay_raw_blob.read_all"]
        <= 4 * (len(_full_snapshots("append-memory-proof")[-1]) + len(plan.payload)) + 128
    )
    phase_names = [phase.name for phase in counter.phases]
    assert phase_names == [
        "watcher_append:before",
        "watcher_append:after",
        "quiescent",
    ], counter.summary()
    for phase in counter.phases:
        assert phase.batch_count == 1
        assert phase.plan_count == 1
    summary = counter.summary()
    for field in ("anon_pss=", "cgroup_anon=", "cgroup_file=", "io_read=", "io_write="):
        assert field in summary
    assert receipt.spec.inputs[0].profile_id == "workload-profile:append-cohort-canary"
    assert receipt.phases[-2].peak_rss_bytes is not None
    assert receipt.phases[-2].quiescent is True
    assert receipt.phases[-1].file_cache_bytes is not None


def test_watcher_append_does_not_reclassify_an_established_cohort(tmp_path: Path) -> None:
    """Anti-vacuity: routing the append through a cohort classifier breaks this route."""
    plan = _seed_cohort_and_append_plan(tmp_path)

    original = revision_governance._classify_full_revision_byte_inputs
    classifications: list[tuple[int, int]] = []

    def observe_classification(rows: Any, open_input: Any) -> Any:
        opened = 0

        def observe_open(raw_id: str, blob_hash: str) -> Any:
            nonlocal opened
            opened += 1
            return open_input(raw_id, blob_hash)

        output = original(rows, observe_open)
        classifications.append((len(rows), opened))
        return output

    with patch.object(revision_governance, "_classify_full_revision_byte_inputs", side_effect=observe_classification):
        result = run_owned_append_plans(tmp_path, _owner(tmp_path), [plan])

    assert result.succeeded == [plan]
    assert classifications == [], classifications


def test_watcher_append_counter_preserves_multi_plan_batch(tmp_path: Path) -> None:
    """Observation wraps one production batch instead of serializing its plans."""
    first = _seed_cohort_and_append_plan(tmp_path)
    second = _seed_cohort_and_append_plan(tmp_path, session_id="append-memory-proof-second")

    with append_cohort_memory_counter() as counter:
        result = run_owned_append_plans(tmp_path, _owner(tmp_path), [first, second])

    assert result.succeeded == [first, second]
    assert counter.batch_count == 1
    assert counter.plan_count == 2
    assert counter.calls_by_site["watcher_append_payload"] == 1
    assert counter.calls_by_site["raw_append_revision_parent"] == 4
    assert counter.calls_by_site["historical_full_blob.read_all"] == 0


def test_watcher_append_defers_incomplete_cohort_without_historical_reads(tmp_path: Path) -> None:
    """Incomplete metadata keeps authority proof and must not advance the append cursor."""
    plan = _seed_cohort_and_append_plan(tmp_path, full_authority=RawRevisionAuthority.ASSERTED)

    with append_cohort_memory_counter() as counter:
        result = run_owned_append_plans(tmp_path, _owner(tmp_path), [plan])

    assert result.succeeded == []
    assert result.deferred == [plan]
    assert result.failed == []
    # Deferral is decided from durable metadata: the retained asserted
    # snapshots are never reread to prove or refuse the append. Its replay
    # publishes nothing, so no terminal receipt is looked up afterwards.
    assert counter.calls_by_site["raw_append_revision_parent"] == 1
    assert counter.calls_by_site["historical_full_blob.read_all"] == 0
    # A deferred append's bytes are sound: it is never settled as a refusal.
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT parse_error FROM raw_sessions WHERE revision_kind = 'append'").fetchall() == [
            (None,)
        ]


def test_convergence_refuses_an_append_over_an_asserted_baseline_as_no_progress(tmp_path: Path) -> None:
    """Anti-vacuity: let the replay report success and the key fails as a broken publication.

    The append extends a baseline whose authority is only asserted, so the
    cohort has no accepted chain and the replay applies nothing. The outcome
    must be the typed no-progress refusal, not a success with no effect.
    """
    plan = _seed_cohort_and_append_plan(tmp_path, full_authority=RawRevisionAuthority.ASSERTED)
    assert run_owned_append_plans(tmp_path, _owner(tmp_path), [plan]).deferred == [plan]
    with sqlite3.connect(tmp_path / "source.db") as source:
        (append_raw,) = source.execute("SELECT raw_id FROM raw_sessions WHERE revision_kind = 'append'").fetchone()

    async def converge() -> DerivationReport:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            return await owner.converge_raw_id(append_raw)

    report = asyncio.run(converge())

    (outcome,) = report.outcomes
    assert outcome.outcome is Outcome.FAILED
    assert outcome.transient is False
    assert outcome.error is not None and RetainedPreparationNoProgressError.__name__ in outcome.error
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone() == (0,)


def test_watcher_append_defers_when_a_newer_asserted_full_hides_the_append(tmp_path: Path) -> None:
    """A newer asserted full must not let the watcher advance an omitted append cursor."""
    plan = _seed_partially_classified_cohort_and_append_plan(tmp_path)

    with append_cohort_memory_counter() as counter:
        result = run_owned_append_plans(tmp_path, _owner(tmp_path), [plan])

    assert result.succeeded == []
    assert result.deferred == [plan]
    assert counter.calls_by_site["raw_append_revision_parent"] == 2
    assert counter.calls_by_site["historical_full_blob.read_all"] == 0
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "index.db") as index:
        (append_raw,) = source.execute("SELECT raw_id FROM raw_sessions WHERE revision_kind = 'append'").fetchone()
        (proven_head,) = source.execute(
            "SELECT raw_id FROM raw_sessions WHERE source_revision = 'full-1' AND revision_authority = 'byte_proven'"
        ).fetchone()
        # Single-pass replay publishes the byte-proven prefix; the head stops
        # there and the append past the asserted full stays deferred.
        assert index.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
            ("codex-session:append-partial-classification-proof",),
        ).fetchone() == (proven_head,)
        assert index.execute(
            "SELECT decision FROM raw_revision_applications WHERE raw_id = ?", (append_raw,)
        ).fetchone() == ("deferred",)
