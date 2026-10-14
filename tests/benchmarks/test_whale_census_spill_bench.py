"""Measure actual managed retained preparation/publication of large message values.

The retired pickle cache and resident-whale alternatives have no current
producer. This measurement binds the shared Raw worker route, records actual
elapsed time and process memory, and verifies complete output bytes. It does
not claim scalar-independent RSS from this synthetic input size.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sqlite3
from pathlib import Path
from time import perf_counter

import pytest

from devtools.measurement_receipts import emit_receipt
from polylogue.core.metrics import read_current_rss_mb, read_peak_rss_self_mb
from polylogue.operations.session_records import read_session_records
from polylogue.sources import revision_backfill
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.retained_replay import replay_retained_components
from tests.infra.revision_backfill_benchmark import build_whale_bearing_corpus


@pytest.mark.benchmark
@pytest.mark.parametrize("whale_payload_bytes", [2_000_000, 4_000_000])
def test_managed_whale_replay_records_complete_output_and_actual_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, whale_payload_bytes: int
) -> None:
    root = tmp_path / "archive"
    small_raw_ids, whale_raw_id = build_whale_bearing_corpus(
        root,
        small_raw_count=6,
        small_avg_payload_bytes=5_000,
        whale_payload_bytes=whale_payload_bytes,
    )
    with sqlite3.connect(root / "source.db") as source:
        hashes = {
            str(raw_id): bytes(blob_hash).hex()
            for raw_id, blob_hash in source.execute("SELECT raw_id, blob_hash FROM raw_sessions")
        }
    store = BlobStore(root / "blob")
    expected = {}
    for raw_id, blob_hash in hashes.items():
        text = json.loads(store.read_all(blob_hash).splitlines()[-1])["payload"]["content"][0]["text"]
        expected[raw_id] = hashlib.sha256(text.encode()).hexdigest()
    del text
    monkeypatch.setattr(
        revision_backfill,
        "parse_retained_raw_sessions",
        lambda *_args: pytest.fail("managed publication parsed inline instead of consuming its sealed worker carrier"),
    )
    before_cpu = resource.getrusage(resource.RUSAGE_SELF)
    rss_before = read_current_rss_mb()
    started = perf_counter()
    result = replay_retained_components(root)
    elapsed = perf_counter() - started
    rss_after = read_current_rss_mb()
    peak_process_lifetime = read_peak_rss_self_mb()
    after_cpu = resource.getrusage(resource.RUSAGE_SELF)
    assert result.quarantined == result.adoption_deferred == 0
    observed = {}
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        sessions = tuple(archive._conn.execute("SELECT session_id, raw_id FROM sessions"))
        assert {str(row[1]) for row in sessions} == {*small_raw_ids, whale_raw_id}
        for session_id, raw_id in sessions:
            page = read_session_records(archive, str(session_id), limit=None, offset=0)
            assert page is not None and page.lineage_complete
            assert len(page.session.messages) == 1
            observed[str(raw_id)] = hashlib.sha256(page.session.messages[0].text.encode()).hexdigest()
    assert observed == expected
    receipt = emit_receipt(
        f"managed-retained-whale-{whale_payload_bytes}",
        {
            "route": "make_raw_observation_derivation.compute/publish",
            "input": {
                "raw_count": len(hashes),
                "whale_payload_bytes": whale_payload_bytes,
                "blob_hashes": sorted(hashes.values()),
            },
            "resources": {
                "elapsed_s": elapsed,
                "self_cpu_s": after_cpu.ru_utime + after_cpu.ru_stime - before_cpu.ru_utime - before_cpu.ru_stime,
                "rss_before_mib": rss_before,
                "rss_after_mib": rss_after,
                "peak_rss_self_process_lifetime_mib": peak_process_lifetime,
            },
            "output": {
                "session_count": len(observed),
                "message_sha256": observed,
                "quarantined": result.quarantined,
                "adoption_deferred": result.adoption_deferred,
            },
            "memory_qualification": "unqualified: process-lifetime peak and finite synthetic sizes do not establish whale RSS",
        },
    )
    print(f"managed retained whale resource receipt: {receipt}")
