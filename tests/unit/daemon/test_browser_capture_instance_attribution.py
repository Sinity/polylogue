"""Concurrent browser-capture instance delivery regressions."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import (
    BrowserCaptureWriteResult,
    summarize_capture_envelope,
    write_capture_envelope,
)
from polylogue.sources.live import WatchSource
from polylogue.sources.source_layout import export_drop_layout
from tests.infra.live_batch import prepared_live_batch_processor


def _envelope(instance_id: str, *, backfill_job_id: str) -> BrowserCaptureEnvelope:
    return BrowserCaptureEnvelope.model_validate(
        {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "provenance": {
                "source_url": "https://chatgpt.com/c/concurrent-1",
                "captured_at": "2026-07-12T20:00:00+00:00",
                "adapter_name": "chatgpt-native",
                "extension_instance_id": instance_id,
            },
            "session": {
                "provider": "chatgpt",
                "provider_session_id": "concurrent-1",
                "provider_meta": {
                    "capture_fidelity": "native_full",
                    "backfill": {
                        "job_id": backfill_job_id,
                        "queue_id": f"queue-{backfill_job_id}",
                        "instance_id": instance_id,
                    },
                },
                "turns": [{"provider_turn_id": "m1", "role": "assistant", "text": "same snapshot"}],
            },
            "provider_meta": {
                "backfill": {
                    "job_id": backfill_job_id,
                    "queue_id": f"queue-{backfill_job_id}",
                    "instance_id": instance_id,
                }
            },
        }
    )


def _write_capture_in_process(payload: dict[str, object], spool_path: str, barrier: Any, result_queue: Any) -> None:
    """Run one receiver write in a fresh interpreter for the flock regression."""
    try:
        envelope = BrowserCaptureEnvelope.model_validate(payload)
        barrier.wait(timeout=10)
        result = write_capture_envelope(envelope, spool_path=Path(spool_path))
        result_queue.put({"deduplicated": result.deduplicated})
    except BaseException as exc:
        result_queue.put({"error": repr(exc)})


def test_multiple_receiver_processes_deduplicate_without_corrupting_spool(tmp_path: Path) -> None:
    """The advisory spool lock serializes writers beyond one receiver process."""
    first = _envelope("extension-instance-a", backfill_job_id="job-a")
    second = _envelope("extension-instance-b", backfill_job_id="job-b")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_write_capture_in_process,
            args=(envelope.model_dump(mode="json"), str(tmp_path), barrier, result_queue),
        )
        for envelope in (first, second)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    outcomes = [result_queue.get(timeout=5) for _ in processes]
    assert all("error" not in outcome for outcome in outcomes)
    assert {outcome["deduplicated"] for outcome in outcomes} == {False, True}
    artifacts = list(tmp_path.rglob("*.json"))
    assert len(artifacts) == 1
    assert BrowserCaptureEnvelope.model_validate_json(artifacts[0].read_bytes()).provider_session_id == "concurrent-1"
    assert not list(tmp_path.rglob(".*.tmp"))


@pytest.mark.asyncio
async def test_concurrent_extension_instances_deduplicate_without_corrupting_spool(tmp_path: Path) -> None:
    """Concurrent receiver writes converge to one artifact and archived session."""
    first = _envelope("extension-instance-a", backfill_job_id="job-a")
    second = _envelope("extension-instance-b", backfill_job_id="job-b")
    assert summarize_capture_envelope(first).dedup_content_hash == summarize_capture_envelope(second).dedup_content_hash

    barrier = Barrier(2)

    def post(envelope: BrowserCaptureEnvelope) -> BrowserCaptureWriteResult:
        barrier.wait()
        return write_capture_envelope(envelope, spool_path=tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(post, (first, second)))

    assert {result.deduplicated for result in results} == {False, True}
    assert len(list(tmp_path.rglob("*.json"))) == 1
    payload = json.loads(results[0].path.read_text(encoding="utf-8"))
    assert payload["provenance"]["extension_instance_id"] in {
        "extension-instance-a",
        "extension-instance-b",
    }
    assert not list(tmp_path.rglob(".*.tmp"))

    # The supplied live owners: writer, retained publication and convergence,
    # on an archive the fixture bootstraps with its durable Source tier.
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    async with prepared_live_batch_processor(
        archive_root,
        (WatchSource(name="browser-capture", root=tmp_path, layout=export_drop_layout((".json",))),),
        parser_fingerprint="test-parser",
    ) as processor:
        metrics = await processor.ingest_files([results[0].path], emit_event=False)
    with sqlite3.connect(archive_root / "index.db") as conn:
        archived_count = conn.execute("SELECT COUNT(*) FROM sessions WHERE native_id = 'concurrent-1'").fetchone()[0]
    assert metrics.ingested_session_count == 1
    assert archived_count == 1
