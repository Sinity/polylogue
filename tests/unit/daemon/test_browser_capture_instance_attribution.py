"""Concurrent browser-capture instance delivery regressions."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import (
    BrowserCaptureWriteResult,
    capture_dedup_content_hash,
    write_capture_envelope,
)


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


def test_concurrent_extension_instances_deduplicate_without_corrupting_spool(tmp_path: Path) -> None:
    """The real receiver writer ignores observer-only backfill attribution."""
    first = _envelope("extension-instance-a", backfill_job_id="job-a")
    second = _envelope("extension-instance-b", backfill_job_id="job-b")
    assert capture_dedup_content_hash(first) == capture_dedup_content_hash(second)

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
