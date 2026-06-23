from __future__ import annotations

from pathlib import Path
from typing import Any

from polylogue.sources.live.batch_observability import record_attempt_progress


class _RecordingCursor:
    def __init__(self) -> None:
        self.progress_kwargs: dict[str, Any] | None = None
        self.event_kwargs: dict[str, Any] | None = None

    def update_ingest_attempt(self, attempt_id: str, **kwargs: Any) -> bool:
        self.progress_kwargs = {"attempt_id": attempt_id, **kwargs}
        return True

    def record_ingest_stage_event(self, attempt_id: str, **kwargs: Any) -> bool:
        self.event_kwargs = {"attempt_id": attempt_id, **kwargs}
        return True


def test_record_attempt_progress_preserves_stage_payload_in_durable_event() -> None:
    cursor = _RecordingCursor()

    record_attempt_progress(
        cursor,
        "attempt-1",
        phase="full_archive_write",
        status="running",
        queued_file_count=1,
        needed_file_count=1,
        succeeded_file_count=0,
        failed_file_count=0,
        input_bytes=128,
        source_payload_read_bytes=64,
        cursor_fingerprint_read_bytes=32,
        archive_write_bytes_delta=16,
        parse_time_s=1.25,
        current_source="chatgpt",
        current_path=Path("/archive/browser-capture/capture.json"),
        stage_payload={
            "storage_route": "archive_full",
            "storage_write_tiers": "source,index",
            "payload_available_file_count": 1,
        },
    )

    assert cursor.progress_kwargs is not None
    assert cursor.progress_kwargs["stage_payload"] == {
        "storage_route": "archive_full",
        "storage_write_tiers": "source,index",
        "payload_available_file_count": 1,
    }
    assert cursor.event_kwargs is not None
    assert cursor.event_kwargs["stage_payload"] == {
        "storage_route": "archive_full",
        "storage_write_tiers": "source,index",
        "payload_available_file_count": 1,
    }
