from __future__ import annotations

from polylogue.sources.live.metrics import LiveBatchMetrics


def test_live_batch_metrics_payload_includes_memory_pressure_fields() -> None:
    metrics = LiveBatchMetrics(
        queued_file_count=1,
        needed_file_count=1,
        skipped_file_count=0,
        succeeded_file_count=1,
        failed_file_count=0,
        source_group_count=1,
        input_bytes=100,
        source_payload_read_bytes=100,
        cursor_fingerprint_read_bytes=0,
        ingest_worker_count_max=1,
        append_file_count=0,
        full_file_count=1,
        archive_bytes_before=0,
        archive_bytes_after=10,
        archive_write_bytes_delta=10,
        parse_time_s=0.1,
        convergence_time_s=0.2,
        total_time_s=0.3,
        cgroup_path="/user.slice/test.scope",
        cgroup_memory_peak_mb=128.0,
    )

    payload = metrics.to_payload()

    assert payload["cgroup_path"] == "/user.slice/test.scope"
    assert payload["cgroup_memory_peak_mb"] == 128.0
