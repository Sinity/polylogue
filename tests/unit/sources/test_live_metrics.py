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
        ingested_session_count=2,
        ingested_message_count=12,
        changed_session_count=1,
        wal_bytes_before_checkpoint_max=4096,
        wal_bytes_after_checkpoint_max=1024,
        wal_checkpointed_pages_total=3,
        wal_busy_pages_total=1,
        wal_checkpoint_elapsed_s=0.04,
        wal_checkpoint_modes={"passive": 1},
        wal_checkpoint_errors=["database is locked"],
        cgroup_path="/user.slice/test.scope",
        cgroup_memory_peak_mb=128.0,
        new_sessions=(("codex", "codex:conv-1"),),
        updated_sessions=(("claude-code", "claude-code:conv-2"),),
    )

    payload = metrics.to_payload()

    assert payload["cgroup_path"] == "/user.slice/test.scope"
    assert payload["cgroup_memory_peak_mb"] == 128.0
    assert payload["ingested_session_count"] == 2
    assert payload["ingested_message_count"] == 12
    assert payload["changed_session_count"] == 1
    assert payload["wal_bytes_before_checkpoint_max"] == 4096
    assert payload["wal_bytes_after_checkpoint_max"] == 1024
    assert payload["wal_checkpointed_pages_total"] == 3
    assert payload["wal_busy_pages_total"] == 1
    assert payload["wal_checkpoint_elapsed_s"] == 0.04
    assert payload["wal_checkpoint_modes"] == {"passive": 1}
    assert payload["wal_checkpoint_errors"] == ["database is locked"]
    # Identity-scoped session touches (polylogue-20d.13): real refs, not an
    # unscoped aggregate, so consumers can tell session A from session B.
    assert payload["new_sessions"] == [{"source_name": "codex", "session_id": "codex:conv-1"}]
    assert payload["updated_sessions"] == [{"source_name": "claude-code", "session_id": "claude-code:conv-2"}]


def test_live_batch_metrics_defaults_to_empty_session_touches() -> None:
    metrics = LiveBatchMetrics(
        queued_file_count=0,
        needed_file_count=0,
        skipped_file_count=0,
        succeeded_file_count=0,
        failed_file_count=0,
        source_group_count=0,
        input_bytes=0,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        ingest_worker_count_max=0,
        append_file_count=0,
        full_file_count=0,
        archive_bytes_before=0,
        archive_bytes_after=0,
        archive_write_bytes_delta=0,
        parse_time_s=0.0,
        convergence_time_s=0.0,
        total_time_s=0.0,
    )

    payload = metrics.to_payload()

    assert payload["new_sessions"] == []
    assert payload["updated_sessions"] == []
