from __future__ import annotations

from collections.abc import Sized
from typing import cast

from polylogue.sources.live.metrics import LIVE_BATCH_IDENTITY_LIST_LIMIT, LiveBatchMetrics


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


def test_split_offered_bytes_names_a_path_no_route_reached() -> None:
    """A file the time budget never attempted is refused, not silently absorbed.

    The 3.5 GB whale in the rehearsal receipts was offered as one chunk, left
    unattempted when the pass budget expired, and still counted in the bytes
    the run reported as input. Named as its own refusal reason it stops
    inflating throughput and says why.

    Anti-vacuity: drop the ``remaining`` bucket at the end of
    ``split_offered_bytes`` and the whale's bytes vanish from the split while
    still counting in the offered total -- the reconciliation below goes red.
    """
    from pathlib import Path

    from polylogue.sources.live.metrics import REFUSED_UNATTEMPTED_TIME_BUDGET, split_offered_bytes

    ingested_path = Path("/src/small.jsonl")
    failed_path = Path("/src/broken.jsonl")
    excluded_path = Path("/src/notes.json")
    whale = Path("/src/whale.jsonl")
    path_sizes = {ingested_path: 100, failed_path: 20, excluded_path: 30, whale: 3_557_000_000}

    ingested, failed, refused = split_offered_bytes(
        path_sizes,
        succeeded=[ingested_path],
        failed=[failed_path],
        excluded={excluded_path: "unsupported source class"},
        deferred=[],
        unattempted_reason=REFUSED_UNATTEMPTED_TIME_BUDGET,
    )

    assert ingested == 100
    assert failed == 20
    assert refused == {"unsupported source class": 30, REFUSED_UNATTEMPTED_TIME_BUDGET: 3_557_000_000}
    assert ingested + failed + sum(refused.values()) == sum(path_sizes.values())


def test_split_offered_bytes_counts_each_path_once() -> None:
    """A path reported under two outcomes is credited to the first only.

    Anti-vacuity: sum each bucket independently over the offered sizes instead
    of consuming from one shared pool, and a path that appears in both
    ``succeeded`` and ``failed`` is counted twice -- which makes the split
    exceed the offered total and lets a residual-derived refused bucket read
    as zero.
    """
    from pathlib import Path

    from polylogue.sources.live.metrics import REFUSED_UNATTEMPTED, split_offered_bytes

    contested = Path("/src/contested.jsonl")
    path_sizes = {contested: 512}

    ingested, failed, refused = split_offered_bytes(
        path_sizes,
        succeeded=[contested],
        failed=[contested],
        excluded={contested: "archive write skipped this raw"},
        deferred=[contested],
        unattempted_reason=REFUSED_UNATTEMPTED,
    )

    assert (ingested, failed, refused) == (512, 0, {})


def test_live_batch_metrics_payload_reports_excluded_files() -> None:
    """A refused file is a counted outcome of the batch, not an absence.

    Anti-vacuity: drop either key from ``to_payload`` and this goes red --
    which is the state every ``ingestion_batch`` event consumer, the daemon's
    ``last_ingestion_batch`` payload and the CLI ratio were in, so a
    time-budget refusal read as if nothing had been offered.
    """
    metrics = LiveBatchMetrics(
        queued_file_count=4,
        needed_file_count=4,
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
        excluded_file_count=3,
        excluded_reasons={"refused_unattempted_time_budget": 3},
    )

    payload = metrics.to_payload()

    assert payload["excluded_file_count"] == 3
    assert payload["excluded_reasons"] == {"refused_unattempted_time_budget": 3}


def test_live_batch_metrics_payload_caps_identity_lists_with_a_counted_omission() -> None:
    """One admitted export cannot amplify into an unbounded persisted event.

    Anti-vacuity: drop the cap and the payload embeds every identity, so the
    ``ingestion_batch`` event and ``last_ingestion_batch`` grow with the export
    rather than with the declared limit. Drop the ``*_omitted`` counts instead
    and a shortened list renders as if it were the whole truth.
    """
    over = LIVE_BATCH_IDENTITY_LIST_LIMIT + 7
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
        new_sessions=tuple(("src", f"s{index}") for index in range(over)),
        updated_sessions=tuple(("src", f"u{index}") for index in range(over)),
        changed_session_count=2 * over,
    )

    payload = metrics.to_payload()

    assert len(cast("Sized", payload["new_sessions"])) == LIVE_BATCH_IDENTITY_LIST_LIMIT
    assert payload["new_sessions_omitted"] == 7
    assert len(cast("Sized", payload["updated_sessions"])) == LIVE_BATCH_IDENTITY_LIST_LIMIT
    assert payload["updated_sessions_omitted"] == 7
    assert len(cast("Sized", payload["changed_session_ids"])) == LIVE_BATCH_IDENTITY_LIST_LIMIT
    assert payload["changed_session_ids_omitted"] == 2 * over - LIVE_BATCH_IDENTITY_LIST_LIMIT
    # The scalar counts stay exact; only the embedded identities are capped.
    assert payload["changed_session_count"] == 2 * over
    assert len(metrics.changed_session_ids) == 2 * over
