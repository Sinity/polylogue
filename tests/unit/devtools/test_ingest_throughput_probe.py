"""Tests for ``devtools bench ingest-throughput``.

Throughput is host-variable, so these assertions only cover the report shape,
type wellformedness, and the *deterministic* count fields — never wall-clock
durations or messages/second values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.ingest_throughput_probe import (
    REPORT_VERSION,
    main,
    measure_ingest_throughput,
)

_PER_BATCH_MS_KEYS = {"min", "max", "mean", "p90"}


@pytest.mark.parametrize("provider", ["codex", "chatgpt"])
def test_measure_emits_expected_shape(provider: str, tmp_path: Path) -> None:
    report = measure_ingest_throughput(provider=provider, batches=3, seed=7, workdir=tmp_path)

    assert report["ok"] is True
    assert report["report_version"] == REPORT_VERSION
    assert report["tool"] == "bench ingest-throughput"
    assert report["provider"] == provider
    assert report["batches"] == 3
    assert report["seed"] == 7

    # Deterministic count fields are non-negative ints.
    for key in ("total_sessions", "total_messages", "batches", "seed", "messages_min", "messages_max"):
        assert isinstance(report[key], int)
        assert report[key] >= 0
    assert report["total_sessions"] == 3
    assert report["total_messages"] >= 3

    # Timing fields are non-negative floats; values themselves are not asserted.
    for key in ("total_wall_s", "messages_per_s", "sessions_per_s"):
        assert isinstance(report[key], float)
        assert report[key] >= 0.0

    per_batch_ms = report["per_batch_ms"]
    assert set(per_batch_ms) == _PER_BATCH_MS_KEYS
    for value in per_batch_ms.values():
        assert isinstance(value, float)
        assert value >= 0.0
    # min <= mean <= max (allow equality).
    assert per_batch_ms["min"] <= per_batch_ms["mean"] <= per_batch_ms["max"]

    per_batch = report["per_batch"]
    assert len(per_batch) == 3
    for index, batch in enumerate(per_batch):
        assert batch["batch_index"] == index
        assert batch["sessions_ingested"] == 1
        assert batch["messages_ingested"] >= 1
        assert isinstance(batch["batch_ms"], float)
        assert batch["batch_ms"] >= 0.0

    # Workload tag is the corpus path for these fixtures.
    assert report["workload"] == "corpus"

    # Per-stage attribution is populated for a real ingest run.
    stage_timings = report["stage_timings_s"]
    assert isinstance(stage_timings, dict)
    assert stage_timings  # non-empty
    for stage_name, seconds in stage_timings.items():
        assert isinstance(stage_name, str)
        assert isinstance(seconds, float)
        assert seconds >= 0.0

    top_stages = report["top_stages"]
    assert isinstance(top_stages, list)
    assert top_stages  # non-empty when stage_timings is non-empty
    for entry in top_stages:
        assert set(entry) >= {"stage", "seconds", "pct_of_total_stage_time"}
        assert isinstance(entry["stage"], str)
        assert isinstance(entry["seconds"], float)
        assert entry["seconds"] >= 0.0
        assert 0.0 <= entry["pct_of_total_stage_time"] <= 100.0
    assert len(top_stages) <= 8

    # CPU / memory headline metrics — shape and non-negativity only.
    assert isinstance(report["cpu_seconds_total"], float)
    assert report["cpu_seconds_total"] >= 0.0
    assert isinstance(report["cpu_utilization"], float)
    assert report["cpu_utilization"] >= 0.0
    assert isinstance(report["peak_rss_mb"], float)
    assert report["peak_rss_mb"] > 0.0

    # rusage-derived resource deltas are present and well-typed.
    resources = report["resources"]
    assert isinstance(resources, dict)
    for float_key in ("ru_utime_s", "ru_stime_s"):
        assert isinstance(resources[float_key], float)
        assert resources[float_key] >= 0.0
    for int_key in (
        "ru_minflt_delta",
        "ru_majflt_delta",
        "ru_inblock_delta",
        "ru_oublock_delta",
    ):
        assert isinstance(resources[int_key], int)

    # /proc/self/io is best-effort; the flag governs whether the dict is filled.
    assert isinstance(report["proc_io_available"], bool)
    proc_io = report["proc_io"]
    assert isinstance(proc_io, dict)
    if report["proc_io_available"]:
        for field in ("rchar", "wchar", "read_bytes", "write_bytes", "syscr", "syscw"):
            assert isinstance(proc_io[field], int)
        assert isinstance(proc_io["write_mb"], float)
        assert isinstance(proc_io["read_mb"], float)

    # SQLite storage growth is reported with non-negative sizes.
    storage = report["storage"]
    assert isinstance(storage, dict)
    for size_key in ("index_db_bytes", "index_wal_peak_bytes", "source_db_bytes"):
        assert isinstance(storage[size_key], int)
        assert storage[size_key] >= 0
    for ratio_key in ("bytes_written_per_message", "db_growth_per_message"):
        assert isinstance(storage[ratio_key], float)


def test_counts_are_deterministic(tmp_path: Path) -> None:
    first = measure_ingest_throughput(provider="codex", batches=3, seed=99, workdir=tmp_path / "a")
    second = measure_ingest_throughput(provider="codex", batches=3, seed=99, workdir=tmp_path / "b")

    # Counts are deterministic for fixed (provider, batches, seed); timings are not.
    assert first["total_sessions"] == second["total_sessions"]
    assert first["total_messages"] == second["total_messages"]
    assert [b["messages_ingested"] for b in first["per_batch"]] == [b["messages_ingested"] for b in second["per_batch"]]


def test_main_json_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--json", "--batches", "2", "--seed", "3", "--workdir", str(tmp_path)])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["tool"] == "bench ingest-throughput"
    assert len(payload["per_batch"]) == 2


def test_rejects_unavailable_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not available"):
        measure_ingest_throughput(provider="nope-not-real", batches=1, workdir=tmp_path)


def test_lineage_workload_composes(tmp_path: Path) -> None:
    report = measure_ingest_throughput(
        lineage=True,
        batches=4,
        seed=7,
        messages_max=12,
        workdir=tmp_path,
    )

    assert report["ok"] is True
    assert report["workload"] == "lineage"
    # One parent + four forks were written.
    assert report["total_sessions"] == 5
    assert len(report["per_batch"]) == 4
    # total_messages reflects parent prefix + every fork's replayed prefix + tail.
    assert report["total_messages"] > report["messages_max"]
    # Stage attribution still populated through the direct ArchiveStore path.
    assert isinstance(report["stage_timings_s"], dict)
    assert report["stage_timings_s"]
    assert report["cpu_utilization"] >= 0.0
    assert report["peak_rss_mb"] > 0.0
