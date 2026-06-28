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
