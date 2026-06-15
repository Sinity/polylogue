"""Tests for ``devtools ingest-amplification-probe`` (#1851)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.ingest_amplification_probe import (
    REPORT_VERSION,
    main,
    measure_ingest_amplification,
)

_COMPONENTS = {"source", "index", "embeddings", "user", "ops", "blob_store"}
_DELTA_KEYS = {"allocated_bytes", "wal_bytes", "total_bytes"}


def test_measure_emits_expected_shape(tmp_path: Path) -> None:
    report = measure_ingest_amplification(batches=3, seed=7, workdir=tmp_path)

    assert report["ok"] is True
    assert report["report_version"] == REPORT_VERSION
    assert report["tool"] == "ingest-amplification-probe"
    assert set(report["components"]) == _COMPONENTS
    assert report["tiers"] == ["source", "index", "embeddings", "user", "ops"]

    batches = report["batches"]
    assert len(batches) == 3
    for index, batch in enumerate(batches):
        assert batch["batch_index"] == index
        assert batch["payload_bytes"] > 0
        assert batch["sessions_ingested"] == 1
        assert batch["messages_ingested"] >= 1
        # Every component carries a full byte-delta triple.
        assert set(batch["component_byte_delta"]) == _COMPONENTS
        for delta in batch["component_byte_delta"].values():
            assert set(delta) == _DELTA_KEYS
        # The reported ratio matches the attributed bytes / payload.
        expected_ratio = round(batch["total_bytes_written"] / batch["payload_bytes"], 4)
        assert batch["amplification_ratio"] == expected_ratio


def test_summary_attributes_bytes_and_shares(tmp_path: Path) -> None:
    report = measure_ingest_amplification(batches=4, seed=11, workdir=tmp_path)
    summary = report["summary"]

    assert summary["batch_count"] == 4
    assert summary["total_payload_bytes"] > 0
    # The parsed index tier is the durable write target on this ingest path.
    assert summary["total_bytes_written"] > 0
    assert summary["per_component_total_bytes"]["index"] > 0
    assert set(summary["per_component_total_bytes"]) == _COMPONENTS

    # Shares are fractions of the attributed bytes and sum to ~1.0.
    shares = summary["per_component_share"]
    assert set(shares) == _COMPONENTS
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)

    assert summary["overall_amplification_ratio"] == pytest.approx(
        summary["total_bytes_written"] / summary["total_payload_bytes"], abs=1e-3
    )


def test_main_json_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--json", "--batches", "2", "--seed", "3", "--workdir", str(tmp_path)])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert len(payload["batches"]) == 2


def test_measurement_is_deterministic(tmp_path: Path) -> None:
    first = measure_ingest_amplification(batches=3, seed=99, workdir=tmp_path / "a")
    second = measure_ingest_amplification(batches=3, seed=99, workdir=tmp_path / "b")

    assert first["summary"] == second["summary"]
    assert [b["total_bytes_written"] for b in first["batches"]] == [b["total_bytes_written"] for b in second["batches"]]


def test_rejects_unavailable_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not available"):
        measure_ingest_amplification(provider="nope-not-real", batches=1, workdir=tmp_path)
