"""Cold measurement receipts preserve the observation, not the exporting host."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.infra.daemon_cold_start import _INSTRUMENTED_BOOTSTRAP, write_retained_measurement_receipt


def test_retained_measurement_carries_original_host(tmp_path: Path) -> None:
    host = {
        "python_implementation": "CPython",
        "python_version": "3.14.0",
        "machine": "synthetic-arm",
        "free_threaded": True,
    }
    receipt: dict[str, object] = {
        "fixture": {"rejected": 4096},
        "candidate": {"sha": "a" * 40},
        "host": host,
        "discovery_measurement": {},
        "intake_counts": {},
    }
    path = write_retained_measurement_receipt(receipt, tmp_path / "report.json")
    assert path is not None
    assert json.loads(path.read_text())["host"] == host
    del receipt["host"]
    with pytest.raises(AssertionError):
        write_retained_measurement_receipt(receipt, tmp_path / "other.json")


def test_discovery_trace_io_follows_measured_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: Any
) -> None:
    import polylogue.daemon.commands as daemon_cli
    import polylogue.operations.intake_adapters as adapters
    import polylogue.sources.live.discovery as discovery

    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("COLD_DISCOVERY_TRACE", str(trace))
    monkeypatch.delenv("COLD_HOLD_DISCOVERY", raising=False)
    # Register originals with monkeypatch so the actual bootstrap's assignments
    # are undone even if the anti-vacuity assertion fails inside its main call.
    monkeypatch.setattr(discovery, "_ordered_children", discovery._ordered_children)
    monkeypatch.setattr(adapters, "_source_path_steps", discovery._source_path_steps)
    namespace: dict[str, Any] = {}
    events = ("root_listing_start", "root_listing_end", "root_sort_end", "first_yielded_entry")

    def measured_main() -> None:
        for event in events:
            namespace["record"](event)
            frozen_clock.advance(0.25)
            assert not trace.exists()
        # Repeated traversal must not overwrite the first yielded timestamp.
        namespace["record"]("first_yielded_entry")

    monkeypatch.setattr(daemon_cli, "main", measured_main)
    exec(_INSTRUMENTED_BOOTSTRAP, namespace)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [row["name"] for row in rows] == list(events)
    assert [row["monotonic"] - rows[0]["monotonic"] for row in rows] == [0.0, 0.25, 0.5, 0.75]
