"""Diagnostic delivery loss is visible on the metrics surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.daemon import metrics
from polylogue.operations.storage_io_observation import IoPhaseObservation, StorageIoObservation


def test_metrics_expose_maintained_sink_loss_without_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Omitting the loss series or hiding drops as failures makes this red."""
    monkeypatch.setattr(
        metrics,
        "diagnostic_snapshot",
        lambda: {"queued": 3, "dropped": 7, "failures": 2, "delivered": 11, "undrained": 1, "high_water": 9},
    )
    body = metrics.format_metrics(tmp_path / "missing.db")
    assert 'polylogue_diagnostic_delivery_total{outcome="dropped"} 7' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="failures"} 2' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="delivered"} 11' in body
    assert 'polylogue_diagnostic_delivery_total{outcome="undrained"} 1' in body
    assert "polylogue_diagnostic_queue_depth 3" in body


def test_metrics_distinguish_measured_io_phases_from_unavailable_sqlite_internals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An uninstrumentable fsync must not masquerade as a measured zero."""
    monkeypatch.setattr(
        metrics,
        "storage_io_observation",
        lambda: StorageIoObservation(
            samples=(
                IoPhaseObservation("ops", "commit", True, True, 2, 250_000_000),
                IoPhaseObservation("ops", "commit", True, False, 1, 10_000_000),
            ),
            unavailable_phases=("sqlite_file_fsync",),
        ),
    )
    body = metrics.format_metrics(tmp_path / "missing.db")
    assert (
        'polylogue_storage_io_phase_total{inside_writer_lease="true",phase="commit",succeeded="true",tier="ops"} 2'
    ) in body
    assert (
        'polylogue_storage_io_phase_seconds_total{inside_writer_lease="true",phase="commit",'
        'succeeded="true",tier="ops"} 0.25'
    ) in body
    assert 'polylogue_storage_io_phase_observable{phase="sqlite_file_fsync"} 0' in body
    assert 'polylogue_storage_io_phase_observable{phase="commit"} 1' in body
