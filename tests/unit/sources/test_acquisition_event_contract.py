"""Acquisition decisions survive the event validator and renderer."""

from __future__ import annotations

import io
import json
from pathlib import Path

from polylogue.logging import add_sink, make_stream_sink, remove_sink
from polylogue.sources.live.acquisition_log import (
    AcquisitionStageTimings,
    log_file_acquisition_decision,
    log_unclaimed_file,
)


def test_acquisition_decision_fields_reach_json_sink() -> None:
    """Dropping source, bytes, evidence, or timings from the real route makes this red."""
    stream = io.StringIO()
    timings = AcquisitionStageTimings()
    timings.record("detect", 1.25)
    sink = add_sink(make_stream_sink(stream, fmt="json"))
    try:
        log_file_acquisition_decision(
            path=Path("/fixture/session.jsonl"),
            size=3072,
            mtime=1234.5,
            origin="codex-session",
            source_name="fixture",
            evidence="codex-envelope",
            stage_timings=timings,
        )
    finally:
        remove_sink(sink)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert not [record for record in records if record["event"] == "log.field_rejected"]
    record = next(record for record in records if record["event"] == "file_acquisition_decision")
    assert record["size"] == 3072
    assert record["mtime"] == 1234.5
    assert record["origin"] == "codex-session"
    assert record["source_name"] == "fixture"
    assert record["evidence"] == "codex-envelope"
    assert record["stage_timings_ms"] == {"detect": 1.25}


def test_unclaimed_reason_uses_quarantined_field() -> None:
    stream = io.StringIO()
    sink = add_sink(make_stream_sink(stream, fmt="json", redact=True))
    try:
        log_unclaimed_file(
            path="/fixture/unknown.jsonl",
            size=None,
            mtime=None,
            source_name="fixture",
            reason="unreadable: ValueError: possible source text",
        )
    finally:
        remove_sink(sink)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert not [record for record in records if record["event"] == "log.field_rejected"]
    record = next(record for record in records if record["event"] == "file_acquisition_unclaimed")
    assert record["reason"] == "unclaimed_file"
    assert "error_detail" not in record
