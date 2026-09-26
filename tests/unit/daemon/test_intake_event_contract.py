"""Structured intake page evidence through the production dispatcher and sink."""

from __future__ import annotations

import asyncio
import io
import json

from polylogue.daemon.intake import AdmissionOutcome, AdmissionResult, FairIntakeDispatcher, IntakeClassSpec, IntakeItem
from polylogue.logging import add_sink, make_stream_sink, remove_sink


class _Adapter:
    async def discover(self, *, limit: int) -> list[IntakeItem]:
        return [
            IntakeItem(item_id=str(index), class_name="fixture", estimated_cost=100 + index)
            for index in range(min(limit, 4))
        ]

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        return AdmissionResult(
            {
                "0": AdmissionOutcome.ADMITTED,
                "1": AdmissionOutcome.RETRYABLE,
                "2": AdmissionOutcome.EXCLUDED,
                "3": AdmissionOutcome.DEFERRED,
            }[item.item_id]
        )

    async def acknowledge(self, item: IntakeItem) -> None:
        pass


def test_intake_page_fields_survive_validation_and_json_sink() -> None:
    """Removing a byte or disposition from the real event makes this red."""
    stream = io.StringIO()
    sink = add_sink(make_stream_sink(stream, fmt="json"))
    try:
        dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="fixture", adapter=_Adapter(), page_size=4)])
        asyncio.run(dispatcher.run_once())
    finally:
        remove_sink(sink)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert not [record for record in records if record["event"] == "log.field_rejected"]
    event = next(record for record in records if record["event"] == "daemon.intake.page")
    assert {key: event[key] for key in ("files", "bytes", "succeeded", "failed", "retried", "refused", "deferred")} == {
        "files": 4,
        "bytes": 406,
        "succeeded": 1,
        "failed": 0,
        "retried": 1,
        "refused": 1,
        "deferred": 1,
    }
    assert event["duration_ms"] >= 0
    assert set(event["stage_timings_ms"]) == {"discovery", "admission"}
