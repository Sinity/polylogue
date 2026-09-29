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


class _PartialAdapter:
    async def discover(self, *, limit: int) -> list[IntakeItem]:
        return [IntakeItem(item_id=str(index), class_name="fixture", estimated_cost=10) for index in range(2)]

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        from polylogue.core.raw_failure_evidence import PARTIAL_TRUNCATED_TAIL, PartialAdmission

        if item.item_id == "0":
            return AdmissionResult(AdmissionOutcome.ADMITTED)
        return AdmissionResult(
            AdmissionOutcome.ADMITTED,
            partial=PartialAdmission(
                reason=PARTIAL_TRUNCATED_TAIL,
                complete_record_count=3,
                complete_prefix_bytes=120,
                source_bytes=150,
            ),
        )

    async def acknowledge(self, item: IntakeItem) -> None:
        pass


def test_a_partial_admission_is_counted_and_reported_apart_from_plain_success() -> None:
    """A partial admission counts as admitted and as partial, with a typed per-item event.

    Anti-vacuity: without the partial count the page reads ``ok`` with two
    plain successes, and the truncated tail of the second item is invisible.
    """
    stream = io.StringIO()
    sink = add_sink(make_stream_sink(stream, fmt="json"))
    try:
        dispatcher = FairIntakeDispatcher([IntakeClassSpec(name="fixture", adapter=_PartialAdapter(), page_size=2)])
        result = asyncio.run(dispatcher.run_once())
    finally:
        remove_sink(sink)
    report = result.require_report("fixture")
    assert (report.admitted, report.partially_admitted) == (2, 1)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert not [record for record in records if record["event"] == "log.field_rejected"]
    page = next(record for record in records if record["event"] == "daemon.intake.page")
    assert (page["outcome"], page["succeeded"], page["partially_admitted"]) == ("degraded", 2, 1)
    (item,) = [record for record in records if record["event"] == "daemon.intake.item_partial"]
    assert {
        key: item[key]
        for key in ("reason", "source_id", "complete_record_count", "complete_prefix_bytes", "source_bytes")
    } == {
        "reason": "truncated_tail",
        "source_id": "1",
        "complete_record_count": 3,
        "complete_prefix_bytes": 120,
        "source_bytes": 150,
    }
