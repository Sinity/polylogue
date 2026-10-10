"""Measured latency includes zero; missing timestamp pairs stay absent."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

from polylogue.analysis.archive import SessionLatencyProfileInsight, SessionLatencyProfileInsightQuery
from polylogue.analysis.archive_models import ArchiveInsightProvenance, SessionLatencyProfilePayload
from polylogue.api.insights import PolylogueInsightsMixin
from polylogue.archive.semantic.timing import compute_session_latency_profile
from polylogue.archive.session.events import SessionEvent
from polylogue.archive.session.session_profile import build_session_profile
from polylogue.core.enums import MaterialOrigin, Origin
from polylogue.core.types import SessionEventId, SessionId
from polylogue.storage.derived.session.latency_profiles import build_session_latency_profile_record
from polylogue.storage.derived.session.storage import replace_session_latency_profiles_bulk_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.queries.session_latency_profile_reads import get_session_latency_profile
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.builders import make_conv, make_msg

STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _events(session_id: str, kind: str, duration_ms: int = 0) -> tuple[SessionEvent, ...]:
    if kind == "empty":
        return ()
    return tuple(
        SessionEvent(
            id=SessionEventId(f"{session_id}:event-{index}"),
            session_id=SessionId(session_id),
            origin=Origin.CODEX_SESSION,
            event_index=index,
            event_type=event_type,
            timestamp=None if kind == "undated" else STAMP + timedelta(milliseconds=delta),
            payload={"call_id": "call-1"},
        )
        for index, (event_type, delta) in enumerate(
            [("function_call", 0)]
            if kind == "unpaired"
            else [("function_call", 0), ("function_call_output", duration_ms)]
        )
    )


@pytest.mark.parametrize(
    "kind,duration,expected",
    [("paired", 0, 0), ("paired", 100, 100), ("unpaired", 0, None), ("undated", 0, None), ("empty", 0, None)],
)
def test_latency_presence_comes_from_timestamped_pairs(kind: str, duration: int, expected: int | None) -> None:
    facts = compute_session_latency_profile([], _events("codex-session:presence", kind, duration))
    assert facts.median_tool_call_ms == facts.p90_tool_call_ms == facts.max_tool_call_ms == expected
    assert facts.median_agent_response_ms is None
    assert facts.median_user_response_ms is None
    assert facts.to_dict()["median_tool_call_ms"] == expected


@pytest.mark.parametrize("duration", [0, 100])
def test_response_latency_preserves_measured_zero(duration: int) -> None:
    messages = [
        make_msg(id="u1", role="user", timestamp=STAMP, material_origin=MaterialOrigin.HUMAN_AUTHORED),
        make_msg(id="a1", role="assistant", timestamp=STAMP + timedelta(milliseconds=duration)),
        make_msg(
            id="u2",
            role="user",
            timestamp=STAMP + timedelta(milliseconds=2 * duration),
            material_origin=MaterialOrigin.HUMAN_AUTHORED,
        ),
    ]
    facts = compute_session_latency_profile(messages, [])
    assert facts.median_agent_response_ms == facts.median_user_response_ms == duration
    assert facts.median_tool_call_ms is None


class _Reader(PolylogueInsightsMixin):
    def __init__(self, insights: list[SessionLatencyProfileInsight]) -> None:
        self.insights = insights

    async def list_session_latency_profile_insights(
        self, query: SessionLatencyProfileInsightQuery | None = None
    ) -> list[SessionLatencyProfileInsight]:
        return self.insights


@pytest.mark.asyncio
async def test_timed_zero_population_participates_in_public_percentiles() -> None:
    insights = []
    for index, duration in enumerate([0] * 9 + [100]):
        session_id = f"codex-session:timed-{index}"
        facts = compute_session_latency_profile([], _events(session_id, "paired", duration))
        payload = SessionLatencyProfilePayload.model_validate(facts.to_dict())
        insights.append(
            SessionLatencyProfileInsight(
                session_id=session_id,
                origin="codex-session",
                latency=payload,
                provenance=ArchiveInsightProvenance(materializer_version=None),
            )
        )
    rollup = await _Reader(insights).tool_call_latency_distribution()
    assert rollup["median_tool_call_ms"] == rollup["p90_tool_call_ms"] == 0
    assert rollup["max_tool_call_ms"] == 100
    assert rollup["total_sessions"] == rollup["measured_sessions"] == 10


@pytest.mark.asyncio
async def test_measured_zero_survives_materialization_both_readers_and_public_rollup(tmp_path: Path) -> None:
    """Dropping zero or converting absent timing to zero changes these populations."""
    cases = [("paired", 0)] * 9 + [("paired", 100), ("unpaired", 0), ("undated", 0), ("empty", 0)]
    root = tmp_path / "archive"

    def materialize() -> list[SessionLatencyProfileInsight]:
        bootstrap_archive_root(root)
        with sqlite3.connect(root / "index.db") as connection:
            records = []
            for index, (kind, duration) in enumerate(cases):
                native_id = f"latency-{index}"
                session_id = f"codex-session:{native_id}"
                connection.execute(
                    "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, 'codex-session', zeroblob(32))",
                    (native_id,),
                )
                events = _events(session_id, kind, duration)
                session = make_conv(id=session_id, origin="codex-session", messages=[], session_events=events)
                facts = compute_session_latency_profile(
                    [], events, tool_call_count_by_category={} if kind == "empty" else {"shell": 1}
                )
                records.append(build_session_latency_profile_record(session, build_session_profile(session), facts))
            replace_session_latency_profiles_bulk_sync(connection, records)
        with ArchiveStore.open_existing(root, read_only=True) as archive:
            insights = []
            for record in records:
                insight = archive.get_session_latency_profile_insight(str(record.session_id))
                assert insight is not None
                insights.append(insight)
            return insights

    synchronous = run_off_event_loop(materialize)
    assert all(insight is not None for insight in synchronous)
    async with aiosqlite.connect(root / "index.db") as connection:
        connection.row_factory = sqlite3.Row
        records = [
            await get_session_latency_profile(connection, f"codex-session:latency-{i}") for i in range(len(cases))
        ]
    asynchronous = []
    for record in records:
        assert record is not None
        asynchronous.append(SessionLatencyProfileInsight.from_record(record))
    for metric in (
        "median_tool_call_ms",
        "p90_tool_call_ms",
        "max_tool_call_ms",
        "median_agent_response_ms",
        "median_user_response_ms",
    ):
        assert [getattr(item.latency, metric) for item in synchronous] == [
            getattr(item.latency, metric) for item in asynchronous
        ]
    assert [item.latency.median_tool_call_ms for item in asynchronous] == [0] * 9 + [100, None, None, None]

    payload = await _Reader(asynchronous).tool_call_latency_distribution()
    assert payload["total_sessions"] == 13
    assert payload["measured_sessions"] == 10
    assert payload["median_tool_call_ms"] == payload["p90_tool_call_ms"] == 0
    assert payload["max_tool_call_ms"] == 100
    shell = await _Reader(asynchronous).tool_call_latency_distribution(tool_category="shell")
    assert shell["total_sessions"] == 12 and shell["measured_sessions"] == 10
    assert shell["median_tool_call_ms"] == shell["p90_tool_call_ms"] == 0
    absent = await _Reader(asynchronous[-3:]).tool_call_latency_distribution()
    assert absent["total_sessions"] == 3 and absent["measured_sessions"] == 0
    assert absent["median_tool_call_ms"] is absent["p90_tool_call_ms"] is absent["max_tool_call_ms"] is None
