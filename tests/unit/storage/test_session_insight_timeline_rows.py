from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from polylogue.archive.models import Conversation
from polylogue.archive.session.session_profile import build_session_analysis, build_session_profile
from polylogue.storage.insights.session.timeline_rows import (
    build_session_phase_records,
    build_session_work_event_records,
    hydrate_session_phase,
    hydrate_work_event,
)
from polylogue.types import Provider
from tests.infra.builders import make_conv, make_msg

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_PATH = REPO_ROOT / "polylogue" / "facade.py"


def _timeline_conversation() -> Conversation:
    return make_conv(
        id="conv-timeline",
        provider=Provider.CLAUDE_CODE,
        title="Timeline",
        created_at=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 2, 12, 6, tzinfo=timezone.utc),
        messages=[
            make_msg(
                id="u1",
                role="user",
                provider=Provider.CLAUDE_CODE,
                text=f"Please inspect {APP_PATH} and fix the failing tests.",
                timestamp=datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc),
            ),
            make_msg(
                id="a1",
                role="assistant",
                provider=Provider.CLAUDE_CODE,
                text="I inspected the file and ran pytest.",
                timestamp=datetime(2026, 4, 2, 12, 1, tzinfo=timezone.utc),
                content_blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "tool_input": {"file_path": str(APP_PATH)},
                    }
                ],
            ),
            make_msg(
                id="u2",
                role="user",
                provider=Provider.CLAUDE_CODE,
                text="Now patch it and verify the fix.",
                timestamp=datetime(2026, 4, 2, 12, 6, tzinfo=timezone.utc),
            ),
            make_msg(
                id="a2",
                role="assistant",
                provider=Provider.CLAUDE_CODE,
                text="Patched the file and reran pytest successfully.",
                timestamp=datetime(2026, 4, 2, 12, 7, tzinfo=timezone.utc),
            ),
        ],
    )


def test_session_timeline_row_builders_roundtrip_work_events_and_phases() -> None:
    conversation = _timeline_conversation()
    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis)

    work_event_records = build_session_work_event_records(
        profile,
        materialized_at="2026-04-02T12:08:00+00:00",
    )
    phase_records = build_session_phase_records(
        profile,
        materialized_at="2026-04-02T12:08:00+00:00",
    )

    assert work_event_records
    assert phase_records

    work_event = hydrate_work_event(work_event_records[0])
    phase = hydrate_session_phase(phase_records[0])

    assert work_event.summary == work_event_records[0].summary
    assert work_event_records[0].inference_payload.summary == work_event.summary
    assert work_event.kind.value == work_event_records[0].kind
    assert work_event_records[0].search_text

    assert phase.message_range == (phase_records[0].start_index, phase_records[0].end_index)
    assert phase.tool_counts == phase_records[0].tool_counts
    assert phase_records[0].search_text
