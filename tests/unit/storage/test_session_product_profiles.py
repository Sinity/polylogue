from __future__ import annotations

from datetime import datetime, timezone

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.lib.session_profile import build_session_analysis, build_session_profile
from polylogue.storage.session_product_profiles import (
    assistant_turn_texts,
    blocker_texts,
    session_enrichment_payload,
    user_turn_texts,
)


def _enrichment_conversation() -> Conversation:
    return Conversation(
        id="conv-enrichment",
        provider="claude-code",
        title="Enrichment",
        created_at=datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 2, 10, 3, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="The tests failed with traceback in /realm/project/polylogue/app.py. Please fix it.",
                    timestamp=datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc),
                ),
                Message(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="I inspected app.py and ran pytest.",
                    timestamp=datetime(2026, 4, 2, 10, 1, tzinfo=timezone.utc),
                    content_blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Read",
                            "tool_input": {"file_path": "/realm/project/polylogue/app.py"},
                        }
                    ],
                ),
                Message(
                    id="a2",
                    role="assistant",
                    provider="claude-code",
                    text="Patched it and reran pytest successfully.",
                    timestamp=datetime(2026, 4, 2, 10, 2, tzinfo=timezone.utc),
                ),
            ]
        ),
    )


def test_session_enrichment_payload_reuses_text_band_outputs() -> None:
    conversation = _enrichment_conversation()
    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis)

    assert user_turn_texts(analysis) == (
        "The tests failed with traceback in /realm/project/polylogue/app.py. Please fix it.",
    )
    assert assistant_turn_texts(analysis) == (
        "Patched it and reran pytest successfully.",
    )
    assert blocker_texts(analysis) == (
        "The tests failed with traceback in /realm/project/polylogue/app.py. Please fix it.",
    )

    payload = session_enrichment_payload(profile, analysis)

    assert payload["intent_summary"] == user_turn_texts(analysis)[0]
    assert payload["outcome_summary"] == assistant_turn_texts(analysis)[-1]
    assert payload["blockers"] == list(blocker_texts(analysis))
    assert payload["input_band_summary"] == {
        "user_turns": 1,
        "assistant_turns": 1,
        "action_events": 1,
        "touched_paths": 1,
        "canonical_projects": 1,
    }
    assert tuple(payload["support_signals"]) == (
        "user_turns",
        "action_events",
        "touched_paths",
        "canonical_projects",
        "heuristic_work_events",
        "assistant_outcome_text",
    )
