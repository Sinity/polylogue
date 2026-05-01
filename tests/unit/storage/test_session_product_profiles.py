from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from polylogue.archive.models import Conversation
from polylogue.archive.session.session_profile import build_session_analysis, build_session_profile
from polylogue.storage.products.session.profiles import (
    assistant_turn_texts,
    blocker_texts,
    session_enrichment_payload,
    user_turn_texts,
)
from polylogue.types import Provider
from tests.infra.builders import make_conv, make_msg

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_PATH = REPO_ROOT / "polylogue" / "facade.py"


def _enrichment_conversation() -> Conversation:
    return make_conv(
        id="conv-enrichment",
        provider=Provider.CLAUDE_CODE,
        title="Enrichment",
        created_at=datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 2, 10, 3, tzinfo=timezone.utc),
        messages=[
            make_msg(
                id="u1",
                role="user",
                provider=Provider.CLAUDE_CODE,
                text=f"The tests failed with traceback in {APP_PATH}. Please fix it.",
                timestamp=datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc),
            ),
            make_msg(
                id="a1",
                role="assistant",
                provider=Provider.CLAUDE_CODE,
                text="I inspected app.py and ran pytest.",
                timestamp=datetime(2026, 4, 2, 10, 1, tzinfo=timezone.utc),
                content_blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "tool_input": {"file_path": str(APP_PATH)},
                    }
                ],
            ),
            make_msg(
                id="a2",
                role="assistant",
                provider=Provider.CLAUDE_CODE,
                text="Patched it and reran pytest successfully.",
                timestamp=datetime(2026, 4, 2, 10, 2, tzinfo=timezone.utc),
            ),
        ],
    )


def test_session_enrichment_payload_reuses_text_band_outputs() -> None:
    conversation = _enrichment_conversation()
    analysis = build_session_analysis(conversation)
    profile = build_session_profile(conversation, analysis=analysis)

    assert user_turn_texts(analysis) == (f"The tests failed with traceback in {APP_PATH}. Please fix it.",)
    assert assistant_turn_texts(analysis) == ("Patched it and reran pytest successfully.",)
    assert blocker_texts(analysis) == (f"The tests failed with traceback in {APP_PATH}. Please fix it.",)

    payload = session_enrichment_payload(profile, analysis)

    assert payload.intent_summary == user_turn_texts(analysis)[0]
    assert payload.outcome_summary == assistant_turn_texts(analysis)[-1]
    assert payload.blockers == blocker_texts(analysis)
    assert payload.input_band_summary == {
        "user_turns": 1,
        "assistant_turns": 1,
        "action_events": 1,
        "touched_paths": 1,
        "repo_names": 1,
    }
    support_signals = payload.support_signals
    assert isinstance(support_signals, tuple)
    assert support_signals == (
        "user_turns",
        "action_events",
        "touched_paths",
        "repo_names",
        "heuristic_work_events",
        "assistant_outcome_text",
    )
