from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from polylogue.archive.models import Session
from polylogue.archive.session.session_profile import build_session_analysis, build_session_profile
from polylogue.core.enums import MaterialOrigin, Provider
from polylogue.storage.insights.session.profiles import (
    assistant_turn_texts,
    blocker_texts,
    build_session_profile_record,
    profile_evidence_payload,
    profile_inference_payload,
    profile_inference_search_text,
    session_enrichment_payload,
    user_turn_texts,
)
from tests.infra.builders import make_conv, make_msg

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_PATH = REPO_ROOT / "polylogue" / "facade.py"


def _enrichment_session() -> Session:
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
                blocks=[
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
    session = _enrichment_session()
    analysis = build_session_analysis(session)
    profile = build_session_profile(session, analysis=analysis)

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
        "actions": 1,
        "touched_paths": 1,
        "repo_names": 1,
    }
    support_signals = payload.support_signals
    assert isinstance(support_signals, tuple)
    assert support_signals == (
        "user_turns",
        "actions",
        "touched_paths",
        "repo_names",
        "heuristic_work_events",
        "assistant_outcome_text",
    )


def test_session_enrichment_ignores_provider_user_runtime_protocol() -> None:
    session = make_conv(
        id="conv-runtime-user",
        provider=Provider.CLAUDE_CODE,
        title="Runtime output",
        messages=[
            make_msg(
                id="u-runtime",
                role="user",
                provider=Provider.CLAUDE_CODE,
                text="<local-command-stdout>Traceback: generated runtime output</local-command-stdout>",
                material_origin=MaterialOrigin.RUNTIME_PROTOCOL,
            ),
            make_msg(
                id="u-authored",
                role="user",
                provider=Provider.CLAUDE_CODE,
                text="Please fix the failing tests.",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
            ),
            make_msg(
                id="a1",
                role="assistant",
                provider=Provider.CLAUDE_CODE,
                text="I will inspect the failure.",
                material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
            ),
        ],
    )

    analysis = build_session_analysis(session)
    profile = build_session_profile(session, analysis=analysis)
    payload = session_enrichment_payload(profile, analysis)

    assert user_turn_texts(analysis) == ("Please fix the failing tests.",)
    assert payload.intent_summary == "Please fix the failing tests."
    assert payload.input_band_summary["user_turns"] == 1
    assert payload.blockers == ()


def test_session_profile_infers_topic_from_first_substantive_user_turn() -> None:
    session = make_conv(
        id="conv-topic",
        provider=Provider.CLAUDE_CODE,
        title="Caveat: The messages below were generated by the user while running local commands.",
        messages=[
            make_msg(
                id="u1",
                role="user",
                provider=Provider.CLAUDE_CODE,
                text=(
                    "Caveat: The messages below were generated by the user while running local commands. "
                    "Fix the FTS freshness repair path."
                ),
            ),
            make_msg(id="a1", role="assistant", provider=Provider.CLAUDE_CODE, text="I will inspect the repair code."),
        ],
    )

    profile = build_session_profile(session)
    inference = profile_inference_payload(profile)

    assert profile.inferred_topic == "Fix the FTS freshness repair path."
    assert inference.inferred_topic == "Fix the FTS freshness repair path."
    assert inference.inferred_topic_source == "first_substantive_user_turn"
    assert "Fix the FTS freshness repair path." in profile_inference_search_text(profile)


def test_session_profile_topic_uses_authored_turn_not_provider_user_protocol() -> None:
    session = make_conv(
        id="conv-authored-topic",
        provider=Provider.CLAUDE_CODE,
        title="Caveat: The messages below were generated by the user while running local commands.",
        messages=[
            make_msg(
                id="u-runtime",
                role="user",
                provider=Provider.CLAUDE_CODE,
                text="<command-name>/status</command-name><command-message>generated wrapper</command-message>",
                material_origin=MaterialOrigin.RUNTIME_PROTOCOL,
            ),
            make_msg(
                id="u-authored",
                role="user",
                provider=Provider.CLAUDE_CODE,
                text="Design the material-origin facet split.",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
            ),
        ],
    )

    profile = build_session_profile(session)

    assert profile.inferred_topic == "Design the material-origin facet split."
    assert profile.inferred_topic_source == "first_substantive_user_turn"


def test_session_profile_infers_codex_topic_from_repo_and_first_user_turn(tmp_path: Path) -> None:
    # Seed a deterministic fake repo so repo-name inference (which probes the real
    # filesystem for a `.git` root, see archive/session/repo_identity._find_git_root)
    # resolves to "polylogue" regardless of the checkout directory name. Using the
    # ambient REPO_ROOT couples the assertion to the worktree basename, which is
    # "polylogue" in CI but differs in agent worktrees.
    fake_repo = tmp_path / "polylogue"
    (fake_repo / ".git").mkdir(parents=True)
    fake_app_path = fake_repo / "polylogue" / "facade.py"
    fake_app_path.parent.mkdir(parents=True, exist_ok=True)

    session = make_conv(
        id="conv-codex-topic",
        provider=Provider.CODEX,
        title="2c2c8d0c-3b9b-4de8-a3ee-7e17e5070e31",
        messages=[
            make_msg(
                id="u1",
                role="user",
                provider=Provider.CODEX,
                text="Implement the daemon workload probe evidence surface and update docs.",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Read",
                        "tool_input": {"file_path": str(fake_app_path)},
                    }
                ],
            ),
        ],
    )

    profile = build_session_profile(session)

    assert profile.repo_names == ("polylogue",)
    assert profile.inferred_topic == "polylogue — Implement the daemon workload probe evidence surface and upd"
    assert profile.inferred_topic_source == "repo_plus_first_user_turn"


def test_session_profile_record_exposes_tool_active_duration() -> None:
    session = _enrichment_session()
    profile = build_session_profile(session)
    profile = profile.__class__.from_dict(
        {
            **profile.to_dict(),
            "tool_active_duration_ms": 180_000,
        }
    )

    record = build_session_profile_record(profile)

    assert record.tool_active_duration_ms == 180_000
    assert record.evidence_payload.tool_active_duration_ms == 180_000
    assert record.inference_payload.tool_active_duration_ms == 180_000
    assert record.inference_payload.tool_active_minutes == 3.0


def test_session_profile_record_exposes_shape_and_terminal_state() -> None:
    profile = build_session_profile(_enrichment_session())
    profile = profile.__class__.from_dict(
        {
            **profile.to_dict(),
            "workflow_shape": "agentic_loop",
            "workflow_shape_confidence": 0.86,
            "workflow_shape_features": {"edit_count": 1, "tool_ratio": 0.4},
            "terminal_state": "clean_finish",
            "terminal_state_confidence": 0.68,
            "terminal_state_evidence": {"message_id": "a2"},
        }
    )

    record = build_session_profile_record(profile)

    assert record.workflow_shape == "agentic_loop"
    assert record.terminal_state == "clean_finish"
    assert record.evidence_payload.workflow_shape_features == {"edit_count": 1, "tool_ratio": 0.4}
    assert record.evidence_payload.terminal_state_evidence == {"message_id": "a2"}
    assert record.inference_payload.workflow_shape == "agentic_loop"
    assert record.inference_payload.terminal_state == "clean_finish"
    assert "agentic_loop" in record.search_text
    assert "clean_finish" in record.search_text


def test_stored_inference_payload_json_omits_native_mirrored_fields() -> None:
    """#14: inference_payload_json must not persist a second copy of the native
    workflow_shape / terminal_state columns. The native session_profiles columns
    are the single source of truth; the read paths reconcile onto them.
    """
    import json

    from polylogue.storage.insights.session.storage import (
        _INFERENCE_NATIVE_MIRRORED_FIELDS,
        _stored_inference_payload_json,
    )

    profile = build_session_profile(_enrichment_session())
    profile = profile.__class__.from_dict(
        {
            **profile.to_dict(),
            "workflow_shape": "agentic_loop",
            "workflow_shape_confidence": 0.86,
            "terminal_state": "clean_finish",
            "terminal_state_confidence": 0.68,
        }
    )
    record = build_session_profile_record(profile)

    stored = _stored_inference_payload_json(record)
    assert stored is not None
    payload = json.loads(stored)
    for field in _INFERENCE_NATIVE_MIRRORED_FIELDS:
        assert field not in payload, f"{field} must not be persisted in inference_payload_json"
    # Non-mirrored inference fields are still persisted.
    assert "inferred_topic_source" in payload


def test_session_profile_insight_reads_native_columns_over_payload() -> None:
    """#14: when the (stored-then-rehydrated) inference payload disagrees with the
    authoritative native columns, the read insight returns the native value.
    """
    from polylogue.insights.archive import SessionProfileInsight

    profile = build_session_profile(_enrichment_session())
    profile = profile.__class__.from_dict(
        {
            **profile.to_dict(),
            "workflow_shape": "agentic_loop",
            "workflow_shape_confidence": 0.86,
            "terminal_state": "clean_finish",
            "terminal_state_confidence": 0.68,
        }
    )
    record = build_session_profile_record(profile)
    # Simulate the rehydrated state: inference_payload no longer carries the
    # native-mirrored fields, so it parses back to the model defaults.
    rehydrated = record.model_copy(
        update={
            "inference_payload": record.inference_payload.model_copy(
                update={
                    "workflow_shape": "unknown",
                    "workflow_shape_confidence": 0.0,
                    "terminal_state": "unknown",
                    "terminal_state_confidence": 0.0,
                }
            )
        }
    )

    insight = SessionProfileInsight.from_record(rehydrated)
    assert insight.inference is not None
    assert insight.inference.workflow_shape == "agentic_loop"
    assert insight.inference.workflow_shape_confidence == 0.86
    assert insight.inference.terminal_state == "clean_finish"
    assert insight.inference.terminal_state_confidence == 0.68


def test_session_profile_uses_session_timestamp_when_messages_are_untimestamped() -> None:
    session = make_conv(
        id="conv-fallback-time",
        provider=Provider.CODEX,
        title="Untimestamped messages",
        created_at=datetime(2026, 5, 12, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 12, 9, 30, tzinfo=timezone.utc),
        messages=[
            make_msg(id="u1", role="user", provider=Provider.CODEX, text="Start"),
            make_msg(id="a1", role="assistant", provider=Provider.CODEX, text="Done"),
        ],
    )

    profile = build_session_profile(session)

    assert profile.first_message_at == datetime(2026, 5, 12, 9, 0, tzinfo=timezone.utc)
    assert profile.last_message_at == datetime(2026, 5, 12, 9, 30, tzinfo=timezone.utc)
    assert profile.timestamp_source == "session_timestamp_fallback"
    assert profile.timestamp_coverage == "none"


def test_session_profile_evidence_exposes_timestamp_source() -> None:
    session = make_conv(
        id="conv-evidence-time",
        provider=Provider.CODEX,
        title="Evidence timestamp",
        created_at=datetime(2026, 5, 12, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 12, 9, 30, tzinfo=timezone.utc),
        messages=[make_msg(id="u1", role="user", provider=Provider.CODEX, text="No message timestamp")],
    )

    profile = build_session_profile(session)
    record_evidence = profile_evidence_payload(profile)

    assert record_evidence.session_timestamp == "2026-05-12T09:00:00+00:00"
    assert record_evidence.timestamp_source == "session_timestamp_fallback"
