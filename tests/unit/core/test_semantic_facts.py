"""Tests for the canonical semantic facts layer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.archive.conversation import extraction as work_event_extraction
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.models import Conversation as ConversationModel
from polylogue.archive.models import ConversationSummary
from polylogue.archive.phase.extraction import SessionPhase
from polylogue.archive.phase.extraction import extract_phases as phase_extract_phases
from polylogue.archive.semantic.facts import (
    ConversationSemanticFacts,
    build_conversation_semantic_facts,
    build_mcp_summary_semantic_facts,
    build_projection_semantic_facts,
)
from polylogue.archive.semantic.pricing import harmonize_session_cost
from polylogue.archive.session import runtime as session_profile_runtime
from polylogue.archive.session.session_profile import build_session_profile
from polylogue.storage.archive_views import ConversationRenderProjection
from polylogue.types import ConversationId, Provider
from tests.infra.builders import make_conv, make_msg
from tests.infra.storage_records import make_attachment, make_conversation, make_message

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_REPO_NAME = REPO_ROOT.name
README_PATH = REPO_ROOT / "README.md"
ARCHIVE_SESSION_PATH = REPO_ROOT / "polylogue" / "archive" / "session"
ARCHIVE_ACTION_EVENTS_PATH = REPO_ROOT / "polylogue" / "archive" / "action_event" / "events.py"
SHOWCASE_REPORT_PATH = REPO_ROOT / "polylogue" / "showcase" / "report.py"


def _semantic_conversation() -> ConversationModel:
    return make_conv(
        id="conv-semantic-facts",
        provider="claude-code",
        title="Semantic Facts",
        created_at=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 23, 9, 5, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="Please inspect README.md and summarize the result clearly.",
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="I will inspect the file",
                    timestamp=datetime(2026, 3, 23, 9, 1, tzinfo=timezone.utc),
                    provider_meta={
                        "raw": {
                            "type": "assistant",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": "I will inspect the file"},
                                    {
                                        "type": "tool_use",
                                        "id": "tool-1",
                                        "name": "Read",
                                        "input": {"file_path": str(README_PATH)},
                                    },
                                ],
                            },
                        }
                    },
                ),
                make_msg(
                    id="a2",
                    role="assistant",
                    provider="claude-code",
                    text="",
                    timestamp=datetime(2026, 3, 23, 9, 4, tzinfo=timezone.utc),
                    branch_index=1,
                    attachments=[],
                    provider_meta={
                        "raw": {
                            "type": "assistant",
                            "uuid": "a2",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "thinking", "thinking": "step by step"}],
                            },
                        }
                    },
                ),
            ]
        ),
    )


def _protocol_summary_conversation() -> ConversationModel:
    return make_conv(
        id="conv-work-event-summary",
        provider="claude-code",
        title="Work Event Summary",
        created_at=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 23, 10, 5, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="<system-reminder>skip this</system-reminder>\n"
                    + (f"Please inspect {README_PATH} and summarize the findings clearly. " * 3),
                    timestamp=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="I will inspect the file.",
                    timestamp=datetime(2026, 3, 23, 10, 1, tzinfo=timezone.utc),
                    provider_meta={
                        "raw": {
                            "type": "assistant",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": "I will inspect the file."},
                                    {
                                        "type": "tool_use",
                                        "id": "tool-1",
                                        "name": "Read",
                                        "input": {"file_path": str(README_PATH)},
                                    },
                                ],
                            },
                        }
                    },
                ),
                make_msg(
                    id="u2",
                    role="user",
                    provider="claude-code",
                    text=(
                        "Second user summary that should still fit after cleanup. "
                        "It repeats a bit so we hit the summary truncation boundary."
                    ),
                    timestamp=datetime(2026, 3, 23, 10, 2, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="u3",
                    role="user",
                    provider="claude-code",
                    text="This trailing note should be truncated away once the summary is already full.",
                    timestamp=datetime(2026, 3, 23, 10, 3, tzinfo=timezone.utc),
                ),
            ]
        ),
    )


def test_build_projection_semantic_facts_counts_renderable_and_empty_messages() -> None:
    projection = ConversationRenderProjection(
        conversation=make_conversation(
            conversation_id="conv-projection",
            provider_name="chatgpt",
            provider_conversation_id="provider-conv-projection",
            title="Projection Facts",
            content_hash="hash-projection",
        ),
        messages=[
            make_message(
                message_id="m1",
                conversation_id="conv-projection",
                role="user",
                text="hello",
                sort_key=1.0,
                content_hash="hash-m1",
            ),
            make_message(
                message_id="m2",
                conversation_id="conv-projection",
                role="assistant",
                text=None,
                sort_key=2.0,
                content_hash="hash-m2",
                has_thinking=1,
            ),
            make_message(
                message_id="m3",
                conversation_id="conv-projection",
                role="assistant",
                text=None,
                content_hash="hash-m3",
                has_tool_use=1,
            ),
        ],
        attachments=[
            make_attachment(
                attachment_id="att-m2",
                conversation_id="conv-projection",
                message_id="m2",
                path="/tmp/att-m2.txt",
            )
        ],
    )

    facts = build_projection_semantic_facts(projection)

    assert facts.total_messages == 3
    assert facts.renderable_messages == 2
    assert facts.timestamped_renderable_messages == 2
    assert facts.attachment_count == 1
    assert facts.empty_messages == 1
    assert facts.thinking_messages == 1
    assert facts.tool_messages == 1
    assert facts.renderable_role_counts == {"assistant": 1, "user": 1}


def test_build_conversation_semantic_facts_collects_semantic_counts() -> None:
    conversation = _semantic_conversation()

    facts = build_conversation_semantic_facts(conversation)

    assert facts.conversation_id == "conv-semantic-facts"
    assert facts.provider == "claude-code"
    assert facts.total_messages == 3
    assert facts.substantive_messages == 1
    assert facts.text_messages == 2
    assert facts.message_ids == ("u1", "a1", "a2")
    assert facts.text_message_ids == ("u1", "a1")
    assert facts.text_role_counts == {"assistant": 1, "user": 1}
    assert facts.timestamped_text_messages == 2
    assert facts.timestamped_messages == 3
    assert facts.untimestamped_messages == 0
    assert facts.timestamp_coverage == "complete"
    assert facts.thinking_messages == 1
    assert facts.tool_messages == 1
    assert facts.branch_messages == 1
    assert facts.word_count == 13
    assert facts.tool_category_counts == {"file_read": 1}
    assert facts.message_facts[1].tool_category_counts == {"file_read": 1}
    assert facts.message_facts[1].affected_paths == (str(README_PATH),)
    assert len(facts.action_events) == 1
    assert facts.action_events[0].kind.value == "file_read"
    assert facts.action_events[0].affected_paths == (str(README_PATH),)
    assert facts.action_events[0].message_id == "a1"
    assert facts.action_events[0].sequence_index == 0
    assert facts.first_message_at == datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc)
    assert facts.last_message_at == datetime(2026, 3, 23, 9, 4, tzinfo=timezone.utc)
    assert facts.wall_duration_ms == 240000


def test_build_session_profile_reuses_shared_semantic_facts() -> None:
    profile = build_session_profile(_semantic_conversation())

    assert profile.message_count == 3
    assert profile.substantive_count == 1
    assert profile.tool_use_count == 1
    assert profile.thinking_count == 1
    assert profile.tool_categories == {"file_read": 1}
    assert profile.repo_names == (EXPECTED_REPO_NAME,)
    assert profile.first_message_at == datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc)
    assert profile.last_message_at == datetime(2026, 3, 23, 9, 4, tzinfo=timezone.utc)
    assert profile.timestamped_message_count == 3
    assert profile.untimestamped_message_count == 0
    assert profile.timestamp_coverage == "complete"
    assert profile.canonical_session_date is not None
    assert profile.canonical_session_date.isoformat() == "2026-03-23"
    assert profile.engaged_duration_ms > 0
    assert profile.wall_duration_ms == 240000


def test_build_conversation_semantic_facts_marks_partial_timestamp_coverage() -> None:
    conversation = make_conv(
        id="conv-partial-timestamps",
        provider="claude-code",
        title="Partial Timestamps",
        created_at=datetime(2026, 3, 24, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 24, 9, 10, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="Start the task",
                    timestamp=datetime(2026, 3, 24, 9, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="Working on it",
                    timestamp=None,
                ),
                make_msg(
                    id="a2",
                    role="assistant",
                    provider="claude-code",
                    text="Finished",
                    timestamp=datetime(2026, 3, 24, 9, 7, tzinfo=timezone.utc),
                ),
            ]
        ),
    )

    facts = build_conversation_semantic_facts(conversation)
    profile = build_session_profile(conversation)

    assert facts.timestamped_messages == 2
    assert facts.untimestamped_messages == 1
    assert facts.timestamp_coverage == "partial"
    assert facts.wall_duration_ms == 420000
    assert profile.timestamp_coverage == "partial"
    assert profile.timestamped_message_count == 2
    assert profile.untimestamped_message_count == 1


def test_build_conversation_semantic_facts_marks_missing_and_single_message_timestamp_cases() -> None:
    no_timestamp = make_conv(
        id="conv-no-timestamps",
        provider="claude-code",
        title="No Timestamps",
        messages=MessageCollection(
            messages=[
                make_msg(id="u1", role="user", provider="claude-code", text="Hello", timestamp=None),
                make_msg(id="a1", role="assistant", provider="claude-code", text="Hi", timestamp=None),
            ]
        ),
    )
    single_timestamp = make_conv(
        id="conv-single-timestamp",
        provider="claude-code",
        title="Single Timestamp",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="One message",
                    timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
                )
            ]
        ),
    )

    no_timestamp_facts = build_conversation_semantic_facts(no_timestamp)
    single_timestamp_facts = build_conversation_semantic_facts(single_timestamp)

    assert no_timestamp_facts.first_message_at is None
    assert no_timestamp_facts.last_message_at is None
    assert no_timestamp_facts.wall_duration_ms == 0
    assert no_timestamp_facts.timestamp_coverage == "none"
    assert no_timestamp_facts.timestamped_messages == 0
    assert no_timestamp_facts.untimestamped_messages == 2

    assert single_timestamp_facts.first_message_at == datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc)
    assert single_timestamp_facts.last_message_at == datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc)
    assert single_timestamp_facts.wall_duration_ms == 0
    assert single_timestamp_facts.timestamp_coverage == "complete"
    assert single_timestamp_facts.timestamped_messages == 1
    assert single_timestamp_facts.untimestamped_messages == 0


def test_build_session_analysis_reuses_precomputed_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    conversation = _semantic_conversation()
    original_extract_phases = phase_extract_phases
    runtime_phase_calls = 0
    work_event_phase_calls = 0

    def counting_runtime_extract_phases(
        conv: ConversationModel,
        *,
        facts: ConversationSemanticFacts | None = None,
    ) -> list[SessionPhase]:
        nonlocal runtime_phase_calls
        runtime_phase_calls += 1
        return original_extract_phases(conv, facts=facts)

    def unexpected_work_event_extract_phases(
        conv: ConversationModel,
        *,
        facts: ConversationSemanticFacts | None = None,
    ) -> list[SessionPhase]:
        nonlocal work_event_phase_calls
        work_event_phase_calls += 1
        raise AssertionError("work-event extraction should reuse precomputed phases")

    monkeypatch.setattr(session_profile_runtime, "extract_phases", counting_runtime_extract_phases)
    monkeypatch.setattr(work_event_extraction, "extract_phases", unexpected_work_event_extract_phases)

    analysis = session_profile_runtime.build_session_analysis(conversation)

    assert runtime_phase_calls == 1
    assert work_event_phase_calls == 0
    assert analysis.work_events
    assert analysis.phases


def test_extract_work_events_strips_protocol_noise_and_respects_summary_cap() -> None:
    events = work_event_extraction.extract_work_events(_protocol_summary_conversation())

    assert events
    summary = events[0].summary
    assert "<system-reminder>" not in summary
    assert "skip this" not in summary
    assert f"Please inspect {README_PATH}" in summary
    assert "This trailing note should be truncated away" not in summary
    assert len(summary) <= 200


def test_build_conversation_semantic_facts_uses_canonical_db_content_blocks() -> None:
    conversation = make_conv(
        id="conv-db-semantic-facts",
        provider="claude-code",
        title="DB Semantic Facts",
        created_at=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 23, 9, 5, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="Inspect README.md and summarize the result.",
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="I checked the file.",
                    timestamp=datetime(2026, 3, 23, 9, 1, tzinfo=timezone.utc),
                    content_blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Read",
                            "tool_id": "tool-1",
                            "tool_input": {"file_path": str(README_PATH)},
                            "semantic_type": "file_read",
                            "metadata": {"path": str(README_PATH)},
                        },
                        {
                            "type": "tool_result",
                            "tool_id": "tool-1",
                            "text": "README contents",
                        },
                    ],
                ),
            ]
        ),
    )

    facts = build_conversation_semantic_facts(conversation)
    profile = build_session_profile(conversation)

    assert facts.tool_category_counts == {"file_read": 1}
    assert facts.message_facts[1].tool_category_counts == {"file_read": 1}
    assert facts.message_facts[1].affected_paths == (str(README_PATH),)
    assert facts.message_facts[1].tool_calls[0].output == "README contents"
    assert profile.tool_categories == {"file_read": 1}
    assert profile.repo_names == (EXPECTED_REPO_NAME,)


def test_build_conversation_semantic_facts_preserves_tool_results_before_tool_use() -> None:
    conversation = make_conv(
        id="conv-db-tool-result-before-use",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="I checked the file.",
                    content_blocks=[
                        {
                            "type": "tool_result",
                            "tool_id": "tool-1",
                            "text": "README contents",
                        },
                        {
                            "type": "tool_use",
                            "tool_name": "Read",
                            "tool_id": "tool-1",
                            "tool_input": {"file_path": str(README_PATH)},
                            "semantic_type": "file_read",
                        },
                    ],
                ),
            ]
        ),
    )

    facts = build_conversation_semantic_facts(conversation)

    assert facts.message_facts[0].tool_calls[0].output == "README contents"


def test_build_conversation_semantic_facts_upgrades_stale_other_semantic_type() -> None:
    conversation = make_conv(
        id="conv-db-upgrade-other",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="Use the edit tools.",
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="Applying edits.",
                    timestamp=datetime(2026, 3, 23, 9, 1, tzinfo=timezone.utc),
                    content_blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "MultiEdit",
                            "tool_input": {"file_path": str(README_PATH)},
                            "semantic_type": "other",
                            "metadata": {"path": str(README_PATH)},
                        },
                        {
                            "type": "tool_use",
                            "tool_name": "TodoWrite",
                            "tool_input": {"todos": [{"id": "1"}, {"id": "2"}]},
                            "semantic_type": "other",
                        },
                    ],
                ),
            ]
        ),
    )

    facts = build_conversation_semantic_facts(conversation)

    assert facts.tool_category_counts == {"agent": 1, "file_edit": 1}
    assert facts.message_facts[1].affected_paths == (str(README_PATH),)


def test_action_events_capture_normalized_command_query_branch_and_cwd() -> None:
    conversation = make_conv(
        id="conv-action-facts",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="Running git and search actions.",
                    timestamp=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
                    content_blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Bash",
                            "tool_input": {
                                "command": "git checkout feature/action-facts",
                                "cwd": str(REPO_ROOT),
                            },
                            "semantic_type": "git",
                        },
                        {
                            "type": "tool_use",
                            "tool_name": "Grep",
                            "tool_input": {
                                "pattern": "build_session_profile",
                                "path": str(ARCHIVE_SESSION_PATH),
                            },
                            "semantic_type": "search",
                        },
                    ],
                )
            ]
        ),
    )

    facts = build_conversation_semantic_facts(conversation)

    assert facts.tool_category_counts == {"git": 1, "search": 1}
    git_action, search_action = facts.action_events
    assert git_action.kind.value == "git"
    assert git_action.event_id.startswith("act-")
    assert git_action.command == "git checkout feature/action-facts"
    assert git_action.cwd_path == str(REPO_ROOT)
    assert git_action.branch_names == ("feature/action-facts",)
    assert "feature/action-facts" in git_action.search_text
    assert search_action.kind.value == "search"
    assert search_action.query == "build_session_profile"
    assert search_action.affected_paths == (str(ARCHIVE_SESSION_PATH),)
    assert str(ARCHIVE_SESSION_PATH) in search_action.search_text

    profile = build_session_profile(conversation)
    assert profile.repo_names == (EXPECTED_REPO_NAME,)


def test_action_events_do_not_treat_checkout_pathspec_as_branch_or_commit_message_words_as_paths() -> None:
    conversation = make_conv(
        id="conv-action-noise-guard",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="Running git maintenance.",
                    timestamp=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
                    content_blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Bash",
                            "tool_input": {
                                "command": (
                                    "git add modules/services/sinex/bridge.nix && git commit -m "
                                    "\"$(cat <<'EOF'\\nfix(sinex): cap schema-apply pool\\nEOF\\n)\""
                                ),
                            },
                            "metadata": {
                                "files": [
                                    "modules/services/sinex/bridge.nix",
                                    "&&",
                                    "git",
                                    "commit",
                                    '"$(cat',
                                    "<<'EOF'",
                                    "fix(sinex):",
                                    "(1M",
                                ]
                            },
                            "semantic_type": "git",
                        },
                        {
                            "type": "tool_use",
                            "tool_name": "Bash",
                            "tool_input": {
                                "command": "git checkout dots/transmission/settings.json",
                                "cwd": str(REPO_ROOT),
                            },
                            "semantic_type": "git",
                        },
                    ],
                )
            ]
        ),
    )

    facts = build_conversation_semantic_facts(conversation)

    commit_action, checkout_action = facts.action_events
    assert commit_action.affected_paths == ("modules/services/sinex/bridge.nix",)
    assert checkout_action.branch_names == ()

    profile = build_session_profile(conversation)
    assert profile.branch_names == ()
    assert profile.file_paths_touched == ("modules/services/sinex/bridge.nix",)
    assert profile.repo_names == (EXPECTED_REPO_NAME,)


def test_build_session_profile_does_not_infer_repos_from_dialogue_paths() -> None:
    conversation = make_conv(
        id="conv-user-paths",
        provider="gemini",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="gemini",
                    text=f"Please inspect {README_PATH} and summarize it.",
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="gemini",
                    text="I can do that.",
                    timestamp=datetime(2026, 3, 23, 9, 1, tzinfo=timezone.utc),
                ),
            ]
        ),
    )

    profile = build_session_profile(conversation)

    assert profile.repo_names == ()
    assert profile.repo_paths == ()
    assert profile.file_paths_touched == ()


def test_build_session_profile_ignores_persisted_output_paths_in_dialogue_text() -> None:
    conversation = make_conv(
        id="conv-user-path-noise",
        provider="claude-code",
        provider_meta={"working_directories": [str(REPO_ROOT)]},
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text=(
                        "Output too large. Full output saved to: "
                        "/home/sinity/.claude/projects/-realm-project-polylogue/example/tool-results/toolu_123.txt"
                    ),
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="I will inspect the concrete file.",
                    timestamp=datetime(2026, 3, 23, 9, 1, tzinfo=timezone.utc),
                ),
            ]
        ),
    )

    profile = build_session_profile(conversation)

    assert profile.repo_names == (EXPECTED_REPO_NAME,)
    assert profile.repo_paths == (str(REPO_ROOT),)
    assert profile.file_paths_touched == ()


def test_build_session_profile_discards_shell_path_noise_from_action_events() -> None:
    conversation = make_conv(
        id="conv-action-path-noise",
        provider="claude-code",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="Running a shell command.",
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                    content_blocks=[
                        {
                            "type": "tool_use",
                            "tool_name": "Bash",
                            "tool_input": {
                                "command": (
                                    f"echo /# /12 /AFK/browser /Codex /DAG && "
                                    f"cat {README_PATH} {ARCHIVE_ACTION_EVENTS_PATH}"
                                )
                            },
                            "semantic_type": "shell",
                        }
                    ],
                )
            ]
        ),
    )

    profile = build_session_profile(conversation)

    assert str(README_PATH) in profile.file_paths_touched
    assert str(ARCHIVE_ACTION_EVENTS_PATH) in profile.file_paths_touched
    assert str(REPO_ROOT) in profile.repo_paths
    assert "/#" not in profile.file_paths_touched
    assert "/12" not in profile.file_paths_touched
    assert "/AFK/browser" not in profile.file_paths_touched
    assert "/Codex" not in profile.file_paths_touched
    assert "/DAG" not in profile.file_paths_touched


def test_build_session_profile_uses_conversation_level_git_context() -> None:
    conversation = make_conv(
        id="conv-git-context",
        provider="codex",
        provider_meta={
            "git": {
                "branch": "feature/runtime-cleanup",
                "repository_url": str(REPO_ROOT),
            }
        },
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u1",
                    role="user",
                    provider="codex",
                    text="Please continue.",
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="codex",
                    text="Continuing.",
                    timestamp=datetime(2026, 3, 23, 9, 1, tzinfo=timezone.utc),
                ),
            ]
        ),
    )

    profile = build_session_profile(conversation)

    assert profile.branch_names == ("feature/runtime-cleanup",)
    assert profile.repo_names == (EXPECTED_REPO_NAME,)


def test_build_session_profile_ignores_context_dump_wrappers_for_work_event_intent() -> None:
    conversation = make_conv(
        id="conv-context-dump",
        provider="codex",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="u0",
                    role="user",
                    provider="codex",
                    text="<environment_context>\nerror: cached tool output\n</environment_context>",
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="u1",
                    role="user",
                    provider="codex",
                    text="Please plan the refactor and lay out the implementation strategy.",
                    timestamp=datetime(2026, 3, 23, 9, 1, tzinfo=timezone.utc),
                ),
                make_msg(
                    id="a1",
                    role="assistant",
                    provider="codex",
                    text="I will plan the refactor in detail.",
                    timestamp=datetime(2026, 3, 23, 9, 2, tzinfo=timezone.utc),
                ),
            ]
        ),
    )

    profile = build_session_profile(conversation)

    assert profile.work_events
    assert profile.work_events[0].kind.value == "planning"


def test_build_mcp_summary_semantic_facts_uses_canonical_summary_shape() -> None:
    summary = ConversationSummary(
        id=ConversationId("conv-semantic-facts"),
        provider=Provider.CLAUDE_CODE,
        title="Semantic Facts",
        created_at=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 23, 9, 5, tzinfo=timezone.utc),
    )

    facts = build_mcp_summary_semantic_facts(summary, message_count=3)

    assert facts.conversation_id == "conv-semantic-facts"
    assert facts.provider == "claude-code"
    assert facts.title == "Semantic Facts"
    assert facts.messages == 3
    assert facts.created_at == "2026-03-23T09:00:00+00:00"
    assert facts.updated_at == "2026-03-23T09:05:00+00:00"
    assert facts.tags == ()
    assert facts.summary is None


def test_harmonize_session_cost_uses_canonical_harmonized_model_and_tokens() -> None:
    conversation = make_conv(
        id="conv-cost",
        provider="chatgpt",
        messages=MessageCollection(
            messages=[
                make_msg(
                    id="m1",
                    role="assistant",
                    provider="chatgpt",
                    text="Estimated response",
                    provider_meta={
                        "model": "gpt-4o",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 500,
                        },
                    },
                )
            ]
        ),
    )

    cost_usd, is_estimated = harmonize_session_cost(conversation)

    assert is_estimated is True
    assert cost_usd == pytest.approx(0.0075)
