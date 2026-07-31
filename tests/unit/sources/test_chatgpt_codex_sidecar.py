"""Tests for codex.json parsing (bd polylogue-2m2e): Codex Cloud tasks
delivered inside the ChatGPT export.
"""

from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider, SessionRefKind
from polylogue.sources.parsers import codex as codex_parser
from polylogue.sources.parsers.chatgpt import looks_like_fragment as chatgpt_looks_like_fragment
from polylogue.sources.parsers.chatgpt_codex_sidecar import INGEST_FLAG, looks_like, parse_codex_task

_TASK: dict[str, object] = {
    "archived": False,
    "id": "task_e_6a513f0f28e08320826bb4a333c36457",
    "title": "Fix polylogue daemon correctness bug",
    "turns": [
        {
            "custom_instructions": None,
            "id": "task_e_6a513f0f28e08320826bb4a333c36457~usertrn_e_6a513f0f457483209ed4fcdc7e91e41b",
            "input_items": [
                {
                    "content": [{"content_type": "text", "text": "Fix the bug."}],
                    "role": "user",
                    "type": "message",
                }
            ],
            "role": "user",
        },
        {
            "branch": "master",
            "branch_name": None,
            "external_pull_request_id": "None",
            "id": "task_e_6a513f0f28e08320826bb4a333c36457~assttrn_e_6a513f1032a883208d5bf03461f555f4",
            "output_items": [
                {
                    "content": [
                        {"content_type": "text", "text": "### Summary\n\nFixed it."},
                        {
                            "content_type": "repo_file_citation",
                            "line_range_end": 136,
                            "line_range_start": 1,
                            "path": "polylogue/archive/write_coordinator.py",
                        },
                    ],
                }
            ],
            "previous_turn_id": "task_e_6a513f0f28e08320826bb4a333c36457~usertrn_e_6a513f0f457483209ed4fcdc7e91e41b",
            "pull_request_status": "not_created",
            "role": "assistant",
            "turn_status": "TaskTurnStatusEnum.COMPLETED",
        },
    ],
}


class TestLooksLike:
    def test_matches_real_codex_task_shape(self) -> None:
        assert looks_like(_TASK) is True

    def test_rejects_conversation_fragment(self) -> None:
        fragment: dict[str, object] = {"mapping": {"n1": {"id": "n1", "parent": None, "children": []}}}
        assert looks_like(fragment) is False

    def test_rejects_non_task_id(self) -> None:
        assert looks_like({**_TASK, "id": "conv-abc"}) is False

    def test_rejects_missing_turns(self) -> None:
        payload = {k: v for k, v in _TASK.items() if k != "turns"}
        assert looks_like(payload) is False

    def test_rejects_empty_turns(self) -> None:
        assert looks_like({**_TASK, "turns": []}) is False

    def test_rejects_turn_with_bad_role(self) -> None:
        bad = {**_TASK, "turns": [{**_TASK["turns"][0], "role": "system"}]}  # type: ignore[index]
        assert looks_like(bad) is False

    def test_rejects_non_dict(self) -> None:
        assert looks_like("not-a-dict") is False
        assert looks_like(None) is False

    def test_disjoint_from_local_codex_session_detector(self) -> None:
        """codex.json tasks must never be claimed by the local-rollout Codex parser.

        Identity does not collide: local Codex CLI sessions are rollout
        session_id UUIDs; codex.json tasks are ``task_e_<hex>`` ids. If this
        ever started matching, ingesting a ChatGPT export could silently
        create phantom entries under the wrong provider/parser.
        """
        assert codex_parser.looks_like([dict(_TASK)]) is False

    def test_disjoint_from_chatgpt_conversation_fragment_detector(self) -> None:
        assert chatgpt_looks_like_fragment(_TASK) is False


class TestParseCodexTask:
    def test_two_messages_user_then_assistant(self) -> None:
        session = parse_codex_task(_TASK, "fallback-0")
        assert [m.role for m in session.messages] == [Role.USER, Role.ASSISTANT]
        assert session.messages[0].text == "Fix the bug."
        assert "Fixed it." in (session.messages[1].text or "")

    def test_identity_and_tagging(self) -> None:
        session = parse_codex_task(_TASK, "fallback-0")
        assert session.source_name is Provider.CHATGPT
        assert session.provider_session_id == "task_e_6a513f0f28e08320826bb4a333c36457"
        assert session.title == "Fix polylogue daemon correctness bug"
        assert INGEST_FLAG in session.ingest_flags

    def test_parent_link_from_assistant_to_user_turn(self) -> None:
        session = parse_codex_task(_TASK, "fallback-0")
        user_id, assistant_id = (m.provider_message_id for m in session.messages)
        assert session.messages[1].parent_message_provider_id == user_id

    def test_branch_recorded_as_git_branch(self) -> None:
        session = parse_codex_task(_TASK, "fallback-0")
        assert session.git_branch == "master"

    def test_turn_metadata_recorded_as_session_event(self) -> None:
        session = parse_codex_task(_TASK, "fallback-0")
        events = [e for e in session.session_events if e.event_type == "chatgpt_codex_cloud_turn"]
        assert len(events) == 1
        assert events[0].payload["turn_status"] == "TaskTurnStatusEnum.COMPLETED"
        assert events[0].payload["pull_request_status"] == "not_created"
        assert events[0].payload["branch"] == "master"

    def test_repo_file_citation_becomes_web_construct(self) -> None:
        session = parse_codex_task(_TASK, "fallback-0")
        assistant_message = session.messages[1]
        constructs = [c for block in assistant_message.blocks for c in block.web_constructs]
        assert len(constructs) == 1
        assert constructs[0].text == "polylogue/archive/write_coordinator.py"
        assert constructs[0].start_index == 1
        assert constructs[0].end_index == 136

    def test_literal_none_string_pull_request_id_is_not_a_ref(self) -> None:
        # Real export data carries the literal string "None" for tasks with
        # no PR (measured 2026-07-31) -- must not become a fake session_ref.
        session = parse_codex_task(_TASK, "fallback-0")
        assert session.session_refs == []

    def test_real_pull_request_id_becomes_session_ref(self) -> None:
        task = {
            **_TASK,
            "turns": [
                _TASK["turns"][0],  # type: ignore[index]
                {**_TASK["turns"][1], "external_pull_request_id": "1234"},  # type: ignore[index]
            ],
        }
        session = parse_codex_task(task, "fallback-0")
        assert len(session.session_refs) == 1
        assert session.session_refs[0].kind == SessionRefKind.PULL_REQUEST.value
        assert session.session_refs[0].url == "1234"

    def test_missing_id_falls_back(self) -> None:
        task = {k: v for k, v in _TASK.items() if k != "id"}
        session = parse_codex_task(task, "fallback-9")
        assert session.provider_session_id == "fallback-9"
