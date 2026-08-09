"""Contract tests for the Claude Design chat parser (bd polylogue-tbun).

Claude Design is a distinct product/backend from claude.ai: camelCase
``contentBlocks``/``authorAccountUuid``/``turnChanges``, ``content`` is a
dict not a list. Fixtures here are synthetic but shaped after the measured
wire spec (bd polylogue-tbun notes) -- no real export data is committed.
"""

from __future__ import annotations

from typing import Any

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Origin, Provider, TitleSource
from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers.claude import looks_like_ai, looks_like_claude_design, parse_design


def test_looks_like_claude_design_detects_the_shape_and_rejects_chat_messages() -> None:
    design_payload = {
        "uuid": "design-1",
        "project": {"uuid": "project-1", "name": "Sinex Design System"},
        "messages": [],
    }
    assert looks_like_claude_design(design_payload)
    assert not looks_like_ai(design_payload)

    claude_ai_payload = {"uuid": "conv-1", "chat_messages": []}
    assert not looks_like_claude_design(claude_ai_payload)


def test_claude_design_origin_is_distinct_from_claude_ai() -> None:
    assert origin_from_provider(Provider.CLAUDE_DESIGN) is Origin.CLAUDE_DESIGN_SESSION
    assert origin_from_provider(Provider.CLAUDE_DESIGN) is not origin_from_provider(Provider.CLAUDE_AI)


def test_detect_provider_claims_design_chat_shape() -> None:
    payload = {
        "uuid": "design-1",
        "project": {"uuid": "project-1", "name": "Sinex Design System"},
        "messages": [{"uuid": "m1", "role": "user", "content": {"role": "user", "content": "hi"}}],
    }
    assert detect_provider(payload) is Provider.CLAUDE_DESIGN
    sessions = parse_payload(Provider.CLAUDE_DESIGN, payload, "fallback")
    assert len(sessions) == 1
    assert sessions[0].source_name is Provider.CLAUDE_DESIGN


def _user_message(uuid: str, text: str, *, attachments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "role": "user",
        "content": {
            "role": "user",
            "content": text,
            "attachments": attachments or [],
            "authorAccountUuid": "acct-1",
            "authorName": "Sinity",
            "timestamp": "2026-07-09T11:30:00.000Z",
        },
    }


def _assistant_message(uuid: str, content_blocks: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    content: dict[str, Any] = {
        "id": f"content-{uuid}",
        "role": "assistant",
        "content": "",
        "contentBlocks": content_blocks,
        "timestamp": "2026-07-09T11:32:00.000Z",
    }
    content.update(extra)
    return {"uuid": uuid, "role": "assistant", "content": content}


def test_tool_call_thinking_text_and_error_blocks_are_parsed() -> None:
    payload = {
        "uuid": "design-2",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [
            _user_message("u1", "build it"),
            _assistant_message(
                "a1",
                [
                    {"type": "thinking", "text": "let me plan"},
                    {"type": "text", "text": "Working on it."},
                    {
                        "type": "tool_call",
                        "toolCall": {
                            "id": "toolu_01ABC",
                            "type": "edit",
                            "name": "write_file",
                            "input": {"path": "a.css"},
                            "output": "wrote 12 bytes",
                        },
                    },
                    {"type": "error", "message": "The model refused to respond to this request."},
                ],
                turnInputTokens=4200,
            ),
        ],
    }
    session = parse_design(payload, "fallback")
    assert len(session.messages) == 2
    assistant = session.messages[1]
    assert assistant.role is Role.ASSISTANT
    assert assistant.input_tokens == 4200
    types = [block.type for block in assistant.blocks]
    assert types == [BlockType.THINKING, BlockType.TEXT, BlockType.TOOL_USE, BlockType.TOOL_RESULT, BlockType.TEXT]

    tool_use, tool_result = assistant.blocks[2], assistant.blocks[3]
    assert tool_use.tool_id == "toolu_01ABC"
    assert tool_use.tool_name == "write_file"
    assert tool_use.tool_input == {"path": "a.css"}
    assert tool_result.tool_id == "toolu_01ABC"
    assert tool_result.text == "wrote 12 bytes"

    error_block = assistant.blocks[4]
    assert error_block.is_error is True
    assert error_block.text == "The model refused to respond to this request."

    # text field concatenates only real TEXT segments, not the error text
    assert assistant.text == "Working on it."


def test_tool_call_with_no_output_yields_only_tool_use() -> None:
    payload = {
        "uuid": "design-3",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [
            _user_message("u1", "go"),
            _assistant_message(
                "a1",
                [
                    {
                        "type": "tool_call",
                        "toolCall": {"id": "toolu_02", "type": "edit", "name": "web_fetch", "input": {"url": "x"}},
                    }
                ],
            ),
        ],
    }
    session = parse_design(payload, "fallback")
    assert [block.type for block in session.messages[1].blocks] == [BlockType.TOOL_USE]


def test_assistant_turn_with_only_tool_calls_is_not_dropped() -> None:
    """Regression for bd polylogue-je9t: the pre-rewrite parser only ever
    read ``content.content`` (a flat string), which is empty for a turn made
    entirely of tool calls -- 7 of 95 real messages were silently dropped
    this way. contentBlocks-based parsing must keep the message."""
    payload = {
        "uuid": "design-4",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [
            _user_message("u1", "go"),
            _assistant_message(
                "a1",
                [
                    {
                        "type": "tool_call",
                        "toolCall": {"id": "toolu_1", "type": "edit", "name": "snip", "input": {}, "output": "ok"},
                    },
                    {
                        "type": "tool_call",
                        "toolCall": {
                            "id": "toolu_2",
                            "type": "edit",
                            "name": "web_fetch",
                            "input": {"url": "y"},
                            "output": "fetched",
                        },
                    },
                ],
            ),
        ],
    }
    session = parse_design(payload, "fallback")
    assert len(session.messages) == 2
    assert session.messages[1].text is None
    assert len(session.messages[1].blocks) == 4


def test_user_attachment_only_message_is_not_dropped() -> None:
    """Regression for bd polylogue-je9t: a user message with empty ``content``
    but real attachments was previously dropped by the empty-text guard."""
    payload = {
        "uuid": "design-5",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [
            {
                "uuid": "u1",
                "role": "user",
                "content": {
                    "role": "user",
                    "content": "",
                    "attachments": [{"id": "att-1", "name": "brief.md", "type": "text", "content": "brief text"}],
                    "timestamp": "2026-07-09T11:00:00.000Z",
                },
            }
        ],
    }
    session = parse_design(payload, "fallback")
    assert len(session.messages) == 1
    assert session.messages[0].text is None
    assert len(session.attachments) == 1
    assert session.attachments[0].provider_attachment_id == "att-1"
    assert session.attachments[0].inline_bytes == b"brief text"


def test_skill_and_folder_attachment_kinds_are_tagged() -> None:
    payload = {
        "uuid": "design-6",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [
            {
                "uuid": "u1",
                "role": "user",
                "content": {
                    "role": "user",
                    "content": "use these",
                    "attachments": [
                        {"id": "skill-1", "name": "Make a deck", "type": "skill", "content": "Build slides."},
                        {"id": "folder-1", "name": "polylogue", "type": "folder"},
                    ],
                    "timestamp": "2026-07-09T11:00:00.000Z",
                },
            }
        ],
    }
    session = parse_design(payload, "fallback")
    by_id = {attachment.provider_attachment_id: attachment for attachment in session.attachments}
    assert by_id["skill-1"].attachment_kind == "skill"
    assert by_id["skill-1"].inline_bytes == b"Build slides."
    assert by_id["folder-1"].attachment_kind == "folder"


def test_turn_changes_become_a_session_event() -> None:
    payload = {
        "uuid": "design-7",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [
            _user_message("u1", "go"),
            _assistant_message(
                "a1",
                [{"type": "text", "text": "done"}],
                turnChanges={
                    "created": ["support.js", "index.html"],
                    "edited": [],
                    "deleted": [],
                    "moved": [],
                },
            ),
        ],
    }
    session = parse_design(payload, "fallback")
    events = [event for event in session.session_events if event.event_type == "claude_design_turn_changes"]
    assert len(events) == 1
    event = events[0]
    assert event.payload["created"] == ["support.js", "index.html"]
    assert event.payload["edited"] == []
    assert event.source_message_provider_id == session.messages[1].provider_message_id


def test_message_author_becomes_a_session_event_and_sender_name() -> None:
    payload = {
        "uuid": "design-8",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [_user_message("u1", "hi")],
    }
    session = parse_design(payload, "fallback")
    assert session.messages[0].sender_name == "Sinity"
    author_events = [event for event in session.session_events if event.event_type == "claude_design_message_author"]
    assert len(author_events) == 1
    assert author_events[0].payload == {"author_account_uuid": "acct-1", "author_name": "Sinity"}


def test_user_interjection_splits_the_assistant_turn_and_preserves_ordering() -> None:
    """bd polylogue-tbun AC4: a user_interjection nested inside an assistant
    turn must land as a real role=user message physically between the two
    half-turns, not flattened into an ordinary same-position message."""
    payload = {
        "uuid": "design-9",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [
            _user_message("u1", "start"),
            _assistant_message(
                "a1",
                [
                    {"type": "text", "text": "repeat"},
                    {
                        "type": "user_interjection",
                        "message": {
                            "id": "interjection-1",
                            "role": "user",
                            "content": "actually, do X instead",
                            "attachments": [],
                            "timestamp": "2026-04-22T16:30:16.922Z",
                        },
                    },
                    {"type": "text", "text": "repeat"},
                ],
            ),
        ],
    }
    session = parse_design(payload, "fallback")
    roles = [message.role for message in session.messages]
    texts = [message.text for message in session.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]
    assert texts == ["start", "repeat", "actually, do X instead", "repeat"]
    assert session.messages[2].provider_message_id == "interjection-1"
    # Segments have no independent provider id. Their content-derived ids are
    # stable across re-export ordering and do not encode the segment position.
    assert session.messages[1].provider_message_id == ""
    assert session.messages[3].provider_message_id == ""
    # active leaf is the true last message, not the raw turn's nominal end
    assert session.active_leaf_message_provider_id is None
    assert session.messages[-1].is_active_leaf is True
    assert sum(message.is_active_leaf is True for message in session.messages) == 1


def test_unrecognized_content_block_type_is_dropped_not_guessed(caplog: Any) -> None:
    payload = {
        "uuid": "design-10",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [
            _user_message("u1", "go"),
            _assistant_message(
                "a1",
                [{"type": "text", "text": "known"}, {"type": "some_future_block_type", "text": "unknown"}],
            ),
        ],
    }
    session = parse_design(payload, "fallback")
    assert len(session.messages[1].blocks) == 1
    assert session.messages[1].blocks[0].text == "known"


def test_title_is_origin_when_provider_asserts_one() -> None:
    payload = {
        "uuid": "design-11",
        "title": "Chat",
        "project": {"uuid": "p1", "name": "Proj"},
        "messages": [_user_message("u1", "hi")],
    }
    session = parse_design(payload, "fallback")
    assert session.title == "Chat"
    assert session.title_source is TitleSource.ORIGIN
