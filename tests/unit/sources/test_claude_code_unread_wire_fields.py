"""Coverage for polylogue-2qx.4 / polylogue-cgfy: the unread-wire batch (index v46).

Each test drives the REAL production parser (``polylogue.sources.parsers.
claude.code_parser.parse_code``) on a synthesized fixture shaped like the real
wire, asserting the value the parser contract (``base_models.py``) declares is
actually populated. These pin the parser side of the v46 batch; the writer/
reader side is pinned separately by
``tests/unit/storage/test_unread_wire_batch_v46.py``.
"""

from __future__ import annotations

from polylogue.core.enums import BlockType
from polylogue.sources.parsers.claude import parse_code


def test_stop_reason_lands_on_the_assistant_message() -> None:
    """``message.stop_reason`` must reach ``ParsedMessage.stop_reason``.

    Deleting the ``msg_stop_reason`` extraction/assignment in
    ``_parse_code_records`` (a revert to only feeding it into the
    ``message_usage`` event payload) makes this assert ``None``.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-stop-reason",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                },
            },
        ],
        "sess-stop-reason",
    )
    assert len(parsed.messages) == 1
    assert parsed.messages[0].stop_reason == "end_turn"


def test_stop_reason_absent_stays_none() -> None:
    """Anti-vacuity: a record with no ``stop_reason`` must not fabricate one."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-no-stop-reason",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            },
        ],
        "sess-no-stop-reason",
    )
    assert parsed.messages[0].stop_reason is None


def test_slug_lands_as_session_display_name() -> None:
    """The top-level ``slug`` field (stamped on every record) becomes ``display_name``.

    Deleting the ``session_slug_value`` tracking block or its wiring into the
    ``ParsedSession`` return makes this assert ``None`` -- exactly the
    "5ecdb160-...:agent-af4e" vs "greedy-squishing-hamming" gap polylogue-cgfy
    measured.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "5ecdb160-agent",
                "slug": "greedy-squishing-hamming",
                "message": {"role": "user", "content": "hello"},
            },
        ],
        "5ecdb160-agent",
    )
    assert parsed.display_name == "greedy-squishing-hamming"


def test_slug_absent_leaves_display_name_none() -> None:
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-no-slug",
                "message": {"role": "user", "content": "hello"},
            },
        ],
        "sess-no-slug",
    )
    assert parsed.display_name is None


def test_pr_link_becomes_a_typed_session_ref() -> None:
    """``pr-link`` must persist as a tracker-agnostic ``ParsedSessionRef``, not only an event.

    Deleting the ``pr-link`` branch added to the skipped-sidecar-record
    handling in ``_parse_code_records`` makes ``session_refs`` empty even
    though the existing ``claude_pr_link`` event still fires.
    """
    parsed = parse_code(
        [
            {
                "type": "pr-link",
                "sessionId": "sess-pr",
                "prNumber": 3126,
                "prUrl": "https://github.com/Sinity/polylogue/pull/3126",
                "prRepository": "Sinity/polylogue",
            },
        ],
        "sess-pr",
    )
    assert len(parsed.session_refs) == 1
    ref = parsed.session_refs[0]
    assert ref.kind == "pull_request"
    assert ref.url == "https://github.com/Sinity/polylogue/pull/3126"
    assert ref.repo == "Sinity/polylogue"
    assert ref.number == 3126
    # The pre-existing audit-trail event must still fire (dual evidence).
    pr_events = [e for e in parsed.session_events if e.event_type == "claude_pr_link"]
    assert len(pr_events) == 1


def test_file_edit_attaches_to_the_tool_result_block() -> None:
    """``toolUseResult``'s structuredPatch/originalFile/oldString/newString must
    reach ``ParsedContentBlock.file_edit`` on the TOOL_RESULT block, not just
    the bounded hunk-count summary in ``claude_tool_execution_result``.

    Deleting ``_file_edit_from_tool_result``/``_attach_file_edit`` or their
    wiring into the record loop makes ``file_edit`` None on every block.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-file-edit",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "edit-tool-1", "name": "Edit", "input": {"file_path": "/tmp/x.py"}}
                    ],
                },
            },
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-file-edit",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "edit-tool-1", "content": "applied", "is_error": False}
                    ],
                },
                "toolUseResult": {
                    "filePath": "/tmp/x.py",
                    "originalFile": "old contents\n",
                    "oldString": "old",
                    "newString": "new",
                    "replaceAll": False,
                    "userModified": True,
                    "structuredPatch": [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}],
                },
            },
        ],
        "sess-file-edit",
    )
    result_blocks = [b for m in parsed.messages for b in m.blocks if b.type is BlockType.TOOL_RESULT]
    assert len(result_blocks) == 1
    edit = result_blocks[0].file_edit
    assert edit is not None
    assert edit.file_path == "/tmp/x.py"
    assert edit.original_file == "old contents\n"
    assert edit.old_string == "old"
    assert edit.new_string == "new"
    assert edit.replace_all is False
    assert edit.user_modified is True
    assert edit.structured_patch == [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}]


def test_tool_result_without_edit_shaped_fields_has_no_file_edit() -> None:
    """Anti-vacuity: a Bash result's ``toolUseResult`` must not fabricate an edit."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-no-edit",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "bash-1", "name": "Bash", "input": {"command": "ls"}}],
                },
            },
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-no-edit",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "bash-1", "content": "file.py", "is_error": False}
                    ],
                },
                "toolUseResult": {"stdout": "file.py", "stderr": "", "interrupted": False, "isImage": False},
            },
        ],
        "sess-no-edit",
    )
    result_blocks = [b for m in parsed.messages for b in m.blocks if b.type is BlockType.TOOL_RESULT]
    assert len(result_blocks) == 1
    assert result_blocks[0].file_edit is None


def test_tool_result_missing_is_error_gets_not_reported_reason() -> None:
    """A tool_result segment with no ``is_error`` key gets NOT_REPORTED, not a bare unknown.

    Deleting the ``outcome_unknown_reason`` assignment in
    ``content_blocks_from_segments`` (base_support.py) makes this None.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-not-reported",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": "echo hi"}}],
                },
            },
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-not-reported",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "hi"}],
                },
            },
        ],
        "sess-not-reported",
    )
    result_blocks = [b for m in parsed.messages for b in m.blocks if b.type is BlockType.TOOL_RESULT]
    assert len(result_blocks) == 1
    assert result_blocks[0].is_error is None
    assert result_blocks[0].outcome_unknown_reason == "not_reported"


def test_request_id_lands_on_message_usage_event() -> None:
    """The top-level ``requestId`` (Anthropic API request id, 1,171 sampled
    occurrences per polylogue-cgfy) must reach the ``message_usage`` session
    event as ``request_id``.

    Deleting the ``record=item`` wiring at the ``message_usage`` append site
    (or the ``requestId`` extraction inside ``_message_usage_event_payload``)
    makes this key absent from every event.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-request-id",
                "requestId": "req_011CPuYvnLASUV8W7nChG4jH",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        ],
        "sess-request-id",
    )
    usage_events = [e for e in parsed.session_events if e.event_type == "message_usage"]
    assert len(usage_events) == 1
    assert usage_events[0].payload["request_id"] == "req_011CPuYvnLASUV8W7nChG4jH"


def test_request_id_absent_omits_the_key() -> None:
    """Anti-vacuity: no ``requestId`` on the record must not fabricate one."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-no-request-id",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        ],
        "sess-no-request-id",
    )
    usage_events = [e for e in parsed.session_events if e.event_type == "message_usage"]
    assert len(usage_events) == 1
    assert "request_id" not in usage_events[0].payload


def test_thinking_metadata_max_tokens_lands_on_message_usage_event() -> None:
    """``thinkingMetadata.maxThinkingTokens`` (extended-thinking budget) must
    reach the ``message_usage`` event as ``max_thinking_tokens``.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-thinking-metadata",
                "thinkingMetadata": {"maxThinkingTokens": 31999},
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        ],
        "sess-thinking-metadata",
    )
    usage_events = [e for e in parsed.session_events if e.event_type == "message_usage"]
    assert len(usage_events) == 1
    assert usage_events[0].payload["max_thinking_tokens"] == 31999


def test_thinking_metadata_absent_omits_the_key() -> None:
    """Anti-vacuity: no ``thinkingMetadata`` on the record must not fabricate one."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-no-thinking-metadata",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        ],
        "sess-no-thinking-metadata",
    )
    usage_events = [e for e in parsed.session_events if e.event_type == "message_usage"]
    assert len(usage_events) == 1
    assert "max_thinking_tokens" not in usage_events[0].payload


def test_thinking_metadata_negative_tokens_omits_the_key() -> None:
    """A negative ``maxThinkingTokens`` is not a real token budget -- omit it
    rather than persisting a nonsensical value (CodeRabbit review, PR #3465)."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-negative-thinking-metadata",
                "thinkingMetadata": {"maxThinkingTokens": -1},
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        ],
        "sess-negative-thinking-metadata",
    )
    usage_events = [e for e in parsed.session_events if e.event_type == "message_usage"]
    assert len(usage_events) == 1
    assert "max_thinking_tokens" not in usage_events[0].payload


def test_background_task_start_ack_gets_distrusted_reason() -> None:
    """The backgrounded-task start acknowledgement's ``is_error=false`` is
    positively distrusted (it only confirms the task started), not merely
    absent -- must record DISTRUSTED, matching ``_mark_background_task_start``.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-distrusted",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "bg-tool-1", "name": "Bash", "input": {"command": "sleep 100 &"}}
                    ],
                },
            },
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-distrusted",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "bg-tool-1", "content": "started", "is_error": False}
                    ],
                },
                "toolUseResult": {"backgroundTaskId": "bgtask-1"},
            },
        ],
        "sess-distrusted",
    )
    result_blocks = [b for m in parsed.messages for b in m.blocks if b.type is BlockType.TOOL_RESULT]
    assert len(result_blocks) == 1
    assert result_blocks[0].is_error is None
    assert result_blocks[0].outcome_unknown_reason == "distrusted"
