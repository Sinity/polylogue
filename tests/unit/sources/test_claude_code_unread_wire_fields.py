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


def test_team_name_is_captured_once_as_a_session_event() -> None:
    """The top-level ``teamName`` is session identity, not message metadata."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-team",
                "teamName": "structural-consolidation",
                "message": {"role": "user", "content": "hello"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-team",
                "teamName": "structural-consolidation",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            },
        ],
        "sess-team",
    )
    team_events = [event for event in parsed.session_events if event.event_type == "claude_team_name"]
    assert parsed.team_name == "structural-consolidation"
    assert len(team_events) == 1
    assert team_events[0].payload["team_name"] == "structural-consolidation"


def test_advisor_model_is_kept_on_the_usage_turn() -> None:
    """The real top-level ``advisorModel`` shape remains linked to its turn."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a-advisor",
                "sessionId": "sess-advisor",
                "advisorModel": "claude-opus-4-7",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "advised"}],
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            },
        ],
        "sess-advisor",
    )
    advisor_events = [event for event in parsed.session_events if event.event_type == "claude_advisor_model"]
    assert len(advisor_events) == 1
    assert advisor_events[0].source_message_provider_id == "a-advisor"
    assert advisor_events[0].payload["advisor_model"] == "claude-opus-4-7"
    usage_events = [event for event in parsed.session_events if event.event_type == "message_usage"]
    assert usage_events[0].payload["advisor_model"] == "claude-opus-4-7"


def test_aborted_mid_stream_is_a_linked_turn_outcome() -> None:
    """Top-level ``isAbortedMidStream`` must not read as a complete answer."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a-aborted",
                "sessionId": "sess-aborted",
                "isAbortedMidStream": True,
                "message": {"role": "assistant", "content": [{"type": "text", "text": "partial"}]},
            },
        ],
        "sess-aborted",
    )
    aborted_events = [event for event in parsed.session_events if event.event_type == "claude_aborted_mid_stream"]
    assert parsed.messages[0].is_aborted_mid_stream is True
    assert len(aborted_events) == 1
    assert aborted_events[0].source_message_provider_id == "a-aborted"
    assert aborted_events[0].payload["aborted"] is True


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


def test_thinking_metadata_lands_on_its_own_event_from_a_user_record() -> None:
    """``thinkingMetadata`` must reach ``claude_thinking_budget``.

    The fixture is the shape the corpus actually carries: a ``user`` record
    with no ``message.usage`` (3,620 occurrences over 14,536 session files, 0
    of them on an assistant record and 0 carrying usage). Deleting the
    ``_thinking_budget_payload`` call in ``_fold_code_record`` leaves no
    event at all.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-thinking-metadata",
                "thinkingMetadata": {"maxThinkingTokens": 31999},
                "message": {"role": "user", "content": "go"},
            },
        ],
        "sess-thinking-metadata",
    )
    events = [e for e in parsed.session_events if e.event_type == "claude_thinking_budget"]
    assert len(events) == 1
    assert events[0].source_message_provider_id == "u1"
    assert events[0].payload["max_thinking_tokens"] == 31999


def test_thinking_metadata_level_and_triggers_land_on_the_event() -> None:
    """The second live shape: the effort level and the prompt span that raised it.

    ``triggers`` names where in the user's own prompt the escalation token
    appeared -- dropping it would leave the level with no evidence of why it
    was set. ``disabled`` is false on every record measured and is recorded
    only when true, so it must be absent here.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-thinking-level",
                "thinkingMetadata": {
                    "level": "high",
                    "disabled": False,
                    "triggers": [{"start": 0, "end": 10, "text": "ultrathink"}],
                },
                "message": {"role": "user", "content": "ultrathink about the parser"},
            },
        ],
        "sess-thinking-level",
    )
    events = [e for e in parsed.session_events if e.event_type == "claude_thinking_budget"]
    assert len(events) == 1
    assert events[0].payload["level"] == "high"
    assert events[0].payload["triggers"] == [{"start": 0, "end": 10, "text": "ultrathink"}]
    assert "disabled" not in events[0].payload


def test_thinking_metadata_absent_emits_no_event() -> None:
    """Anti-vacuity: no ``thinkingMetadata`` on the record must not fabricate one."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-no-thinking-metadata",
                "message": {"role": "user", "content": "go"},
            },
        ],
        "sess-no-thinking-metadata",
    )
    assert [e for e in parsed.session_events if e.event_type == "claude_thinking_budget"] == []


def test_thinking_metadata_negative_tokens_omits_the_key() -> None:
    """A negative ``maxThinkingTokens`` is not a real token budget -- omit it
    rather than persisting a nonsensical value (CodeRabbit review, PR #3465)."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-negative-thinking-metadata",
                "thinkingMetadata": {"maxThinkingTokens": -1},
                "message": {"role": "user", "content": "go"},
            },
        ],
        "sess-negative-thinking-metadata",
    )
    assert [e for e in parsed.session_events if e.event_type == "claude_thinking_budget"] == []


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


def test_capability_attribution_skill_and_plugin_land_on_their_own_event() -> None:
    """The ``attribution*`` cluster must reach ``claude_capability_attribution``.

    The fixture is the shape the corpus carries: top-level keys on a
    ``type:"assistant"`` record, not nested under ``message``, and with no
    ``message.usage`` -- the event must not be gated on usage the way
    ``request_id`` is. Deleting the ``_capability_attribution_payload`` call
    in ``_fold_code_record`` leaves no event at all.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-attribution-skill",
                "attributionSkill": "superpowers:dispatching-parallel-agents",
                "attributionPlugin": "superpowers",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "dispatching"}]},
            },
        ],
        "sess-attribution-skill",
    )
    assert [e for e in parsed.session_events if e.event_type == "message_usage"] == []
    events = [e for e in parsed.session_events if e.event_type == "claude_capability_attribution"]
    assert len(events) == 1
    assert events[0].source_message_provider_id == "a1"
    assert events[0].payload["skill"] == "superpowers:dispatching-parallel-agents"
    assert events[0].payload["plugin"] == "superpowers"
    assert events[0].payload["summary"] == "superpowers:dispatching-parallel-agents"
    assert "mcp_server" not in events[0].payload
    assert "agent" not in events[0].payload


def test_capability_attribution_mcp_pair_names_the_called_tool() -> None:
    """``attributionMcpServer``/``attributionMcpTool`` co-occur on every record
    that carries either (7,683 each, the same rows) and together name one MCP
    call -- the summary must join them rather than keep only the server."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-attribution-mcp",
                "attributionMcpServer": "context7",
                "attributionMcpTool": "resolve-library-id",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "looking it up"}]},
            },
        ],
        "sess-attribution-mcp",
    )
    events = [e for e in parsed.session_events if e.event_type == "claude_capability_attribution"]
    assert len(events) == 1
    assert events[0].payload["mcp_server"] == "context7"
    assert events[0].payload["mcp_tool"] == "resolve-library-id"
    assert events[0].payload["summary"] == "context7:resolve-library-id"


def test_capability_attribution_plugin_without_a_skill_is_kept() -> None:
    """``attributionPlugin`` is not redundant with ``attributionSkill``: a
    plugin-shipped skill carries the plugin prefix in its own value, but
    ``feature-dev`` stamps the plugin on 541 records with no skill at all.
    Dropping plugin as derivable from skill loses those turns entirely."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-attribution-plugin",
                "attributionPlugin": "feature-dev",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "batching"}]},
            },
        ],
        "sess-attribution-plugin",
    )
    events = [e for e in parsed.session_events if e.event_type == "claude_capability_attribution"]
    assert len(events) == 1
    assert events[0].payload["plugin"] == "feature-dev"
    assert events[0].payload["summary"] == "feature-dev"
    assert "skill" not in events[0].payload


def test_capability_attribution_agent_is_read_per_turn_not_per_session() -> None:
    """A subagent transcript that replays a parent's prefix carries the parent's
    agent on the replayed head and its own on the divergent tail. Collapsing
    ``attributionAgent`` to one session-level value -- or reading only the
    first or the last record's -- loses the split this asserts.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-attribution-agent",
                "attributionAgent": "triage",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "replayed prefix"}]},
            },
            {
                "type": "assistant",
                "uuid": "a2",
                "sessionId": "sess-attribution-agent",
                "attributionAgent": "fork",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "divergent tail"}]},
            },
        ],
        "sess-attribution-agent",
    )
    events = [e for e in parsed.session_events if e.event_type == "claude_capability_attribution"]
    assert [(e.source_message_provider_id, e.payload["agent"]) for e in events] == [("a1", "triage"), ("a2", "fork")]
    assert [e.payload["summary"] for e in events] == ["triage", "fork"]


def test_capability_attribution_survives_the_empty_content_drop() -> None:
    """An assistant record whose message never materializes -- an API error
    row with no content -- still carries the capability that produced the
    attempt. Emitting after the empty-content drop instead of before it would
    lose exactly these turns.
    """
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-attribution-empty",
                "attributionSkill": "code-review",
                "isApiErrorMessage": True,
                "message": {"role": "assistant", "content": []},
            },
        ],
        "sess-attribution-empty",
    )
    assert parsed.messages == []
    events = [e for e in parsed.session_events if e.event_type == "claude_capability_attribution"]
    assert len(events) == 1
    assert events[0].payload["skill"] == "code-review"


def test_capability_attribution_absent_emits_no_event() -> None:
    """Anti-vacuity: a record carrying none of the five fields must not
    fabricate an event, and a blank value is not a capability."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-no-attribution",
                "attributionSkill": "",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            },
        ],
        "sess-no-attribution",
    )
    assert [e for e in parsed.session_events if e.event_type == "claude_capability_attribution"] == []


def test_unknown_stop_reason_remains_evidence_without_entering_constrained_column() -> None:
    """An unrecognized provider token must not make the archive writer reject the session."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "unknown-stop",
                "sessionId": "stop-session",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "synthetic"}],
                    "stop_reason": "new_provider_reason",
                    "usage": {"input_tokens": 1},
                },
            }
        ],
        "stop-session",
    )
    assert len(parsed.messages) == 1
    assert parsed.messages[0].stop_reason is None
    assert any(event.payload.get("stop_reason") == "new_provider_reason" for event in parsed.session_events)
