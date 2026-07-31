"""Coverage for polylogue-pbuh: sidecar records must persist as typed evidence.

Each test exercises ``polylogue.sources.parsers.claude.code_parser``, the
sole production code path that decides what happens to a Claude Code JSONL
record whose ``type`` is in ``_SKIPPED_SIDECAR_RECORD_TYPES``. Before this
bead, every record type below was silently dropped by a bare frozenset
membership check with no per-type rationale; these tests pin that dropping
one of the evidence-bearing types (or reverting the ai-title title-override)
reproduces exactly that silent-loss bug.
"""

from __future__ import annotations

from polylogue.core.enums import TitleSource
from polylogue.sources.parsers.base import ParsedSession, ParsedSessionEvent
from polylogue.sources.parsers.claude import parse_code

# polylogue-pbuh AC5: every parse now also emits one bounded
# ``claude_parse_coverage`` event per session (seen/persisted counts by
# record type). It is orthogonal to the specific-record-type behavior each
# test below pins, so the assertions here look through it rather than
# hard-coding it into every expected event list -- see
# ``test_parse_coverage_event_reports_seen_and_persisted_counts`` for the
# dedicated coverage-event test.
_COVERAGE_EVENT_TYPE = "claude_parse_coverage"


def _typed_events(session: ParsedSession) -> list[ParsedSessionEvent]:
    return [e for e in session.session_events if e.event_type != _COVERAGE_EVENT_TYPE]


def test_ai_title_wins_session_title_over_uuid_fallback() -> None:
    """``ai-title`` must resolve the session title, not just exist as a sidecar.

    Deleting the ``latest_ai_title`` precedence block in
    ``_parse_code_records`` (or reverting to dropping ``ai-title`` records
    entirely) makes this session fall back to the raw UUID title -- the
    exact defect polylogue-pbuh measured as 84.6% of Claude Code sessions.
    """
    parsed = parse_code(
        [
            {
                "type": "ai-title",
                "sessionId": "sess-title",
                "aiTitle": "Recover what was lost",
            },
        ],
        "sess-title",
    )
    assert parsed.title == "Recover what was lost"
    assert parsed.title_source is TitleSource.ORIGIN
    assert parsed.title_ref == "claude-ai-title:sess-title"
    assert parsed.title_confidence == 1.0
    # Also persisted as an independently queryable audit-trail event, not
    # only consumed for title resolution -- AC2 requires it survive as typed
    # evidence in its own right.
    ai_title_events = [e for e in parsed.session_events if e.event_type == "claude_ai_title"]
    assert [e.payload for e in ai_title_events] == [
        {"ai_title": "Recover what was lost", "summary": "Recover what was lost"}
    ]


def test_ai_title_absent_falls_back_to_heuristic_title() -> None:
    """Without an ``ai-title`` record, the pre-existing heuristic still wins.

    Anti-vacuity: this pins the *other* side of the precedence -- if the
    ai-title override block ran unconditionally (e.g. on an empty string),
    it would clobber this heuristic-derived title.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-no-title",
                "message": {"role": "user", "content": "please refactor the parser"},
            },
        ],
        "sess-no-title",
    )
    assert parsed.title == "please refactor the parser"
    assert parsed.title_source is TitleSource.HEURISTIC


def test_agent_name_persists_as_typed_event() -> None:
    """Removing the ``agent-name`` branch of ``_sidecar_evidence_payload``
    (or its entry in ``_SIDECAR_EVENT_TYPES``) makes this list empty."""
    parsed = parse_code(
        [{"type": "agent-name", "sessionId": "sess-agent", "agentName": "orchestration-docs-6np"}],
        "sess-agent",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        ("claude_agent_name", {"agent_name": "orchestration-docs-6np", "summary": "orchestration-docs-6np"})
    ]


def test_agent_name_wins_session_title_over_uuid_fallback() -> None:
    """``agent-name`` must resolve the session title, not only exist as a
    sidecar event -- polylogue-pbuh AC3: a background/agent-mode session's
    provider-assigned label (a readable task name, e.g.
    "orchestration-docs-6np") is a strictly better title than the raw
    ``sessionId``/``sessionId:agent-suffix`` fallback this loop otherwise
    leaves in place. Deleting the ``latest_agent_name`` precedence block
    reverts this session's title to the raw session id.
    """
    parsed = parse_code(
        [{"type": "agent-name", "sessionId": "sess-agent", "agentName": "orchestration-docs-6np"}],
        "sess-agent",
    )
    assert parsed.title == "orchestration-docs-6np"
    assert parsed.title_source is TitleSource.ORIGIN
    assert parsed.title_ref == "claude-agent-name:sess-agent"
    assert parsed.title_confidence == 0.9


def test_agent_name_yields_to_ai_title_and_custom_title() -> None:
    """``agent-name`` is a fallback, not the strongest signal: an explicit
    provider ``ai-title`` or user ``custom-title`` on the same session still
    wins -- this pins the precedence order rather than only the presence of
    each individual override.
    """
    parsed = parse_code(
        [
            {"type": "agent-name", "sessionId": "sess-agent", "agentName": "orchestration-docs-6np"},
            {"type": "ai-title", "sessionId": "sess-agent", "aiTitle": "Provider suggested title"},
        ],
        "sess-agent",
    )
    assert parsed.title == "Provider suggested title"
    assert parsed.title_ref == "claude-ai-title:sess-agent"


def test_pr_link_persists_typed_pr_fields() -> None:
    """Deleting the pr-link branch collapses this to an empty session_events
    list -- exactly the producer gap polylogue-pbuh AC4 names."""
    parsed = parse_code(
        [
            {
                "type": "pr-link",
                "sessionId": "sess-pr",
                "prNumber": 3126,
                "prUrl": "https://github.com/Sinity/polylogue/pull/3126",
                "prRepository": "Sinity/polylogue",
            }
        ],
        "sess-pr",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_pr_link",
            {
                "pr_number": 3126,
                "pr_url": "https://github.com/Sinity/polylogue/pull/3126",
                "pr_repository": "Sinity/polylogue",
                "summary": "PR #3126: https://github.com/Sinity/polylogue/pull/3126",
            },
        )
    ]


def test_bridge_session_persists_cross_session_link() -> None:
    parsed = parse_code(
        [
            {
                "type": "bridge-session",
                "sessionId": "sess-bridge",
                "bridgeSessionId": "cse_01YHHspKPVi2QYy1na2Cgvos",
                "lastSequenceNum": 4,
            }
        ],
        "sess-bridge",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_bridge_session",
            {
                "bridge_session_id": "cse_01YHHspKPVi2QYy1na2Cgvos",
                "last_sequence_num": 4,
                "summary": "cse_01YHHspKPVi2QYy1na2Cgvos",
            },
        )
    ]


def test_file_history_snapshot_persists_tracked_file_count() -> None:
    """Reverting to dropping this type loses the 'checkpointed' trajectory
    evidence entirely -- the file paths and count below would vanish."""
    parsed = parse_code(
        [
            {
                "type": "file-history-snapshot",
                "sessionId": "sess-fhs",
                "messageId": "msg-1",
                "isSnapshotUpdate": False,
                "snapshot": {
                    "trackedFileBackups": {
                        "/realm/project/sinnix/modules/foundation.nix": {"version": 1},
                        "/realm/project/sinnix/hosts/sinnix-prime/storage.nix": {"version": 1},
                    },
                    "timestamp": "2026-02-05T17:32:58.442Z",
                },
            }
        ],
        "sess-fhs",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_file_history_snapshot",
            {
                "is_snapshot_update": False,
                "file_count": 2,
                "files": [
                    "/realm/project/sinnix/hosts/sinnix-prime/storage.nix",
                    "/realm/project/sinnix/modules/foundation.nix",
                ],
                "summary": "2 tracked file backup(s)",
            },
        )
    ]


def test_permission_mode_persists_operational_signal() -> None:
    """permission-mode values genuinely vary in the live corpus (auto,
    default, acceptEdits, plan, bypassPermissions) -- unlike ``mode``, which
    stays transient because it was a corpus-wide constant."""
    parsed = parse_code(
        [{"type": "permission-mode", "sessionId": "sess-perm", "permissionMode": "bypassPermissions"}],
        "sess-perm",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        ("claude_permission_mode", {"permission_mode": "bypassPermissions", "summary": "bypassPermissions"})
    ]


def test_last_prompt_persists_resume_continuity_signal() -> None:
    parsed = parse_code(
        [{"type": "last-prompt", "sessionId": "sess-lp", "lastPrompt": "hello world."}],
        "sess-lp",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [("claude_last_prompt", {"last_prompt": "hello world.", "summary": "hello world."})]


def test_queue_operation_enqueue_persists_content_dequeue_does_not() -> None:
    """enqueue carries the actual drafted prompt text (real evidence);
    dequeue/remove/popAll carry no content field at all in the live corpus --
    this pins that both operations still get a summary-only event rather
    than being silently dropped, without fabricating content for the ones
    that never had any."""
    parsed = parse_code(
        [
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "sessionId": "sess-queue",
                "content": "hello world.",
            },
            {
                "type": "queue-operation",
                "operation": "dequeue",
                "sessionId": "sess-queue",
            },
        ],
        "sess-queue",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_queue_operation",
            {"operation": "enqueue", "content": "hello world.", "summary": "enqueue: hello world."},
        ),
        ("claude_queue_operation", {"operation": "dequeue", "summary": "dequeue"}),
    ]


def test_attachment_persists_generic_typed_payload() -> None:
    """attachment.type covers 20 distinct shapes in the live corpus; this
    pins that the generic pass-through keeps the full typed payload (not an
    opaque blob truncation) for at least one representative shape."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-att",
                "attachment": {"type": "file", "path": "/tmp/example.txt", "sizeBytes": 42},
            }
        ],
        "sess-att",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_attachment",
            {"type": "file", "path": "/tmp/example.txt", "sizeBytes": 42, "summary": "file"},
        )
    ]


def test_progress_agent_progress_dedups_into_one_delegation_event() -> None:
    """Three ``agent_progress`` ticks under the same dispatching tool_use
    collapse into ONE ``claude_delegation_progress`` event with a tick count
    -- not three duplicated rows carrying the full re-streamed subagent
    message payload. Reverting the dedup accumulator (persisting one event
    per progress record instead) makes the event count assertion fail.
    """
    records: list[object] = [
        {
            "type": "progress",
            "sessionId": "sess-progress",
            "toolUseID": "agent_msg_1",
            "parentToolUseID": "toolu_dispatch_1",
            "timestamp": "2026-02-05T17:37:04.000Z",
            "data": {"type": "agent_progress", "message": {"type": "assistant"}},
        },
        {
            "type": "progress",
            "sessionId": "sess-progress",
            "toolUseID": "agent_msg_2",
            "parentToolUseID": "toolu_dispatch_1",
            "timestamp": "2026-02-05T17:37:05.000Z",
            "data": {"type": "agent_progress", "message": {"type": "assistant"}},
        },
        {
            "type": "progress",
            "sessionId": "sess-progress",
            "toolUseID": "agent_msg_3",
            "parentToolUseID": "toolu_dispatch_1",
            "timestamp": "2026-02-05T17:37:06.000Z",
            "data": {"type": "agent_progress", "message": {"type": "assistant"}},
        },
    ]
    parsed = parse_code(records, "sess-progress")
    delegation_events = [e for e in parsed.session_events if e.event_type == "claude_delegation_progress"]
    assert len(delegation_events) == 1
    event = delegation_events[0]
    assert event.payload["parent_tool_use_id"] == "toolu_dispatch_1"
    assert event.payload["progress_tick_count"] == 3
    assert event.payload["first_seen"] == "2026-02-05T17:37:04+00:00"
    assert event.payload["last_seen"] == "2026-02-05T17:37:06+00:00"
    # No messages were produced for the progress ticks themselves -- they
    # stay non-message sidecar evidence, matching pre-existing behavior.
    assert parsed.messages == []


def test_progress_bash_progress_and_hook_progress_stay_transient() -> None:
    """bash_progress/hook_progress ticks reference an already-captured tool
    call via a synthetic per-tick id, not a real delegation edge -- adding a
    session_event for them would be pure duplication. This test would start
    failing (event list non-empty) if that classification were reverted."""
    records: list[object] = [
        {
            "type": "progress",
            "sessionId": "sess-bash",
            "toolUseID": "bash-progress-0",
            "parentToolUseID": "toolu_bash_1",
            "data": {"type": "bash_progress", "output": "", "elapsedTimeSeconds": 1},
        },
        {
            "type": "progress",
            "sessionId": "sess-bash",
            "toolUseID": "a4d08c84-hook",
            "parentToolUseID": "a4d08c84-hook",
            "data": {"type": "hook_progress", "hookEvent": "SessionStart", "hookName": "SessionStart:startup"},
        },
    ]
    parsed = parse_code(records, "sess-bash")
    assert _typed_events(parsed) == []
    assert parsed.messages == []


def test_custom_title_wins_over_ai_title() -> None:
    """An explicit user rename (``custom-title``) outranks the provider's own
    ``ai-title`` suggestion -- deleting the ``latest_custom_title`` precedence
    block makes this session keep the (weaker-intent) ai-title instead."""
    parsed = parse_code(
        [
            {"type": "ai-title", "sessionId": "sess-title", "aiTitle": "Provider suggested title"},
            {"type": "custom-title", "sessionId": "sess-title", "customTitle": "My deliberate rename"},
        ],
        "sess-title",
    )
    assert parsed.title == "My deliberate rename"
    assert parsed.title_ref == "claude-custom-title:sess-title"
    event_types = [event.event_type for event in parsed.session_events]
    assert "claude_custom_title" in event_types
    assert "claude_ai_title" in event_types


def test_file_history_delta_persists_tracking_path() -> None:
    """Removing the ``file-history-delta`` branch of ``_sidecar_evidence_payload``
    (or its ``_SIDECAR_EVENT_TYPES`` entry) makes this list empty."""
    parsed = parse_code(
        [
            {
                "type": "file-history-delta",
                "sessionId": "sess-delta",
                "messageId": "m1",
                "snapshotMessageId": "snap-1",
                "trackingPath": "src/lib.rs",
                "backup": {"backupFileName": "lib.rs.bak", "version": 2, "backupTime": "2026-01-01T00:00:00Z"},
            }
        ],
        "sess-delta",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_file_history_delta",
            {
                "snapshot_message_id": "snap-1",
                "tracking_path": "src/lib.rs",
                "backup_file_name": "lib.rs.bak",
                "backup_version": 2,
                "summary": "src/lib.rs",
            },
        )
    ]


def test_tool_use_result_structural_facts_persist() -> None:
    """``toolUseResult`` sandbox/interrupted/file facts must survive as an event.

    Removing ``_tool_execution_result_payload`` (or its call site) drops this
    to an empty event list even though the record carries real structured
    outcome data beyond the plain tool_result text block.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-toolresult",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
                "toolUseResult": {
                    "sandbox": True,
                    "interrupted": False,
                    "numFiles": 3,
                    "filenames": ["a.py", "b.py"],
                    "file": {"filePath": "a.py", "numLines": 10, "totalLines": 100},
                    "structuredPatch": [{"oldStart": 1, "oldLines": 2, "newStart": 1, "newLines": 3, "lines": ["+x"]}],
                    "stdout": "should not be persisted -- duplicate of tool_result text",
                },
            }
        ],
        "sess-toolresult",
    )
    events = [
        (e.event_type, e.payload) for e in parsed.session_events if e.event_type == "claude_tool_execution_result"
    ]
    assert len(events) == 1
    _, payload = events[0]
    assert payload["sandbox"] is True
    assert payload["interrupted"] is False
    assert payload["numFiles"] == 3
    assert payload["filenames"] == ["a.py", "b.py"]
    assert payload["file_path"] == "a.py"
    assert payload["file_numLines"] == 10
    assert payload["structured_patch_hunk_count"] == 1
    assert payload["structured_patch_lines_changed"] == 1
    assert "stdout" not in payload


def test_todo_write_result_persists_priority() -> None:
    """TodoWrite's accepted before/after state (with ``priority``) must persist.

    The tool *call* input (``content[].input.todos``) is already captured
    wholesale via ``tool_input``; this covers the *result* state transition,
    which is otherwise read nowhere.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-todo",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
                "toolUseResult": {
                    "oldTodos": [{"content": "write tests", "status": "pending", "priority": "high"}],
                    "newTodos": [{"content": "write tests", "status": "completed", "priority": "high"}],
                },
            }
        ],
        "sess-todo",
    )
    events = [(e.event_type, e.payload) for e in parsed.session_events if e.event_type == "claude_todo_state"]
    assert events == [
        (
            "claude_todo_state",
            {
                "new_todos": [{"content": "write tests", "status": "completed", "priority": "high"}],
                "old_todos": [{"content": "write tests", "status": "pending", "priority": "high"}],
            },
        )
    ]


def test_message_usage_event_carries_ttft_stop_reason_and_extended_usage_fields() -> None:
    """``message.ttftMs``, ``stop_reason``, and the extended ``usage`` fields
    (cache_creation TTL split, service_tier, inference_geo, cache_miss_reason)
    must reach the ``message_usage`` event -- these were read nowhere before."""
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "sess-usage",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "model": "claude-opus-5",
                    "ttftMs": 3817,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "diagnostics": {"cache_miss_reason": {"type": "previous_message_not_found"}},
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 167,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 50007,
                        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 50007},
                        "service_tier": "standard",
                        "inference_geo": "not_available",
                    },
                },
            }
        ],
        "sess-usage",
    )
    usage_events = [e.payload for e in parsed.session_events if e.event_type == "message_usage"]
    assert len(usage_events) == 1
    payload = usage_events[0]
    assert payload["ttft_ms"] == 3817
    assert payload["stop_reason"] == "end_turn"
    assert "stop_sequence" not in payload
    assert payload["cache_miss_reason"] == "previous_message_not_found"
    assert payload["cache_creation_by_ttl"] == {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 50007}
    assert payload["service_tier"] == "standard"
    assert payload["inference_geo"] == "not_available"


def test_session_kind_and_git_branch_persist() -> None:
    """``sessionKind``/``gitBranch`` were stamped on every record and read
    nowhere in the primary JSONL parse path (only via a separate, often-absent
    legacy sessions-index.json sidecar for gitBranch)."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-bg",
                "sessionKind": "bg",
                "gitBranch": "feature/parser-diff",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "run in background"},
            }
        ],
        "sess-bg",
    )
    assert parsed.git_branch == "feature/parser-diff"
    kind_events = [e.payload for e in parsed.session_events if e.event_type == "claude_session_kind"]
    assert kind_events == [{"session_kind": "bg"}]


def test_init_and_mode_records_produce_no_events_or_messages() -> None:
    """init/mode stay genuinely transient: no session_events, no messages.

    init carries no field beyond ``type`` in every live occurrence found;
    mode was a corpus-wide constant ("normal"). Both differ from the other
    ten record types precisely because there is no payload to lose.
    """
    parsed = parse_code(
        [
            {"type": "init", "sessionId": "sess-init"},
            {"type": "mode", "sessionId": "sess-init", "mode": "normal"},
        ],
        "sess-init",
    )
    assert _typed_events(parsed) == []
    assert parsed.messages == []


def test_parse_coverage_event_reports_seen_and_persisted_counts() -> None:
    """polylogue-pbuh AC5: coverage is reported per type -- seen vs. actually
    persisted -- so a future silently-dropped record type is visible in the
    archive itself rather than requiring another corpus rg audit to notice.

    ``permission-mode`` here always persists (its record always carries a
    ``permissionMode`` string), while a ``bash_progress`` tick under
    ``progress`` is seen but never persisted (see the classification comment
    above ``_SKIPPED_SIDECAR_RECORD_TYPES``) -- pinning that seen and
    persisted counts can genuinely diverge, not just mirror each other.
    """
    parsed = parse_code(
        [
            {"type": "permission-mode", "sessionId": "sess-cov", "permissionMode": "plan"},
            {"type": "permission-mode", "sessionId": "sess-cov", "permissionMode": "acceptEdits"},
            {
                "type": "progress",
                "sessionId": "sess-cov",
                "toolUseID": "bash-progress-0",
                "parentToolUseID": "toolu_bash_1",
                "data": {"type": "bash_progress", "output": "", "elapsedTimeSeconds": 1},
            },
        ],
        "sess-cov",
    )
    coverage_events = [e for e in parsed.session_events if e.event_type == _COVERAGE_EVENT_TYPE]
    assert len(coverage_events) == 1
    payload = coverage_events[0].payload
    assert payload["sidecar_seen"] == {"permission-mode": 2, "progress": 1}
    assert payload["sidecar_persisted"] == {"permission-mode": 2}
    assert payload["empty_dropped_by_record_type"] == {}


def test_parse_coverage_event_absent_when_only_ordinary_messages_parsed() -> None:
    """A session with no sidecar/empty-drop activity gets no coverage event
    at all -- keeps the common case from carrying a useless empty payload."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-plain",
                "message": {"role": "user", "content": "plain session, nothing skipped"},
            },
        ],
        "sess-plain",
    )
    assert [e for e in parsed.session_events if e.event_type == _COVERAGE_EVENT_TYPE] == []
