"""Coverage for polylogue-pbuh: sidecar records must persist as typed evidence.

Each test exercises ``polylogue.sources.parsers.claude.code_parser``, the
sole production code path that decides what happens to a Claude Code JSONL
record whose ``type`` is in ``_NON_MESSAGE_SIDECAR_RECORD_TYPES``. Before this
bead, every record type below was silently dropped by a bare frozenset
membership check with no per-type rationale; these tests pin that dropping
one of the evidence-bearing types (or reverting the ai-title title-override)
reproduces exactly that silent-loss bug.
"""

from __future__ import annotations

from polylogue.core.enums import BranchType, SessionKind, TitleSource
from polylogue.sources.dispatch import merge_parsed_session_chunks
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


def test_prompt_suggestion_marker_produces_prompt_suggestion_session() -> None:
    """Removing marker admission would store this real parser shape as primary."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "sessionId": "suggestion-session",
                "uuid": "message-1",
                "timestamp": "2026-08-31T00:00:00Z",
                "message": {
                    "role": "user",
                    "content": "[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.\nReview the failing test.",
                },
            }
        ],
        "suggestion-session",
    )

    assert parsed.session_kind is SessionKind.PROMPT_SUGGESTION
    assert {event.event_type for event in parsed.session_events} >= {"claude_session_kind"}


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


def test_attachment_file_subtype_gets_its_own_event_type() -> None:
    """A real referenced file must not share an event_type with hook chatter.

    ``attachment.type`` covers 38 distinct shapes in the live corpus
    (polylogue lane, 2026-07-31 full-corpus enumeration); this pins that a
    real file attachment is routed to ``claude_attachment_file``, not the
    single collapsed ``claude_attachment`` bucket the dispatch replaced.
    """
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
            "claude_attachment_file",
            {"type": "file", "path": "/tmp/example.txt", "sizeBytes": 42, "summary": "file"},
        )
    ]


def test_attachment_hook_subtypes_share_one_event_type() -> None:
    """Six hook-lifecycle subtypes are the same real-world entity (a hook
    firing) distinguished by outcome, not six near-identical event types."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-hook",
                "attachment": {"type": "hook_success", "hookName": "SessionStart:startup"},
            },
            {
                "type": "attachment",
                "sessionId": "sess-hook",
                "attachment": {"type": "hook_blocking_error", "hookName": "PreToolUse:Bash"},
            },
        ],
        "sess-hook",
    )
    event_types = [e.event_type for e in parsed.session_events if e.event_type != "claude_parse_coverage"]
    assert event_types == ["claude_hook_event", "claude_hook_event"]


def test_attachment_queued_command_reuses_queue_operation_event_type() -> None:
    """``queued_command`` (attachment subtype) and ``queue-operation``
    (top-level record type) describe the same message-queue entity through
    two different provider code paths -- they must land on the same
    event_type, not a sibling type for an identical concept."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-queue",
                "attachment": {"type": "queued_command", "prompt": "run the tests", "commandMode": "prompt"},
            }
        ],
        "sess-queue",
    )
    events = [(e.event_type, e.payload) for e in parsed.session_events if e.event_type != "claude_parse_coverage"]
    assert events == [
        (
            "claude_queue_operation",
            {
                "operation": "queued_command",
                "content": "run the tests",
                "command_mode": "prompt",
                "summary": "queued_command",
            },
        )
    ]


def test_attachment_transient_subtype_emits_no_event() -> None:
    """Confirmed-zero-information subtypes (e.g. the constant
    total_tokens_reminder text) are dropped, not persisted as noise."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-transient",
                "attachment": {
                    "type": "total_tokens_reminder",
                    "text": "<total_tokens>Infinite tokens left</total_tokens>",
                },
            }
        ],
        "sess-transient",
    )
    assert [e for e in parsed.session_events if e.event_type != "claude_parse_coverage"] == []


def test_attachment_unrecognized_subtype_fails_loud() -> None:
    """A future/unrecognized attachment subtype must still surface -- tagged
    distinctly so it is triageable, not silently merged into a known bucket
    or dropped on the floor."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-unknown",
                "attachment": {"type": "some_future_subtype", "value": 1},
            }
        ],
        "sess-unknown",
    )
    events = [(e.event_type, e.payload) for e in parsed.session_events if e.event_type != "claude_parse_coverage"]
    assert events == [
        (
            "claude_attachment_unclassified",
            {"type": "some_future_subtype", "value": 1, "summary": "some_future_subtype"},
        )
    ]


def test_attachment_deferred_tools_delta_drops_body_text_keeps_names() -> None:
    """Capability deltas keep the added tool/skill names but drop the full
    injected instruction-block text (unbounded, duplicative)."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-delta",
                "attachment": {
                    "type": "deferred_tools_delta",
                    "addedNames": ["WebFetch", "WebSearch"],
                    "addedLines": ["full description of WebFetch...", "full description of WebSearch..."],
                },
            }
        ],
        "sess-delta",
    )
    events = [(e.event_type, e.payload) for e in parsed.session_events if e.event_type != "claude_parse_coverage"]
    assert events == [
        (
            "claude_capability_delta",
            {
                "added_names": ["WebFetch", "WebSearch"],
                "added_body_count": 2,
                "summary": "deferred_tools_delta",
            },
        )
    ]


def test_attachment_skill_listing_extracts_names_not_full_text() -> None:
    """``skill_listing`` keeps skill names, not the full concatenated
    markdown description of every available skill."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-skills",
                "attachment": {
                    "type": "skill_listing",
                    "content": "- update-config: long description here\n- keybindings-help: another description",
                },
            }
        ],
        "sess-skills",
    )
    events = [(e.event_type, e.payload) for e in parsed.session_events if e.event_type != "claude_parse_coverage"]
    assert events == [
        (
            "claude_capability_snapshot",
            {
                "skill_names": ["update-config", "keybindings-help"],
                "skill_count": 2,
                "summary": "skill_listing",
            },
        )
    ]


def test_attachment_diagnostics_bounds_to_per_file_counts() -> None:
    """``diagnostics`` keeps per-file finding counts, not the full LSP
    message text/ranges/codes for every finding."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-diag",
                "attachment": {
                    "type": "diagnostics",
                    "files": [
                        {
                            "uri": "/repo/foo.py",
                            "diagnostics": [
                                {"message": "long pyright message", "severity": "Error"},
                                {"message": "another long message", "severity": "Warning"},
                            ],
                        }
                    ],
                },
            }
        ],
        "sess-diag",
    )
    events = [(e.event_type, e.payload) for e in parsed.session_events if e.event_type != "claude_parse_coverage"]
    assert events == [
        (
            "claude_diagnostics",
            {
                "file_count": 1,
                "diagnostic_count": 2,
                "files": [{"uri": "/repo/foo.py", "diagnostic_count": 2}],
                "summary": "diagnostics",
            },
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


# ---------------------------------------------------------------------------
# Record types admitted 2026-09-06 (polylogue-p81hr / -amftr / -z87qb).
# Each fixture uses the exact key-set the exhaustive corpus walk found -- the
# defect these cover survived because earlier fixtures asserted a shape the
# provider never emits.
# ---------------------------------------------------------------------------


def test_result_record_persists_subagent_output() -> None:
    """A dispatched subagent's returned text becomes evidence, not a drop.

    Anti-vacuity: remove ``result`` from ``_NON_MESSAGE_SIDECAR_RECORD_TYPES``
    and the record falls back to the empty-content drop -- ``_typed_events``
    goes empty and ``empty_dropped_by_record_type`` gains ``result``.
    """
    parsed = parse_code(
        [
            {
                "type": "result",
                "key": "dispatch-7",
                "agentId": "agent_01SYNTHETIC",
                "result": "Findings: the selector matched no tests.\nSecond line.",
            }
        ],
        "sess-result",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_subagent_result",
            {
                "agent_id": "agent_01SYNTHETIC",
                "dispatch_key": "dispatch-7",
                "result": "Findings: the selector matched no tests.\nSecond line.",
                "summary": "Findings: the selector matched no tests.",
            },
        )
    ]
    coverage = [e for e in parsed.session_events if e.event_type == _COVERAGE_EVENT_TYPE][0]
    assert coverage.payload["empty_dropped_by_record_type"] == {}


def test_started_record_pairs_a_dispatch_with_no_result() -> None:
    parsed = parse_code(
        [{"type": "started", "key": "dispatch-7", "agentId": "agent_01SYNTHETIC"}],
        "sess-started",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_subagent_started",
            {"agent_id": "agent_01SYNTHETIC", "dispatch_key": "dispatch-7", "summary": "agent_01SYNTHETIC"},
        )
    ]


def test_relocated_record_corrects_the_session_working_directory() -> None:
    """The provider's own cwd correction reaches the working-directory set.

    Anti-vacuity: drop the ``acc.cwds.add`` in the ``relocated`` branch and
    ``working_directories`` keeps only the stale original.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-moved",
                "cwd": "/work/original",
                "message": {"role": "user", "content": "hello"},
            },
            {"type": "relocated", "sessionId": "sess-moved", "relocatedCwd": "/work/moved"},
        ],
        "sess-moved",
    )
    assert parsed.working_directories == ["/work/moved", "/work/original"]
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [("claude_session_relocated", {"relocated_cwd": "/work/moved", "summary": "/work/moved"})]


def test_worktree_state_record_persists_session_topology() -> None:
    parsed = parse_code(
        [
            {
                "type": "worktree-state",
                "sessionId": "sess-main",
                "worktreeSession": "11111111-2222-4333-8444-555555555555",
            }
        ],
        "sess-main",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_worktree_state",
            {
                "worktree_session": "11111111-2222-4333-8444-555555555555",
                "summary": "11111111-2222-4333-8444-555555555555",
            },
        )
    ]


def test_cost_state_record_keeps_the_producers_own_incompleteness_flag() -> None:
    """``hasUnknownModelCost`` is the producer saying its total is partial.

    Anti-vacuity: drop the ``cost-state`` branch and the whole ledger --
    including the qualification that makes the total safe to read -- becomes
    an empty-content drop.
    """
    parsed = parse_code(
        [
            {
                "type": "cost-state",
                "sessionId": "sess-cost",
                "totalCostUSD": 1.25,
                "totalAPIDuration": 900,
                "totalAPIDurationWithoutRetries": 850,
                "totalToolDuration": 120,
                "totalLinesAdded": 40,
                "totalLinesRemoved": 12,
                "totalDuration": 4000,
                "startTime": 1750000000000,
                "modelUsage": {"claude-synthetic-1": {"inputTokens": 10, "outputTokens": 20}},
                "hasUnknownModelCost": True,
            }
        ],
        "sess-cost",
    )
    events = [(e.event_type, e.payload) for e in _typed_events(parsed)]
    assert events == [
        (
            "claude_cost_state",
            {
                "total_cost_usd": 1.25,
                "has_unknown_model_cost": True,
                "model_usage": {"claude-synthetic-1": {"inputTokens": 10, "outputTokens": 20}},
                "total_duration_ms": 4000,
                "total_api_duration_ms": 900,
                "total_api_duration_without_retries_ms": 850,
                "total_tool_duration_ms": 120,
                "total_lines_added": 40,
                "total_lines_removed": 12,
                "start_time": 1750000000000,
                "summary": "$1.25",
            },
        )
    ]


# ---------------------------------------------------------------------------
# Record types admitted 2026-09-06 (polylogue-chemh), completing the inventory
# of what the live corpus carries. Each fixture uses the exact key-set the
# exhaustive 14,536-file walk found for that type.
# ---------------------------------------------------------------------------


def test_frame_link_persists_the_published_artifact_reference() -> None:
    """``frame-link`` names an artifact published from a local file.

    Dropping ``frame-link`` from the disposition tables returns it to an
    empty-content drop, losing the URL, the local path and the title.
    """
    parsed = parse_code(
        [
            {
                "type": "frame-link",
                "sessionId": "sess-frame",
                "path": "/realm/tmp/report.html",
                "frameUrl": "https://claude.ai/code/artifact/1a899ee9",
                "title": "One Wakeup Short",
                "timestamp": "2026-08-01T15:02:48.175Z",
            }
        ],
        "sess-frame",
    )
    assert [(e.event_type, e.payload) for e in _typed_events(parsed)] == [
        (
            "claude_frame_link",
            {
                "frame_url": "https://claude.ai/code/artifact/1a899ee9",
                "path": "/realm/tmp/report.html",
                "title": "One Wakeup Short",
                "artifact_count": None,
                "summary": "One Wakeup Short",
            },
        )
    ]


def test_frame_link_count_only_shape_still_persists() -> None:
    """The second live ``frame-link`` shape carries only ``artifactCount``.

    105 of 234 corpus records take this shape; requiring a URL would drop
    them all.
    """
    parsed = parse_code(
        [
            {
                "type": "frame-link",
                "sessionId": "sess-frame-count",
                "artifactCount": 3,
                "timestamp": "2026-08-01T15:02:48.175Z",
            }
        ],
        "sess-frame-count",
    )
    payloads = [e.payload for e in _typed_events(parsed)]
    assert payloads == [
        {
            "frame_url": None,
            "path": None,
            "title": None,
            "artifact_count": 3,
            "summary": "3 artifact(s)",
        }
    ]


def test_artifact_comment_monitor_persists_artifact_identity() -> None:
    """``artifact-comment-monitor`` names the artifacts a session published."""
    parsed = parse_code(
        [
            {
                "type": "artifact-comment-monitor",
                "v": 1,
                "sessionId": "sess-artifacts",
                "artifacts": {
                    "2b084816-93f9-486b-91a5-ac70d69fb235": {
                        "state": "armed",
                        "writtenAtMs": 1788139487957,
                        "title": "Pairwise Ranking Redesign",
                    }
                },
            }
        ],
        "sess-artifacts",
    )
    assert [(e.event_type, e.payload) for e in _typed_events(parsed)] == [
        (
            "claude_artifact_comment_monitor",
            {
                "version": 1,
                "artifacts": [
                    {
                        "artifact_id": "2b084816-93f9-486b-91a5-ac70d69fb235",
                        "state": "armed",
                        "title": "Pairwise Ranking Redesign",
                        "written_at_ms": 1788139487957,
                    }
                ],
                "summary": "1 monitored artifact(s)",
            },
        )
    ]


def test_artifact_autoreact_ledger_bounds_threads_to_counts() -> None:
    """Per-artifact ``threads``/``turnTimestamps`` are unbounded activity logs.

    Their length is the queryable fact; persisting the raw lists would copy
    every comment thread's body into the index. Reverting the bounding makes
    this assert the raw lists instead of the counts.
    """
    parsed = parse_code(
        [
            {
                "type": "artifact-autoreact-ledger",
                "v": 1,
                "sessionId": "sess-ledger",
                "accountUuid": "e1f27e58-33e5-4a5d-836b-40a2b78055aa",
                "artifacts": {
                    "2b084816-93f9-486b-91a5-ac70d69fb235": {
                        "savedAt": 1788139493629,
                        "stampHighWater": None,
                        "everBaselined": True,
                        "everHadThreads": True,
                        "turnTimestamps": [1788139493629, 1788139494000],
                        "threads": [{"id": "t1", "comments": ["long comment body"]}],
                    }
                },
            }
        ],
        "sess-ledger",
    )
    assert [(e.event_type, e.payload) for e in _typed_events(parsed)] == [
        (
            "claude_artifact_autoreact_ledger",
            {
                "version": 1,
                "account_uuid": "e1f27e58-33e5-4a5d-836b-40a2b78055aa",
                "artifacts": [
                    {
                        "artifact_id": "2b084816-93f9-486b-91a5-ac70d69fb235",
                        "saved_at_ms": 1788139493629,
                        "stamp_high_water": None,
                        "ever_baselined": True,
                        "ever_had_threads": True,
                        "thread_count": 1,
                        "turn_count": 2,
                    }
                ],
                "summary": "1 artifact(s) in ledger",
            },
        )
    ]


def test_atis_latch_and_agent_color_stay_transient() -> None:
    """Both are declared transient on measured evidence, and stay counted.

    ``atis`` is the empty string in every corpus occurrence and
    ``agentColor`` is a presentation attribute with no agent id to attach it
    to. They must produce no typed event, yet still appear in the coverage
    event's seen counts -- a transient classification is a decision on
    record, not a silent drop.
    """
    parsed = parse_code(
        [
            {"type": "atis-latch", "atis": "", "sessionId": "sess-transient"},
            {"type": "agent-color", "agentColor": "cyan", "sessionId": "sess-transient"},
        ],
        "sess-transient",
    )
    assert _typed_events(parsed) == []
    assert parsed.messages == []
    coverage = [e for e in parsed.session_events if e.event_type == _COVERAGE_EVENT_TYPE]
    assert coverage[0].payload["sidecar_seen"] == {"agent-color": 1, "atis-latch": 1}
    assert coverage[0].payload["sidecar_persisted"] == {}


def test_unknown_record_type_persists_its_shape_instead_of_vanishing() -> None:
    """A record type with no disposition must fail loud, not fall through.

    Every type the 2026-09-06 corpus walk found is classified; this pins what
    happens to the next one a CLI version introduces. Deleting the
    ``_UNCLASSIFIED_RECORD_EVENT_TYPE`` branch returns it to a silent
    empty-content drop, which is the defect polylogue-chemh measured across
    ten record types. Long strings and containers are bounded here because
    the record's own bytes stay durable in source.db.
    """
    parsed = parse_code(
        [
            {
                "type": "record-type-from-a-newer-cli",
                "sessionId": "sess-unknown",
                "uuid": "r1",
                "someField": "x" * 500,
                "someList": [1, 2, 3],
                "someCount": 7,
            }
        ],
        "sess-unknown",
    )
    events = _typed_events(parsed)
    assert [e.event_type for e in events] == ["claude_unclassified_record"]
    assert events[0].source_message_provider_id == "r1"
    assert events[0].payload == {
        "record_type": "record-type-from-a-newer-cli",
        "fields": {"someCount": 7, "someField": "x" * 200, "someList": {"size": 3}},
        "summary": "record-type-from-a-newer-cli",
    }
    coverage = [e for e in parsed.session_events if e.event_type == _COVERAGE_EVENT_TYPE]
    assert coverage[0].payload["empty_dropped_by_record_type"] == {"record-type-from-a-newer-cli": 1}


def test_every_live_record_type_has_a_disposition() -> None:
    """Every record type the corpus carries is classified, none falls through.

    The list is the exhaustive 2026-09-06 walk of 14,536 Claude Code session
    files. A type reaching ``claude_unclassified_record`` here is one whose
    disposition was never decided -- the shape polylogue-chemh measured on ten
    types at once.
    """
    live_record_types = [
        "assistant",
        "user",
        "progress",
        "attachment",
        "queue-operation",
        "last-prompt",
        "permission-mode",
        "mode",
        "file-history-snapshot",
        "ai-title",
        "system",
        "pr-link",
        "bridge-session",
        "atis-latch",
        "custom-title",
        "agent-name",
        "summary",
        "file-history-delta",
        "relocated",
        "worktree-state",
        "frame-link",
        "started",
        "result",
        "agent-color",
        "cost-state",
        "artifact-comment-monitor",
        "artifact-autoreact-ledger",
        "init",
    ]
    parsed = parse_code(
        [{"type": record_type, "sessionId": "sess-inventory"} for record_type in live_record_types],
        "sess-inventory",
    )
    assert [e for e in parsed.session_events if e.event_type == "claude_unclassified_record"] == []


# ---------------------------------------------------------------------------
# polylogue-esvzb: forkedFrom is the session's own fork parent.
# ---------------------------------------------------------------------------


def test_forked_from_resolves_the_session_parent_edge() -> None:
    """``forkedFrom`` must become the session's parent, deduplicated.

    The provider stamps it on nearly every record of a forked session
    (10,561 records across 9 forked sessions in the corpus walk), including
    ``progress``/``attachment`` records that never reach message parsing.
    Deleting the ``acc.forked_from`` read leaves the session parentless,
    which is what the archive stored before: a fork whose lineage is not
    recoverable from its own bytes afterwards.
    """
    fork_edge = {"sessionId": "parent-sess", "messageUuid": "parent-msg-9"}
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "child-sess",
                "forkedFrom": fork_edge,
                "message": {"role": "user", "content": "continue from the fork"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "child-sess",
                "forkedFrom": fork_edge,
                "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            },
            {
                "type": "attachment",
                "uuid": "at1",
                "sessionId": "child-sess",
                "forkedFrom": fork_edge,
                "attachment": {"type": "output_style", "outputStyle": "default"},
            },
        ],
        "child-sess",
    )
    assert parsed.parent_session_provider_id == "parent-sess"
    assert parsed.branch_type is BranchType.FORK
    fork_events = [e for e in parsed.session_events if e.event_type == "claude_forked_from"]
    assert [e.payload for e in fork_events] == [
        {
            "parent_session_provider_id": "parent-sess",
            "branch_point_message_provider_id": "parent-msg-9",
            "record_count": 3,
            "summary": "parent-sess",
        }
    ]


def test_forked_from_absent_leaves_the_session_parentless() -> None:
    """Anti-vacuity: no ``forkedFrom`` must not fabricate a parent edge."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "lone-sess",
                "message": {"role": "user", "content": "no fork here"},
            }
        ],
        "lone-sess",
    )
    assert parsed.parent_session_provider_id is None
    assert parsed.branch_type is None
    assert [e for e in parsed.session_events if e.event_type == "claude_forked_from"] == []


def test_forked_from_does_not_displace_a_subagent_parent() -> None:
    """A route that knows this file's identity outranks a content claim.

    A subagent transcript (an ``agent-`` fallback id) resolves its parent from
    its own composed identity; ``forkedFrom`` is a claim carried in replayed
    content and must not overwrite it, but is still persisted as its own event.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "main-sess",
                "forkedFrom": {"sessionId": "other-sess", "messageUuid": "m1"},
                "message": {"role": "user", "content": "subagent work"},
            }
        ],
        "agent-abc123",
    )
    assert parsed.parent_session_provider_id == "main-sess"
    assert parsed.branch_type is BranchType.SUBAGENT
    assert [e.event_type for e in parsed.session_events if e.event_type == "claude_forked_from"] == [
        "claude_forked_from"
    ]


def test_attachment_rendered_content_rides_the_attachment_event() -> None:
    """``rendered[].content`` is the text injected into the model's context.

    Dropping the ``rendered`` assignment in ``_attachment_sidecar_event``
    reproduces the loss this pins: the structured payload survives and what
    the model actually read does not. The two are different text here on
    purpose -- a ``file`` attachment is rendered with line numbers the payload
    has not got, which is why the payload cannot stand in for it.
    """
    rendered_text = "<system-reminder>\n     1\tfrom polylogue import archive\n</system-reminder>"
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-rendered",
                "attachment": {"type": "file", "filename": "a.py", "content": "from polylogue import archive"},
                "rendered": [{"content": rendered_text}],
            }
        ],
        "sess-rendered",
    )
    events = _typed_events(parsed)
    assert [e.event_type for e in events] == ["claude_attachment_file"]
    assert events[0].payload["rendered"] == [rendered_text]
    assert events[0].payload["content"] == "from polylogue import archive"


def test_attachment_rendered_content_keeps_every_entry_in_wire_order() -> None:
    """A record may carry more than one rendered entry; none of them is a spare."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-rendered-multi",
                "attachment": {"type": "file", "filename": "a.py"},
                "rendered": [{"content": "first injected"}, {"content": "second injected"}],
            }
        ],
        "sess-rendered-multi",
    )
    assert _typed_events(parsed)[0].payload["rendered"] == ["first injected", "second injected"]


def test_bounded_subtype_rendered_content_is_counted_not_stored() -> None:
    """The rendered text of a bounded subtype IS the body its builder excludes.

    ``skill_listing``'s payload builder keeps skill names and drops the
    concatenated skill bodies; the rendered text is those same bodies inside a
    system-reminder wrapper. Storing it verbatim would reinstate exactly what
    the builder excludes, so only its size is kept -- delete the
    ``rendered_char_count`` branch and the bounding is decorative.
    """
    body = "- polylogue: Query or develop Polylogue session archives."
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-bounded-rendered",
                "attachment": {"type": "skill_listing", "content": body, "skillCount": 1},
                "rendered": [{"content": f"<system-reminder>\n{body}\n</system-reminder>"}],
            }
        ],
        "sess-bounded-rendered",
    )
    payload = _typed_events(parsed)[0].payload
    assert payload["rendered_char_count"] == len(f"<system-reminder>\n{body}\n</system-reminder>")
    assert payload["rendered_count"] == 1
    assert "rendered" not in payload
    assert payload["skill_names"] == ["polylogue"]


def test_transient_subtype_drops_its_rendered_content_with_the_event() -> None:
    """A dropped subtype emits no event, so its rendered text has nothing to ride.

    This is the declared consequence of the transient ruling, not an oversight:
    if a transient subtype ever starts emitting an event, its rendered content
    rides it and this expectation flips.
    """
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-transient-rendered",
                "attachment": {"type": "total_tokens_reminder", "text": "<total_tokens>7 tokens left</total_tokens>"},
                "rendered": [
                    {"content": "<system-reminder>\n<total_tokens>7 tokens left</total_tokens>\n</system-reminder>"}
                ],
            }
        ],
        "sess-transient-rendered",
    )
    assert _typed_events(parsed) == []


def test_failed_mcp_servers_survive_the_bounded_delta_payload() -> None:
    """A capability the session expected and did not get must stay visible.

    ``_bounded_delta_payload`` keeps names and drops body text; reverting it to
    the names/count-only dict loses the only record that an MCP server's tools
    were absent, which is otherwise indistinguishable from never having
    configured the server.
    """
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-mcp-failure",
                "attachment": {
                    "type": "deferred_tools_delta",
                    "addedNames": ["Monitor"],
                    "addedLines": ["Monitor: watch a condition"],
                    "failedMcpServers": [
                        {"name": "polylogue", "errorCode": "CONNECT_TIMEOUT", "error": "Connection timed out"}
                    ],
                },
            }
        ],
        "sess-mcp-failure",
    )
    payload = _typed_events(parsed)[0].payload
    assert payload["failed_mcp_servers"] == [
        {"name": "polylogue", "error_code": "CONNECT_TIMEOUT", "error": "Connection timed out"}
    ]
    assert payload["added_names"] == ["Monitor"]


def test_delta_payload_omits_failed_mcp_servers_when_every_server_connected() -> None:
    """The key is evidence of a failure, so a clean session must not carry an empty one."""
    parsed = parse_code(
        [
            {
                "type": "attachment",
                "sessionId": "sess-mcp-clean",
                "attachment": {"type": "deferred_tools_delta", "addedNames": ["Monitor"], "failedMcpServers": []},
            }
        ],
        "sess-mcp-clean",
    )
    assert "failed_mcp_servers" not in _typed_events(parsed)[0].payload


def test_session_environment_event_separates_a_dispatched_lane_from_an_operator_session() -> None:
    """``entrypoint``/``version``/``permissionMode``/``promptSource`` are read.

    Without ``entrypoint``, an SDK-dispatched lane and an operator's own
    interactive session produce identical archived sessions -- delete the fold
    reads and this session is indistinguishable from a ``cli`` one. The counts
    are per value because ``permissionMode``/``promptSource`` are per-record
    state, not one session-wide setting.
    """
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-env",
                "entrypoint": "sdk-cli",
                "version": "2.1.263",
                "permissionMode": "bypassPermissions",
                "promptSource": "sdk",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "implement the bead"},
            },
            {
                "type": "attachment",
                "sessionId": "sess-env",
                "entrypoint": "sdk-cli",
                "version": "2.1.263",
                "timestamp": "2026-01-01T00:00:01Z",
                "attachment": {"type": "date", "date": "2026-01-01"},
            },
        ],
        "sess-env",
    )
    environment = [e for e in parsed.session_events if e.event_type == "claude_session_environment"]
    assert [e.payload for e in environment] == [
        {
            "entrypoints": {"sdk-cli": 2},
            "cli_versions": {"2.1.263": 2},
            "permission_modes": {"bypassPermissions": 1},
            "prompt_sources": {"sdk": 1},
        }
    ]


def test_session_environment_event_is_absent_when_no_record_carries_provenance() -> None:
    """An older CLI build stamps none of these keys; that must not fabricate an event."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-no-env",
                "message": {"role": "user", "content": "hello"},
            }
        ],
        "sess-no-env",
    )
    assert [e for e in parsed.session_events if e.event_type == "claude_session_environment"] == []


def test_session_environment_counts_every_version_a_resumed_session_spans() -> None:
    """177 of 14,287 corpus files carry more than one ``version``: a resume across
    a CLI upgrade. Reading only the first value would name the wrong producer for
    the tail of the session."""
    parsed = parse_code(
        [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "sess-upgrade",
                "version": "2.1.261",
                "message": {"role": "user", "content": "before"},
            },
            {
                "type": "user",
                "uuid": "u2",
                "sessionId": "sess-upgrade",
                "version": "2.1.263",
                "message": {"role": "user", "content": "after"},
            },
        ],
        "sess-upgrade",
    )
    environment = [e for e in parsed.session_events if e.event_type == "claude_session_environment"]
    assert environment[0].payload["cli_versions"] == {"2.1.261": 1, "2.1.263": 1}


def test_session_environment_survives_a_chunked_stream_as_one_row() -> None:
    """The eager route and a chunk-split stream must agree on the event list.

    ``claude_session_environment`` summarizes the complete record set, so a
    stream split into chunks must reduce its per-chunk rows to one -- otherwise
    the archived session's event count and every count inside them depend on
    the read size that happened to be used.
    """
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-chunked-env",
            "entrypoint": "sdk-cli",
            "version": "2.1.261",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "before the upgrade"},
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "sess-chunked-env",
            "entrypoint": "sdk-cli",
            "version": "2.1.263",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"role": "user", "content": "after the upgrade"},
        },
    ]

    eager = parse_code(records, "sess-chunked-env")
    chunked = merge_parsed_session_chunks(
        [parse_code(records[:1], "sess-chunked-env"), parse_code(records[1:], "sess-chunked-env")]
    )[0]

    assert [e.event_type for e in chunked.session_events] == [e.event_type for e in eager.session_events]
    environment = [e for e in chunked.session_events if e.event_type == "claude_session_environment"]
    assert [e.payload for e in environment] == [
        {"entrypoints": {"sdk-cli": 2}, "cli_versions": {"2.1.261": 1, "2.1.263": 1}}
    ]
