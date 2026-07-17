from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.message.types import MessageType
from polylogue.core.enums import MaterialOrigin, Role
from polylogue.sources.parsers.claude import parse_code
from polylogue.sources.parsers.claude.common import normalize_timestamp
from polylogue.sources.parsers.claude.orchestration import (
    inventory_claude_orchestration_artifacts,
    parse_claude_orchestration_artifact,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def test_normalize_timestamp_returns_canonical_iso_text() -> None:
    assert normalize_timestamp(1704067200) == "2024-01-01T00:00:00+00:00"
    assert normalize_timestamp(1704067200000) == "2024-01-01T00:00:00+00:00"


def test_parse_code_classifies_runtime_artifacts() -> None:
    items: list[object] = [
        {
            "type": "user",
            "uuid": "task-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "user",
                "content": "<task-notification><status>completed</status></task-notification>",
            },
        },
        {
            "type": "user",
            "uuid": "stdout-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201,
            "message": {
                "role": "user",
                "content": "<local-command-stdout>pytest passed</local-command-stdout>",
            },
        },
        {
            "type": "user",
            "uuid": "stderr-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201.1,
            "message": {
                "role": "user",
                "content": "<local-command-stderr>warning: generated runtime stderr</local-command-stderr>",
            },
        },
        {
            "type": "user",
            "uuid": "bash-stdout-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201.2,
            "message": {
                "role": "user",
                "content": "<bash-stdout>cargo test passed</bash-stdout>",
            },
        },
        {
            "type": "user",
            "uuid": "bash-stderr-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201.3,
            "message": {
                "role": "user",
                "content": "<bash-stderr>traceback from shell wrapper</bash-stderr>",
            },
        },
        {
            "type": "user",
            "uuid": "command-1",
            "sessionId": "sess-1",
            "timestamp": 1704067202,
            "message": {
                "role": "user",
                "content": "<command-name>status</command-name>\n<command-message>status</command-message>",
            },
        },
        {
            "type": "user",
            "uuid": "skill-1",
            "sessionId": "sess-1",
            "timestamp": 1704067203,
            "message": {
                "role": "user",
                "content": "Base directory for this skill: /tmp/skill\n\n# Skill Body",
            },
        },
        {
            "type": "user",
            "uuid": "commit-pack-1",
            "sessionId": "sess-1",
            "timestamp": 1704067204,
            "message": {
                "role": "user",
                "content": "# Commit N: Generate all artifacts\n\nlarge generated context",
            },
        },
        {
            "type": "user",
            "uuid": "retro-pack-1",
            "sessionId": "sess-1",
            "timestamp": 1704067205,
            "message": {
                "role": "user",
                "content": "Generate a retrospective for: long generated analysis bundle",
            },
        },
        {
            "type": "user",
            "uuid": "prompt-1",
            "sessionId": "sess-1",
            "timestamp": 1704067206,
            "message": {
                "role": "user",
                "content": "<system-reminder>model-only context</system-reminder>\n\nActual user prompt.",
            },
        },
    ]

    result = parse_code(items, "fallback")

    by_id = {message.provider_message_id: message for message in result.messages}
    assert len(result.messages) == len(by_id) == 10
    assert by_id["task-1"].message_type is MessageType.PROTOCOL
    assert by_id["task-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["stdout-1"].message_type is MessageType.PROTOCOL
    assert by_id["stdout-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["stderr-1"].role is Role.USER
    assert by_id["stderr-1"].message_type is MessageType.PROTOCOL
    assert by_id["stderr-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["bash-stdout-1"].role is Role.USER
    assert by_id["bash-stdout-1"].message_type is MessageType.PROTOCOL
    assert by_id["bash-stdout-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["bash-stderr-1"].role is Role.USER
    assert by_id["bash-stderr-1"].message_type is MessageType.PROTOCOL
    assert by_id["bash-stderr-1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["command-1"].message_type is MessageType.PROTOCOL
    assert by_id["command-1"].material_origin is MaterialOrigin.OPERATOR_COMMAND
    assert by_id["skill-1"].message_type is MessageType.CONTEXT
    assert by_id["skill-1"].material_origin is MaterialOrigin.RUNTIME_CONTEXT
    assert by_id["commit-pack-1"].message_type is MessageType.MESSAGE
    assert by_id["commit-pack-1"].material_origin is MaterialOrigin.GENERATED_CONTEXT_PACK
    assert by_id["retro-pack-1"].message_type is MessageType.MESSAGE
    assert by_id["retro-pack-1"].material_origin is MaterialOrigin.GENERATED_ANALYSIS_PACK
    assert by_id["prompt-1"].message_type is MessageType.MESSAGE
    # Claude Code has provider-native provenance for real user turns: type=user,
    # !isMeta, no toolUseResult, no non-human origin.
    assert by_id["prompt-1"].material_origin is MaterialOrigin.HUMAN_AUTHORED
    assert result.title == "Actual user prompt."


def test_parse_code_preserves_tool_result_reclassification_material_origin() -> None:
    result = parse_code(
        [
            {
                "type": "user",
                "uuid": "tool-result-1",
                "sessionId": "sess-tool",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}],
                },
            }
        ],
        "fallback-tool",
    )

    assert result.messages[0].role is Role.TOOL
    assert result.messages[0].material_origin is MaterialOrigin.TOOL_RESULT


def test_claude_workflow_artifact_parser_retains_native_facts_and_coverage_gaps() -> None:
    run = parse_claude_orchestration_artifact(
        "/tmp/.claude/projects/x/workflows/wf-54.json",
        json.dumps({"runId": "wf-54", "taskId": "task-7", "resumeFromRunId": "wf-53", "scriptHash": "abc"}),
    )
    journal = parse_claude_orchestration_artifact(
        "/tmp/.claude/projects/x/subagents/workflows/wf-54/journal.jsonl",
        json.dumps({"contentKey": "call-1", "agentId": "agent-a", "structuredResult": {"ok": True}}) + "\n",
    )

    assert run is not None and run.facts[0].run_id == "wf-54"
    assert run.facts[0].payload["resumeFromRunId"] == "wf-53"
    assert journal is not None and journal.facts[0].content_key == "call-1"
    assert journal.facts[0].payload["structuredResult"] == {"ok": True}

    coverage = inventory_claude_orchestration_artifacts(
        (
            "/tmp/.claude/projects/x/workflows/wf-54.json",
            "/tmp/.claude/projects/x/subagents/workflows/wf-54/journal.jsonl",
            "/tmp/.claude/projects/x/subagents/agent-a.jsonl",
            "/tmp/.claude/projects/x/subagents/agent-b.meta.json",
            "/tmp/.claude/projects/x/jobs/session-a/adopt.json",
        )
    )
    assert coverage.artifact_counts == {
        "adopt_manifest": 1,
        "agent_sidecar_meta": 1,
        "agent_transcript": 1,
        "workflow_journal": 1,
        "workflow_run_snapshot": 1,
    }
    assert coverage.gaps == (
        "missing agent metadata for transcript agent-a",
        "missing agent transcript for metadata agent-b",
    )


def test_claude_agent_prompt_needs_positive_human_provenance() -> None:
    generated = parse_code(
        [{"type": "user", "uuid": "u1", "sessionId": "agent-session", "message": {"role": "user", "content": "work"}}],
        "agent-a",
    )
    direct = parse_code(
        [{"type": "user", "uuid": "u2", "sessionId": "direct-session", "message": {"role": "user", "content": "work"}}],
        "direct-session",
    )

    assert generated.messages[0].material_origin is MaterialOrigin.GENERATED_CONTEXT_PACK
    assert direct.messages[0].material_origin is MaterialOrigin.HUMAN_AUTHORED


def test_claude_coordinator_workflow_tool_use_preserves_invocation_evidence() -> None:
    parsed = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "workflow-tool-use",
                "sessionId": "coordinator",
                "timestamp": 1704067200,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "workflow-1",
                            "name": "Workflow",
                            "input": {
                                "runId": "wf-54",
                                "taskId": "task-7",
                                "resumeFromRunId": "wf-53",
                                "scriptHash": "abc",
                            },
                        }
                    ],
                },
            }
        ],
        "coordinator",
    )

    assert [(event.event_type, event.source_message_provider_id, event.payload) for event in parsed.session_events] == [
        (
            "claude_workflow_invocation",
            "workflow-tool-use",
            {"runId": "wf-54", "taskId": "task-7", "resumeFromRunId": "wf-53", "scriptHash": "abc"},
        )
    ]


def _background_task_records() -> list[object]:
    """Real Claude Code protocol shape, reduced from persisted JSONL evidence."""
    return [
        {
            "type": "assistant",
            "uuid": "assistant-ok",
            "sessionId": "background-outcomes",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-ok",
                        "name": "Bash",
                        "input": {"command": "true"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "start-ok",
            "sessionId": "background-outcomes",
            "toolUseResult": {"backgroundTaskId": "task-ok"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-ok",
                        "content": "Command running in background with ID: task-ok.",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "assistant-fail",
            "sessionId": "background-outcomes",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-fail",
                        "name": "Bash",
                        "input": {"command": "false"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "start-fail",
            "sessionId": "background-outcomes",
            "toolUseResult": {"backgroundTaskId": "task-fail"},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-fail",
                        "content": "Command running in background with ID: task-fail.",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "assistant-foreground",
            "sessionId": "background-outcomes",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-foreground",
                        "name": "Bash",
                        "input": {"command": "printf foreground"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "result-foreground",
            "sessionId": "background-outcomes",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-foreground",
                        "content": "foreground",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "notification-ok",
            "sessionId": "background-outcomes",
            "origin": {"kind": "task-notification"},
            "message": {
                "role": "user",
                "content": """<task-notification>
<task-id>task-ok</task-id>
<tool-use-id>tool-ok</tool-use-id>
<output-file>/tmp/task-ok.output</output-file>
<status>completed</status>
<summary>Background command \"true\" completed (exit code 0)</summary>
</task-notification>""",
            },
        },
        {
            "type": "user",
            "uuid": "notification-fail",
            "sessionId": "background-outcomes",
            "origin": {"kind": "task-notification"},
            "message": {
                "role": "user",
                "content": """<task-notification>
<task-id>task-fail</task-id>
<tool-use-id>tool-fail</tool-use-id>
<output-file>/tmp/task-fail.output</output-file>
<status>failed</status>
<summary>Background command \"false\" failed with exit code 1</summary>
</task-notification>""",
            },
        },
        {
            "type": "user",
            "uuid": "notification-fail-update",
            "sessionId": "background-outcomes",
            "origin": {"kind": "task-notification"},
            "message": {
                "role": "user",
                "content": """<task-notification>
<task-id>task-fail</task-id>
<tool-use-id>tool-fail</tool-use-id>
<output-file>/tmp/task-fail-final.output</output-file>
<status>failed</status>
<summary>Background command \"false\" failed with exit code 1</summary>
</task-notification>""",
            },
        },
    ]


def _background_start_records(*, task_id: str, tool_id: str, command: str, suffix: str) -> list[object]:
    return [
        {
            "type": "assistant",
            "uuid": f"assistant-{suffix}",
            "sessionId": "background-sidecars",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"command": command}}],
            },
        },
        {
            "type": "user",
            "uuid": f"start-{suffix}",
            "sessionId": "background-sidecars",
            "toolUseResult": {"backgroundTaskId": task_id},
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Command running in background with ID: {task_id}.",
                        "is_error": False,
                    }
                ],
            },
        },
    ]


def _task_notification(*, task_id: str, status: str, summary: str, tool_id: str | None = None, suffix: str = "") -> str:
    tool_use = f"<tool-use-id>{tool_id}</tool-use-id>\n" if tool_id else ""
    return (
        "<task-notification>\n"
        f"<task-id>{task_id}</task-id>\n"
        f"{tool_use}"
        f"<output-file>/tmp/{task_id}{suffix}.output</output-file>\n"
        f"<status>{status}</status>\n"
        f"<summary>{summary}</summary>\n"
        "</task-notification>"
    )


def test_parse_code_projects_queue_attachment_and_user_completion_shapes(tmp_path: Path) -> None:
    """Production route: Claude parser -> ``ArchiveStore`` -> ``actions``.

    This fails if notification extraction is moved below sidecar suppression,
    the failed-template matcher is removed, or unique task-id fallback is
    removed from ``_project_background_task_completions``.
    """
    records: list[object] = []
    records.extend(
        _background_start_records(task_id="task-queue", tool_id="tool-queue", command="true", suffix="queue")
    )
    records.extend(
        _background_start_records(
            task_id="task-attachment", tool_id="tool-attachment", command="false", suffix="attachment"
        )
    )
    records.extend(
        _background_start_records(task_id="task-user", tool_id="tool-user", command="printf user", suffix="user")
    )
    records.extend(
        [
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "sessionId": "background-sidecars",
                "content": _task_notification(
                    task_id="task-queue",
                    tool_id="tool-queue",
                    status="completed",
                    summary='Background command "true" completed (exit code 0)',
                ),
            },
            {
                "type": "attachment",
                "uuid": "attachment-failed",
                "sessionId": "background-sidecars",
                "attachment": {
                    "type": "queued_command",
                    "commandMode": "task-notification",
                    "prompt": _task_notification(
                        task_id="task-attachment",
                        tool_id="tool-attachment",
                        status="failed",
                        summary='Background command "false" failed with exit code 1',
                    ),
                },
            },
            {
                "type": "user",
                "uuid": "user-fallback",
                "sessionId": "background-sidecars",
                "origin": {"kind": "task-notification"},
                "message": {
                    "role": "user",
                    "content": _task_notification(
                        task_id="task-user",
                        status="completed",
                        summary='Background command "printf user" completed (exit code 0)',
                    )
                    + "\nRead the output file before reporting completion.",
                },
            },
        ]
    )

    parsed = parse_code(records, "background-sidecars")
    provider_ids = {message.provider_message_id for message in parsed.messages}
    assert "attachment-failed" not in provider_ids
    assert "user-fallback" in provider_ids
    assert [
        message.provider_message_id for message in parsed.messages if "<task-notification>" in (message.text or "")
    ] == ["user-fallback"]

    archive_root = tmp_path / "background-sidecars"
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(parsed)
        actions = archive.query_session_actions([session_id], limit=10)

    assert {action.tool_command: (action.is_error, action.exit_code) for action in actions} == {
        "true": (0, 0),
        "false": (1, 1),
        "printf user": (0, 0),
    }
    completion_events = [event for event in parsed.session_events if event.event_type == "background_task_completion"]
    assert [event.source_message_provider_id for event in completion_events] == [
        None,
        "attachment-failed",
        "user-fallback",
    ]


def test_parse_code_keeps_task_only_completion_unknown_when_background_start_is_ambiguous() -> None:
    """Production dependency: the parser's unique task-id fallback.

    This fails if task-only notifications are correlated to the first matching
    start instead of requiring exactly one ``backgroundTaskId`` match.
    """
    records: list[object] = []
    records.extend(_background_start_records(task_id="shared-task", tool_id="tool-one", command="true", suffix="one"))
    records.extend(_background_start_records(task_id="shared-task", tool_id="tool-two", command="false", suffix="two"))
    records.append(
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "sessionId": "background-sidecars",
            "content": _task_notification(
                task_id="shared-task",
                status="completed",
                summary='Background command "unknown" completed (exit code 0)',
            ),
        }
    )

    parsed = parse_code(records, "background-ambiguous")
    by_id = {message.provider_message_id: message for message in parsed.messages}
    assert by_id["start-one"].blocks[0].exit_code is None
    assert by_id["start-one"].blocks[0].is_error is None
    assert by_id["start-two"].blocks[0].exit_code is None
    assert by_id["start-two"].blocks[0].is_error is None


def test_parse_code_keeps_exact_pair_completion_unknown_when_start_is_duplicated() -> None:
    """Production dependency: exact-pair start indexing in ``parse_code``.

    This fails if exact-pair starts are collapsed with ``setdefault`` or if a
    completion is applied to the first duplicate instead of requiring one
    candidate just like the task-only fallback.
    """
    records: list[object] = []
    records.extend(_background_start_records(task_id="same-task", tool_id="same-tool", command="true", suffix="first"))
    records.extend(
        _background_start_records(task_id="same-task", tool_id="same-tool", command="false", suffix="second")
    )
    records.append(
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "sessionId": "background-sidecars",
            "content": _task_notification(
                task_id="same-task",
                tool_id="same-tool",
                status="failed",
                summary='Background command "false" failed with exit code 1',
            ),
        }
    )

    parsed = parse_code(records, "background-exact-ambiguous")
    by_id = {message.provider_message_id: message for message in parsed.messages}
    assert by_id["start-first"].blocks[0].exit_code is None
    assert by_id["start-first"].blocks[0].is_error is None
    assert by_id["start-second"].blocks[0].exit_code is None
    assert by_id["start-second"].blocks[0].is_error is None


def test_parse_code_projects_background_completion_outcomes_through_actions(tmp_path: Path) -> None:
    """Production route: ``parse_code`` -> ``ArchiveStore`` -> ``actions``.

    This fails if ``_project_background_task_completions`` or the
    ``background_task_completion`` event emission is removed, if the exact
    task-id/tool-use-id join is replaced with no correlation, or if the
    known-template exit-code extraction is removed from
    ``ClaudeCodeBackgroundTaskNotification``.
    """
    parsed = parse_code(_background_task_records(), "background-outcomes")
    by_id = {message.provider_message_id: message for message in parsed.messages}
    failed_start = by_id["start-fail"].blocks[0]
    assert failed_start.exit_code == 1
    assert failed_start.is_error is True
    assert failed_start.metadata == {
        "claude_background_task_id": "task-fail",
        "claude_background_completion_status": "failed",
        "claude_background_output_file": "/tmp/task-fail-final.output",
    }
    assert by_id["notification-fail"].text is not None
    assert "<task-notification>" in by_id["notification-fail"].text
    completion_events = [event for event in parsed.session_events if event.event_type == "background_task_completion"]
    assert [event.payload for event in completion_events] == [
        {
            "task_id": "task-ok",
            "tool_use_id": "tool-ok",
            "output_file": "/tmp/task-ok.output",
            "status": "completed",
            "summary": 'Background command "true" completed (exit code 0)',
            "exit_code": 0,
        },
        {
            "task_id": "task-fail",
            "tool_use_id": "tool-fail",
            "output_file": "/tmp/task-fail-final.output",
            "status": "failed",
            "summary": 'Background command "false" failed with exit code 1',
            "exit_code": 1,
        },
    ]

    archive_root = tmp_path / "background-outcomes"
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(parsed)
        actions = archive.query_session_actions([session_id], limit=10)

    with sqlite3.connect(archive_root / "index.db") as conn:
        event_rows = conn.execute(
            """
            SELECT source_message_provider_id, payload_json
            FROM session_events
            WHERE session_id = ? AND event_type = 'background_task_completion'
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()

    outcomes = {action.tool_command: (action.is_error, action.exit_code) for action in actions}
    assert outcomes == {
        "true": (0, 0),
        "false": (1, 1),
        "printf foreground": (0, None),
    }
    assert len(actions) == 3  # Duplicate notification updates one source action, never creates another.
    assert [(source_id, json.loads(payload)) for source_id, payload in event_rows] == [
        ("notification-ok", completion_events[0].payload),
        ("notification-fail-update", completion_events[1].payload),
    ]


def test_parse_code_degrades_changed_background_completion_template_to_unknown() -> None:
    """Production dependency: the strict typed notification parser in ``parse_code``.

    This fails if the known-template ``fullmatch`` is loosened to extract an
    arbitrary ``exit code 7`` phrase, or if background start acknowledgements
    retain their provider ``is_error=false`` success flag.
    """
    records = _background_task_records()
    notification = records[-2]
    assert isinstance(notification, dict)
    message = notification["message"]
    assert isinstance(message, dict)
    message["content"] = """<task-notification>
<task-id>task-fail</task-id>
<tool-use-id>tool-fail</tool-use-id>
<output-file>/tmp/task-fail.output</output-file>
<status>completed</status>
<summary>Background command \"false\" ended unexpectedly; exit code 7</summary>
</task-notification>"""
    # Remove the duplicate so this drifted provider update is terminal.
    records.pop()

    parsed = parse_code(records, "background-outcomes-drift")
    by_id = {message.provider_message_id: message for message in parsed.messages}
    failed_start = by_id["start-fail"].blocks[0]
    foreground = by_id["result-foreground"].blocks[0]
    assert failed_start.exit_code is None
    assert failed_start.is_error is None
    assert foreground.exit_code is None
    assert foreground.is_error is False


def test_parse_code_drops_progress_hook_records() -> None:
    """#1617: ``type=progress`` is a hook lifecycle event, not message content."""
    items: list[object] = [
        {
            "type": "user",
            "uuid": "u-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {"role": "user", "content": "real prompt"},
        },
        {
            "type": "progress",
            "uuid": "prog-1",
            "sessionId": "sess-1",
            "timestamp": 1704067201,
            "data": {"hookEvent": "PreToolUse", "hookName": "test-hook", "command": "echo"},
            "parentToolUseID": "tool-1",
            "toolUseID": "tool-1",
        },
        {
            "type": "progress",
            "uuid": "prog-2",
            "sessionId": "sess-1",
            "timestamp": 1704067202,
            "data": {"hookEvent": "PostToolUse"},
        },
        {
            "type": "assistant",
            "uuid": "a-1",
            "sessionId": "sess-1",
            "timestamp": 1704067203,
            "message": {"role": "assistant", "content": "reply"},
        },
    ]

    result = parse_code(items, "fallback-progress")

    # Two real messages survive; both progress hook events are dropped.
    provider_ids = {m.provider_message_id for m in result.messages}
    assert provider_ids == {"u-1", "a-1"}
    assert all("prog" not in pid for pid in provider_ids)


def test_parse_code_drops_non_message_sidecars() -> None:
    items: list[object] = [
        {
            "type": "user",
            "uuid": "u-1",
            "sessionId": "sess-sidecar",
            "message": {"role": "user", "content": "real prompt"},
        },
        {"type": "attachment", "uuid": "att-1", "sessionId": "sess-sidecar", "attachment": {"name": "x.py"}},
        {"type": "mode", "mode": "default", "sessionId": "sess-sidecar"},
        {"type": "last-prompt", "lastPrompt": "real prompt", "sessionId": "sess-sidecar"},
        {"type": "ai-title", "aiTitle": "Title", "sessionId": "sess-sidecar"},
        {"type": "bridge-session", "sessionId": "sess-sidecar", "bridgeSessionId": "b1"},
        {"type": "permission-mode", "permissionMode": "acceptEdits", "sessionId": "sess-sidecar"},
        {"type": "pr-link", "sessionId": "sess-sidecar", "prNumber": 1},
    ]

    result = parse_code(items, "fallback-sidecar")

    assert [message.provider_message_id for message in result.messages] == ["u-1"]
    assert result.messages[0].material_origin is MaterialOrigin.HUMAN_AUTHORED


def test_parse_code_drops_empty_system_sidecars() -> None:
    result = parse_code(
        [
            {"type": "system", "uuid": "sys-empty", "sessionId": "sess-empty"},
            {
                "type": "user",
                "uuid": "u-1",
                "sessionId": "sess-empty",
                "message": {"role": "user", "content": "real prompt"},
            },
        ],
        "fallback-empty",
    )

    assert [message.provider_message_id for message in result.messages] == ["u-1"]


def test_parse_code_continuation_summary_is_runtime_context() -> None:
    result = parse_code(
        [
            {
                "type": "user",
                "uuid": "continued",
                "sessionId": "sess-continued",
                "message": {
                    "role": "user",
                    "content": "This session is being continued from a previous conversation that ran out of context.\n\nSummary...",
                },
            }
        ],
        "fallback-continued",
    )

    assert result.messages[0].message_type is MessageType.CONTEXT
    assert result.messages[0].material_origin is MaterialOrigin.RUNTIME_CONTEXT


def test_parse_code_compaction_summary_is_generated_context() -> None:
    result = parse_code(
        [
            {
                "type": "summary",
                "uuid": "summary-1",
                "sessionId": "sess-summary",
                "summary": "Compressed context for the next turn.",
                "timestamp": 1704067200,
            }
        ],
        "fallback-summary",
    )

    assert result.messages[0].message_type is MessageType.SUMMARY
    assert result.messages[0].material_origin is MaterialOrigin.GENERATED_CONTEXT_PACK


def test_parse_code_non_human_user_origin_is_runtime_protocol() -> None:
    result = parse_code(
        [
            {
                "type": "user",
                "uuid": "task-origin",
                "sessionId": "sess-origin",
                "origin": {"kind": "task-notification"},
                "message": {"role": "user", "content": "background task completed"},
            }
        ],
        "fallback-origin",
    )

    assert result.messages[0].message_type is MessageType.PROTOCOL
    assert result.messages[0].material_origin is MaterialOrigin.RUNTIME_PROTOCOL


def test_parse_code_interrupt_marker_is_protocol_not_human_prompt() -> None:
    result = parse_code(
        [
            {
                "type": "user",
                "uuid": "interrupt",
                "sessionId": "sess-interrupt",
                "message": {"role": "user", "content": "[Request interrupted by user]"},
            }
        ],
        "fallback-interrupt",
    )

    assert result.messages[0].message_type is MessageType.PROTOCOL
    assert result.messages[0].material_origin is MaterialOrigin.RUNTIME_PROTOCOL


def test_message_input_tokens_stays_raw_unlike_codex_disjoint_fix() -> None:
    """Control case for polylogue-f2qv.2: Anthropic's native ``input_tokens``
    already excludes the cached portion (unlike Codex's inclusive convention),
    so the Claude Code parser must NOT subtract ``cache_read_input_tokens``
    out of it -- doing so would under-count fresh input. ``input_tokens`` and
    ``cache_read_input_tokens`` are already disjoint additive lanes as
    reported."""
    items: list[object] = [
        {
            "type": "assistant",
            "uuid": "asst-1",
            "sessionId": "sess-cache",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": "done",
                "model": "claude-opus-4-8",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 2400,
                    "cache_creation_input_tokens": 10,
                },
            },
        },
    ]

    result = parse_code(items, "fallback-cache")

    assert result.messages[0].input_tokens == 100
    assert result.messages[0].cache_read_tokens == 2400
    assert result.messages[0].cache_write_tokens == 10


def test_message_usage_payload_captures_server_tool_use() -> None:
    """web_search/web_fetch request counts are billed separately and must be
    preserved in the usage event payload (they ride into payload_json)."""
    from polylogue.sources.parsers.claude.code_parser import _message_usage_event_payload

    payload = _message_usage_event_payload(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "server_tool_use": {"web_search_requests": 3, "web_fetch_requests": 1},
        },
        model_name="claude-opus-4-8",
        model_effort=None,
    )
    assert payload["server_tool_use"] == {"web_search_requests": 3, "web_fetch_requests": 1}


def test_message_usage_payload_omits_all_zero_server_tool_use() -> None:
    """Most CLI sessions never call web tools; an all-zero sub-dict is dropped so
    payload_json is not bloated on every message."""
    from polylogue.sources.parsers.claude.code_parser import _message_usage_event_payload

    payload = _message_usage_event_payload(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        },
        model_name=None,
        model_effort=None,
    )
    assert "server_tool_use" not in payload


def test_message_usage_payload_without_server_tool_use_key() -> None:
    from polylogue.sources.parsers.claude.code_parser import _message_usage_event_payload

    payload = _message_usage_event_payload(
        {"input_tokens": 10, "output_tokens": 5},
        model_name=None,
        model_effort=None,
    )
    assert "server_tool_use" not in payload
