"""Neutral native Claude records with asynchronous task and child identities."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.core.json import JSONDocument, JSONValue


def claude_async_records(number: int) -> list[JSONDocument]:
    session = f"neutral-parent-{number}"
    rows: list[JSONDocument] = []

    def append(kind: str, content: JSONValue = None, **fields: JSONValue) -> None:
        row: JSONDocument = {
            "type": kind,
            "uuid": f"neutral-{number}-{len(rows)}",
            "sessionId": session,
            "timestamp": f"2025-01-01T00:00:{len(rows):02d}Z",
            **fields,
        }
        if content is not None:
            row["message"] = {"role": kind, "content": content}
        rows.append(row)

    def call(identity: str, name: str, operand: JSONDocument) -> None:
        append("assistant", [{"type": "tool_use", "id": identity, "name": name, "input": operand}])

    def result(identity: str, evidence: JSONDocument) -> None:
        append(
            "user",
            [{"type": "tool_result", "tool_use_id": identity, "content": "Neutral result.", "is_error": False}],
            toolUseResult=evidence,
        )

    def notification(task: str, tool: str | None) -> None:
        append(
            "user",
            "<task-notification>\n"
            f"<task-id>{task}</task-id>\n"
            + (f"<tool-use-id>{tool}</tool-use-id>\n" if tool is not None else "")
            + "<output-file>/neutral/task.output</output-file>\n"
            "<status>failed</status>\n"
            '<summary>Background command "false" failed with exit code 7</summary>\n'
            "</task-notification>",
            origin={"kind": "task-notification"},
        )

    append("user", "Review the neutral asynchronous work.")
    call("call-foreground", "Bash", {"command": "printf neutral"})
    # No Agent is open. A progress record may exist without claiming a child.
    append("progress", parentToolUseID="call-foreground", data={"type": "agent_progress"})
    result("call-foreground", {"stdout": "neutral", "stderr": "", "interrupted": False})
    for identity, task in (
        ("call-failure", "task-failure"),
        ("call-unknown", "task-unknown"),
        ("call-ambiguous-a", "task-ambiguous"),
        ("call-ambiguous-b", "task-ambiguous"),
    ):
        call(identity, "Bash", {"command": "false", "run_in_background": True})
        result(identity, {"backgroundTaskId": task})
    notification("task-failure", "call-failure")
    notification("task-ambiguous", None)
    call("call-agent", "Agent", {"prompt": "Review the neutral child.", "subagent_type": "general-purpose"})
    for _ in range(2):
        append(
            "progress",
            parentToolUseID="call-agent",
            agentId=f"neutral-parent-emitter-{number}",
            data={"type": "agent_progress", "agentId": f"neutral-child-{number}"},
        )
    result("call-agent", {"agentId": f"neutral-child-{number}"})
    append("assistant", "Neutral work complete.")
    return rows


def write_claude_async_sources(root: Path) -> tuple[Path, ...]:
    root.mkdir()
    mains: list[Path] = []
    for number in range(1, 4):
        main = root / f"neutral-parent-{number}.jsonl"
        main.write_text("".join(json.dumps(row) + "\n" for row in claude_async_records(number)), encoding="utf-8")
        child_root = root / main.stem / "subagents"
        child_root.mkdir(parents=True)
        child = {
            "type": "assistant",
            "uuid": f"neutral-child-message-{number}",
            "sessionId": f"neutral-parent-{number}",
            "agentId": f"neutral-child-{number}",
            "isSidechain": True,
            "timestamp": "2025-01-01T00:01:00Z",
            "message": {"role": "assistant", "content": "Neutral child review."},
        }
        (child_root / f"agent-neutral-child-{number}.jsonl").write_text(json.dumps(child) + "\n", encoding="utf-8")
        (child_root / f"agent-neutral-child-{number}.meta.json").write_text(
            json.dumps({"agentId": f"neutral-child-{number}", "toolUseId": "call-agent"}), encoding="utf-8"
        )
        mains.append(main)
    return tuple(mains)
