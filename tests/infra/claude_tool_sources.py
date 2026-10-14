"""Neutral native Claude tool records for measured semantic generation."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from polylogue.core.json import JSONDocument, JSONValue


def mixed_claude_tool_records(
    number: int, *, multiple_calls: bool = False, result_arrays: bool = False
) -> list[JSONDocument]:
    session = str(uuid.UUID(int=number, version=4))
    rows: list[JSONDocument] = []

    def append(kind: str, content: JSONValue, **fields: JSONValue) -> None:
        rows.append(
            {
                "type": kind,
                "uuid": str(uuid.UUID(int=number * 100 + len(rows) + 1, version=4)),
                "parentUuid": rows[-1]["uuid"] if rows else None,
                "sessionId": session,
                "timestamp": "2025-01-01T00:00:00Z",
                "message": {"role": kind, "content": content},
                **fields,
            }
        )

    append("user", [{"type": "text", "text": "Review the neutral example."}])
    cases: tuple[tuple[str, JSONDocument, JSONDocument], ...] = (
        ("Read", {"file_path": "neutral.txt"}, {}),
        (
            "Edit",
            {"file_path": "neutral.txt", "old_string": "old", "new_string": "new"},
            {
                "filePath": "neutral.txt",
                "originalFile": "old\n",
                "oldString": "old",
                "newString": "new",
                "replaceAll": False,
                "structuredPatch": [
                    {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1, "lines": ["-old", "+new"]}
                ],
            },
        ),
        (
            "Write",
            {"file_path": "created.txt", "content": "neutral content\n"},
            {"filePath": "created.txt", "originalFile": "", "newString": "neutral content\n"},
        ),
        (
            "TodoWrite",
            {
                "todos": [
                    {"content": "Review neutral example", "status": "completed", "activeForm": "Reviewing example"}
                ]
            },
            {
                "oldTodos": [{"content": "Review neutral example", "status": "pending", "priority": "high"}],
                "newTodos": [{"content": "Review neutral example", "status": "completed", "priority": "high"}],
            },
        ),
        ("Bash", {"command": "printf neutral"}, {"stdout": "neutral", "stderr": "", "interrupted": False}),
        ("Bash", {"command": "false", "run_in_background": True}, {"backgroundTaskId": "task-neutral"}),
        (
            "Agent",
            {"prompt": "Review the neutral example", "subagent_type": "general-purpose"},
            {"agentId": "neutral-child"},
        ),
        ("private_tool_neutral", {}, {}),
    )
    for index, (name, tool_input, result) in enumerate(cases):
        tool_id = f"call-neutral-{index}"
        if multiple_calls and index == 0:
            append(
                "assistant",
                [
                    {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input},
                    {"type": "tool_use", "id": "call-neutral-7", "name": "private_tool_neutral", "input": {}},
                ],
            )
        elif not multiple_calls or index != 7:
            append("assistant", [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}])
        content: JSONValue = (
            [{"type": "text", "text": "Neutral tool result."}]
            if result_arrays and name == "Edit"
            else "Command running in background with ID: task-neutral."
            if index == 5
            else "Neutral tool result."
        )
        append(
            "user",
            [{"type": "tool_result", "tool_use_id": tool_id, "content": content, "is_error": False}],
            **({"toolUseResult": result} if result else {}),
        )
    append("assistant", [{"type": "text", "text": "Neutral review complete."}])
    return rows


def write_mixed_claude_tool_sources(
    root: Path, *, multiple_calls: bool = False, result_arrays: bool = False
) -> list[list[JSONDocument]]:
    root.mkdir()
    sessions = [
        mixed_claude_tool_records(number, multiple_calls=multiple_calls, result_arrays=result_arrays)
        for number in range(1, 4)
    ]
    for number, records in enumerate(sessions, 1):
        (root / f"session-{number}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    return sessions
