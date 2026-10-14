"""Neutral native Codex calls and results, independent of the generator."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from polylogue.core.json import JSONDocument, JSONValue

PRIVATE_TOOL = "neutral_private_function"


def mixed_codex_tool_records(number: int, *, custom: bool = False) -> list[JSONDocument]:
    records: list[JSONDocument] = [
        {
            "type": "session_meta",
            "payload": {"id": str(uuid.UUID(int=number, version=4)), "timestamp": "2025-01-01T00:00:00Z"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Neutral tools."}],
            },
        },
    ]
    cases: tuple[tuple[str, JSONValue], ...] = (
        ("exec_command", {"output": "neutral success", "exit_code": 0, "is_error": False}),
        ("exec_command", {"output": "neutral failure", "exit_code": 7, "is_error": True}),
        (PRIVATE_TOOL, {"output": "neutral plain"}),
        ("exec_command", "Wall time: 0.1 seconds\nProcess exited with code 0\nOutput:\nneutral success"),
        ("exec_command", "Wall time: 0.2 seconds\nProcess exited with code 7\nOutput:\nneutral failure"),
        (PRIVATE_TOOL, "neutral plain response"),
    )
    # Simultaneous public and anonymous calls make final identity-class
    # selection observable independently of result-mode marginals.
    for group in (range(3), range(3, 6)):
        for ordinal in group:
            name, _ = cases[ordinal]
            records.append(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": f"call-{ordinal}",
                        "name": name,
                        "arguments": '{"cmd":"printf neutral"}',
                    },
                }
            )
        for ordinal in group:
            _, output = cases[ordinal]
            records.append(
                {
                    "type": "response_item",
                    "payload": {"type": "function_call_output", "call_id": f"call-{ordinal}", "output": output},
                }
            )
    if custom:
        for record in records:
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "function_call":
                payload["type"] = "custom_tool_call"
                payload.pop("arguments", None)
                if payload["name"] == "exec_command":
                    payload["name"] = "apply_patch"
                    payload["input"] = "*** Begin Patch\n*** Update File: neutral.txt\n@@\n-old\n+new\n*** End Patch"
                else:
                    payload["input"] = "neutral private input"
            elif payload.get("type") == "function_call_output":
                payload["type"] = "custom_tool_call_output"
                output = payload.get("output")
                if isinstance(output, str) and output.startswith("Wall time:"):
                    code = "0" if "code 0" in output else "7"
                    payload["output"] = f"Exit code: {code}\nWall time: 0.1 seconds\nOutput:\nneutral patch"
    return records


def write_mixed_codex_tool_sources(root: Path, *, custom: bool = False) -> Path:
    root.mkdir()
    for number in range(1, 4):
        records = mixed_codex_tool_records(number, custom=custom)
        (root / f"session-{number}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    return root
