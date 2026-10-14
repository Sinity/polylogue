"""Neutral native streams for inferred bookkeeping semantics."""

from __future__ import annotations

import uuid

from polylogue.core.json import JSONDocument


def bookkeeping_records(provider: str, number: int) -> list[JSONDocument]:
    """Keep repeated and changed facts in the same observed session."""
    session_id = str(uuid.UUID(int=number, version=4))
    if provider == "claude-code":
        user_id = str(uuid.UUID(int=100 + number, version=4))
        return [
            {
                "type": "user",
                "uuid": user_id,
                "parentUuid": None,
                "origin": {"kind": "human"},
                "sessionId": session_id,
                "message": {"role": "user", "content": [{"type": "text", "text": "Neutral question."}]},
            },
            {"type": "relocated", "sessionId": session_id, "relocatedCwd": "/workspace/neutral-relocated"},
            {
                "type": "assistant",
                "uuid": str(uuid.UUID(int=200 + number, version=4)),
                "parentUuid": user_id,
                "sessionId": session_id,
                "attributionAgent": "neutral-agent",
                "attributionSkill": "neutral-skill",
                "attributionMcpServer": "neutral-server",
                "attributionMcpTool": "neutral-tool",
                "attributionPlugin": "neutral-plugin",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Neutral answer."}]},
            },
        ]
    if provider != "codex":
        raise ValueError(provider)
    rows: list[JSONDocument] = [
        {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2025-01-01T00:00:00Z"}}
    ]
    for turn, instructions in enumerate(
        ("Neutral first instruction.", "Neutral first instruction.", "Neutral changed instruction.")
    ):
        turn_id = str(uuid.UUID(int=1000 + number * 10 + turn, version=4))
        rows.extend(
            [
                {"type": "turn_context", "payload": {"turn_id": turn_id, "user_instructions": instructions}},
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "message": f"Neutral event {turn}.",
                        "turn_id": turn_id,
                        "thread_id": session_id,
                        "metadata": {"turn_id": turn_id},
                        "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"Neutral request {turn}."}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": f"Neutral reasoning {turn}."}],
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "item_completed",
                        "item": {
                            "type": "CommandExecution",
                            "id": f"command-{turn}",
                            "command": "printf neutral",
                            "exit_code": 0,
                        },
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "item_completed",
                        "item": {
                            "type": "FileChange",
                            "id": f"patch-{turn}",
                            "changes": [{"path": "neutral.txt", "kind": "update"}],
                        },
                    },
                },
            ]
        )
    return rows
