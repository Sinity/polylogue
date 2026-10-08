"""Neutral complete-document representatives for canonical sequence detection."""

from __future__ import annotations

from polylogue.core.enums import Provider
from polylogue.sources.decoder_json import JsonValue


def sequence_document_cases() -> list[tuple[str, Provider, dict[str, JsonValue]]]:
    return [
        ("gemini-cli", Provider.GEMINI_CLI, gemini_document("sequence-one")),
        (
            "hermes-atof",
            Provider.HERMES,
            {
                "atof_version": "0.1",
                "kind": "mark",
                "uuid": "atof-one",
                "timestamp": "2026-01-01T00:00:00Z",
                "name": "neutral",
            },
        ),
        (
            "chatgpt",
            Provider.CHATGPT,
            {
                "id": "chat-one",
                "current_node": "node",
                "create_time": 1,
                "mapping": {"node": {"id": "node", "parent": None, "children": [], "message": None}},
            },
        ),
        ("claude-ai", Provider.CLAUDE_AI, claude_document("claude-one")),
        (
            "claude-memory",
            Provider.CLAUDE_AI,
            {"account_uuid": "account-one", "conversations_memory": "neutral memory"},
        ),
        ("claude-project", Provider.CLAUDE_AI, {"uuid": "project-one", "docs": [], "prompt_template": "neutral"}),
        (
            "claude-design",
            Provider.CLAUDE_DESIGN,
            {"uuid": "design-one", "project": {"uuid": "project-one"}, "messages": []},
        ),
        ("grok-native", Provider.GROK, {"conversation": {"conversationId": "native-one"}, "responses": []}),
        ("grok-export", Provider.GROK, grok_document("export-one")),
        ("drive", Provider.GEMINI, {"chunkedPrompt": {"chunks": [{"role": "user", "text": "neutral prompt"}]}}),
        (
            "browser",
            Provider.CHATGPT,
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "capture_id": "chatgpt:browser-one",
                "provenance": {
                    "source_url": "https://chatgpt.com/c/browser-one",
                    "captured_at": "2026-01-01T00:00:00Z",
                    "adapter_name": "chatgpt-dom-v1",
                    "capture_mode": "snapshot",
                },
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "browser-one",
                    "title": "neutral",
                    "turns": [{"provider_turn_id": "turn-one", "role": "user", "text": "neutral", "ordinal": 0}],
                },
            },
        ),
    ]


def gemini_document(identity: str) -> dict[str, JsonValue]:
    return {
        "sessionId": identity,
        "kind": "main",
        "startTime": "2026-01-01T00:00:00Z",
        "messages": [
            {
                "id": identity + "-message",
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "content": "neutral prompt",
            }
        ],
    }


def claude_document(identity: str) -> dict[str, JsonValue]:
    return {
        "uuid": identity,
        "name": "neutral",
        "chat_messages": [
            {
                "uuid": identity + "-message",
                "sender": "human",
                "text": "neutral prompt",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
    }


def grok_document(identity: str) -> dict[str, JsonValue]:
    return {
        "conversations": [
            {
                "conversation": {"id": identity, "title": "neutral"},
                "responses": [
                    {
                        "response": {
                            "id": identity + "-message",
                            "sender": "human",
                            "message": "neutral prompt " + identity,
                        }
                    }
                ],
            }
        ]
    }


def large_chatgpt_document(identity: str, text: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "current_node": "node",
        "create_time": 1,
        "mapping": {
            "node": {
                "id": "node",
                "parent": None,
                "children": [],
                "message": {
                    "id": identity + "-message",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [text]},
                    "create_time": 1,
                },
            }
        },
    }
