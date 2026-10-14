"""Fresh qualified packages inferred from authored neutral source files.

The caller owns the root and its lifetime. No committed catalogue or operator
source enters this fixture. Expected source content is authored separately from
schema annotations and remains available for independent parser assertions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.json import JSONDocument
from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.generation.workflow import (
    build_provider_bundle_from_sources,
    persist_generated_provider_bundle,
)
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput

QUALIFIED_SCHEMA_PROVIDERS = ("claude-code", "codex", "claude-ai", "chatgpt", "gemini")


@dataclass(frozen=True)
class NeutralSourceExpectation:
    """Authored wire content, independent of inferred schema or generator state."""

    filename: str
    message_count: int
    bodies: tuple[str, ...]
    title: str | None = None

    @property
    def body_count(self) -> int:
        return len(self.bodies)


@dataclass(frozen=True)
class _NeutralSource:
    payload: JSONDocument | list[JSONDocument]
    expected: NeutralSourceExpectation


_INPUT = "Neutral user question."
_OUTPUT = "Neutral assistant answer."
_TOOL_OUTPUT = "Neutral tool output."
_SESSION = "11111111-1111-4111-8111-111111111111"
_SECOND_SESSION = "22222222-2222-4222-8222-222222222222"
_USER = "33333333-3333-4333-8333-333333333333"
_ASSISTANT = "44444444-4444-4444-8444-444444444444"
_TIME = "2025-01-01T00:00:00Z"


def _neutral_sources(provider: str, *, tool_cases: bool = False) -> tuple[_NeutralSource, ...]:
    if provider == "claude-code":
        rows: list[JSONDocument] = [
            {
                "type": "user",
                "uuid": _USER,
                "parentUuid": None,
                "sessionId": _SESSION,
                "timestamp": _TIME,
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": _INPUT}],
                },
            },
            {
                "type": "assistant",
                "uuid": _ASSISTANT,
                "parentUuid": _USER,
                "sessionId": _SESSION,
                "timestamp": _TIME,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-neutral",
                            "name": "Read",
                            "input": {"file_path": "neutral.txt"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "uuid": "55555555-5555-4555-8555-555555555555",
                "parentUuid": _ASSISTANT,
                "sessionId": _SESSION,
                "timestamp": _TIME,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-neutral",
                            "content": _TOOL_OUTPUT,
                            **({"is_error": False} if tool_cases else {}),
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "uuid": "66666666-6666-4666-8666-666666666666",
                "parentUuid": "55555555-5555-4555-8555-555555555555",
                "sessionId": _SESSION,
                "timestamp": _TIME,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": _OUTPUT}],
                },
            },
        ]
        string_rows: list[JSONDocument] = [
            {
                "type": "user",
                "uuid": _USER,
                "parentUuid": None,
                "sessionId": _SECOND_SESSION,
                "timestamp": _TIME,
                "message": {"role": "user", "content": _INPUT},
            },
            {
                "type": "assistant",
                "uuid": _ASSISTANT,
                "parentUuid": _USER,
                "sessionId": _SECOND_SESSION,
                "timestamp": _TIME,
                "message": {"role": "assistant", "content": _OUTPUT},
            },
        ]
        if tool_cases:
            return (_NeutralSource(rows, NeutralSourceExpectation("blocks.jsonl", 4, (_INPUT, _TOOL_OUTPUT, _OUTPUT))),)
        return (
            _NeutralSource(
                rows,
                NeutralSourceExpectation("blocks.jsonl", 4, (_INPUT, _TOOL_OUTPUT, _OUTPUT)),
            ),
            _NeutralSource(
                string_rows,
                NeutralSourceExpectation("strings.jsonl", 2, (_INPUT, _OUTPUT)),
            ),
        )
    if provider == "codex":
        rows = [
            {"type": "session_meta", "payload": {"id": _SESSION, "timestamp": _TIME}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": _INPUT}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call-neutral",
                    "name": "Read",
                    "arguments": '{"file_path":"neutral.txt"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-neutral",
                    "output": {"output": _TOOL_OUTPUT, "exit_code": 0, "is_error": False}
                    if tool_cases
                    else _TOOL_OUTPUT,
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": _OUTPUT}],
                },
            },
        ]
        flat: list[JSONDocument] = [
            {"id": _SECOND_SESSION, "timestamp": _TIME},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": _INPUT}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _OUTPUT}],
            },
        ]
        if tool_cases:
            return (
                _NeutralSource(
                    rows,
                    NeutralSourceExpectation(
                        "enveloped.jsonl",
                        4,
                        (_INPUT, '{"exit_code": 0, "is_error": false, "output": "Neutral tool output."}', _OUTPUT),
                    ),
                ),
            )
        return (
            _NeutralSource(
                rows,
                NeutralSourceExpectation("enveloped.jsonl", 4, (_INPUT, _TOOL_OUTPUT, _OUTPUT)),
            ),
            _NeutralSource(flat, NeutralSourceExpectation("flat.jsonl", 2, (_INPUT, _OUTPUT))),
        )
    if provider == "claude-ai":
        document: JSONDocument = {
            "uuid": _SESSION,
            "name": "Neutral Claude conversation",
            "created_at": _TIME,
            "updated_at": _TIME,
            "chat_messages": [
                {"uuid": _USER, "sender": "human", "text": _INPUT, "created_at": _TIME},
                {
                    "uuid": _ASSISTANT,
                    "sender": "assistant",
                    "text": _OUTPUT,
                    "created_at": _TIME,
                },
            ],
        }
        return (
            _NeutralSource(
                document,
                NeutralSourceExpectation(
                    "conversation.json",
                    2,
                    (_INPUT, _OUTPUT),
                    "Neutral Claude conversation",
                ),
            ),
        )
    if provider == "chatgpt":
        document = {
            "id": _SESSION,
            "title": "Neutral ChatGPT conversation",
            "create_time": 1735689600,
            "update_time": 1735689601,
            "current_node": _ASSISTANT,
            "mapping": {
                _USER: {
                    "id": _USER,
                    "parent": None,
                    "children": [_ASSISTANT],
                    "message": {
                        "id": _USER,
                        "author": {"role": "user"},
                        "create_time": 1735689600,
                        "content": {"content_type": "text", "parts": [_INPUT]},
                    },
                },
                _ASSISTANT: {
                    "id": _ASSISTANT,
                    "parent": _USER,
                    "children": [],
                    "message": {
                        "id": _ASSISTANT,
                        "author": {"role": "assistant"},
                        "create_time": 1735689601,
                        "content": {"content_type": "text", "parts": [_OUTPUT]},
                    },
                },
            },
        }
        return (
            _NeutralSource(
                document,
                NeutralSourceExpectation(
                    "conversation.json",
                    2,
                    (_INPUT, _OUTPUT),
                    "Neutral ChatGPT conversation",
                ),
            ),
        )
    if provider == "gemini":
        document = {
            "id": _SESSION,
            "chunkedPrompt": {
                "chunks": [
                    {"id": _USER, "role": "user", "text": _INPUT},
                    {"id": _ASSISTANT, "role": "model", "text": _OUTPUT},
                ]
            },
        }
        return (
            _NeutralSource(
                document,
                NeutralSourceExpectation("conversation.json", 2, (_INPUT, _OUTPUT)),
            ),
        )
    raise ValueError(f"No authored neutral sources for provider {provider!r}")


def neutral_source_expectations(
    *,
    providers: Sequence[str] = QUALIFIED_SCHEMA_PROVIDERS,
) -> dict[str, tuple[NeutralSourceExpectation, ...]]:
    """Return source-message/body counts and explicitly authored document titles."""
    return {provider: tuple(source.expected for source in _neutral_sources(provider)) for provider in providers}


def build_qualified_schema_registry(
    root: Path,
    *,
    providers: Sequence[str] = QUALIFIED_SCHEMA_PROVIDERS,
    tool_cases: bool = False,
) -> SchemaRegistry:
    """Infer and publish measured packages from caller-owned neutral files.

    Each provider receives one current source inventory and one genuinely
    inferred version. This fixture does not promise multiple versions or more
    element kinds than its observed source inventory supplies.
    """
    providers = tuple(providers)
    if len(set(providers)) != len(providers):
        raise ValueError("Neutral source providers must be unique")
    if tool_cases and not set(providers) <= {"claude-code", "codex"}:
        raise ValueError("Tool outcome fixtures require native Claude Code/Codex sources")
    registry_root = root / "registry"
    for provider in providers:
        sources = _neutral_sources(provider, tool_cases=tool_cases)
        source_root = root / "sources" / provider
        source_root.mkdir(parents=True, exist_ok=True)
        for source in sources:
            path = source_root / source.expected.filename
            if isinstance(source.payload, list):
                text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source.payload)
            else:
                text = json.dumps(source.payload, ensure_ascii=False) + "\n"
            path.write_text(text, encoding="utf-8")
        bundle = build_provider_bundle_from_sources(
            provider,
            source_inputs=(SchemaSourceInput(provider, source_root),),
            cache_path=root / f"{provider}-evidence.sqlite3",
            max_workers=1,
            privacy_config=None,
            prior_catalog=None,
        )
        assert bundle.result.success, (provider, bundle.result.error)
        assert bundle.catalog is not None and bundle.package_workload_profiles, provider
        source_receipt = bundle.result.phase_receipt.get("source")
        assert isinstance(source_receipt, dict), provider
        assert source_receipt["source_terminal_outcomes"] == {"included": len(sources)}, (provider, source_receipt)
        persist_generated_provider_bundle(registry_root, provider, bundle)
    registry = SchemaRegistry(storage_root=registry_root)
    audit = audit_schema_bundle_privacy(registry=registry)
    assert audit.failed == 0, audit.format_text()
    return registry


__all__ = [
    "QUALIFIED_SCHEMA_PROVIDERS",
    "NeutralSourceExpectation",
    "build_qualified_schema_registry",
    "neutral_source_expectations",
]
