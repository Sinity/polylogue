"""Focused tests for schema extraction and observation runtime helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.lib.json import JSONDocument
from polylogue.schemas.extraction import extract_message_from_schema
from polylogue.schemas.observation import ProviderConfig, extract_schema_units_from_payload
from polylogue.types import Provider


class TestExtractMessageFromSchema:
    def test_falls_back_to_well_known_fields_without_pins(self) -> None:
        raw: JSONDocument = {
            "uuid": "msg-1",
            "sender": "user",
            "text": "Hello there",
            "created_at": "2026-01-01T00:00:00Z",
            "costUSD": "0.02",
            "durationMs": "15",
            "message": {
                "model": "claude-3.7-sonnet",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                },
            },
        }

        message = extract_message_from_schema(raw, schema={}, provider=Provider.CLAUDE_AI)

        assert message.id == "msg-1"
        assert str(message.role) == "user"
        assert message.text == "Hello there"
        assert message.model == "claude-3.7-sonnet"
        assert message.tokens is not None
        assert message.tokens.total_tokens == 14
        assert message.cost is not None
        assert message.cost.total_usd == 0.02
        assert message.duration_ms == 15

    def test_extracts_text_and_blocks_from_structured_body(self) -> None:
        raw: JSONDocument = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "First block"},
                    {"type": "thinking", "thinking": "Reasoning trace"},
                    {"type": "tool_use", "name": "write_file", "input": {"path": "a.txt"}},
                ],
            },
            "timestamp": "2026-01-01T00:01:00Z",
        }
        schema: JSONDocument = {
            "properties": {
                "message": {
                    "properties": {
                        "content": {"x-polylogue-semantic-role": "message_body"},
                    }
                }
            }
        }

        message = extract_message_from_schema(raw, schema=schema, provider=Provider.CLAUDE_CODE)

        assert message.text == "First block"
        assert len(message.content_blocks) == 3
        assert len(message.reasoning_traces) == 1
        assert len(message.tool_calls) == 1

    def test_uses_pinned_paths_with_wildcards_and_message_id_aliases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("polylogue.schemas.extraction.load_pins", lambda _provider: object())
        monkeypatch.setattr(
            "polylogue.schemas.extraction.resolve_pinned_paths",
            lambda _schema, _pins: {
                "message_role": ".meta.*.sender",
                "message_body": ".payload.parts",
                "message_timestamp": ".timestamps.anyOf[0].created_at",
            },
        )
        raw: JSONDocument = {
            "messageId": "msg-2",
            "meta": {"nested": {"sender": "assistant"}},
            "payload": {
                "parts": [
                    {"type": "text", "text": "Pinned text"},
                    {"type": "tool_use", "name": "edit", "input": {"path": "a.txt"}},
                ]
            },
            "timestamps": {"created_at": "2026-01-02T00:00:00Z"},
            "message": {"model": 123, "usage": {"output_tokens": 7}},
        }

        message = extract_message_from_schema(raw, schema={}, provider=Provider.CLAUDE_CODE)

        assert message.id == "msg-2"
        assert str(message.role) == "assistant"
        assert message.text == "Pinned text"
        assert message.timestamp == datetime(2026, 1, 2, tzinfo=timezone.utc)
        assert message.model is None
        assert message.tokens is not None
        assert message.tokens.output_tokens == 7
        assert len(message.content_blocks) == 2
        assert len(message.tool_calls) == 1

    def test_extracts_nested_parts_and_scalar_body_fallbacks(self) -> None:
        nested_raw: JSONDocument = {
            "id": "msg-nested",
            "author": "assistant",
            "message": {
                "content": {
                    "parts": [
                        {"text": "Nested"},
                        {"content": "parts"},
                    ]
                }
            },
            "cost_usd": 0.5,
            "duration_ms": 4.9,
        }
        scalar_raw: JSONDocument = {
            "id": "msg-scalar",
            "body": 5,
            "durationMs": True,
            "costUSD": "bad",
        }

        nested = extract_message_from_schema(nested_raw, schema={}, provider=Provider.CHATGPT)
        scalar = extract_message_from_schema(scalar_raw, schema={}, provider=Provider.UNKNOWN)

        assert nested.text == "Nested\nparts"
        assert nested.cost is not None
        assert nested.cost.total_usd == 0.5
        assert nested.duration_ms == 4
        assert scalar.text == "5"
        assert str(scalar.role) == "unknown"
        assert scalar.duration_ms is None
        assert scalar.cost is None


class TestExtractSchemaUnitsFromPayload:
    def test_record_granularity_compacts_and_profiles_samples(self) -> None:
        config = ProviderConfig(
            name=Provider.CLAUDE_CODE,
            description="Claude Code",
            sample_granularity="record",
            record_type_key="type",
            schema_sample_cap=8,
        )
        payload = [
            {"type": "session_meta", "id": "sess-1"},
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "x" * 2048}]},
            {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        ]

        units = extract_schema_units_from_payload(
            payload,
            provider_name=Provider.CLAUDE_CODE,
            source_path="/tmp/session.jsonl",
            raw_id="raw-1",
            observed_at="2026-01-01T00:00:00Z",
            config=config,
        )

        assert len(units) == 1
        unit = units[0]
        assert unit.conversation_id == "raw-1"
        assert unit.bundle_scope == "session"
        assert any(token.startswith("bucket:") for token in unit.profile_tokens)
        content = unit.schema_samples[1]["content"]
        assert isinstance(content, list)
        first_block = content[0]
        assert isinstance(first_block, dict)
        text = first_block.get("text")
        assert isinstance(text, str)
        assert len(text) == 1024

    def test_document_granularity_emits_one_unit_per_document(self) -> None:
        config = ProviderConfig(
            name=Provider.CHATGPT,
            description="ChatGPT",
            sample_granularity="document",
        )
        payload = [
            {"id": "conv-1", "mapping": {"node-1": {"message": {"id": "m1"}}}},
            {"id": "conv-2", "mapping": {"node-9": {"message": {"author": {"role": "user"}}}}},
        ]

        units = extract_schema_units_from_payload(
            payload,
            provider_name=Provider.CHATGPT,
            source_path="/tmp/conversations.json",
            raw_id="raw-docs",
            config=config,
        )

        assert len(units) == 2
        assert {unit.conversation_id for unit in units} == {"conv-1", "conv-2"}
        assert all(unit.artifact_kind == "conversation_document" for unit in units)
