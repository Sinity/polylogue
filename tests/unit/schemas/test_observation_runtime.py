"""Focused tests for schema-observation extraction helpers."""

from __future__ import annotations

from polylogue.core.enums import Provider
from polylogue.schemas.observation import ProviderConfig, extract_schema_units_from_payload


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
            source_name=Provider.CLAUDE_CODE,
            source_path="/tmp/session.jsonl",
            raw_id="raw-1",
            observed_at="2026-01-01T00:00:00Z",
            config=config,
        )

        assert len(units) == 1
        unit = units[0]
        assert unit.session_id == "raw-1"
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
            source_name=Provider.CHATGPT,
            source_path="/tmp/sessions.json",
            raw_id="raw-docs",
            config=config,
        )

        assert len(units) == 2
        assert {unit.session_id for unit in units} == {"conv-1", "conv-2"}
        assert all(unit.artifact_kind == "session_document" for unit in units)
