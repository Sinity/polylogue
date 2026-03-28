"""Focused tests for ValidationService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from polylogue.pipeline.services.validation import ValidationService


class TestValidationService:
    def test_validation_default_mode_is_strict(self, monkeypatch):
        monkeypatch.delenv("POLYLOGUE_SCHEMA_VALIDATION", raising=False)
        service = ValidationService(backend=MagicMock())
        assert service._schema_validation_mode() == "strict"

    async def test_validation_uses_all_record_samples_by_default(self, monkeypatch):
        from polylogue.schemas import ValidationResult

        raw_record = MagicMock(
            raw_id="raw-1",
            raw_content=(
                b'{"type":"session_meta"}\n'
                b'{"type":"response_item","payload":{"type":"message"}}\n'
                b'{"record_type":"state"}'
            ),
            provider_name="codex",
            source_path="/tmp/session.jsonl",
        )
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        backend.mark_raw_validated = AsyncMock()
        backend.mark_raw_parsed = AsyncMock()

        class _CapturingValidator:
            provider = "codex"

            def __init__(self):
                self.max_samples_seen = None

            def validation_samples(self, payload, max_samples=None):
                self.max_samples_seen = max_samples
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        capturing = _CapturingValidator()
        monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_provider", lambda _provider: capturing)

        result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

        assert result.parseable_raw_ids == ["raw-1"]
        assert capturing.max_samples_seen is None

    async def test_validation_strict_detects_malformed_jsonl_beyond_large_prefix(self, monkeypatch):
        from polylogue.schemas import ValidationResult

        raw_record = MagicMock(
            raw_id="raw-1",
            raw_content=(b'{"type":"session_meta"}\n' * 1024) + b"not json at all\n",
            provider_name="codex",
            source_path="/tmp/session.jsonl",
        )
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        backend.mark_raw_validated = AsyncMock()
        backend.mark_raw_parsed = AsyncMock()

        class _AlwaysValidValidator:
            provider = "codex"

            def validation_samples(self, payload, max_samples=16):
                return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "strict")
        monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_provider", lambda _provider: _AlwaysValidValidator())

        result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

        assert result.counts["invalid"] == 1
        assert result.parseable_raw_ids == []
        kwargs = backend.mark_raw_validated.await_args.kwargs
        assert kwargs["status"] == "failed"
        assert "Malformed JSONL lines" in (kwargs.get("error") or "")

    async def test_validation_progress_callback_reports_counts(self, monkeypatch):
        from polylogue.schemas import ValidationResult

        raw_records = [
            MagicMock(raw_id="raw-1", raw_content=b'{"id":"1"}', provider_name="chatgpt", source_path="/tmp/a.json"),
            MagicMock(raw_id="raw-2", raw_content=b'{"id":"2"}', provider_name="chatgpt", source_path="/tmp/b.json"),
        ]
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=raw_records)
        backend.mark_raw_validated = AsyncMock()
        callback = MagicMock()

        class _AlwaysValidValidator:
            provider = "chatgpt"

            def validation_samples(self, payload, max_samples=16):
                return [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_provider", lambda _provider: _AlwaysValidValidator())
        await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1", "raw-2"], progress_callback=callback)

        callback.assert_any_call(0, desc="Validating: 0/2 raw")
        callback.assert_any_call(1, desc="Validating: 1/2 raw")
        callback.assert_any_call(1, desc="Validating: 2/2 raw")

    async def test_validation_persists_payload_provider_from_decoded_payload(self, monkeypatch):
        from polylogue.schemas import ValidationResult

        raw_record = MagicMock(
            raw_id="raw-1",
            raw_content=b'[{"id":"conv-1","mapping":{}}]',
            provider_name="inbox-source",
            source_path="/tmp/conversations.json",
            payload_provider=None,
        )
        backend = MagicMock()
        backend.get_raw_conversations_batch = AsyncMock(return_value=[raw_record])
        backend.mark_raw_validated = AsyncMock()
        backend.mark_raw_parsed = AsyncMock()

        class _AlwaysValidValidator:
            provider = "chatgpt"

            def validation_samples(self, payload, max_samples=16):
                return [payload]

            def validate(self, _sample):
                return ValidationResult(is_valid=True)

        monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_provider", lambda _provider: _AlwaysValidValidator())
        result = await ValidationService(backend=backend).validate_raw_ids(raw_ids=["raw-1"])

        assert result.parseable_raw_ids == ["raw-1"]
        kwargs = backend.mark_raw_validated.await_args.kwargs
        assert kwargs["provider"] == "chatgpt"
        assert kwargs["payload_provider"] == "chatgpt"
