"""Focused validator and validation-contract tests for schema handling, plus raw corpus verification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from polylogue.schemas import ValidationResult
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.validator import SchemaValidator, validate_provider_export
from polylogue.schemas.verification import verify_raw_corpus
from polylogue.storage.backends.connection import open_connection


def test_schema_validator_loads_provider(mock_schema_dir):
    """SchemaValidator.for_provider loads the requested schema."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        SchemaValidator._cache.clear()
        validator = SchemaValidator.for_provider("test-provider")
        assert validator.schema["required"] == ["id"]

        with pytest.raises(FileNotFoundError):
            SchemaValidator.for_provider("nonexistent")


def test_validate_valid_data(mock_schema_dir):
    """Valid payloads should pass with no errors or drift."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")
        result = validator.validate({"id": "123", "count": 10, "meta": {"source": "test"}})

    assert result.is_valid
    assert not result.errors
    assert not result.drift_warnings


def test_validate_detects_errors(mock_schema_dir):
    """Validation should report required-field and type errors."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")

        missing = validator.validate({"count": 10})
        wrong_type = validator.validate({"id": 123})

    assert not missing.is_valid
    assert any("id" in error for error in missing.errors)
    assert not wrong_type.is_valid
    assert any("123" in error for error in wrong_type.errors)


def test_validate_detects_drift(mock_schema_dir):
    """Strict mode should surface unexpected fields as drift."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("open-provider", strict=True)
        result = validator.validate({"id": "123", "extra": "drift"})

    assert result.is_valid
    assert result.has_drift
    assert "extra" in result.drift_warnings[0]


def test_provider_alias_maps_claude_to_claude_ai(mock_schema_dir):
    """Runtime provider aliases should resolve to canonical schema providers."""
    alias_schema = {
        "type": "object",
        "properties": {"uuid": {"type": "string"}},
        "required": ["uuid"],
        "additionalProperties": False,
    }
    (mock_schema_dir / "claude-ai.schema.json").write_text(json.dumps(alias_schema), encoding="utf-8")

    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        SchemaValidator._cache.clear()
        validator = SchemaValidator.for_provider("claude")

    assert validator.provider == "claude-ai"
    assert "uuid" in validator.schema["properties"]


def test_validation_samples_record_mode_skips_non_record_documents():
    """Record-mode validation should ignore top-level document payloads."""
    validator = SchemaValidator(
        {
            "type": "object",
            "x-polylogue-sample-granularity": "record",
            "properties": {"type": {"type": "string"}},
        },
        provider="claude-code",
    )

    assert validator.validation_samples({"version": 1, "entries": []}, max_samples=16) == []


def test_schema_validator_prefers_registry_latest(monkeypatch):
    """Latest registry schemas should override packaged fallback files."""
    fake_schema = {
        "type": "object",
        "properties": {"from_registry": {"type": "string"}},
        "additionalProperties": False,
    }

    class _FakeRegistry:
        def get_schema(self, provider: str, version: str = "latest"):
            if provider == "chatgpt" and version == "latest":
                return fake_schema
            return None

    monkeypatch.setattr("polylogue.schemas.validator.SchemaRegistry", _FakeRegistry)
    SchemaValidator._cache.clear()

    validator = SchemaValidator.for_provider("chatgpt")
    assert "from_registry" in validator.schema["properties"]


def test_dynamic_key_maps_do_not_emit_drift_warnings():
    """Explicit dynamic-key containers should suppress key-level drift noise."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": {"type": "object"},
                    "x-polylogue-dynamic-keys": True,
                }
            },
            "additionalProperties": False,
        },
        strict=True,
    )

    result = validator.validate(
        {
            "mapping": {
                "550e8400-e29b-41d4-a716-446655440000": {"id": "node-1"},
                "660f9511-f3ac-52e5-b827-557766551111": {"id": "node-2"},
            }
        }
    )
    assert result.is_valid
    assert not result.has_drift


def test_dynamic_identifier_keys_are_suppressed_in_additional_properties_maps():
    """Identifier-like additional-property keys should not count as drift."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"role": {"type": "string"}},
                        "additionalProperties": False,
                    },
                }
            },
            "additionalProperties": False,
        },
        strict=True,
    )

    result = validator.validate(
        {"mapping": {"msg-550e8400-e29b-41d4-a716-446655440000": {"role": "assistant"}}}
    )
    assert result.is_valid
    assert not result.has_drift


def test_additional_properties_schema_still_detects_nested_drift():
    """Nested additionalProperties schemas must still recurse for drift."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "additionalProperties": {"type": "string"},
                    },
                }
            },
            "additionalProperties": False,
        },
        strict=True,
    )

    result = validator.validate({"mapping": {"static-key": {"name": "node", "extra": "drift"}}})
    assert result.is_valid
    assert result.has_drift
    assert any("mapping.static-key.extra" in warning for warning in result.drift_warnings)


def test_validate_non_strict_mode_skips_drift_detection():
    """Non-strict validation should ignore unexpected-field drift."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "additionalProperties": {"type": "string"},
        },
        strict=False,
    )

    result = validator.validate({"id": "ok", "extra": "allowed"})
    assert result.is_valid
    assert not result.has_drift


def test_format_error_includes_nested_array_path():
    """Error formatting should preserve nested array/object paths."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                }
            },
        },
        strict=False,
    )

    result = validator.validate({"items": [{}]})
    assert not result.is_valid
    assert any(error.startswith("items.0:") for error in result.errors)


def test_looks_dynamic_key_matches_expected_identifier_patterns():
    """Dynamic-key heuristic should match supported identifier families only."""
    validator = SchemaValidator({"type": "object"})

    assert validator._looks_dynamic_key("550e8400-e29b-41d4-a716-446655440000")
    assert validator._looks_dynamic_key("abcdef0123456789abcdef01")
    assert validator._looks_dynamic_key("msg-550e8400-e29b-41d4-a716-446655440000")
    assert not validator._looks_dynamic_key("title")
    assert not validator._looks_dynamic_key("message_text")


def test_available_providers(mock_schema_dir):
    """available_providers should reflect packaged schema files."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        providers = SchemaValidator.available_providers()

    assert "test-provider" in providers
    assert "open-provider" in providers
    assert "nonexistent" not in providers


def test_validate_helper(mock_schema_dir):
    """validate_provider_export should delegate to SchemaValidator cleanly."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        result = validate_provider_export({"id": "123"}, "test-provider")

    assert result.is_valid


def test_missing_provider_raises():
    """Unknown providers should fail with FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No schema found"):
        SchemaValidator.for_provider("nonexistent-provider")


class TestSyntheticRoundTrip:
    """Synthetic data should remain parser-compatible for supported providers."""

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex", "gemini"])
    def test_synthetic_parses_successfully(self, provider: str, synthetic_source) -> None:
        from polylogue.sources import iter_source_conversations

        source = synthetic_source(provider, count=3, seed=42)
        conversations = list(iter_source_conversations(source))

        assert conversations, f"No conversations parsed for {provider}"
        for conversation in conversations:
            assert conversation.messages, f"Empty conversation for {provider}"
            assert any(message.text for message in conversation.messages), f"No message text for {provider}"


def test_validation_result_properties():
    """ValidationResult convenience behavior should stay stable."""
    valid = ValidationResult(is_valid=True)
    assert valid.is_valid
    assert not valid.has_drift
    valid.raise_if_invalid()

    invalid = ValidationResult(is_valid=False, errors=["missing field"])
    assert not invalid.is_valid
    with pytest.raises(ValueError, match="missing field"):
        invalid.raise_if_invalid()

    with_drift = ValidationResult(is_valid=True, drift_warnings=["new field: foo"])
    assert with_drift.is_valid
    assert with_drift.has_drift


# =============================================================================
# Merged from test_schema_verification.py (2024-03-15)
# =============================================================================


def _insert_raw_record(
    *,
    db_path,
    raw_id: str,
    provider_name: str,
    payload_provider: str | None = None,
    source_name: str,
    source_path: str,
    raw_content: bytes,
) -> None:
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_conversations (
                raw_id, provider_name, payload_provider, source_name, source_path, source_index,
                raw_content, acquired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                provider_name,
                payload_provider,
                source_name,
                source_path,
                0,
                raw_content,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def test_verify_raw_corpus_reports_valid_synthetic_chatgpt(db_path, monkeypatch):
    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload, max_samples=16):
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample):
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.verification.SchemaValidator.for_provider",
        lambda _provider: _AlwaysValidValidator(),
    )

    corpus = SyntheticCorpus.for_provider("chatgpt")
    raw = corpus.generate(count=1, seed=42)[0]
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-1",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt.json",
        raw_content=raw,
    )

    report = verify_raw_corpus(db_path=db_path, providers=["chatgpt"], max_samples=16)
    stats = report.providers["chatgpt"]

    assert report.total_records == 1
    assert stats.total_records == 1
    assert stats.valid_records == 1
    assert stats.invalid_records == 0
    assert stats.decode_errors == 0


def test_verify_raw_corpus_counts_missing_schema_as_skipped(db_path):
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-inbox-1",
        provider_name="inbox",
        source_name="inbox",
        source_path="/tmp/inbox.json",
        raw_content=b'{"hello":"world"}',
    )

    report = verify_raw_corpus(db_path=db_path, providers=["inbox"], max_samples=16)
    stats = report.providers["inbox"]

    assert report.total_records == 1
    assert stats.total_records == 1
    assert stats.skipped_no_schema == 1
    assert stats.valid_records == 0
    assert stats.invalid_records == 0


@pytest.mark.slow
def test_verify_raw_corpus_uses_persisted_payload_provider_for_filters(db_path, monkeypatch):
    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload, max_samples=16):
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample):
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.verification.SchemaValidator.for_provider",
        lambda _provider: _AlwaysValidValidator(),
    )

    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-generic-chatgpt",
        provider_name="inbox",
        payload_provider="chatgpt",
        source_name="inbox",
        source_path="/tmp/raw.json",
        raw_content=b'{"id":"one"}',
    )

    report = verify_raw_corpus(db_path=db_path, providers=["chatgpt"], max_samples=16)

    assert report.total_records == 1
    assert report.providers["chatgpt"].total_records == 1
    assert report.providers["chatgpt"].valid_records == 1


def test_verify_raw_corpus_counts_malformed_jsonl_as_decode_error(db_path):
    raw_id = "raw-codex-1"
    _insert_raw_record(
        db_path=db_path,
        raw_id=raw_id,
        provider_name="codex",
        source_name="codex",
        source_path="/tmp/session.jsonl",
        raw_content=(
            b'{"type":"session_meta"}\n'
            b'not json at all\n'
            b'{"type":"response_item","payload":{"type":"message"}}'
        ),
    )

    report = verify_raw_corpus(db_path=db_path, providers=["codex"], max_samples=16)
    stats = report.providers["codex"]

    assert report.total_records == 1
    assert stats.total_records == 1
    assert stats.decode_errors == 1
    assert stats.valid_records == 0
    assert stats.invalid_records == 0
    assert stats.quarantined_records == 0

    with open_connection(db_path) as conn:
        row = conn.execute(
            "SELECT validation_status, validation_error, validation_mode, parse_error "
            "FROM raw_conversations WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None
    assert row[3] is None


def test_verify_raw_corpus_quarantine_malformed_updates_validation_state(db_path):
    raw_id = "raw-codex-q1"
    _insert_raw_record(
        db_path=db_path,
        raw_id=raw_id,
        provider_name="codex",
        source_name="codex",
        source_path="/tmp/session-q1.jsonl",
        raw_content=(
            b'{"type":"session_meta"}\n'
            b'not json at all\n'
            b'{"type":"response_item","payload":{"type":"message"}}'
        ),
    )

    report = verify_raw_corpus(
        db_path=db_path,
        providers=["codex"],
        max_samples=16,
        quarantine_malformed=True,
    )
    stats = report.providers["codex"]

    assert report.total_records == 1
    assert stats.decode_errors == 1
    assert stats.quarantined_records == 1

    with open_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT validation_status, validation_error, validation_mode, validation_provider,
                   payload_provider,
                   validated_at, parse_error
            FROM raw_conversations
            WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "failed"
    assert isinstance(row[1], str) and "Malformed JSONL lines" in row[1]
    assert row[2] == "strict"
    assert row[3] == "codex"
    assert row[4] == "codex"
    assert row[5] is not None
    assert isinstance(row[6], str) and "Malformed JSONL lines" in row[6]


def test_verify_raw_corpus_honors_record_limit_and_offset(db_path, monkeypatch):
    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload, max_samples=16):
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample):
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.verification.SchemaValidator.for_provider",
        lambda _provider: _AlwaysValidValidator(),
    )

    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-1",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt-1.json",
        raw_content=b'{"id":"one"}',
    )
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-2",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt-2.json",
        raw_content=b'{"id":"two"}',
    )

    report = verify_raw_corpus(
        db_path=db_path,
        providers=["chatgpt"],
        max_samples=16,
        record_limit=1,
        record_offset=1,
    )
    stats = report.providers["chatgpt"]

    assert report.total_records == 1
    assert report.record_limit == 1
    assert report.record_offset == 1
    assert stats.total_records == 1
    assert stats.valid_records == 1
