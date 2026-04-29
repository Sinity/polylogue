"""Focused validator and validation-contract tests for schema handling, plus raw corpus verification."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from unittest.mock import patch

import pytest

from polylogue.lib.json import json_document
from polylogue.scenarios import CorpusSpec
from polylogue.schemas import ValidationResult
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.validation.corpus import verify_raw_corpus
from polylogue.schemas.validation.requests import SchemaVerificationRequest
from polylogue.schemas.validator import SchemaValidator, validate_provider_export
from polylogue.storage.backends.connection import open_connection
from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.config import Source


class SyntheticSourceFactory(Protocol):
    def __call__(
        self,
        provider: str,
        count: int = ...,
        messages_per_conversation: range = ...,
        seed: int = ...,
    ) -> Source: ...


def _patch_validator_registry(mock_schema_dir: Path) -> AbstractContextManager[object]:
    return patch(
        "polylogue.schemas.validator.SchemaRegistry",
        partial(SchemaRegistry, storage_root=mock_schema_dir),
    )


def test_schema_validator_loads_provider(mock_schema_dir: Path) -> None:
    """SchemaValidator.for_provider loads the requested schema."""
    with _patch_validator_registry(mock_schema_dir):
        SchemaValidator._cache.clear()
        validator = SchemaValidator.for_provider("chatgpt")
        assert validator.schema["required"] == ["id"]

        with pytest.raises(FileNotFoundError):
            SchemaValidator.for_provider("nonexistent")


def test_validate_valid_data(mock_schema_dir: Path) -> None:
    """Valid payloads should pass with no errors or drift."""
    with _patch_validator_registry(mock_schema_dir):
        validator = SchemaValidator.for_provider("chatgpt")
        result = validator.validate({"id": "123", "count": 10, "meta": {"source": "test"}})

    assert result.is_valid
    assert not result.errors
    assert not result.drift_warnings


def test_validate_detects_errors(mock_schema_dir: Path) -> None:
    """Validation should report required-field and type errors."""
    with _patch_validator_registry(mock_schema_dir):
        validator = SchemaValidator.for_provider("chatgpt")

        missing = validator.validate({"count": 10})
        wrong_type = validator.validate({"id": 123})

    assert not missing.is_valid
    assert any("id" in error for error in missing.errors)
    assert not wrong_type.is_valid
    assert any("123" in error for error in wrong_type.errors)


def test_validate_detects_drift(mock_schema_dir: Path) -> None:
    """Strict mode should surface unexpected fields as drift."""
    with _patch_validator_registry(mock_schema_dir):
        validator = SchemaValidator.for_provider("codex", strict=True)
        result = validator.validate({"id": "123", "extra": "drift"})

    assert result.is_valid
    assert result.has_drift
    assert "extra" in result.drift_warnings[0]


def test_schema_validator_loads_canonical_claude_ai_provider(mock_schema_dir: Path) -> None:
    """Schema validation should use the canonical Claude AI provider token."""
    schema = {
        "type": "object",
        "properties": {"uuid": {"type": "string"}},
        "required": ["uuid"],
        "additionalProperties": False,
    }
    SchemaRegistry(storage_root=mock_schema_dir).write_schema_version("claude-ai", "v1", schema)

    with _patch_validator_registry(mock_schema_dir):
        SchemaValidator._cache.clear()
        validator = SchemaValidator.for_provider("claude-ai")

    assert validator.provider == "claude-ai"
    assert "uuid" in json_document(validator.schema["properties"])


def test_schema_validator_accepts_provider_enum(mock_schema_dir: Path) -> None:
    """Known provider enums should flow through validator lookup without string routing."""
    schema = {
        "type": "object",
        "properties": {"uuid": {"type": "string"}},
        "required": ["uuid"],
        "additionalProperties": False,
    }
    SchemaRegistry(storage_root=mock_schema_dir).write_schema_version("claude-ai", "v1", schema)

    with _patch_validator_registry(mock_schema_dir):
        SchemaValidator._cache.clear()
        validator = SchemaValidator.for_provider(Provider.CLAUDE_AI)

    assert validator.provider is Provider.CLAUDE_AI


def test_validation_samples_record_mode_skips_non_record_documents() -> None:
    """Record-mode validation should ignore top-level document payloads."""
    validator = SchemaValidator(
        {
            "type": "object",
            "x-polylogue-sample-granularity": "record",
            "properties": {"type": {"type": "string"}},
        },
        provider=Provider.CLAUDE_CODE,
    )

    assert validator.validation_samples({"version": 1, "entries": []}, max_samples=16) == []


def test_schema_validator_prefers_registry_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Latest registry schemas should override packaged fallback files."""
    fake_schema = {
        "type": "object",
        "properties": {"from_registry": {"type": "string"}},
        "additionalProperties": False,
    }

    class _FakeRegistry:
        def get_schema(self, provider: str, version: str = "latest") -> dict[str, object] | None:
            if provider == "chatgpt" and version == "latest":
                return fake_schema
            return None

    monkeypatch.setattr("polylogue.schemas.validator.SchemaRegistry", _FakeRegistry)
    SchemaValidator._cache.clear()

    validator = SchemaValidator.for_provider("chatgpt")
    assert "from_registry" in json_document(validator.schema["properties"])


def test_dynamic_key_maps_do_not_emit_drift_warnings() -> None:
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


def test_dynamic_identifier_keys_are_suppressed_in_additional_properties_maps() -> None:
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

    result = validator.validate({"mapping": {"msg-550e8400-e29b-41d4-a716-446655440000": {"role": "assistant"}}})
    assert result.is_valid
    assert not result.has_drift


def test_additional_properties_schema_still_detects_nested_drift() -> None:
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


def test_validate_non_strict_mode_skips_drift_detection() -> None:
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


def test_format_error_includes_nested_array_path() -> None:
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


def test_looks_dynamic_key_matches_expected_identifier_patterns() -> None:
    """Dynamic-key heuristic should match supported identifier families only."""
    validator = SchemaValidator({"type": "object"})

    assert validator._looks_dynamic_key("550e8400-e29b-41d4-a716-446655440000")
    assert validator._looks_dynamic_key("abcdef0123456789abcdef01")
    assert validator._looks_dynamic_key("msg-550e8400-e29b-41d4-a716-446655440000")
    assert not validator._looks_dynamic_key("title")
    assert not validator._looks_dynamic_key("message_text")


def test_available_providers(mock_schema_dir: Path) -> None:
    """available_providers should reflect packaged schema files."""
    with _patch_validator_registry(mock_schema_dir):
        providers = SchemaValidator.available_providers()

    assert "chatgpt" in providers
    assert "codex" in providers
    assert "nonexistent" not in providers


def test_validate_helper(mock_schema_dir: Path) -> None:
    """validate_provider_export should delegate to SchemaValidator cleanly."""
    with _patch_validator_registry(mock_schema_dir):
        result = validate_provider_export({"id": "123"}, "chatgpt")

    assert result.is_valid


def test_missing_provider_raises() -> None:
    """Unknown providers should fail with FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No schema found"):
        SchemaValidator.for_provider("nonexistent-provider")


@pytest.mark.parametrize(
    ("source_path", "sample"),
    [
        (
            "/tmp/codex-session.jsonl",
            {
                "timestamp": "2026-04-03T17:33:23.644Z",
                "type": "event_msg",
                "payload": {
                    "type": "collab_close_end",
                    "status": {"completed": "done"},
                },
            },
        ),
        (
            "/tmp/codex-session.jsonl",
            {
                "timestamp": "2026-03-12T18:37:50.573Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_GM8XncLg7VVQt5ylQrka6TNy",
                    "output": [{"type": "input_image", "image_url": "data:image/jpeg;base64,abc"}],
                },
            },
        ),
        (
            "/tmp/codex-session.jsonl",
            {
                "timestamp": "2026-03-05T05:33:13.392Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "rate_limits": {
                        "limit_id": "codex",
                        "limit_name": None,
                        "primary": None,
                        "secondary": None,
                        "credits": {"has_credits": False, "unlimited": False, "balance": None},
                        "plan_type": None,
                    },
                },
            },
        ),
        (
            "/tmp/codex-session.jsonl",
            {
                "timestamp": "2026-03-13T12:02:43.171Z",
                "type": "session_meta",
                "payload": {
                    "id": "019ce713-cf30-7aa1-832b-cd953aa7c7ec",
                    "source": {"subagent": "review"},
                },
            },
        ),
        (
            "/tmp/codex-session.jsonl",
            {
                "timestamp": "2026-03-13T12:02:43.171Z",
                "type": "session_meta",
                "payload": {
                    "id": "019ce713-cf30-7aa1-832b-cd953aa7c7ec",
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": "019ce441-45ad-7df2-8d64-fcf9db69002f",
                                "depth": 1,
                                "agent_nickname": "Cicero",
                                "agent_role": None,
                            }
                        }
                    },
                },
            },
        ),
        (
            "/tmp/codex-session.jsonl",
            {
                "timestamp": "2026-03-17T17:47:30.487Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "Could not fetch the URL.",
                    "phase": None,
                },
            },
        ),
    ],
)
def test_live_observed_codex_records_validate(source_path: str, sample: dict[str, object]) -> None:
    """Bundled Codex schema should accept live record shapes already seen in the archive."""
    SchemaValidator._cache.clear()
    validator = SchemaValidator.for_payload("codex", [sample], source_path=source_path)

    result = validator.validate(sample, include_drift=False)

    assert result.is_valid, result.errors


@pytest.mark.parametrize(
    ("source_path", "sample"),
    [
        (
            "/tmp/claude-code-session.jsonl",
            {
                "type": "assistant",
                "message": {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01BYSXJLmQyWsjPU1MQyUiLr",
                            "name": "SendMessage",
                            "input": {
                                "to": "task3-exports-daily",
                                "message": {
                                    "type": "shutdown_request",
                                    "reason": "Task complete, no further work needed",
                                },
                                "type": "shutdown_request",
                                "recipient": "task3-exports-daily",
                                "content": "Task complete, no further work needed",
                            },
                            "caller": {"type": "direct"},
                        }
                    ],
                },
            },
        ),
        (
            "/tmp/claude-code-session.jsonl",
            {
                "type": "progress",
                "data": {
                    "message": {
                        "type": "assistant",
                        "timestamp": "2026-03-27T09:52:50.461Z",
                        "message": {
                            "model": "claude-opus-4-6",
                            "id": "msg_019P6avx5DTbMwX9JkUoxWuJ",
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_015is5FD3Gr43kLnSX2XptrN",
                                    "name": "Edit",
                                    "input": {
                                        "replace_all": False,
                                        "file_path": "/realm/project/polylogue/polylogue/storage/backends/schema_ddl.py",
                                        "old_string": "old",
                                        "new_string": "new",
                                    },
                                    "caller": {"type": "direct"},
                                }
                            ],
                            "stop_reason": "tool_use",
                            "stop_sequence": None,
                            "context_management": {"applied_edits": []},
                        },
                        "requestId": "req_011CZTNyrgNf3M3mS3JoGxXZ",
                        "uuid": "13a14d24-56fc-4655-931a-000000000001",
                    },
                    "type": "agent_progress",
                    "prompt": "",
                    "agentId": "aed179a8ea9073c55",
                },
            },
        ),
    ],
)
def test_live_observed_claude_code_records_validate(source_path: str, sample: dict[str, object]) -> None:
    """Bundled Claude Code schema should accept live record shapes already seen in the archive."""
    SchemaValidator._cache.clear()
    validator = SchemaValidator.for_payload("claude-code", [sample], source_path=source_path)

    result = validator.validate(sample, include_drift=False)

    assert result.is_valid, result.errors


class TestSyntheticRoundTrip:
    """Synthetic data should remain parser-compatible for supported providers."""

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex", "gemini"])
    def test_synthetic_parses_successfully(
        self,
        provider: str,
        synthetic_source: SyntheticSourceFactory,
    ) -> None:
        from polylogue.sources import iter_source_conversations

        source = synthetic_source(provider, count=3, seed=42)
        conversations = list(iter_source_conversations(source))

        assert conversations, f"No conversations parsed for {provider}"
        for conversation in conversations:
            assert conversation.messages, f"Empty conversation for {provider}"
            assert any(message.text for message in conversation.messages), f"No message text for {provider}"


def test_validation_result_properties() -> None:
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
    db_path: Path,
    raw_id: str,
    provider_name: str,
    payload_provider: str | None = None,
    source_name: str,
    source_path: str,
    raw_content: bytes,
) -> str:
    """Insert a raw record and return the actual raw_id (the blob hash)."""
    from polylogue.storage.blob_store import get_blob_store

    # Write content to blob store and get the actual hash
    blob_store = get_blob_store()
    actual_raw_id, blob_size = blob_store.write_from_bytes(raw_content)

    # Use the actual hash as raw_id (required for content-addressed blob store)
    # Ignore the passed raw_id parameter since it must match the blob hash
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_conversations (
                raw_id, provider_name, payload_provider, source_name, source_path, source_index,
                blob_size, acquired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actual_raw_id,
                provider_name,
                payload_provider,
                source_name,
                source_path,
                0,
                blob_size,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    return actual_raw_id


def test_verify_raw_corpus_reports_valid_synthetic_chatgpt(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload: object, max_samples: int = 16) -> list[object]:
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample: object) -> ValidationResult:
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.validation.corpus.SchemaValidator.for_payload",
        lambda *args, **kwargs: _AlwaysValidValidator(),
    )

    raw = SyntheticCorpus.generate_for_spec(
        CorpusSpec.for_provider(
            "chatgpt",
            count=1,
            seed=42,
            origin="generated.test-schema-validation",
            tags=("synthetic", "test", "schema-validation"),
        )
    )[0]
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-1",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt.json",
        raw_content=raw,
    )

    report = verify_raw_corpus(
        db_path=db_path,
        request=SchemaVerificationRequest(providers=["chatgpt"], max_samples=16),
    )
    stats = report.providers["chatgpt"]

    assert report.total_records == 1
    assert stats.total_records == 1
    assert stats.valid_records == 1
    assert stats.invalid_records == 0
    assert stats.decode_errors == 0


def test_verify_raw_corpus_counts_missing_schema_as_skipped(db_path: Path) -> None:
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-inbox-1",
        provider_name="inbox",
        source_name="inbox",
        source_path="/tmp/inbox.json",
        raw_content=b'{"hello":"world"}',
    )

    report = verify_raw_corpus(
        db_path=db_path,
        request=SchemaVerificationRequest(providers=["unknown"], max_samples=16),
    )
    stats = report.providers["unknown"]

    assert report.total_records == 1
    assert stats.total_records == 1
    assert stats.skipped_no_schema == 1
    assert stats.valid_records == 0
    assert stats.invalid_records == 0


@pytest.mark.slow
def test_verify_raw_corpus_uses_persisted_payload_provider_for_filters(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload: object, max_samples: int = 16) -> list[object]:
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample: object) -> ValidationResult:
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.validation.corpus.SchemaValidator.for_payload",
        lambda *args, **kwargs: _AlwaysValidValidator(),
    )

    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-generic-chatgpt",
        provider_name="inbox",
        payload_provider="chatgpt",
        source_name="inbox",
        source_path="/tmp/raw.json",
        raw_content=b'{"id":"one","mapping":{}}',
    )

    report = verify_raw_corpus(
        db_path=db_path,
        request=SchemaVerificationRequest(providers=["chatgpt"], max_samples=16),
    )

    assert report.total_records == 1
    assert report.providers["chatgpt"].total_records == 1
    assert report.providers["chatgpt"].valid_records == 1


def test_verify_raw_corpus_counts_malformed_jsonl_as_decode_error(db_path: Path) -> None:
    raw_id = _insert_raw_record(
        db_path=db_path,
        raw_id="raw-codex-1",
        provider_name="codex",
        source_name="codex",
        source_path="/tmp/session.jsonl",
        raw_content=(
            b'{"type":"session_meta"}\nnot json at all\n{"type":"response_item","payload":{"type":"message"}}'
        ),
    )

    report = verify_raw_corpus(
        db_path=db_path,
        request=SchemaVerificationRequest(providers=["codex"], max_samples=16),
    )
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


def test_verify_raw_corpus_quarantine_malformed_updates_validation_state(db_path: Path) -> None:
    raw_id = _insert_raw_record(
        db_path=db_path,
        raw_id="raw-codex-q1",
        provider_name="codex",
        source_name="codex",
        source_path="/tmp/session-q1.jsonl",
        raw_content=(
            b'{"type":"session_meta"}\nnot json at all\n{"type":"response_item","payload":{"type":"message"}}'
        ),
    )

    report = verify_raw_corpus(
        db_path=db_path,
        request=SchemaVerificationRequest(
            providers=["codex"],
            max_samples=16,
            quarantine_malformed=True,
        ),
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


def test_verify_raw_corpus_quarantine_empty_payload_updates_validation_state(db_path: Path) -> None:
    raw_id = _insert_raw_record(
        db_path=db_path,
        raw_id="raw-codex-empty",
        provider_name="codex",
        source_name="codex",
        source_path="/tmp/empty-session.jsonl",
        raw_content=b"",
    )

    report = verify_raw_corpus(
        db_path=db_path,
        request=SchemaVerificationRequest(
            providers=["codex"],
            max_samples=16,
            quarantine_malformed=True,
        ),
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
    assert isinstance(row[1], str) and "zero-length" in row[1]
    assert row[2] == "strict"
    assert row[3] == "codex"
    assert row[4] is None
    assert row[5] is not None
    assert isinstance(row[6], str) and "zero-length" in row[6]


def test_verify_raw_corpus_honors_record_limit_and_offset(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AlwaysValidValidator:
        provider = "chatgpt"

        def validation_samples(self, payload: object, max_samples: int = 16) -> list[object]:
            return [payload] if isinstance(payload, dict) else []

        def validate(self, _sample: object) -> ValidationResult:
            return ValidationResult(is_valid=True)

    monkeypatch.setattr(
        "polylogue.schemas.validation.corpus.SchemaValidator.for_payload",
        lambda *args, **kwargs: _AlwaysValidValidator(),
    )

    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-1",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt-1.json",
        raw_content=b'{"id":"one","mapping":{}}',
    )
    _insert_raw_record(
        db_path=db_path,
        raw_id="raw-chatgpt-2",
        provider_name="chatgpt",
        source_name="chatgpt",
        source_path="/tmp/chatgpt-2.json",
        raw_content=b'{"id":"two","mapping":{}}',
    )

    report = verify_raw_corpus(
        db_path=db_path,
        request=SchemaVerificationRequest(
            providers=["chatgpt"],
            max_samples=16,
            record_limit=1,
            record_offset=1,
        ),
    )
    stats = report.providers["chatgpt"]

    assert report.total_records == 1
    assert report.record_limit == 1
    assert report.record_offset == 1
    assert stats.total_records == 1
    assert stats.valid_records == 1
