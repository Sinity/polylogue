"""Tests for unified schema extraction and validator modules.

Migrated from test_filters.py during test consolidation — these tests
cover polylogue.schemas.unified and polylogue.schemas.validator, not
filter logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias
from unittest.mock import patch

import pytest

from polylogue.schemas.json_types import JSONDocument
from polylogue.schemas.unified import (
    HarmonizedMessage,
    bulk_harmonize,
    extract_chatgpt_text,
    extract_claude_code_text,
    extract_content_blocks,
    extract_harmonized_message,
    extract_reasoning_traces,
    extract_token_usage,
    harmonize_parsed_message,
    is_message_record,
)
from polylogue.schemas.validator import (
    SchemaValidator,
    ValidationResult,
    validate_provider_export,
)

# =============================================================================
# UNIFIED.PY TESTS
# =============================================================================


ProviderName: TypeAlias = Literal["claude-code", "claude-ai", "chatgpt", "gemini", "codex"]
ProviderPayload: TypeAlias = dict[str, Any]
ProviderContent: TypeAlias = Any
UsagePayload: TypeAlias = dict[str, Any] | None
ExtractTextFn: TypeAlias = Callable[[Any], str]
ReasoningTraceCase: TypeAlias = tuple[ProviderContent, str, int, str | None, str]
ContentBlocksCase: TypeAlias = tuple[ProviderContent, int, list[str], str]
ExtractTextCase: TypeAlias = tuple[ExtractTextFn, ProviderContent | UsagePayload, str, str]
MessageRecordTypeCase: TypeAlias = tuple[str, str, bool, str]
TokenUsageExpectation: TypeAlias = Literal["partial"] | None
TokenUsageCase: TypeAlias = tuple[UsagePayload, TokenUsageExpectation, str]


REASONING_TRACES_CASES: list[ReasoningTraceCase] = [
    (None, "claude-ai", 0, None, "empty_none_content"),
    (["string", 123], "claude-ai", 0, None, "non_dict_block"),
    ([{"type": "thinking", "thinking": "Let me think..."}], "claude-ai", 1, "Let me think...", "thinking_block"),
    ([{"isThought": True, "text": "Gemini thinking"}], "gemini", 1, "Gemini thinking", "gemini_thought"),
    ([{"type": "thinking", "text": "Fallback text"}], "claude-ai", 1, "Fallback text", "thinking_fallback"),
]

CONTENT_BLOCKS_CASES: list[ContentBlocksCase] = [
    (None, 0, [], "empty_none"),
    (["string", 123, None], 0, [], "non_dict_items"),
    ([{"type": "text", "text": "Hello"}], 1, ["text"], "text_block"),
    ([{"type": "thinking", "thinking": "Thought"}], 1, ["thinking"], "thinking_block"),
    (
        [{"type": "tool_use", "name": "bash", "id": "tool1", "input": {"command": "ls"}}],
        1,
        ["tool_use"],
        "tool_use_block",
    ),
    ([{"type": "tool_result", "content": "result data"}], 1, ["tool_result"], "tool_result_block"),
    ([{"type": "tool_result"}], 1, ["tool_result"], "tool_result_no_content"),
    ([{"type": "code", "text": "print('hello')", "language": "python"}], 1, ["code"], "code_block_text"),
    ([{"type": "code", "code": "def test(): pass"}], 1, ["code"], "code_block_code"),
    ([{"type": "unknown", "data": "something"}], 0, [], "unknown_block_type"),
]

EXTRACT_TEXT_CASES: list[ExtractTextCase] = [
    (extract_claude_code_text, None, "", "claude_code_none"),
    (extract_claude_code_text, ["string", 123], "", "claude_code_non_dict"),
    (extract_chatgpt_text, None, "", "chatgpt_none"),
    (extract_chatgpt_text, {}, "", "chatgpt_no_parts"),
    (extract_chatgpt_text, {"parts": "string"}, "string", "chatgpt_parts_as_string"),
    (extract_chatgpt_text, {"parts": [123, "text", {"key": "val"}]}, "text", "chatgpt_non_string_parts"),
]

HARMONIZED_MESSAGE_PROVIDER_CASES: list[tuple[ProviderName, str, str]] = [
    ("claude-code", "claude_code_msg", "Claude Code extraction"),
    ("claude-ai", "claude_ai_msg", "Claude AI extraction"),
    ("chatgpt", "chatgpt_msg", "ChatGPT extraction"),
    ("gemini", "gemini_msg", "Gemini extraction"),
    ("codex", "codex_msg", "Codex extraction"),
]

MESSAGE_RECORD_TYPE_CASES: list[MessageRecordTypeCase] = [
    ("claude-code", "user", True, "Claude Code user"),
    ("claude-code", "assistant", True, "Claude Code assistant"),
    ("claude-code", "metadata", False, "Claude Code metadata"),
    ("chatgpt", "anything", True, "Other providers always True"),
]

TOKEN_USAGE_CASES: list[TokenUsageCase] = [
    (None, None, "None usage"),
    ({}, None, "Empty dict (no tokens)"),
    ({"input_tokens": 100}, "partial", "Partial token fields"),
]


class TestUnifiedMissingRole:
    def test_missing_role_raises_error(self) -> None:
        from polylogue.schemas.unified import _missing_role

        with pytest.raises(ValueError, match="Message has no role"):
            _missing_role()


class TestUnifiedExtractReasoningTraces:
    @pytest.mark.parametrize("content,provider,expected_len,expected_text,description", REASONING_TRACES_CASES)
    def test_extract_reasoning_traces(
        self,
        content: ProviderContent,
        provider: str,
        expected_len: int,
        expected_text: str | None,
        description: str,
    ) -> None:
        result = extract_reasoning_traces(content, provider)
        assert len(result) == expected_len
        if expected_text is not None:
            assert result[0].text == expected_text


class TestUnifiedExtractContentBlocks:
    @pytest.mark.parametrize("content,expected_len,expected_types,description", CONTENT_BLOCKS_CASES)
    def test_extract_content_blocks(
        self,
        content: ProviderContent,
        expected_len: int,
        expected_types: list[str],
        description: str,
    ) -> None:
        result = extract_content_blocks(content)
        assert len(result) == expected_len
        if expected_types:
            for block, expected_type in zip(result, expected_types, strict=True):
                assert block.type.value == expected_type


class TestUnifiedExtractTokenUsage:
    @pytest.mark.parametrize("usage,expected_type,description", TOKEN_USAGE_CASES)
    def test_extract_token_usage(
        self,
        usage: UsagePayload,
        expected_type: TokenUsageExpectation,
        description: str,
    ) -> None:
        result = extract_token_usage(usage)
        if expected_type is None:
            assert result is None
        elif expected_type == "partial":
            assert result is not None
            assert result.input_tokens == 100


class TestUnifiedExtractTextHelpers:
    @pytest.mark.parametrize("extract_fn,content,expected,description", EXTRACT_TEXT_CASES)
    def test_extract_text(
        self,
        extract_fn: ExtractTextFn,
        content: ProviderContent | UsagePayload,
        expected: str,
        description: str,
    ) -> None:
        result = extract_fn(content)
        assert result == expected or expected in result


class TestUnifiedExtractHarmonizedMessage:
    def test_extract_harmonized_message_invalid_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            extract_harmonized_message("unknown_provider", {})

    @pytest.mark.parametrize("provider,msg_key,description", HARMONIZED_MESSAGE_PROVIDER_CASES)
    def test_extract_harmonized_message_by_provider(
        self,
        provider: ProviderName,
        msg_key: str,
        description: str,
    ) -> None:
        raw: JSONDocument
        if provider == "claude-code":
            raw = {
                "uuid": "msg1",
                "timestamp": "2024-01-01T00:00:00Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "Hello"}], "model": "claude-ai"},
            }
        elif provider == "claude-ai":
            raw = {"uuid": "msg1", "sender": "user", "text": "Hello", "created_at": "2024-01-01T00:00:00Z"}
        elif provider == "chatgpt":
            raw = {"id": "msg1", "author": {"role": "user"}, "content": {"parts": ["Hello"]}, "create_time": 1704067200}
        elif provider == "gemini":
            raw = {"role": "user", "text": "Hello"}
        elif provider == "codex":
            raw = {"id": "msg1", "role": "user", "content": [{"text": "Hello"}], "timestamp": "2024-01-01T00:00:00Z"}
        result = extract_harmonized_message(provider, raw)
        assert isinstance(result, HarmonizedMessage)


class TestUnifiedHarmonizeParsedMessage:
    def test_harmonize_parsed_message_none_meta(self) -> None:
        assert harmonize_parsed_message("claude-ai", None) is None

    def test_harmonize_parsed_message_not_message_record(self) -> None:
        assert harmonize_parsed_message("claude-code", {"type": "metadata"}) is None

    def test_harmonize_parsed_message_valid(self) -> None:
        meta = {"raw": {"uuid": "msg1", "sender": "user", "text": "Hello"}}
        result = harmonize_parsed_message("claude-ai", meta)
        assert isinstance(result, HarmonizedMessage)


class TestUnifiedBulkHarmonize:
    def test_bulk_harmonize_no_provider_meta(self) -> None:
        class MockParsedMessage:
            provider_meta: JSONDocument | None = None

            pass

        result = bulk_harmonize("claude-ai", [MockParsedMessage()])
        assert result == []

    def test_bulk_harmonize_mixed_valid_invalid(self) -> None:
        class MockParsedMessage:
            def __init__(self, meta: JSONDocument | None = None) -> None:
                self.provider_meta = meta

        messages = [
            MockParsedMessage({"raw": {"type": "metadata"}}),
            MockParsedMessage(
                {
                    "raw": {
                        "type": "user",
                        "uuid": "1",
                        "message": {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                    }
                }
            ),
        ]
        result = bulk_harmonize("claude-code", messages)
        assert len(result) == 1


class TestUnifiedIsMessageRecord:
    @pytest.mark.parametrize("provider,record_type,expected,description", MESSAGE_RECORD_TYPE_CASES)
    def test_is_message_record(
        self,
        provider: str,
        record_type: str,
        expected: bool,
        description: str,
    ) -> None:
        result = is_message_record(provider, {"type": record_type})
        assert result == expected


# =============================================================================
# VALIDATOR.PY TESTS
# =============================================================================


class TestValidatorImportErrorHandling:
    def test_validator_jsonschema_not_installed(self) -> None:
        with patch("polylogue.schemas.validator.jsonschema", None):
            with pytest.raises(ImportError, match="jsonschema not installed"):
                SchemaValidator({})


class TestValidatorAvailableProviders:
    def test_available_providers_returns_list(self) -> None:
        result = SchemaValidator.available_providers()
        assert isinstance(result, list)


class TestValidatorDetectDrift:
    def test_validate_detects_unexpected_field(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False}
        validator = SchemaValidator(schema, strict=True)
        result = validator.validate({"name": "test", "extra": "field"})
        assert result.has_drift
        assert any("Unexpected field" in w for w in result.drift_warnings)

    def test_validate_additional_properties_true(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": True}
        validator = SchemaValidator(schema, strict=True)
        result = validator.validate({"name": "test", "extra": "field"})
        assert not result.has_drift

    def test_validate_additional_properties_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "string"},
        }
        validator = SchemaValidator(schema, strict=True)
        validator.validate({"name": "test", "extra": "value"})

    def test_validate_nested_object_drift(self) -> None:
        schema = {
            "type": "object",
            "properties": {"user": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
        validator = SchemaValidator(schema, strict=True)
        data = {"user": {"name": "test", "extra": "field"}}
        result = validator.validate(data)
        assert isinstance(result, ValidationResult)

    def test_validate_list_items_drift(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}}
            },
        }
        validator = SchemaValidator(schema, strict=True)
        data = {"items": [{"id": 1, "extra": "field"}]}
        result = validator.validate(data)
        assert isinstance(result, ValidationResult)


class TestValidatorFormatError:
    def test_validate_multiple_errors(self) -> None:
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        validator = SchemaValidator(schema, strict=False)
        result = validator.validate({"age": "not an integer"})
        assert len(result.errors) >= 1


class TestValidatorConvenienceFunction:
    def test_validate_provider_export_raises_on_missing_schema(self) -> None:
        with pytest.raises(FileNotFoundError):
            validate_provider_export({}, "invalid_provider", strict=True)


class TestValidationResult:
    def test_validation_result_has_drift_property(self) -> None:
        result = ValidationResult(is_valid=True, drift_warnings=["Field X is new"])
        assert result.has_drift is True

    def test_validation_result_no_drift(self) -> None:
        result = ValidationResult(is_valid=True)
        assert result.has_drift is False

    def test_validation_result_raise_if_invalid(self) -> None:
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        with pytest.raises(ValueError, match="Schema validation failed"):
            result.raise_if_invalid()

    def test_validation_result_raise_if_valid(self) -> None:
        result = ValidationResult(is_valid=True)
        result.raise_if_invalid()
