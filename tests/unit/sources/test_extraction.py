"""Tests for unified extraction layer."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from polylogue.lib.provider_semantics import (
    extract_chatgpt_text,
    extract_codex_text,
)
from polylogue.lib.roles import normalize_role
from polylogue.lib.viewports import ContentType, ToolCategory
from polylogue.schemas.unified import (
    HarmonizedMessage,
    bulk_harmonize,
    extract_from_provider_meta,
    extract_harmonized_message,
    harmonize_parsed_message,
    is_message_record,
)

# =============================================================================
# Transform Function Tests
# =============================================================================


class TestUnifiedNormalizeRole:
    """Tests for role normalization."""

    def test_user_variants(self):
        assert normalize_role("user") == "user"
        assert normalize_role("human") == "user"
        assert normalize_role("USER") == "user"

    def test_assistant_variants(self):
        assert normalize_role("assistant") == "assistant"
        assert normalize_role("model") == "assistant"
        assert normalize_role("ai") == "assistant"

    def test_system(self):
        assert normalize_role("system") == "system"

    def test_tool(self):
        assert normalize_role("tool") == "tool"
        assert normalize_role("function") == "tool"

    def test_unknown_passthrough(self):
        assert normalize_role("custom_role") == "unknown"


# =============================================================================
# Claude Code Extraction Tests
# =============================================================================


class TestClaudeCodeExtraction:
    """Tests for Claude Code message extraction."""

    def test_extract_basic_message(self):
        raw = {
            "type": "assistant",
            "uuid": "abc-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "model": "claude-3-opus",
                "content": [
                    {"type": "text", "text": "Hello, how can I help?"}
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50
                }
            },
            "costUSD": 0.005
        }

        msg = extract_harmonized_message("claude-code", raw)

        assert isinstance(msg, HarmonizedMessage)
        assert msg.role == "assistant"
        assert msg.text == "Hello, how can I help?"
        assert msg.id == "abc-123"
        assert msg.model == "claude-3-opus"
        assert msg.provider == "claude-code"
        assert msg.cost is not None
        assert msg.cost.total_usd == 0.005

    def test_extract_with_tool_calls(self):
        raw = {
            "type": "assistant",
            "uuid": "def-456",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read that file."},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "Read",
                        "input": {"file_path": "/path/to/file.py"}
                    }
                ]
            }
        }

        msg = extract_harmonized_message("claude-code", raw)

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "Read"
        assert msg.tool_calls[0].category == ToolCategory.FILE_READ
        assert msg.tool_calls[0].affected_paths == ["/path/to/file.py"]
        assert msg.has_tool_use is True

    def test_extract_with_thinking(self):
        raw = {
            "type": "assistant",
            "uuid": "ghi-789",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this..."},
                    {"type": "text", "text": "Here's my conclusion."}
                ]
            }
        }

        msg = extract_harmonized_message("claude-code", raw)

        assert len(msg.reasoning_traces) == 1
        assert msg.reasoning_traces[0].text == "Let me analyze this..."
        assert msg.has_reasoning is True
        # Thinking is surfaced via reasoning_traces, not mixed into text
        assert msg.text == "Here's my conclusion."
        assert "Let me analyze this..." not in msg.text

    def test_content_blocks_extraction(self):
        raw = {
            "type": "assistant",
            "uuid": "jkl-012",
            "timestamp": "2024-01-15T10:30:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Thinking..."},
                    {"type": "text", "text": "Response text"},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}
                ]
            }
        }

        msg = extract_harmonized_message("claude-code", raw)

        assert len(msg.content_blocks) == 3
        assert msg.content_blocks[0].type == ContentType.THINKING
        assert msg.content_blocks[1].type == ContentType.TEXT
        assert msg.content_blocks[2].type == ContentType.TOOL_USE

    def test_is_message_record(self):
        assert is_message_record("claude-code", {"type": "user"}) is True
        assert is_message_record("claude-code", {"type": "assistant"}) is True
        assert is_message_record("claude-code", {"type": "system"}) is True
        assert is_message_record("claude-code", {"type": "progress"}) is False
        assert is_message_record("claude-code", {"type": "file-history-snapshot"}) is False


class TestHarmonizeParsedMessage:
    """Direct contracts for ParsedMessage -> HarmonizedMessage bridging."""

    def test_returns_none_for_missing_provider_meta(self):
        assert harmonize_parsed_message("claude-ai", None) is None

    def test_skips_non_message_claude_code_records(self):
        assert harmonize_parsed_message("claude-code", {"type": "progress"}) is None

    def test_overlays_database_fields_for_raw_payloads(self):
        provider_meta = {
            "raw": {
                "sender": "human",
                "text": "",
            }
        }

        msg = harmonize_parsed_message(
            "claude-ai",
            provider_meta,
            message_id="db-id",
            role="user",
            text="db text",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert msg is not None
        assert msg.id == "db-id"
        assert msg.role == "user"
        assert msg.text == "db text"
        assert msg.timestamp is not None
        assert msg.provider == "claude"

    def test_reuses_extracted_viewports_when_raw_is_absent(self):
        provider_meta = {
            "content_blocks": [
                {"type": "text", "text": "Assistant answer"},
                {"type": "thinking", "text": "Internal trace"},
            ],
            "reasoning_traces": [],
            "tool_calls": [],
        }

        msg = harmonize_parsed_message(
            "claude-code",
            provider_meta,
            message_id="view-id",
            role="assistant",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert msg is not None
        assert msg.id == "view-id"
        assert msg.role == "assistant"
        assert msg.text == "Assistant answer\nInternal trace"
        assert msg.timestamp is not None
        assert [trace.text for trace in msg.reasoning_traces] == ["Internal trace"]

    def test_falls_back_to_provider_meta_harmonization_for_malformed_payloads(self):
        provider_meta = {
            "sender": "assistant",
            "text": "Fallback text",
            "created_at": "2024-01-15T10:30:00Z",
        }

        msg = harmonize_parsed_message("claude-ai", provider_meta)

        assert msg is not None
        assert msg.role == "assistant"
        assert msg.text == "Fallback text"
        assert msg.timestamp is not None


class TestBulkHarmonize:
    """Contracts for bulk ParsedMessage harmonization."""

    def test_skips_entries_without_provider_meta_or_non_messages(self):
        parsed_messages = [
            SimpleNamespace(provider_meta=None),
            SimpleNamespace(provider_meta={"type": "progress"}),
            SimpleNamespace(
                provider_meta={"sender": "assistant", "text": "Kept"},
                provider_message_id="kept-id",
                role="assistant",
                text="Kept",
                timestamp="2024-01-15T10:30:00Z",
            ),
        ]

        result = bulk_harmonize("claude-code", parsed_messages)

        assert len(result) == 1
        assert result[0].id == "kept-id"
        assert result[0].role == "assistant"
        assert result[0].text == "Kept"

    def test_passes_database_context_through_to_harmonized_messages(self):
        parsed_messages = [
            SimpleNamespace(
                provider_meta={"sender": "human", "text": ""},
                provider_message_id="db-id",
                role="user",
                text="db text",
                timestamp="2024-01-15T10:30:00Z",
            )
        ]

        result = bulk_harmonize("claude-ai", parsed_messages)

        assert len(result) == 1
        assert result[0].id == "db-id"
        assert result[0].role == "user"
        assert result[0].text == "db text"
        assert result[0].timestamp is not None


# =============================================================================
# ChatGPT Extraction Tests
# =============================================================================


class TestChatGPTExtraction:
    """Tests for ChatGPT extraction."""

    def test_extract_message(self):
        raw = {
            "id": "chatgpt-123",
            "author": {"role": "assistant"},
            "content": {
                "content_type": "text",
                "parts": ["Here's your answer."]
            },
            "create_time": 1705318200,
            "metadata": {"model_slug": "gpt-4"}
        }

        msg = extract_harmonized_message("chatgpt", raw)

        assert msg.role == "assistant"
        assert msg.text == "Here's your answer."
        assert msg.id == "chatgpt-123"
        assert msg.model == "gpt-4"
        assert msg.provider == "chatgpt"


class TestClaudeAIFallbackExtraction:
    """Tests for malformed Claude AI payloads that require fallback extraction."""

    def test_extract_harmonized_message_falls_back_for_malformed_raw(self):
        raw = {
            "sender": "human",
            "text": "Fallback Claude AI text",
            "created_at": "2024-01-15T10:30:00Z",
        }

        msg = extract_harmonized_message("claude-ai", raw)

        assert msg.id is None
        assert msg.role == "user"
        assert msg.text == "Fallback Claude AI text"
        assert msg.timestamp is not None
        assert msg.provider == "claude"

    def test_extract_from_provider_meta_overlays_db_fields_for_malformed_claude_ai(self):
        provider_meta = {
            "sender": "human",
            "text": "",
        }

        msg = extract_from_provider_meta(
            "claude-ai",
            provider_meta,
            message_id="db-message-id",
            role="user",
            text="DB fallback text",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert msg.id == "db-message-id"
        assert msg.role == "user"
        assert msg.text == "DB fallback text"
        assert msg.timestamp is not None
        assert msg.provider == "claude"


# =============================================================================
# Gemini Extraction Tests
# =============================================================================


class TestGeminiExtraction:
    """Tests for Gemini extraction."""

    def test_extract_message(self):
        raw = {
            "role": "model",
            "text": "I can help with that.",
            "tokenCount": 25
        }

        msg = extract_harmonized_message("gemini", raw)

        assert msg.role == "assistant"
        assert msg.text == "I can help with that."
        assert msg.tokens is not None
        assert msg.tokens.output_tokens == 25

    def test_extract_thinking(self):
        raw = {
            "role": "model",
            "text": "Thinking about this...",
            "isThought": True,
            "thinkingBudget": 1000
        }

        msg = extract_harmonized_message("gemini", raw)

        assert msg.has_reasoning is True
        assert len(msg.reasoning_traces) == 1
        assert msg.reasoning_traces[0].token_count == 1000


class TestGeminiFallbackExtraction:
    """Tests for malformed Gemini payloads that require fallback extraction."""

    def test_extract_harmonized_message_falls_back_for_malformed_raw(self):
        raw = {
            "role": "model",
            "text": "Fallback Gemini text",
            "tokenCount": {"invalid": True},
            "isThought": True,
            "thinkingBudget": 256,
        }

        msg = extract_harmonized_message("gemini", raw)

        assert msg.id is None
        assert msg.role == "assistant"
        assert msg.text == "Fallback Gemini text"
        assert msg.timestamp is None
        assert msg.provider == "gemini"
        assert len(msg.reasoning_traces) == 1
        assert msg.reasoning_traces[0].text == "Fallback Gemini text"
        assert msg.reasoning_traces[0].token_count == 256
        assert msg.tokens is None

    def test_extract_from_provider_meta_overlays_db_fields_for_malformed_gemini(self):
        provider_meta = {
            "role": "model",
            "text": "",
            "tokenCount": {"invalid": True},
        }

        msg = extract_from_provider_meta(
            "gemini",
            provider_meta,
            message_id="db-gemini-id",
            role="assistant",
            text="DB Gemini fallback text",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert msg.id == "db-gemini-id"
        assert msg.role == "assistant"
        assert msg.text == "DB Gemini fallback text"
        assert msg.timestamp is not None
        assert msg.provider == "gemini"


# =============================================================================
# Codex Extraction Tests
# =============================================================================


class TestCodexExtraction:
    """Tests for Codex extraction."""

    def test_extract_direct_format(self):
        raw = {
            "id": "codex-123",
            "role": "user",
            "timestamp": "2024-01-15T10:30:00Z",
            "content": [
                {"type": "input_text", "text": "Hello from Codex"}
            ]
        }

        msg = extract_harmonized_message("codex", raw)

        assert msg.role == "user"
        assert msg.text == "Hello from Codex"
        assert msg.id == "codex-123"

    def test_extract_envelope_format(self):
        raw = {
            "type": "response_item",
            "payload": {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Response from Codex"}
                ]
            }
        }

        msg = extract_harmonized_message("codex", raw)

        assert msg.role == "assistant"
        assert msg.text == "Response from Codex"

    def test_extract_harmonized_message_falls_back_for_malformed_envelope(self):
        raw = {
            "id": "codex-fallback",
            "timestamp": "2024-01-15T10:30:00Z",
            "payload": "not-a-dict",
            "content": [
                {"type": "input_text", "text": "Fallback Codex text"},
            ],
        }

        msg = extract_harmonized_message("codex", raw)

        assert msg.id == "codex-fallback"
        assert msg.role == "unknown"
        assert msg.text == "Fallback Codex text"
        assert msg.timestamp is not None
        assert msg.provider == "codex"

    def test_extract_from_provider_meta_overlays_db_fields_for_malformed_codex(self):
        provider_meta = {
            "payload": "not-a-dict",
            "content": [],
        }

        msg = extract_from_provider_meta(
            "codex",
            provider_meta,
            message_id="db-codex-id",
            role="assistant",
            text="DB Codex fallback text",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert msg.id == "db-codex-id"
        assert msg.role == "assistant"
        assert msg.text == "DB Codex fallback text"
        assert msg.timestamp is not None
        assert msg.provider == "codex"

    def test_extract_from_provider_meta_raw_overlays_missing_codex_fields(self):
        provider_meta = {
            "raw": {
                "payload": "not-a-dict",
                "content": [],
            }
        }

        msg = extract_from_provider_meta(
            "codex",
            provider_meta,
            message_id="db-codex-raw-id",
            role="assistant",
            text="DB raw fallback text",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert msg.id == "db-codex-raw-id"
        assert msg.role == "assistant"
        assert msg.text == "DB raw fallback text"
        assert msg.timestamp is not None
        assert msg.provider == "codex"


# =============================================================================
# Database Integration Tests
# =============================================================================


class TestDatabaseIntegration:
    """Integration tests using seeded database with real fixture data (schema v3)."""

    @pytest.mark.parametrize("provider", ["claude-code", "chatgpt", "codex"])
    def test_messages_have_content_blocks(self, seeded_db, provider):
        """Messages in DB have content_blocks rows with valid types."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.message_id, m.role, m.text
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = ?
            LIMIT 20
            """,
            (provider,),
        )
        rows = cur.fetchall()
        assert rows, f"No {provider} messages in seeded database"

        for (msg_id, role, text) in rows:
            assert role in ("user", "assistant", "system", "tool"), (
                f"Unexpected role '{role}' in {provider} message {msg_id}"
            )
            # text is nullable for tool-only messages, but role must be valid
            assert isinstance(text, (str, type(None)))

        conn.close()

    def test_claude_code_has_tool_use_blocks(self, seeded_db):
        """Claude Code messages produce tool_use content_blocks in the DB."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT cb.type, cb.tool_name, cb.semantic_type
            FROM content_blocks cb
            JOIN messages m ON cb.message_id = m.message_id
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
              AND cb.type = 'tool_use'
            LIMIT 100
            """
        )
        rows = cur.fetchall()
        conn.close()

        # Synthetic corpus generates tool_use blocks for claude-code
        assert len(rows) > 0, "Expected tool_use blocks for claude-code in seeded DB"
        for (block_type, tool_name, semantic_type) in rows:
            assert block_type == "tool_use"
            assert tool_name is not None
            # semantic_type is classified or None (for unknown tool names)
            assert semantic_type is None or isinstance(semantic_type, str)


class TestExtractedProviderMetaFallbackText:
    """Tests for extracted provider_meta text fallback from stored content blocks."""

    def test_extract_from_provider_meta_rebuilds_text_from_supported_content_blocks(self):
        provider_meta = {
            "content_blocks": [
                {"type": "text", "text": "plain text"},
                {"type": "code", "text": "print('x')", "language": "python"},
                {"type": "tool_result", "text": "tool output"},
                {"type": "thinking", "text": "private reasoning"},
                {"type": "tool_use", "text": "ignored tool use"},
                {"type": "text", "text": ""},
                "ignored non-dict",
            ]
        }

        msg = extract_from_provider_meta(
            "claude-code",
            provider_meta,
            message_id="msg-1",
            role="assistant",
        )

        assert msg.id == "msg-1"
        assert msg.role == "assistant"
        assert msg.text == "plain text\nprint('x')\ntool output\nprivate reasoning"

    def test_extract_from_provider_meta_ignores_nonsensical_content_blocks_for_text(self):
        provider_meta = {
            "content_blocks": [
                {"type": "tool_use", "text": "ignored"},
                {"type": "image", "text": "ignored"},
                {"type": "text", "text": None},
                123,
            ]
        }

        msg = extract_from_provider_meta(
            "claude-code",
            provider_meta,
            message_id="msg-2",
            role="assistant",
        )

        assert msg.id == "msg-2"
        assert msg.role == "assistant"
        assert msg.text == ""

    def test_content_blocks_have_semantic_types(self, seeded_db):
        """Tool_use blocks classified by semantic_type; thinking blocks tagged."""
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT semantic_type, COUNT(*) as cnt
            FROM content_blocks
            WHERE semantic_type IS NOT NULL
            GROUP BY semantic_type
            ORDER BY cnt DESC
            """
        )
        rows = cur.fetchall()
        conn.close()

        # The seeded corpus includes tool_use blocks that should be classified
        assert len(rows) > 0, "Expected at least one classified content_block"
        known_types = {"file_read", "file_write", "file_edit", "shell", "git",
                       "search", "web", "agent", "subagent", "thinking", "code", "other"}
        for (sem_type, _count) in rows:
            assert sem_type in known_types, f"Unknown semantic_type '{sem_type}'"


# MERGED FROM test_unified_extraction_edge_cases.py
# =============================================================================
# extract_chatgpt_text
# =============================================================================


class TestExtractChatGPTText:
    """Tests for extract_chatgpt_text with various content structures."""

    def test_normal_string_parts(self):
        """Normal case: list of string parts joined with newlines."""
        content = {"parts": ["Hello", "World"]}
        assert extract_chatgpt_text(content) == "Hello\nWorld"

    def test_single_string_part(self):
        content = {"parts": ["Just one part"]}
        assert extract_chatgpt_text(content) == "Just one part"

    def test_empty_parts_list(self):
        content = {"parts": []}
        assert extract_chatgpt_text(content) == ""

    def test_none_content(self):
        assert extract_chatgpt_text(None) == ""

    def test_empty_dict_content(self):
        assert extract_chatgpt_text({}) == ""

    def test_no_parts_key(self):
        content = {"content_type": "text"}
        assert extract_chatgpt_text(content) == ""

    def test_parts_is_integer(self):
        """REGRESSION: int parts caused TypeError (not iterable)."""
        content = {"parts": 42}
        result = extract_chatgpt_text(content)
        assert result == "42"

    def test_parts_is_string(self):
        """REGRESSION: string parts iterated characters ('h', 'e', 'l', 'l', 'o')."""
        content = {"parts": "hello world"}
        result = extract_chatgpt_text(content)
        assert result == "hello world"

    def test_parts_is_none(self):
        """parts key exists but value is None."""
        content = {"parts": None}
        result = extract_chatgpt_text(content)
        assert result == ""

    def test_parts_is_bool(self):
        """parts key with boolean value."""
        content = {"parts": True}
        result = extract_chatgpt_text(content)
        assert result == "True"

    def test_mixed_parts_with_non_strings(self):
        """Only string parts are included (non-strings like dicts are skipped)."""
        content = {"parts": ["text part", {"type": "image"}, None, "another"]}
        result = extract_chatgpt_text(content)
        assert result == "text part\nanother"

    def test_empty_string_parts(self):
        """Empty strings in parts are included (they are still strings)."""
        content = {"parts": ["", "non-empty", ""]}
        result = extract_chatgpt_text(content)
        assert result == "\nnon-empty\n"


# =============================================================================
# extract_codex_text
# =============================================================================


class TestExtractCodexText:
    """Tests for extract_codex_text with edge cases."""

    def test_normal_text_blocks(self):
        content = [{"text": "Hello"}, {"text": "World"}]
        assert extract_codex_text(content) == "Hello\nWorld"

    def test_input_text_field(self):
        content = [{"input_text": "User input"}]
        assert extract_codex_text(content) == "User input"

    def test_output_text_field(self):
        content = [{"output_text": "Model output"}]
        assert extract_codex_text(content) == "Model output"

    def test_none_content(self):
        assert extract_codex_text(None) == ""

    def test_empty_list(self):
        assert extract_codex_text([]) == ""

    def test_non_list_content(self):
        """Content that's not a list returns empty."""
        assert extract_codex_text("not a list") == ""  # type: ignore[arg-type]

    def test_non_dict_blocks_skipped(self):
        """Non-dict items in the list are silently skipped."""
        content = [{"text": "valid"}, "string block", 42, None, {"text": "also valid"}]
        assert extract_codex_text(content) == "valid\nalso valid"

    def test_block_with_no_text_fields(self):
        """Block without any text fields is skipped."""
        content = [{"type": "image", "url": "http://example.com"}]
        assert extract_codex_text(content) == ""


# =============================================================================
# extract_harmonized_message (edge cases from edge_cases file)
# =============================================================================


class TestExtractHarmonizedMessageEdgeCases:
    """Tests for extract_harmonized_message provider dispatch (edge cases)."""

    def test_claude_code_basic(self):
        raw = {
            "uuid": "msg-1",
            "type": "human",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Hello Claude"}],
            },
            "timestamp": "2025-01-01T00:00:00Z",
        }
        msg = extract_harmonized_message("claude-code", raw)
        assert msg.role == "user"
        assert msg.text == "Hello Claude"
        assert msg.id == "msg-1"

    def test_claude_code_with_cost(self):
        raw = {
            "uuid": "msg-2",
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Reply"}],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            "costUSD": 0.005,
            "durationMs": 1200,
        }
        msg = extract_harmonized_message("claude-code", raw)
        assert msg.role == "assistant"
        assert msg.model == "claude-sonnet-4-20250514"
        assert msg.cost is not None
        assert msg.cost.total_usd == 0.005
        assert msg.duration_ms == 1200

    def test_chatgpt_basic_edge(self):
        raw = {
            "id": "msg-gpt-1",
            "author": {"role": "user"},
            "content": {"content_type": "text", "parts": ["Hello GPT"]},
            "create_time": 1704067200.0,
        }
        msg = extract_harmonized_message("chatgpt", raw)
        assert msg.role == "user"
        assert msg.text == "Hello GPT"

    def test_chatgpt_non_list_parts(self):
        """REGRESSION: non-list parts in ChatGPT content must not crash."""
        raw = {
            "id": "msg-gpt-2",
            "author": {"role": "assistant"},
            "content": {"content_type": "text", "parts": 42},
        }
        msg = extract_harmonized_message("chatgpt", raw)
        assert msg.text == "42"

    def test_gemini_basic_edge(self):
        raw = {
            "role": "model",
            "text": "Gemini response",
        }
        msg = extract_harmonized_message("gemini", raw)
        assert msg.role == "assistant"
        assert msg.text == "Gemini response"

    def test_gemini_thinking_edge(self):
        raw = {
            "role": "model",
            "text": "Thinking deeply...",
            "isThought": True,
            "thinkingBudget": 500,
        }
        msg = extract_harmonized_message("gemini", raw)
        assert msg.has_reasoning
        assert len(msg.reasoning_traces) == 1
        assert msg.reasoning_traces[0].token_count == 500

    def test_missing_provider_raises(self):
        """Unknown provider should raise ValueError."""
        with pytest.raises((ValueError, KeyError)):
            extract_harmonized_message("unknown-provider", {"text": "test"})

    def test_empty_raw_raises_for_missing_role(self):
        """Empty raw dict raises ValueError (role is required)."""
        with pytest.raises(ValueError, match="no role"):
            extract_harmonized_message("chatgpt", {})

    def test_claude_code_type_fallback_when_no_message(self):
        """Claude Code falls back to 'type' field when message is not a dict."""
        raw = {"uuid": "m1", "type": "human", "message": "not-a-dict"}
        msg = extract_harmonized_message("claude-code", raw)
        assert msg.role == "user"  # "human" normalizes to "user"

    def test_claude_code_empty_message_raises(self):
        """Claude Code with empty message dict has no role → raises."""
        raw = {"uuid": "m1", "type": "human", "message": {}}
        with pytest.raises(ValueError, match="no role"):
            extract_harmonized_message("claude-code", raw)
