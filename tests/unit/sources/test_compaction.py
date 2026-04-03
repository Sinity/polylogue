"""Tests for first-class provider event support for compaction.

Covers:
- Legacy and modern compaction detection in semantic_capture
- Claude Code parser emitting provider_events
- Codex provider model compaction/turn_context recognition
- Codex parser emitting provider_events
- Profile builder counting compaction events
"""
from __future__ import annotations

from polylogue.pipeline.semantic_capture import detect_context_compaction
from polylogue.sources.parsers.claude_code_parser import parse_code
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.sources.providers.codex import CodexRecord


# =============================================================================
# detect_context_compaction — legacy format
# =============================================================================

class TestLegacyCompactionDetection:
    def test_summary_type_detected(self) -> None:
        item = {"type": "summary", "message": {"content": "Session summary text"}, "timestamp": "2024-01-01T10:00:00Z"}
        result = detect_context_compaction(item)
        assert result is not None
        assert result["summary"] == "Session summary text"
        assert result["timestamp"] == "2024-01-01T10:00:00Z"
        assert result["is_modern"] is False

    def test_summary_with_content_blocks(self) -> None:
        item = {
            "type": "summary",
            "message": {"content": [{"type": "text", "text": "Block summary"}]},
            "timestamp": "2024-01-01",
        }
        result = detect_context_compaction(item)
        assert result is not None
        assert result["summary"] == "Block summary"
        assert result["is_modern"] is False

    def test_summary_with_empty_message(self) -> None:
        item = {"type": "summary", "message": {}}
        result = detect_context_compaction(item)
        assert result is not None
        assert result["summary"] == ""
        assert result["trigger"] is None
        assert result["pre_tokens"] is None
        assert result["preserved_segment_id"] is None

    def test_non_compaction_returns_none(self) -> None:
        assert detect_context_compaction({"type": "user"}) is None
        assert detect_context_compaction({"type": "assistant"}) is None
        assert detect_context_compaction({}) is None


# =============================================================================
# detect_context_compaction — modern format
# =============================================================================

class TestModernCompactionDetection:
    def test_system_compact_boundary_detected(self) -> None:
        item = {
            "type": "system",
            "subtype": "compact_boundary",
            "message": {"content": "Compacted summary"},
            "timestamp": "2024-06-01T12:00:00Z",
            "compact_metadata": {
                "trigger": "token_limit",
                "preTokens": 128000,
                "preserved_segment": {
                    "head_uuid": "h1",
                    "anchor_uuid": "a1",
                    "tail_uuid": "t1",
                },
            },
        }
        result = detect_context_compaction(item)
        assert result is not None
        assert result["summary"] == "Compacted summary"
        assert result["timestamp"] == "2024-06-01T12:00:00Z"
        assert result["is_modern"] is True
        assert result["trigger"] == "token_limit"
        assert result["pre_tokens"] == 128000
        assert result["preserved_segment_id"] == "a1"

    def test_modern_snake_case_pre_tokens(self) -> None:
        item = {
            "type": "system",
            "subtype": "compact_boundary",
            "message": {"content": "summary"},
            "compact_metadata": {"pre_tokens": 50000},
        }
        result = detect_context_compaction(item)
        assert result is not None
        assert result["pre_tokens"] == 50000

    def test_modern_with_content_blocks(self) -> None:
        item = {
            "type": "system",
            "subtype": "compact_boundary",
            "message": {"content": [{"type": "text", "text": "Modern block"}]},
            "compact_metadata": {},
        }
        result = detect_context_compaction(item)
        assert result is not None
        assert result["summary"] == "Modern block"
        assert result["is_modern"] is True

    def test_system_without_compact_boundary_not_detected(self) -> None:
        """system records without subtype == compact_boundary are not compaction."""
        assert detect_context_compaction({"type": "system"}) is None
        assert detect_context_compaction({"type": "system", "subtype": "other"}) is None

    def test_modern_no_preserved_segment(self) -> None:
        item = {
            "type": "system",
            "subtype": "compact_boundary",
            "message": {"content": "No segment"},
            "compact_metadata": {"trigger": "manual"},
        }
        result = detect_context_compaction(item)
        assert result is not None
        assert result["preserved_segment_id"] is None


# =============================================================================
# ClaudeCodeRecord.is_context_compaction
# =============================================================================

class TestClaudeCodeRecordCompaction:
    def test_legacy_summary(self) -> None:
        record = ClaudeCodeRecord(type="summary")
        assert record.is_context_compaction is True

    def test_modern_compact_boundary(self) -> None:
        record = ClaudeCodeRecord(type="system", subtype="compact_boundary")
        assert record.is_context_compaction is True

    def test_system_without_subtype(self) -> None:
        record = ClaudeCodeRecord(type="system")
        assert record.is_context_compaction is False

    def test_user_not_compaction(self) -> None:
        record = ClaudeCodeRecord(type="user")
        assert record.is_context_compaction is False


# =============================================================================
# Claude Code parser — provider_events emission
# =============================================================================

class TestClaudeCodeParserProviderEvents:
    def test_legacy_compaction_emits_provider_event(self) -> None:
        payload = [
            {"type": "user", "uuid": "u1", "timestamp": "2024-01-01T10:00:00Z",
             "message": {"role": "user", "content": "hello"}},
            {"type": "summary", "uuid": "s1", "timestamp": "2024-01-01T10:05:00Z",
             "message": {"content": "Summary of conversation"}},
            {"type": "assistant", "uuid": "a1", "timestamp": "2024-01-01T10:10:00Z",
             "message": {"role": "assistant", "content": "response"}},
        ]
        result = parse_code(payload, "test-session")
        assert len(result.provider_events) == 1
        event = result.provider_events[0]
        assert event.event_type == "compaction"
        assert event.payload["summary"] == "Summary of conversation"
        assert event.payload["is_modern"] is False

    def test_modern_compaction_emits_provider_event(self) -> None:
        payload = [
            {"type": "user", "uuid": "u1", "timestamp": "2024-01-01T10:00:00Z",
             "message": {"role": "user", "content": "hello"}},
            {"type": "system", "subtype": "compact_boundary", "uuid": "c1",
             "timestamp": "2024-01-01T10:05:00Z",
             "message": {"content": "Modern compacted summary"},
             "compact_metadata": {"trigger": "auto", "preTokens": 100000}},
        ]
        result = parse_code(payload, "test-session")
        assert len(result.provider_events) == 1
        event = result.provider_events[0]
        assert event.event_type == "compaction"
        assert event.payload["is_modern"] is True
        assert event.payload["trigger"] == "auto"
        assert event.payload["pre_tokens"] == 100000

    def test_compaction_backward_compat_provider_meta(self) -> None:
        """provider_meta['context_compactions'] still populated for backward compat."""
        payload = [
            {"type": "summary", "uuid": "s1", "message": {"content": "sum"}},
        ]
        result = parse_code(payload, "test-session")
        assert result.provider_meta is not None
        compactions = result.provider_meta["context_compactions"]
        assert len(compactions) == 1
        assert compactions[0]["summary"] == "sum"

    def test_no_compaction_no_provider_events(self) -> None:
        payload = [
            {"type": "user", "uuid": "u1", "timestamp": "2024-01-01T10:00:00Z",
             "message": {"role": "user", "content": "hello"}},
        ]
        result = parse_code(payload, "test-session")
        assert result.provider_events == []

    def test_compaction_not_counted_as_message(self) -> None:
        """Compaction records should not appear in messages list."""
        payload = [
            {"type": "user", "uuid": "u1", "message": {"role": "user", "content": "hello"}},
            {"type": "summary", "uuid": "s1", "message": {"content": "summary"}},
            {"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "content": "world"}},
        ]
        result = parse_code(payload, "test-session")
        assert len(result.messages) == 2
        roles = [m.role for m in result.messages]
        assert "system" not in roles or all(r in ("user", "assistant") for r in roles)


# =============================================================================
# CodexRecord — compaction and turn_context recognition
# =============================================================================

class TestCodexRecordCompaction:
    def test_compacted_type_recognized(self) -> None:
        record = CodexRecord(type="compacted", payload={"message": "Compacted text", "replacement_history": [{"role": "user"}]})
        assert record.is_compaction is True
        assert record.is_message is False
        assert record.compacted_message == "Compacted text"
        assert record.has_replacement_history is True

    def test_compacted_no_replacement_history(self) -> None:
        record = CodexRecord(type="compacted", payload={"message": "Short"})
        assert record.is_compaction is True
        assert record.has_replacement_history is False

    def test_turn_context_recognized(self) -> None:
        record = CodexRecord(type="turn_context", payload={"context": "data"})
        assert record.is_turn_context is True
        assert record.is_message is False
        assert record.is_compaction is False

    def test_response_item_not_compaction(self) -> None:
        record = CodexRecord(type="response_item", payload={"type": "message", "role": "user", "content": []})
        assert record.is_compaction is False
        assert record.is_turn_context is False

    def test_compacted_message_empty_payload(self) -> None:
        record = CodexRecord(type="compacted", payload=None)
        assert record.compacted_message == ""
        assert record.has_replacement_history is False


# =============================================================================
# Codex parser — provider_events emission
# =============================================================================

class TestCodexParserProviderEvents:
    def test_compaction_emits_provider_event(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2024-01-01"}},
            {"type": "compacted", "payload": {"message": "Context was compacted", "replacement_history": []}},
            {"type": "response_item", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }},
        ]
        result = parse_codex(payload, "fallback")
        assert len(result.provider_events) == 1
        event = result.provider_events[0]
        assert event.event_type == "compaction"
        assert event.payload["summary"] == "Context was compacted"

    def test_turn_context_emits_provider_event(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2024-01-01"}},
            {"type": "turn_context", "payload": {"context": "previous turn data"}},
            {"type": "response_item", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }},
        ]
        result = parse_codex(payload, "fallback")
        assert len(result.provider_events) == 1
        event = result.provider_events[0]
        assert event.event_type == "turn_context"

    def test_messages_still_extracted_alongside_compaction(self) -> None:
        """Compaction records must not break normal message extraction."""
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2024-01-01"}},
            {"type": "response_item", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "first message"}],
            }},
            {"type": "compacted", "payload": {"message": "compacted"}},
            {"type": "response_item", "payload": {
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "second message"}],
            }},
            {"type": "turn_context", "payload": {}},
            {"type": "response_item", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "third message"}],
            }},
        ]
        result = parse_codex(payload, "fallback")
        assert len(result.messages) == 3
        assert result.messages[0].text == "first message"
        assert result.messages[1].text == "second message"
        assert result.messages[2].text == "third message"
        assert len(result.provider_events) == 2
        assert result.provider_events[0].event_type == "compaction"
        assert result.provider_events[1].event_type == "turn_context"

    def test_compaction_backward_compat_provider_meta(self) -> None:
        """provider_meta['context_compactions'] still populated for backward compat."""
        payload = [
            {"type": "compacted", "payload": {"message": "compact text"}},
            {"type": "response_item", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }},
        ]
        result = parse_codex(payload, "fallback")
        assert result.provider_meta is not None
        assert "context_compactions" in result.provider_meta
        compactions = result.provider_meta["context_compactions"]
        assert len(compactions) == 1
        assert compactions[0]["summary"] == "compact text"

    def test_no_compaction_no_provider_events(self) -> None:
        payload = [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2024-01-01"}},
            {"type": "response_item", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }},
        ]
        result = parse_codex(payload, "fallback")
        assert result.provider_events == []


# =============================================================================
# Profile builder — compaction counting
# =============================================================================

class TestProfileCompactionCounting:
    def test_evidence_payload_includes_compaction_fields(self) -> None:
        from polylogue.archive_products import SessionEvidencePayload

        # Default values
        payload = SessionEvidencePayload()
        assert payload.compaction_count == 0
        assert payload.has_compaction is False

        # With compaction
        payload = SessionEvidencePayload(compaction_count=3, has_compaction=True)
        assert payload.compaction_count == 3
        assert payload.has_compaction is True

    def test_evidence_payload_roundtrips(self) -> None:
        from polylogue.archive_products import SessionEvidencePayload

        payload = SessionEvidencePayload(compaction_count=2, has_compaction=True)
        dumped = payload.model_dump()
        restored = SessionEvidencePayload.model_validate(dumped)
        assert restored.compaction_count == 2
        assert restored.has_compaction is True

    def test_profile_evidence_payload_populates_compaction(self) -> None:
        from polylogue.lib.session_profile_models import SessionProfile
        from polylogue.storage.session_product_profiles import profile_evidence_payload

        profile = SessionProfile(
            conversation_id="test",
            provider="claude-code",
            title="Test",
            created_at=None,
            updated_at=None,
            message_count=10,
            substantive_count=5,
            tool_use_count=2,
            thinking_count=1,
            attachment_count=0,
            word_count=100,
            total_cost_usd=0.0,
            total_duration_ms=0,
            tool_categories={},
            repo_paths=(),
            cwd_paths=(),
            branch_names=(),
            file_paths_touched=(),
            languages_detected=(),
            canonical_projects=(),
            work_events=(),
            phases=(),
            compaction_count=3,
        )
        evidence = profile_evidence_payload(profile)
        assert evidence["compaction_count"] == 3
        assert evidence["has_compaction"] is True

    def test_profile_evidence_payload_zero_compaction(self) -> None:
        from polylogue.lib.session_profile_models import SessionProfile
        from polylogue.storage.session_product_profiles import profile_evidence_payload

        profile = SessionProfile(
            conversation_id="test",
            provider="codex",
            title="Test",
            created_at=None,
            updated_at=None,
            message_count=5,
            substantive_count=3,
            tool_use_count=0,
            thinking_count=0,
            attachment_count=0,
            word_count=50,
            total_cost_usd=0.0,
            total_duration_ms=0,
            tool_categories={},
            repo_paths=(),
            cwd_paths=(),
            branch_names=(),
            file_paths_touched=(),
            languages_detected=(),
            canonical_projects=(),
            work_events=(),
            phases=(),
        )
        evidence = profile_evidence_payload(profile)
        assert evidence["compaction_count"] == 0
        assert evidence["has_compaction"] is False
