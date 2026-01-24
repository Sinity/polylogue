"""Tests for polylogue.types module - NewType IDs and Provider enum."""
from __future__ import annotations

import pytest

from polylogue.types import AttachmentId, ContentHash, ConversationId, MessageId, Provider


class TestConversationId:
    """Test ConversationId NewType."""

    def test_creation_from_string(self) -> None:
        """ConversationId can be created from string."""
        cid = ConversationId("conv-123")
        assert cid == "conv-123"

    def test_is_string_type(self) -> None:
        """ConversationId is a string at runtime."""
        cid = ConversationId("conv-123")
        assert isinstance(cid, str)

    def test_equality_same_value(self) -> None:
        """Two ConversationIds with same value are equal."""
        cid1 = ConversationId("conv-123")
        cid2 = ConversationId("conv-123")
        assert cid1 == cid2

    def test_inequality_different_value(self) -> None:
        """Two ConversationIds with different values are not equal."""
        cid1 = ConversationId("conv-123")
        cid2 = ConversationId("conv-456")
        assert cid1 != cid2

    def test_use_as_dict_key(self) -> None:
        """ConversationId can be used as dict key."""
        cid = ConversationId("conv-123")
        d = {cid: "metadata"}
        assert d[cid] == "metadata"

    def test_use_in_set(self) -> None:
        """ConversationId can be used in sets with deduplication."""
        cid1 = ConversationId("conv-123")
        cid2 = ConversationId("conv-123")
        cid3 = ConversationId("conv-456")
        s = {cid1, cid2, cid3}
        assert len(s) == 2

    def test_string_methods_available(self) -> None:
        """ConversationId supports string methods."""
        cid = ConversationId("conv-123")
        assert cid.startswith("conv")
        assert "123" in cid
        assert cid.upper() == "CONV-123"
        assert cid.replace("conv", "conversation") == "conversation-123"

    def test_empty_string_valid(self) -> None:
        """Empty string is valid ConversationId (no validation)."""
        cid = ConversationId("")
        assert cid == ""
        assert isinstance(cid, str)

    def test_special_characters(self) -> None:
        """ConversationId handles special characters."""
        cid = ConversationId("conv-🎯-123")
        assert "🎯" in cid
        assert len(cid) > 0


class TestMessageId:
    """Test MessageId NewType."""

    def test_creation_from_string(self) -> None:
        """MessageId can be created from string."""
        mid = MessageId("msg-456")
        assert mid == "msg-456"

    def test_is_string_type(self) -> None:
        """MessageId is a string at runtime."""
        mid = MessageId("msg-456")
        assert isinstance(mid, str)

    def test_equality_same_value(self) -> None:
        """Two MessageIds with same value are equal."""
        mid1 = MessageId("msg-456")
        mid2 = MessageId("msg-456")
        assert mid1 == mid2

    def test_inequality_different_value(self) -> None:
        """Two MessageIds with different values are not equal."""
        mid1 = MessageId("msg-456")
        mid2 = MessageId("msg-789")
        assert mid1 != mid2

    def test_use_as_dict_key(self) -> None:
        """MessageId can be used as dict key."""
        mid = MessageId("msg-456")
        d = {mid: "content"}
        assert d[mid] == "content"

    def test_string_methods_available(self) -> None:
        """MessageId supports string methods."""
        mid = MessageId("msg-456")
        assert mid.startswith("msg")
        assert "456" in mid


class TestAttachmentId:
    """Test AttachmentId NewType."""

    def test_creation_from_string(self) -> None:
        """AttachmentId can be created from string."""
        aid = AttachmentId("att-789")
        assert aid == "att-789"

    def test_is_string_type(self) -> None:
        """AttachmentId is a string at runtime."""
        aid = AttachmentId("att-789")
        assert isinstance(aid, str)

    def test_equality_same_value(self) -> None:
        """Two AttachmentIds with same value are equal."""
        aid1 = AttachmentId("att-789")
        aid2 = AttachmentId("att-789")
        assert aid1 == aid2

    def test_inequality_different_value(self) -> None:
        """Two AttachmentIds with different values are not equal."""
        aid1 = AttachmentId("att-789")
        aid2 = AttachmentId("att-012")
        assert aid1 != aid2

    def test_use_in_set(self) -> None:
        """AttachmentId can be used in sets."""
        aid1 = AttachmentId("att-789")
        aid2 = AttachmentId("att-789")
        aid3 = AttachmentId("att-012")
        s = {aid1, aid2, aid3}
        assert len(s) == 2


class TestContentHash:
    """Test ContentHash NewType."""

    def test_creation_from_string(self) -> None:
        """ContentHash can be created from string."""
        ch = ContentHash("abc123def456")
        assert ch == "abc123def456"

    def test_is_string_type(self) -> None:
        """ContentHash is a string at runtime."""
        ch = ContentHash("abc123def456")
        assert isinstance(ch, str)

    def test_equality_same_value(self) -> None:
        """Two ContentHashes with same value are equal."""
        ch1 = ContentHash("abc123def456")
        ch2 = ContentHash("abc123def456")
        assert ch1 == ch2

    def test_use_as_dict_key(self) -> None:
        """ContentHash can be used as dict key."""
        ch = ContentHash("abc123def456")
        d = {ch: "conversation_id"}
        assert d[ch] == "conversation_id"


class TestProviderEnum:
    """Test Provider enum."""

    def test_enum_values(self) -> None:
        """Provider enum has expected values."""
        assert Provider.CHATGPT.value == "chatgpt"
        assert Provider.CLAUDE.value == "claude"
        assert Provider.CLAUDE_CODE.value == "claude-code"
        assert Provider.CODEX.value == "codex"
        assert Provider.GEMINI.value == "gemini"
        assert Provider.DRIVE.value == "drive"
        assert Provider.UNKNOWN.value == "unknown"

    def test_is_string_enum(self) -> None:
        """Provider inherits from str and Enum."""
        assert isinstance(Provider.CHATGPT, str)
        assert isinstance(Provider.CLAUDE, str)

    def test_string_methods_available(self) -> None:
        """Provider supports string methods."""
        assert Provider.CHATGPT.startswith("chat")
        assert "gpt" in Provider.CHATGPT
        assert Provider.CHATGPT.upper() == "CHATGPT"

    def test_str_representation(self) -> None:
        """str(Provider) returns value."""
        assert str(Provider.CHATGPT) == "chatgpt"
        assert str(Provider.CLAUDE) == "claude"
        assert str(Provider.UNKNOWN) == "unknown"


class TestProviderFromString:
    """Test Provider.from_string() class method."""

    def test_exact_match_chatgpt(self) -> None:
        """Exact match returns correct enum."""
        assert Provider.from_string("chatgpt") == Provider.CHATGPT

    def test_exact_match_claude(self) -> None:
        """Exact match for claude returns CLAUDE."""
        assert Provider.from_string("claude") == Provider.CLAUDE

    def test_exact_match_claude_code(self) -> None:
        """Exact match for claude-code returns CLAUDE_CODE."""
        assert Provider.from_string("claude-code") == Provider.CLAUDE_CODE

    def test_exact_match_codex(self) -> None:
        """Exact match for codex returns CODEX."""
        assert Provider.from_string("codex") == Provider.CODEX

    def test_exact_match_gemini(self) -> None:
        """Exact match for gemini returns GEMINI."""
        assert Provider.from_string("gemini") == Provider.GEMINI

    def test_exact_match_drive(self) -> None:
        """Exact match for drive returns DRIVE."""
        assert Provider.from_string("drive") == Provider.DRIVE

    def test_case_insensitive_match(self) -> None:
        """Matching is case-insensitive."""
        assert Provider.from_string("CHATGPT") == Provider.CHATGPT
        assert Provider.from_string("ClAuDe") == Provider.CLAUDE
        assert Provider.from_string("GEMINI") == Provider.GEMINI

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert Provider.from_string("  chatgpt  ") == Provider.CHATGPT
        assert Provider.from_string("\tclaude\n") == Provider.CLAUDE

    def test_gpt_alias(self) -> None:
        """'gpt' and 'openai' are aliases for CHATGPT."""
        assert Provider.from_string("gpt") == Provider.CHATGPT
        assert Provider.from_string("openai") == Provider.CHATGPT
        assert Provider.from_string("GPT") == Provider.CHATGPT
        assert Provider.from_string("OPENAI") == Provider.CHATGPT

    def test_claude_ai_alias(self) -> None:
        """'claude-ai' and 'anthropic' are aliases for CLAUDE."""
        assert Provider.from_string("claude-ai") == Provider.CLAUDE
        assert Provider.from_string("anthropic") == Provider.CLAUDE
        assert Provider.from_string("CLAUDE-AI") == Provider.CLAUDE
        assert Provider.from_string("ANTHROPIC") == Provider.CLAUDE

    def test_unknown_for_invalid(self) -> None:
        """Invalid provider returns UNKNOWN."""
        assert Provider.from_string("invalid-provider") == Provider.UNKNOWN
        assert Provider.from_string("foobar") == Provider.UNKNOWN

    def test_none_returns_unknown(self) -> None:
        """None returns UNKNOWN."""
        assert Provider.from_string(None) == Provider.UNKNOWN

    def test_empty_string_returns_unknown(self) -> None:
        """Empty string returns UNKNOWN."""
        assert Provider.from_string("") == Provider.UNKNOWN

    def test_whitespace_only_returns_unknown(self) -> None:
        """Whitespace-only string returns UNKNOWN."""
        assert Provider.from_string("   ") == Provider.UNKNOWN
        assert Provider.from_string("\t\n") == Provider.UNKNOWN


class TestTypeInteroperability:
    """Test NewType IDs work together in realistic scenarios."""

    def test_all_ids_as_dict_keys(self) -> None:
        """All ID types can be used as dict keys simultaneously."""
        cid = ConversationId("conv-1")
        mid = MessageId("msg-1")
        aid = AttachmentId("att-1")
        ch = ContentHash("hash-1")

        data = {
            cid: {"messages": [mid], "attachments": [aid]},
            ch: "dedup-info",
        }

        assert data[cid]["messages"][0] == mid
        assert data[ch] == "dedup-info"

    def test_ids_in_list_and_set(self) -> None:
        """ID types work in collections."""
        cids = [
            ConversationId("conv-1"),
            ConversationId("conv-2"),
            ConversationId("conv-1"),
        ]
        unique_cids = set(cids)

        assert len(cids) == 3
        assert len(unique_cids) == 2

    def test_provider_enum_interop_with_ids(self) -> None:
        """Provider enum works with ID types in data structures."""
        cid = ConversationId("conv-1")
        provider = Provider.CLAUDE

        metadata = {"conversation_id": cid, "provider": provider}

        assert metadata["conversation_id"] == cid
        assert metadata["provider"] == Provider.CLAUDE
        assert str(metadata["provider"]) == "claude"
