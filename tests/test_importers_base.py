"""Tests for importers.base module and DialoguePair validation."""

from __future__ import annotations

import pytest

from polylogue.importers.base import (
    ParsedAttachment,
    attachment_from_meta,
    normalize_role,
)
from polylogue.lib.models import DialoguePair, Message

# =============================================================================
# Parametrized test cases for normalize_role
# =============================================================================

NORMALIZE_ROLE_STANDARD_CASES = [
    ("user", "user", "user stays user"),
    ("assistant", "assistant", "assistant stays assistant"),
    ("system", "system", "system stays system"),
    ("human", "user", "human aliased to user"),
    ("model", "assistant", "model aliased to assistant"),
]

NORMALIZE_ROLE_CASE_INSENSITIVE_CASES = [
    ("USER", "user", "upper USER"),
    ("Human", "user", "mixed Human"),
    ("ASSISTANT", "assistant", "upper ASSISTANT"),
    ("Model", "assistant", "mixed Model"),
    ("SYSTEM", "system", "upper SYSTEM"),
]

NORMALIZE_ROLE_EDGE_CASES = [
    (None, "message", "None returns message"),
    ("", "message", "empty returns message"),
    ("   ", "message", "whitespace only returns message"),
    ("\t\n", "message", "tabs and newlines return message"),
    ("  user  ", "user", "whitespace stripped user"),
    ("\tassistant\n", "assistant", "whitespace stripped assistant"),
    ("custom_role", "custom_role", "unknown lowercased"),
    ("BOT", "bot", "unknown BOT lowercased"),
]


class TestNormalizeRole:
    """Parametrized tests for role normalization."""

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_STANDARD_CASES)
    def test_normalize_role_standard(self, role: str, expected: str, description: str):
        """Standard role mappings."""
        assert normalize_role(role) == expected

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_CASE_INSENSITIVE_CASES)
    def test_normalize_role_case_insensitive(self, role: str, expected: str, description: str):
        """Role normalization is case-insensitive."""
        assert normalize_role(role) == expected

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_EDGE_CASES)
    def test_normalize_role_edge_cases(self, role: str | None, expected: str, description: str):
        """Edge cases: None, empty, whitespace, unknown roles."""
        assert normalize_role(role) == expected


# =============================================================================
# Tests for attachment_from_meta
# =============================================================================


class TestAttachmentFromMeta:
    """Tests for attachment metadata parsing."""

    def test_attachment_from_meta_basic(self):
        """Creates ParsedAttachment from minimal metadata."""
        meta = {"id": "att123", "name": "file.txt"}
        result = attachment_from_meta(meta, "msg1", 0)

        assert result is not None
        assert isinstance(result, ParsedAttachment)
        assert result.provider_attachment_id == "att123"
        assert result.message_provider_id == "msg1"
        assert result.name == "file.txt"
        assert result.provider_meta == meta

    def test_attachment_from_meta_with_all_fields(self):
        """Creates ParsedAttachment with all supported fields."""
        meta = {
            "id": "att456",
            "name": "document.pdf",
            "mimeType": "application/pdf",
            "size": 1024,
        }
        result = attachment_from_meta(meta, "msg2", 1)

        assert result is not None
        assert result.provider_attachment_id == "att456"
        assert result.message_provider_id == "msg2"
        assert result.name == "document.pdf"
        assert result.mime_type == "application/pdf"
        assert result.size_bytes == 1024

    def test_attachment_from_meta_missing_id(self):
        """Generates fallback ID when id is missing but name exists."""
        meta = {"name": "image.png"}
        result = attachment_from_meta(meta, "msg3", 2)

        assert result is not None
        assert result.provider_attachment_id.startswith("att-")
        assert result.name == "image.png"

    def test_attachment_from_meta_empty_dict(self):
        """Returns None for empty metadata dict."""
        result = attachment_from_meta({}, "msg4", 0)
        assert result is None

    def test_attachment_from_meta_not_dict(self):
        """Returns None when meta is not a dict."""
        result = attachment_from_meta("not_a_dict", "msg5", 0)
        assert result is None

        result = attachment_from_meta(None, "msg6", 0)
        assert result is None

    def test_attachment_from_meta_alternative_id_fields(self):
        """Recognizes alternative ID field names."""
        meta1 = {"file_id": "file123", "name": "doc.txt"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.provider_attachment_id == "file123"

        meta2 = {"fileId": "file456", "name": "doc.txt"}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.provider_attachment_id == "file456"

        meta3 = {"uuid": "uuid789", "name": "doc.txt"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.provider_attachment_id == "uuid789"

    def test_attachment_from_meta_alternative_name_fields(self):
        """Recognizes alternative name field names."""
        meta = {"id": "att", "filename": "report.docx"}
        result = attachment_from_meta(meta, "msg", 0)
        assert result.name == "report.docx"

    def test_attachment_from_meta_size_conversion(self):
        """Converts size from string to int."""
        meta1 = {"id": "att", "name": "file", "size": "2048"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.size_bytes == 2048

        meta2 = {"id": "att", "name": "file", "size_bytes": 4096}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.size_bytes == 4096

        meta3 = {"id": "att", "name": "file", "sizeBytes": "8192"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.size_bytes == 8192

    def test_attachment_from_meta_invalid_size(self):
        """Handles invalid size gracefully."""
        meta = {"id": "att", "name": "file", "size": "invalid"}
        result = attachment_from_meta(meta, "msg", 0)
        assert result.size_bytes is None

    def test_attachment_from_meta_mime_type_variations(self):
        """Recognizes different mime_type field names."""
        meta1 = {"id": "att", "name": "file", "mimeType": "text/plain"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.mime_type == "text/plain"

        meta2 = {"id": "att", "name": "file", "mime_type": "image/jpeg"}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.mime_type == "image/jpeg"

        meta3 = {"id": "att", "name": "file", "content_type": "application/json"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.mime_type == "application/json"


# =============================================================================
# Tests for DialoguePair validation
# =============================================================================


class TestDialoguePairValidation:
    """Tests for DialoguePair validation."""

    def test_dialogue_pair_valid(self):
        """Valid user + assistant pair is accepted."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.user.role == "user"
        assert pair.assistant.role == "assistant"

    def test_dialogue_pair_wrong_user_role(self):
        """Raises ValueError if user message doesn't have user role."""
        user_msg = Message(id="u1", role="assistant", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        with pytest.raises(ValueError, match="user message must have user role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_wrong_assistant_role(self):
        """Raises ValueError if assistant message doesn't have assistant role."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="user", text="Hi there")

        with pytest.raises(ValueError, match="assistant message must have assistant role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_human_alias_valid(self):
        """Human role is accepted for user message."""
        user_msg = Message(id="u1", role="human", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.user.is_user

    def test_dialogue_pair_model_alias_valid(self):
        """Model role is accepted for assistant message."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="model", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.assistant.is_assistant

    def test_dialogue_pair_system_role_invalid(self):
        """System role is not valid for dialogue pair."""
        user_msg = Message(id="u1", role="system", text="System prompt")
        assistant_msg = Message(id="a1", role="assistant", text="Response")

        with pytest.raises(ValueError, match="user message must have user role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_exchange_property(self):
        """Exchange property renders the dialogue correctly."""
        user_msg = Message(id="u1", role="user", text="What is 2+2?")
        assistant_msg = Message(id="a1", role="assistant", text="4")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        exchange = pair.exchange

        assert "User: What is 2+2?" in exchange
        assert "Assistant: 4" in exchange
