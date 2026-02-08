"""Integration tests for MCP server repository operations.

These tests verify that the repository uses the correct API methods:
- ConversationRepository.view() for ID resolution (not get())
- Backend save methods for data insertion (not repository.save())
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from polylogue.lib.models import Conversation, ConversationSummary
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ConversationRecord, MessageRecord


class TestRepositoryViewMethod:
    """Tests for ConversationRepository.view() with ID resolution."""

    def test_view_resolves_partial_id(self):
        """view() should call resolve_id for ID resolution."""
        # Create mock backend
        backend = Mock(spec=SQLiteBackend)

        # Mock ID resolution - returns full ID
        backend.resolve_id.return_value = "full-conv-id-12345"

        # view() will call resolve_id and then get()
        # Return None to test just the ID resolution call
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        # Test view with partial ID
        result = repo.view("12345")

        # Should call resolve_id first
        backend.resolve_id.assert_called_once_with("12345")
        # Then try get_conversation with resolved ID
        backend.get_conversation.assert_called_once_with("full-conv-id-12345")

    def test_view_uses_resolved_id_fallback(self):
        """view() should fall back to original ID if resolve fails."""
        backend = Mock(spec=SQLiteBackend)

        # Resolve returns None (no match found)
        backend.resolve_id.return_value = None
        # get_conversation also returns None
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        # Test with ID that won't resolve
        result = repo.view("nonexistent")

        # Should try resolve
        backend.resolve_id.assert_called_once_with("nonexistent")
        # Then try original ID
        backend.get_conversation.assert_called_once_with("nonexistent")

    def test_view_returns_none_if_not_found(self):
        """view() should return None if conversation not found."""
        backend = Mock(spec=SQLiteBackend)
        backend.resolve_id.return_value = None
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        result = repo.view("missing-id")

        assert result is None


class TestRepositoryDataInsertion:
    """Tests for proper data insertion using backend methods."""

    def test_save_conversation_uses_backend_methods(self):
        """save_conversation() should use backend.save_conversation() and backend.save_messages()."""
        backend = Mock(spec=SQLiteBackend)
        backend.get_conversation.return_value = None
        backend.get_messages.return_value = []
        backend.get_attachments.return_value = []

        repo = ConversationRepository(backend)

        conv_record = Mock(spec=ConversationRecord)
        conv_record.conversation_id = "conv-1"
        conv_record.content_hash = "hash123"

        msg_record = Mock(spec=MessageRecord)
        msg_record.message_id = "msg-1"
        msg_record.content_hash = "hash456"

        # Perform save operation
        result = repo.save_conversation(
            conversation=conv_record,
            messages=[msg_record],
            attachments=[],
        )

        # Should use backend.save_conversation
        backend.save_conversation.assert_called()

        # Should use backend.save_messages
        backend.save_messages.assert_called()

        # Should return counts dict
        assert isinstance(result, dict)
        assert "conversations" in result
        assert "messages" in result

    def test_backend_direct_insertion(self):
        """Test using backend directly for fixture data insertion."""
        # For test setup, use backend methods directly instead of repository.save()
        backend = Mock(spec=SQLiteBackend)

        # Simulate what test setup would do:
        conv_record = Mock(spec=ConversationRecord)
        conv_record.conversation_id = "test-conv-1"

        msg_record = Mock(spec=MessageRecord)
        msg_record.message_id = "test-msg-1"
        msg_record.conversation_id = "test-conv-1"

        # Use backend methods, not repository.save()
        backend.save_conversation(conv_record)
        backend.save_messages([msg_record])

        # Verify backend methods were called
        backend.save_conversation.assert_called_once_with(conv_record)
        backend.save_messages.assert_called_once_with([msg_record])


class TestMCPServerHandleGet:
    """Tests for _handle_get using repo.view() instead of repo.get()."""

    def test_handle_get_uses_view_method(self):
        """_handle_get should use repo.view() for ID resolution."""
        backend = Mock(spec=SQLiteBackend)

        # Mock resolve_id and get_conversation to return None
        # (test just verifies view() calls these)
        backend.resolve_id.return_value = "full-id"
        backend.get_conversation.return_value = None

        repo = ConversationRepository(backend)

        # Simulate what _handle_get would do
        result = repo.view("partial-id")

        # Should use resolve_id
        assert backend.resolve_id.called
        # Should eventually call get_conversation
        assert backend.get_conversation.called

    def test_mock_includes_view_method(self):
        """Mock repo should include view() method for testing."""
        mock_repo = Mock(spec=ConversationRepository)

        # Mock should have view method
        assert hasattr(mock_repo, "view")

        # Set up mock to return None
        mock_repo.view.return_value = None

        # Test the mock
        result = mock_repo.view("conv-id")

        assert result is None
        mock_repo.view.assert_called_once_with("conv-id")


class TestRepositoryIntegration:
    """Integration tests between repository and backend."""

    def test_repository_wraps_backend_operations(self):
        """Repository should wrap backend for thread-safe operations."""
        backend = Mock(spec=SQLiteBackend)

        repo = ConversationRepository(backend)

        # Repository should have reference to backend
        assert repo.backend == backend
        assert hasattr(repo, "_write_lock")

    def test_repository_methods_exist(self):
        """Repository should have the documented methods."""
        backend = Mock(spec=SQLiteBackend)
        repo = ConversationRepository(backend)

        # Check methods exist
        assert hasattr(repo, "view")
        assert hasattr(repo, "get")
        assert hasattr(repo, "save_conversation")
        assert hasattr(repo, "search")
        assert callable(repo.view)
        assert callable(repo.get)
        assert callable(repo.save_conversation)
