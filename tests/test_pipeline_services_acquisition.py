"""Tests for pipeline/services/acquisition.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Source
from polylogue.importers.base import RawConversationData
from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
from polylogue.storage.backends.sqlite import SQLiteBackend


class TestAcquireResult:
    """Tests for AcquireResult."""

    def test_counts_initialized_to_zero(self):
        """All count fields start at zero."""
        result = AcquireResult()

        assert result.counts["acquired"] == 0
        assert result.counts["skipped"] == 0
        assert result.counts["errors"] == 0

    def test_raw_ids_initialized_empty(self):
        """raw_ids starts as empty list."""
        result = AcquireResult()

        assert result.raw_ids == []
        assert isinstance(result.raw_ids, list)


class TestAcquisitionServiceInit:
    """Tests for AcquisitionService initialization."""

    def test_init_sets_backend(self, tmp_path: Path):
        """Backend is stored on init."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        service = AcquisitionService(backend=backend)

        assert service.backend is backend


class TestAcquisitionServiceAcquireSources:
    """Tests for AcquisitionService.acquire_sources method."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a SQLiteBackend with a temp database."""
        return SQLiteBackend(db_path=tmp_path / "test.db")

    def test_acquire_empty_sources(self, backend: SQLiteBackend):
        """Empty sources list returns empty result."""
        service = AcquisitionService(backend=backend)

        result = service.acquire_sources([])

        assert result.counts["acquired"] == 0
        assert result.counts["skipped"] == 0
        assert result.counts["errors"] == 0
        assert result.raw_ids == []

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_single_conversation(self, mock_iter, backend: SQLiteBackend):
        """Single conversation is acquired correctly."""
        raw_bytes = b'{"id": "test-conv", "messages": []}'
        raw_data = RawConversationData(
            raw_bytes=raw_bytes,
            source_path="/tmp/test.json",
            source_index=0,
            provider_hint="chatgpt",
        )
        mock_iter.return_value = iter([(raw_data, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["acquired"] == 1
        assert result.counts["skipped"] == 0
        assert len(result.raw_ids) == 1

        # Verify stored in database
        stored = backend.get_raw_conversation(result.raw_ids[0])
        assert stored is not None
        assert stored.raw_content == raw_bytes
        assert stored.provider_name == "chatgpt"

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_multiple_conversations(self, mock_iter, backend: SQLiteBackend):
        """Multiple conversations are acquired correctly."""
        convos = [
            RawConversationData(
                raw_bytes=f'{{"id": "conv-{i}"}}'.encode(),
                source_path="/tmp/test.json",
                source_index=i,
                provider_hint="chatgpt",
            )
            for i in range(3)
        ]
        mock_iter.return_value = iter([(r, MagicMock()) for r in convos])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["acquired"] == 3
        assert len(result.raw_ids) == 3

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_skips_duplicates(self, mock_iter, backend: SQLiteBackend):
        """Duplicate raw_ids are skipped."""
        raw_bytes = b'{"id": "same-conv"}'
        raw_data = RawConversationData(
            raw_bytes=raw_bytes,
            source_path="/tmp/test.json",
            source_index=0,
            provider_hint="chatgpt",
        )
        # Same conversation twice
        mock_iter.return_value = iter([(raw_data, MagicMock()), (raw_data, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["acquired"] == 1
        assert result.counts["skipped"] == 1

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_uses_source_name_as_fallback_provider(self, mock_iter, backend: SQLiteBackend):
        """Source name is used as provider_name when provider_hint is None."""
        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
            provider_hint=None,  # No hint
        )
        mock_iter.return_value = iter([(raw_data, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="my-inbox", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        stored = backend.get_raw_conversation(result.raw_ids[0])
        assert stored is not None
        assert stored.provider_name == "my-inbox"

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_progress_callback_called(self, mock_iter, backend: SQLiteBackend):
        """Progress callback is invoked for each conversation."""
        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
        )
        mock_iter.return_value = iter([(raw_data, MagicMock())])

        callback = MagicMock()

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        service.acquire_sources([source], progress_callback=callback)

        callback.assert_called_with(1, desc="Acquiring")

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_handles_iteration_error(self, mock_iter, backend: SQLiteBackend):
        """Errors during source iteration are counted."""
        mock_iter.side_effect = ValueError("File not found")

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["errors"] == 1
        assert result.counts["acquired"] == 0

    @patch("polylogue.pipeline.services.acquisition.iter_source_conversations_with_raw")
    def test_acquire_handles_none_raw_data(self, mock_iter, backend: SQLiteBackend):
        """None raw_data is counted as error."""
        mock_iter.return_value = iter([(None, MagicMock())])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        result = service.acquire_sources([source])

        assert result.counts["errors"] == 1
        assert result.counts["acquired"] == 0


class TestAcquisitionServiceIntegration:
    """Integration tests for AcquisitionService with real files."""

    def test_acquire_real_chatgpt_file(self, tmp_path: Path):
        """Acquire from a real ChatGPT conversations.json file."""
        # Create a realistic ChatGPT export
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        conv_data = [
            {
                "id": "conv-1",
                "title": "Test Chat",
                "create_time": 1700000000,
                "update_time": 1700000100,
                "mapping": {
                    "root": {"id": "root", "message": None, "children": ["msg1"]},
                    "msg1": {
                        "id": "msg1",
                        "message": {
                            "id": "msg1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Hello"]},
                            "create_time": 1700000050,
                        },
                        "parent": "root",
                        "children": [],
                    },
                },
            },
        ]
        (inbox / "conversations.json").write_text(json.dumps(conv_data))

        # Create backend and service
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        service = AcquisitionService(backend=backend)

        source = Source(name="chatgpt-inbox", path=inbox)

        result = service.acquire_sources([source])

        assert result.counts["acquired"] == 1
        assert result.counts["errors"] == 0
        assert len(result.raw_ids) == 1

        # Verify the raw content is stored
        stored = backend.get_raw_conversation(result.raw_ids[0])
        assert stored is not None
        assert stored.provider_name == "chatgpt"

        # Parse the stored content to verify it's valid JSON
        stored_data = json.loads(stored.raw_content)
        assert stored_data["id"] == "conv-1"
        assert stored_data["title"] == "Test Chat"

    def test_acquire_multiple_json_files(self, tmp_path: Path):
        """Acquire from multiple JSON files in a directory."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        # Create two separate JSON files
        conv1 = {
            "id": "conv-1",
            "title": "Chat 1",
            "create_time": 1700000000,
            "update_time": 1700000100,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["msg1"]},
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                        "create_time": 1700000050,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
        conv2 = {
            "id": "conv-2",
            "title": "Chat 2",
            "create_time": 1700000200,
            "update_time": 1700000300,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["msg2"]},
                "msg2": {
                    "id": "msg2",
                    "message": {
                        "id": "msg2",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["World"]},
                        "create_time": 1700000250,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }

        # Write as separate files (ChatGPT exports multiple conversations in one file)
        (inbox / "conversations.json").write_text(json.dumps([conv1, conv2]))

        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        service = AcquisitionService(backend=backend)

        source = Source(name="chatgpt-export", path=inbox)

        result = service.acquire_sources([source])

        # ChatGPT bundle: each conversation in the array is acquired separately
        assert result.counts["acquired"] == 2
        assert len(result.raw_ids) == 2

        # Verify both are stored with distinct raw_ids
        assert len(set(result.raw_ids)) == 2  # All unique
