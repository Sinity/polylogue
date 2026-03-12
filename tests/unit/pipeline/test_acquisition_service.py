"""Focused tests for AcquisitionService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import Source
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.backends.async_sqlite import SQLiteBackend


class TestAcquisitionServiceAcquireSources:
    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def test_acquire_empty_sources(self, backend: SQLiteBackend):
        result = await AcquisitionService(backend=backend).acquire_sources([])
        assert all(result.counts[key] == 0 for key in ["acquired", "skipped", "errors"])
        assert result.raw_ids == []

    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_progress_callback_called(self, mock_iter, backend: SQLiteBackend):
        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
        )
        mock_iter.return_value = iter([raw_data])
        backend.get_known_source_mtimes = AsyncMock(return_value={})
        callback = MagicMock()
        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        await service.acquire_sources([source], progress_callback=callback)

        callback.assert_any_call(1, desc="Acquiring [test-source]")
        assert mock_iter.call_args is not None
        assert mock_iter.call_args.kwargs.get("known_mtimes") is not None

    @pytest.mark.parametrize("error_scenario", ["iteration_error", "none_raw_data"])
    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_acquire_handles_errors(self, mock_iter, backend: SQLiteBackend, error_scenario: str):
        if error_scenario == "iteration_error":
            mock_iter.side_effect = ValueError("File not found")
        else:
            mock_iter.return_value = iter([None])

        service = AcquisitionService(backend=backend)
        source = Source(name="test-source", path=Path("/tmp/inbox"))
        result = await service.acquire_sources([source])

        assert result.counts["errors"] == 1
        assert result.counts["acquired"] == 0


class TestAcquisitionServiceIntegration:
    def _make_conv(self, conversation_id: str, title: str, timestamp: int, message: str) -> dict:
        return {
            "id": conversation_id,
            "title": title,
            "create_time": timestamp,
            "update_time": timestamp + 100,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["msg1"]},
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": [message]},
                        "create_time": timestamp + 50,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }

    async def test_acquire_real_chatgpt_file(self, tmp_path: Path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "conversations.json").write_text(
            json.dumps([self._make_conv("conv-1", "Test Chat", 1700000000, "Hello")])
        )
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = await AcquisitionService(backend=backend).acquire_sources(
            [Source(name="chatgpt-inbox", path=inbox)]
        )

        assert result.counts["acquired"] == 1
        assert result.counts["errors"] == 0
        assert len(result.raw_ids) == 1
        stored = await backend.get_raw_conversation(result.raw_ids[0])
        data = json.loads(stored.raw_content)
        assert stored.provider_name == "chatgpt"
        assert isinstance(data, list)
        assert data[0]["id"] == "conv-1"
        assert data[0]["title"] == "Test Chat"

    async def test_acquire_multiple_json_files(self, tmp_path: Path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        conversations = [
            self._make_conv("conv-1", "Chat 1", 1700000000, "Hello"),
            self._make_conv("conv-2", "Chat 2", 1700000200, "World"),
        ]
        (inbox / "conversations.json").write_text(json.dumps(conversations))
        result = await AcquisitionService(backend=SQLiteBackend(db_path=tmp_path / "test.db")).acquire_sources(
            [Source(name="chatgpt-export", path=inbox)]
        )

        assert result.counts["acquired"] == 1
        assert len(result.raw_ids) == 1
