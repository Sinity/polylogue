"""Focused tests for ParseResult and ParsingService."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
from polylogue.pipeline.services.parsing import ParseResult, ParsingService
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository


class TestParseResultMerge:
    async def test_merge_thread_safe(self):
        result = ParseResult()

        async def merge_batch(start_id: int) -> None:
            for index in range(100):
                await result.merge_result(
                    f"conv-{start_id}-{index}",
                    {
                        "conversations": 1,
                        "messages": 2,
                        "attachments": 1,
                        "skipped_conversations": 0,
                        "skipped_messages": 0,
                        "skipped_attachments": 0,
                    },
                    content_changed=True,
                )

        await asyncio.gather(*(merge_batch(index) for index in range(5)))

        assert result.counts["conversations"] == 500
        assert result.counts["messages"] == 1000
        assert result.counts["attachments"] == 500
        assert len(result.processed_ids) == 500


class TestParsingServiceParseSources:
    async def test_ingest_empty_sources_returns_empty_result(self):
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)
        acquire_result = AcquireResult()

        with patch("polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources", new=AsyncMock(return_value=acquire_result)) as mock_acquire:
            with patch("polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog", new=AsyncMock(return_value=[])) as mock_collect_validate:
                with patch("polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog", new=AsyncMock(return_value=[])) as mock_collect_parse:
                    with patch.object(service, "parse_from_raw", new_callable=AsyncMock) as mock_parse:
                        result = await service.parse_sources([])

        mock_acquire.assert_awaited_once_with([], ui=None, progress_callback=None, drive_config=mock_config.drive_config)
        mock_collect_validate.assert_awaited_once_with(source_names=None, exclude_raw_ids=[])
        mock_collect_parse.assert_awaited_once_with(source_names=None, exclude_raw_ids=[])
        mock_parse.assert_not_called()
        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0
        assert len(result.processed_ids) == 0

    async def test_ingest_calls_acquire_then_parse(self):
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)

        acquire_result = AcquireResult()
        acquire_result.raw_ids = ["raw-1", "raw-2"]
        acquire_result.counts["acquired"] = 2
        validation_result = MagicMock(parseable_raw_ids=["raw-1", "raw-2"])

        with patch("polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources", new=AsyncMock(return_value=acquire_result)) as mock_acquire:
            with patch("polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog", new=AsyncMock(return_value=[])) as mock_collect_validate:
                with patch("polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog", new=AsyncMock(return_value=[])) as mock_collect_parse:
                    with patch("polylogue.pipeline.services.validation.ValidationService.validate_raw_ids", new=AsyncMock(return_value=validation_result)) as mock_validate:
                        parse_result = ParseResult()
                        parse_result.counts["conversations"] = 2
                        parse_result.counts["messages"] = 5
                        parse_result.processed_ids = {"conv-1", "conv-2"}
                        with patch.object(service, "parse_from_raw", new_callable=AsyncMock, return_value=parse_result) as mock_parse:
                            source = Source(name="test-source", path=Path("/tmp/inbox"))
                            result = await service.parse_sources([source])

        mock_acquire.assert_awaited_once_with(
            [Source(name="test-source", path=Path("/tmp/inbox"))],
            ui=None,
            progress_callback=None,
            drive_config=mock_config.drive_config,
        )
        mock_collect_validate.assert_awaited_once_with(source_names=["test-source"], exclude_raw_ids=["raw-1", "raw-2"])
        mock_collect_parse.assert_awaited_once_with(source_names=["test-source"], exclude_raw_ids=["raw-1", "raw-2"])
        mock_validate.assert_awaited_once_with(raw_ids=["raw-1", "raw-2"], progress_callback=None)
        mock_parse.assert_awaited_once_with(raw_ids=["raw-1", "raw-2"], progress_callback=None)
        assert result.counts["conversations"] == 2
        assert result.counts["messages"] == 5
        assert result.processed_ids == {"conv-1", "conv-2"}

    async def test_ingest_skips_parse_when_nothing_acquired(self):
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)

        acquire_result = AcquireResult()
        acquire_result.counts["skipped"] = 5
        with patch("polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources", new=AsyncMock(return_value=acquire_result)) as mock_acquire:
            with patch("polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog", new=AsyncMock(return_value=[])) as mock_collect_validate:
                with patch("polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog", new=AsyncMock(return_value=[])) as mock_collect_parse:
                    with patch.object(service, "parse_from_raw", new_callable=AsyncMock) as mock_parse:
                        source = Source(name="test-source", path=Path("/tmp/inbox"))
                        result = await service.parse_sources([source])

        mock_acquire.assert_awaited_once()
        mock_collect_validate.assert_awaited_once_with(source_names=["test-source"], exclude_raw_ids=[])
        mock_collect_parse.assert_awaited_once_with(source_names=["test-source"], exclude_raw_ids=[])
        mock_parse.assert_not_called()
        assert result.counts["conversations"] == 0

    async def test_progress_callback_passed_to_both_stages(self):
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)

        callback = MagicMock()
        acquire_result = AcquireResult()
        acquire_result.raw_ids = ["raw-1"]
        validation_result = MagicMock(parseable_raw_ids=["raw-1"])

        with patch("polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources", new=AsyncMock(return_value=acquire_result)) as mock_acquire:
            with patch("polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog", new=AsyncMock(return_value=[])):
                with patch("polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog", new=AsyncMock(return_value=[])):
                    with patch("polylogue.pipeline.services.validation.ValidationService.validate_raw_ids", new=AsyncMock(return_value=validation_result)) as mock_validate:
                        parse_result = ParseResult()
                        with patch.object(service, "parse_from_raw", new_callable=AsyncMock, return_value=parse_result) as mock_parse:
                            source = Source(name="test-source", path=Path("/tmp/inbox"))
                            await service.parse_sources([source], progress_callback=callback)

        mock_acquire.assert_awaited_once_with(
            [Source(name="test-source", path=Path("/tmp/inbox"))],
            ui=None,
            progress_callback=callback,
            drive_config=mock_config.drive_config,
        )
        mock_validate.assert_awaited_once_with(raw_ids=["raw-1"], progress_callback=callback)
        mock_parse.assert_awaited_once_with(raw_ids=["raw-1"], progress_callback=callback)

    async def test_backend_not_initialized_raises(self):
        mock_repository = MagicMock()
        mock_repository.backend = None
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        with pytest.raises(RuntimeError, match="backend is not initialized"):
            await service.parse_sources([source])


class TestParsingServiceIntegration:
    def _conversation_json(self, conversation_id: str, title: str, message: str) -> dict:
        return {
            "id": conversation_id,
            "title": title,
            "create_time": 1700000000,
            "update_time": 1700000100,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["msg1"]},
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": [message]},
                        "create_time": 1700000050,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }

    async def test_ingest_with_real_database(self, cli_workspace, monkeypatch):
        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
        inbox = cli_workspace["inbox_dir"]
        (inbox / "conversations.json").write_text(
            json.dumps([self._conversation_json("test-conv-1", "Test Conversation", "Hello, world!")])
        )
        backend = SQLiteBackend(db_path=cli_workspace["db_path"])
        config = Config(
            archive_root=cli_workspace["archive_root"],
            render_root=cli_workspace["render_root"],
            sources=[Source(name="test-inbox", path=inbox)],
        )
        result = await ParsingService(
            repository=ConversationRepository(backend=backend),
            archive_root=cli_workspace["archive_root"],
            config=config,
        ).parse_sources(config.sources)
        assert result.counts["conversations"] >= 1
        assert result.processed_ids

    async def test_parse_from_raw_parses_stored_conversations(self, cli_workspace):
        inbox = cli_workspace["inbox_dir"]
        (inbox / "conversations.json").write_text(
            json.dumps([self._conversation_json("test-conv-raw", "Test Raw Conversation", "Hello from raw!")])
        )
        backend = SQLiteBackend(db_path=cli_workspace["db_path"])
        config = Config(
            archive_root=cli_workspace["archive_root"],
            render_root=cli_workspace["render_root"],
            sources=[Source(name="test-inbox", path=inbox)],
        )
        acquire_result = await AcquisitionService(backend=backend).acquire_sources(config.sources)
        raw_ids = acquire_result.raw_ids
        assert len(raw_ids) == 1

        parse_result = await ParsingService(
            repository=ConversationRepository(backend=backend),
            archive_root=cli_workspace["archive_root"],
            config=config,
        ).parse_from_raw(raw_ids=raw_ids)
        assert parse_result.counts["conversations"] >= 1
        assert parse_result.processed_ids
        async with backend.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT raw_id FROM conversations WHERE conversation_id = ?",
                    (list(parse_result.processed_ids)[0],),
                )
            ).fetchone()
        assert row is not None
        assert row["raw_id"] == raw_ids[0]


class TestParsingServiceStreaming:
    async def test_parse_from_raw_uses_raw_ids_without_prefetching_full_records(self, tmp_path):
        backend = MagicMock()
        backend.iter_raw_conversations.side_effect = AssertionError("iter_raw_conversations should not be used")

        async def raw_ids():
            for raw_id in ("raw-1", "raw-2"):
                yield raw_id

        backend.iter_raw_ids = MagicMock(return_value=raw_ids())
        repository = MagicMock()
        repository.backend = backend
        config = Config(archive_root=tmp_path / "archive", render_root=tmp_path / "render", sources=[])
        service = ParsingService(repository=repository, archive_root=config.archive_root, config=config)

        with patch.object(service, "_process_raw_batch", new_callable=AsyncMock) as mock_process:
            await service.parse_from_raw(provider="chatgpt")

        backend.iter_raw_ids.assert_called_once_with(provider_name="chatgpt")
        assert mock_process.await_count == 1
