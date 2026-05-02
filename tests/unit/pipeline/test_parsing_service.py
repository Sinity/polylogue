"""Focused tests for ParseResult and ParsingService."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.raw_payload.decode import JSONValue
from polylogue.config import Config, Source
from polylogue.errors import DatabaseError
from polylogue.pipeline.payload_types import ParseBatchObservation
from polylogue.pipeline.services.acquisition import AcquireResult, AcquisitionService
from polylogue.pipeline.services.acquisition_records import ScanResult
from polylogue.pipeline.services.parsing import ParseResult, ParsingService
from polylogue.pipeline.services.planning import PlanningService
from polylogue.pipeline.services.validation import ValidationService  # used by TestPlanningService
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.runtime import RawConversationRecord
from polylogue.types import Provider

WorkspacePaths = dict[str, Path]
ConversationPayload = dict[str, JSONValue]
VisitSourcesCallback = Callable[[RawConversationRecord], Awaitable[None]]


def _parse_batch_observation(
    *,
    elapsed_ms: float,
    blob_mb: float,
    rss_end_mb: float,
    peak_rss_growth_mb: float,
) -> ParseBatchObservation:
    return {
        "elapsed_ms": elapsed_ms,
        "blob_mb": blob_mb,
        "rss_end_mb": rss_end_mb,
        "peak_rss_growth_mb": peak_rss_growth_mb,
    }


class TestParseResultMerge:
    async def test_merge_thread_safe(self) -> None:
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
    async def test_ingest_empty_sources_returns_empty_result(self) -> None:
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)
        acquire_result = AcquireResult()

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources",
            new=AsyncMock(return_value=acquire_result),
        ) as mock_acquire:
            with patch(
                "polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog",
                new=AsyncMock(return_value=[]),
            ) as mock_collect_validate:
                with patch(
                    "polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog",
                    new=AsyncMock(return_value=[]),
                ) as mock_collect_parse:
                    with patch.object(service, "parse_from_raw", new_callable=AsyncMock) as mock_parse:
                        result = await service.parse_sources([])

        mock_acquire.assert_awaited_once_with(
            [], ui=None, progress_callback=None, drive_config=mock_config.drive_config
        )
        mock_collect_validate.assert_awaited_once_with(source_names=None, exclude_raw_ids=[])
        mock_collect_parse.assert_awaited_once_with(source_names=None, exclude_raw_ids=[])
        mock_parse.assert_not_called()
        assert result.counts["conversations"] == 0
        assert result.counts["messages"] == 0
        assert len(result.processed_ids) == 0

    async def test_ingest_calls_acquire_then_parse(self) -> None:
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)

        acquire_result = AcquireResult()
        acquire_result.raw_ids = ["raw-1", "raw-2"]
        acquire_result.acquired = 2

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources",
            new=AsyncMock(return_value=acquire_result),
        ) as mock_acquire:
            with patch(
                "polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog",
                new=AsyncMock(return_value=[]),
            ) as mock_collect_validate:
                with patch(
                    "polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog",
                    new=AsyncMock(return_value=[]),
                ) as mock_collect_parse:
                    parse_result = ParseResult()
                    parse_result.counts["conversations"] = 2
                    parse_result.counts["messages"] = 5
                    parse_result.processed_ids = {"conv-1", "conv-2"}
                    with patch.object(
                        service, "parse_from_raw", new_callable=AsyncMock, return_value=parse_result
                    ) as mock_parse:
                        source = Source(name="test-source", path=Path("/tmp/inbox"))
                        result = await service.parse_sources([source])

        mock_acquire.assert_awaited_once_with(
            [Source(name="test-source", path=Path("/tmp/inbox"))],
            ui=None,
            progress_callback=None,
            drive_config=mock_config.drive_config,
        )
        # In the unified ingest flow, validation backlog is collected as part of parse candidates
        mock_collect_validate.assert_awaited_once()
        mock_collect_parse.assert_awaited_once()
        mock_parse.assert_awaited_once_with(raw_ids=["raw-1", "raw-2"], progress_callback=None)
        assert result.counts["conversations"] == 2
        assert result.counts["messages"] == 5
        assert result.processed_ids == {"conv-1", "conv-2"}

    async def test_ingest_sources_surfaces_batch_diagnostics_only(self) -> None:
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)

        acquire_result = AcquireResult()
        acquire_result.raw_ids = ["raw-1"]

        parse_result = ParseResult()
        parse_result.batch_observations = [
            _parse_batch_observation(
                elapsed_ms=123.4,
                blob_mb=1.5,
                rss_end_mb=42.0,
                peak_rss_growth_mb=12.5,
            )
        ]

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources",
            new=AsyncMock(return_value=acquire_result),
        ):
            with patch(
                "polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog",
                new=AsyncMock(return_value=[]),
            ):
                with patch(
                    "polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog",
                    new=AsyncMock(return_value=[]),
                ):
                    with patch.object(service, "parse_from_raw", new_callable=AsyncMock, return_value=parse_result):
                        result = await service.ingest_sources(
                            sources=[Source(name="test-source", path=Path("/tmp/inbox"))],
                        )

        assert result.diagnostics["batch_observations"]["batch_count"] == 1
        assert result.diagnostics["batch_observations"]["max_elapsed_ms"] == 123.4
        assert result.diagnostics["batch_observations"]["max_peak_rss_growth_mb"] == 12.5
        assert "session_insight_refresh" not in result.diagnostics

    async def test_ingest_dedupes_backlog_without_rebuilding_raw_id_list(self) -> None:
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)

        acquire_result = AcquireResult()
        acquire_result.raw_ids = ["raw-1", "raw-2", "raw-1"]
        acquire_result.acquired = 3

        parse_backlog_calls: list[list[str]] = []
        validation_backlog_calls: list[list[str]] = []

        async def _parse_backlog(*, source_names: list[str] | None, exclude_raw_ids: list[str]) -> list[str]:
            assert source_names == ["test-source"]
            parse_backlog_calls.append(list(exclude_raw_ids))
            return ["raw-2", "raw-3", "raw-3"]

        async def _validation_backlog(*, source_names: list[str] | None, exclude_raw_ids: list[str]) -> list[str]:
            assert source_names == ["test-source"]
            validation_backlog_calls.append(list(exclude_raw_ids))
            return ["raw-3", "raw-4", "raw-1"]

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources",
            new=AsyncMock(return_value=acquire_result),
        ):
            with patch(
                "polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog",
                new=AsyncMock(side_effect=_parse_backlog),
            ):
                with patch(
                    "polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog",
                    new=AsyncMock(side_effect=_validation_backlog),
                ):
                    parse_result = ParseResult()
                    with patch.object(
                        service, "parse_from_raw", new_callable=AsyncMock, return_value=parse_result
                    ) as mock_parse:
                        source = Source(name="test-source", path=Path("/tmp/inbox"))
                        await service.parse_sources([source])

        assert parse_backlog_calls == [["raw-1", "raw-2"]]
        assert validation_backlog_calls == [["raw-1", "raw-2", "raw-3"]]
        mock_parse.assert_awaited_once_with(
            raw_ids=["raw-1", "raw-2", "raw-3", "raw-4"],
            progress_callback=None,
        )

    async def test_ingest_skips_parse_when_nothing_acquired(self) -> None:
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)

        acquire_result = AcquireResult()
        acquire_result.skipped = 5
        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources",
            new=AsyncMock(return_value=acquire_result),
        ) as mock_acquire:
            with patch(
                "polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog",
                new=AsyncMock(return_value=[]),
            ) as mock_collect_validate:
                with patch(
                    "polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog",
                    new=AsyncMock(return_value=[]),
                ) as mock_collect_parse:
                    with patch.object(service, "parse_from_raw", new_callable=AsyncMock) as mock_parse:
                        source = Source(name="test-source", path=Path("/tmp/inbox"))
                        result = await service.parse_sources([source])

        mock_acquire.assert_awaited_once()
        mock_collect_validate.assert_awaited_once_with(source_names=["test-source"], exclude_raw_ids=[])
        mock_collect_parse.assert_awaited_once_with(source_names=["test-source"], exclude_raw_ids=[])
        mock_parse.assert_not_called()
        assert result.counts["conversations"] == 0

    async def test_progress_callback_passed_to_both_stages(self) -> None:
        mock_repository = MagicMock()
        mock_backend = MagicMock()
        mock_repository.backend = mock_backend
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)

        callback = MagicMock()
        acquire_result = AcquireResult()
        acquire_result.raw_ids = ["raw-1"]

        with patch(
            "polylogue.pipeline.services.acquisition.AcquisitionService.acquire_sources",
            new=AsyncMock(return_value=acquire_result),
        ) as mock_acquire:
            with patch(
                "polylogue.pipeline.services.planning.PlanningService.collect_validation_backlog",
                new=AsyncMock(return_value=[]),
            ):
                with patch(
                    "polylogue.pipeline.services.planning.PlanningService.collect_parse_backlog",
                    new=AsyncMock(return_value=[]),
                ):
                    parse_result = ParseResult()
                    with patch.object(
                        service, "parse_from_raw", new_callable=AsyncMock, return_value=parse_result
                    ) as mock_parse:
                        source = Source(name="test-source", path=Path("/tmp/inbox"))
                        await service.parse_sources([source], progress_callback=callback)

        mock_acquire.assert_awaited_once_with(
            [Source(name="test-source", path=Path("/tmp/inbox"))],
            ui=None,
            progress_callback=callback,
            drive_config=mock_config.drive_config,
        )
        # In unified ingest, validation is inline — callback passed to parse_from_raw
        mock_parse.assert_awaited_once_with(raw_ids=["raw-1"], progress_callback=callback)

    async def test_backend_not_initialized_raises(self) -> None:
        mock_repository = MagicMock()
        mock_repository.backend = None
        mock_config = MagicMock(spec=Config)
        service = ParsingService(repository=mock_repository, archive_root=Path("/tmp/archive"), config=mock_config)
        source = Source(name="test-source", path=Path("/tmp/inbox"))

        with pytest.raises(DatabaseError, match="backend is not initialized"):
            await service.parse_sources([source])


class TestParsingServiceIntegration:
    def _conversation_json(self, conversation_id: str, title: str, message: str) -> ConversationPayload:
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

    async def test_ingest_with_real_database(
        self, cli_workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    async def test_parse_from_raw_parses_stored_conversations(self, cli_workspace: WorkspacePaths) -> None:
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
    async def test_parse_from_raw_uses_raw_headers_without_prefetching_full_records(self, tmp_path: Path) -> None:
        backend = MagicMock()
        backend.iter_raw_conversations.side_effect = AssertionError("iter_raw_conversations should not be used")

        async def raw_headers() -> AsyncGenerator[tuple[str, int], None]:
            for raw_header in (("raw-1", 1), ("raw-2", 1)):
                yield raw_header

        repository = MagicMock()
        repository.backend = backend
        repository.iter_raw_headers = MagicMock(return_value=raw_headers())
        config = Config(archive_root=tmp_path / "archive", render_root=tmp_path / "render", sources=[])
        service = ParsingService(repository=repository, archive_root=config.archive_root, config=config)

        with patch(
            "polylogue.pipeline.services.ingest_batch.process_ingest_batch", new_callable=AsyncMock
        ) as mock_process:
            await service.parse_from_raw(provider="chatgpt")

        repository.iter_raw_headers.assert_called_once_with(provider_name="chatgpt")
        assert mock_process.await_count == 1

    async def test_parse_from_raw_splits_explicit_raw_ids_by_blob_budget(self, tmp_path: Path) -> None:
        backend = MagicMock()
        repository = MagicMock()
        repository.backend = backend
        repository.get_raw_blob_sizes = AsyncMock(
            return_value=[
                ("raw-1", 96 * 1024 * 1024),
                ("raw-2", 96 * 1024 * 1024),
                ("raw-3", 8 * 1024 * 1024),
            ]
        )
        config = Config(archive_root=tmp_path / "archive", render_root=tmp_path / "render", sources=[])
        service = ParsingService(repository=repository, archive_root=config.archive_root, config=config)

        with patch(
            "polylogue.pipeline.services.ingest_batch.process_ingest_batch",
            new=AsyncMock(side_effect=[None, None]),
        ) as mock_process:
            await service.parse_from_raw(raw_ids=["raw-1", "raw-2", "raw-3"])

        repository.get_raw_blob_sizes.assert_awaited_once_with(["raw-1", "raw-2", "raw-3"])
        assert mock_process.await_args_list[0].args[2] == ["raw-1"]
        assert mock_process.await_args_list[1].args[2] == ["raw-2", "raw-3"]

    async def test_parse_from_raw_splits_streamed_backlog_by_blob_budget(self, tmp_path: Path) -> None:
        backend = MagicMock()

        async def raw_headers() -> AsyncGenerator[tuple[str, int], None]:
            for raw_header in (
                ("raw-1", 96 * 1024 * 1024),
                ("raw-2", 96 * 1024 * 1024),
                ("raw-3", 8 * 1024 * 1024),
            ):
                yield raw_header

        repository = MagicMock()
        repository.backend = backend
        repository.iter_raw_headers = MagicMock(return_value=raw_headers())
        config = Config(archive_root=tmp_path / "archive", render_root=tmp_path / "render", sources=[])
        service = ParsingService(repository=repository, archive_root=config.archive_root, config=config)

        with patch(
            "polylogue.pipeline.services.ingest_batch.process_ingest_batch",
            new=AsyncMock(side_effect=[None, None]),
        ) as mock_process:
            await service.parse_from_raw(provider="chatgpt")

        repository.iter_raw_headers.assert_called_once_with(provider_name="chatgpt")
        assert mock_process.await_args_list[0].args[2] == ["raw-1"]
        assert mock_process.await_args_list[1].args[2] == ["raw-2", "raw-3"]


# =====================================================================
# Merged from test_planning_service.py (service tests)
# =====================================================================


class TestPlanningService:
    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_parse_plan_uses_existing_raw_scope_without_scanning_sources(
        self, mock_iter: MagicMock, tmp_path: Path
    ) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="raw-scoped",
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path="/tmp/a.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
        mock_iter.side_effect = AssertionError("parse planning must not scan sources")

        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="parse")

        assert plan.summary.counts["validate"] == 1
        assert plan.summary.counts["parse"] == 1
        assert set(plan.validate_raw_ids) == {"raw-scoped"}
        assert set(plan.parse_ready_raw_ids) == {"raw-scoped"}
        mock_iter.assert_not_called()

    async def test_planning_includes_scoped_validation_backlog(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        for raw_id, provider_name, source_name, source_path in (
            ("raw-scoped", "chatgpt", "inbox-a", "/tmp/a.json"),
            ("raw-legacy-provider", "inbox-a", None, "/tmp/legacy.json"),
            ("raw-other", "chatgpt", "inbox-b", "/tmp/b.json"),
        ):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=raw_id,
                    provider_name=provider_name,
                    source_name=source_name,
                    source_path=source_path,
                    blob_size=len(b'{"id":"x"}'),
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )

        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="parse")

        assert plan.summary.counts["validate"] == 1
        assert plan.summary.details["backlog_validate"] == 1
        assert set(plan.validate_raw_ids) == {"raw-scoped"}

    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_build_plan_dedupes_duplicate_scanned_raw_ids(self, mock_iter: MagicMock, tmp_path: Path) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        raw_data = RawConversationData(
            raw_bytes=b'{"id":"duplicate"}',
            source_path="/tmp/duplicate.json",
            source_index=0,
            provider_hint=Provider.CHATGPT,
        )
        mock_iter.return_value = iter([raw_data, raw_data])

        plan = await planner.build_plan(
            sources=[Source(name="inbox-a", path=source_dir)],
            stage="acquire",
            preview=True,
        )

        assert plan.summary.counts["scan"] == 2
        assert plan.summary.counts["store_raw"] == 1
        assert plan.summary.details["new_raw"] == 1
        assert plan.summary.details["duplicate_raw"] == 1

    async def test_build_plan_execution_does_not_load_backlog_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        backlog_ids = [hashlib.sha256(f"backlog-{index}".encode()).hexdigest() for index in range(5)]
        for index in range(5):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=backlog_ids[index],
                    provider_name="chatgpt",
                    source_name="inbox-a",
                    source_path=f"/tmp/backlog-{index}.json",
                    blob_size=len(b'{"id":"x"}'),
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )

        call_count = [0]
        from polylogue.storage.repository import ConversationRepository

        original = ConversationRepository.get_raw_conversations_batch

        async def spy_batch(
            repository: ConversationRepository,
            ids: list[str],
            *args: object,
            **kwargs: object,
        ) -> list[RawConversationRecord]:
            call_count[0] += 1
            return await original(repository, ids, *args, **kwargs)

        monkeypatch.setattr(ConversationRepository, "get_raw_conversations_batch", spy_batch)
        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="all", preview=False)

        assert set(plan.validate_raw_ids) == set(backlog_ids)
        assert call_count[0] == 0

    async def test_build_plan_preview_validates_backlog_in_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        total_backlog = ValidationService.RAW_BATCH_SIZE + 5
        for index in range(total_backlog):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=hashlib.sha256(f"raw-preview-{index}".encode()).hexdigest(),
                    provider_name="chatgpt",
                    source_name="inbox-a",
                    source_path=f"/tmp/p-{index}.json",
                    blob_size=len(b'{"id":"x"}'),
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )

        call_count = [0]
        batch_sizes: list[int] = []
        from polylogue.storage.repository import ConversationRepository

        original = ConversationRepository.get_raw_conversations_batch

        async def spy_batch(
            repository: ConversationRepository,
            ids: list[str],
            *args: object,
            **kwargs: object,
        ) -> list[RawConversationRecord]:
            call_count[0] += 1
            batch_sizes.append(len(ids))
            return await original(repository, ids, *args, **kwargs)

        monkeypatch.setattr(ConversationRepository, "get_raw_conversations_batch", spy_batch)
        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="all", preview=True)

        assert len(plan.validate_raw_ids) == total_backlog
        assert call_count[0] >= 2
        assert max(batch_sizes) <= ValidationService.RAW_BATCH_SIZE

    async def test_planning_includes_only_parseable_backlog_statuses(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        for raw_id, status in (("raw-passed", "passed"), ("raw-skipped", "skipped"), ("raw-failed", "failed")):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=raw_id,
                    provider_name="chatgpt",
                    source_name="inbox-a",
                    source_path=f"/tmp/{raw_id}.json",
                    blob_size=len(b'{"id":"x"}'),
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )
            await backend.mark_raw_validated(raw_id, status=status, provider="chatgpt", mode="strict")

        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="parse")

        assert plan.summary.counts["parse"] == 2
        assert plan.summary.details["backlog_parse"] == 2
        assert set(plan.parse_ready_raw_ids) == {"raw-passed", "raw-skipped"}

    async def test_build_plan_force_reparse_simulates_reset_for_parse_backlog(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="raw-validated",
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path="/tmp/validated.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
        await backend.mark_raw_validated("raw-validated", status="passed", provider="chatgpt", mode="strict")
        await backend.mark_raw_parsed("raw-validated", payload_provider="chatgpt")

        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="raw-unvalidated",
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path="/tmp/unvalidated.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
        await backend.mark_raw_parsed("raw-unvalidated", payload_provider="chatgpt")

        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id="raw-validation-failed",
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path="/tmp/validation-failed.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        )
        await backend.mark_raw_validated(
            "raw-validation-failed",
            status="failed",
            error="Malformed JSONL lines: 1",
            provider="chatgpt",
            mode="strict",
        )

        ordinary = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="parse")
        forced = await planner.build_plan(
            sources=[Source(name="inbox-a", path=source_dir)],
            stage="parse",
            force_reparse=True,
        )

        assert ordinary.summary.counts == {}
        assert ordinary.validate_raw_ids == []
        assert ordinary.parse_ready_raw_ids == []

        assert forced.summary.counts["validate"] == 1
        assert forced.summary.counts["parse"] == 3
        assert forced.summary.details["backlog_parse"] == 2
        assert set(forced.validate_raw_ids) == {"raw-unvalidated"}
        assert set(forced.parse_ready_raw_ids) == {
            "raw-unvalidated",
            "raw-validated",
            "raw-validation-failed",
        }

    @patch("polylogue.pipeline.services.acquisition.AcquisitionService.visit_sources")
    async def test_build_plan_force_reparse_counts_scanned_existing_records(
        self,
        mock_visit_sources: MagicMock,
        tmp_path: Path,
    ) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        record = RawConversationRecord(
            raw_id="raw-existing",
            provider_name="chatgpt",
            source_name="inbox-a",
            source_path="/tmp/existing.json",
            blob_size=len(b'{"id":"x"}'),
            acquired_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        await backend.save_raw_conversation(record)
        await backend.mark_raw_validated("raw-existing", status="passed", provider="chatgpt", mode="strict")
        await backend.mark_raw_parsed("raw-existing", payload_provider="chatgpt")

        async def _visit_sources(
            _sources: list[Source],
            *,
            on_record: VisitSourcesCallback | None = None,
            **_kwargs: object,
        ) -> ScanResult:
            assert on_record is not None
            await on_record(record)
            result = ScanResult()
            result.counts["scanned"] = 1
            return result

        mock_visit_sources.side_effect = _visit_sources

        ordinary = await planner.build_plan(
            sources=[Source(name="inbox-a", path=source_dir)],
            stage="all",
            preview=True,
        )
        forced = await planner.build_plan(
            sources=[Source(name="inbox-a", path=source_dir)],
            stage="all",
            preview=True,
            force_reparse=True,
        )

        assert ordinary.summary.counts["scan"] == 1
        assert "parse" not in ordinary.summary.counts
        assert forced.summary.counts["scan"] == 1
        assert forced.summary.counts["parse"] == 1
        assert forced.summary.details["existing_raw"] == 1

    @patch("polylogue.pipeline.services.acquisition.AcquisitionService.visit_sources")
    async def test_build_plan_force_reparse_reuses_shared_backlog_semantics_for_scanned_records(
        self,
        mock_visit_sources: MagicMock,
        tmp_path: Path,
    ) -> None:
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        records = [
            RawConversationRecord(
                raw_id=hashlib.sha256(b"existing-passed").hexdigest(),
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path="/tmp/existing-passed.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            ),
            RawConversationRecord(
                raw_id=hashlib.sha256(b"existing-unvalidated").hexdigest(),
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path="/tmp/existing-unvalidated.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            ),
            RawConversationRecord(
                raw_id=hashlib.sha256(b"existing-failed").hexdigest(),
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path="/tmp/existing-failed.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=datetime.now(tz=timezone.utc).isoformat(),
            ),
        ]
        for record in records:
            await backend.save_raw_conversation(record)

        passed_id = records[0].raw_id
        unvalidated_id = records[1].raw_id
        failed_id = records[2].raw_id
        await backend.mark_raw_validated(passed_id, status="passed", provider="chatgpt", mode="strict")
        await backend.mark_raw_parsed(passed_id, payload_provider="chatgpt")
        await backend.mark_raw_parsed(unvalidated_id, payload_provider="chatgpt")
        await backend.mark_raw_validated(
            failed_id,
            status="failed",
            error="Malformed JSONL lines: 1",
            provider="chatgpt",
            mode="strict",
        )

        async def _visit_sources(
            _sources: list[Source],
            *,
            on_record: VisitSourcesCallback | None = None,
            **_kwargs: object,
        ) -> ScanResult:
            assert on_record is not None
            for record in records:
                await on_record(record)
            result = ScanResult()
            result.counts["scanned"] = len(records)
            return result

        mock_visit_sources.side_effect = _visit_sources

        forced = await planner.build_plan(
            sources=[Source(name="inbox-a", path=source_dir)],
            stage="all",
            preview=True,
            force_reparse=True,
        )

        assert forced.summary.counts["scan"] == 3
        assert forced.summary.counts["validate"] == 1
        assert forced.summary.counts["parse"] == 3
        assert forced.summary.details["existing_raw"] == 3
        assert set(forced.validate_raw_ids) == {unvalidated_id}
        assert set(forced.parse_ready_raw_ids) == {
            passed_id,
            unvalidated_id,
            failed_id,
        }

    @pytest.mark.parametrize(("stage", "count_key"), [("render", "render"), ("index", "index")])
    async def test_build_plan_uses_count_query_for_render_and_index(
        self, tmp_path: Path, stage: str, count_key: str
    ) -> None:
        backend = MagicMock(spec=SQLiteBackend)
        backend.queries = MagicMock()
        backend.count_conversation_ids = AsyncMock(return_value=7)
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)

        plan = await planner.build_plan(sources=[], stage=stage)

        assert plan.summary.counts == {count_key: 7}
        backend.count_conversation_ids.assert_awaited_once_with(source_names=None)
