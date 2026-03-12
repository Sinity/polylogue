"""Focused tests for PlanningService and plan construction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.services.planning import PlanningService
from polylogue.pipeline.services.validation import ValidationService
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import RawConversationRecord


class TestPlanningService:
    async def test_planning_includes_scoped_validation_backlog(self, tmp_path: Path):
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
                    raw_content=b'{"id":"x"}',
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )

        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="validate")

        assert plan.summary.counts["validate"] == 2
        assert plan.summary.details["backlog_validate"] == 2
        assert set(plan.validate_raw_ids) == {"raw-scoped", "raw-legacy-provider"}

    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_build_plan_dedupes_duplicate_scanned_raw_ids(self, mock_iter, tmp_path: Path):
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        raw_data = RawConversationData(
            raw_bytes=b'{"id":"duplicate"}',
            source_path="/tmp/duplicate.json",
            source_index=0,
            provider_hint="chatgpt",
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

    async def test_build_plan_execution_does_not_load_backlog_content(self, tmp_path: Path):
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        for index in range(5):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"raw-backlog-{index}",
                    provider_name="chatgpt",
                    source_name="inbox-a",
                    source_path=f"/tmp/backlog-{index}.json",
                    raw_content=b'{"id":"x"}',
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )

        call_count = [0]
        original = backend.get_raw_conversations_batch

        async def spy_batch(ids, *args, **kwargs):
            call_count[0] += 1
            return await original(ids, *args, **kwargs)

        backend.get_raw_conversations_batch = spy_batch  # type: ignore[method-assign]
        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="all", preview=False)

        assert set(plan.validate_raw_ids) == {f"raw-backlog-{index}" for index in range(5)}
        assert call_count[0] == 0

    async def test_build_plan_preview_validates_backlog_in_batches(self, tmp_path: Path):
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)
        source_dir = tmp_path / "inbox-a"
        source_dir.mkdir()

        total_backlog = ValidationService.RAW_BATCH_SIZE + 5
        for index in range(total_backlog):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"raw-preview-{index}",
                    provider_name="chatgpt",
                    source_name="inbox-a",
                    source_path=f"/tmp/p-{index}.json",
                    raw_content=b'{"id":"x"}',
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )

        call_count = [0]
        batch_sizes: list[int] = []
        original = backend.get_raw_conversations_batch

        async def spy_batch(ids, *args, **kwargs):
            call_count[0] += 1
            batch_sizes.append(len(ids))
            return await original(ids, *args, **kwargs)

        backend.get_raw_conversations_batch = spy_batch  # type: ignore[method-assign]
        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="all", preview=True)

        assert len(plan.validate_raw_ids) == total_backlog
        assert call_count[0] >= 2
        assert max(batch_sizes) <= ValidationService.RAW_BATCH_SIZE

    async def test_planning_includes_only_parseable_backlog_statuses(self, tmp_path: Path):
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
                    raw_content=b'{"id":"x"}',
                    acquired_at=datetime.now(tz=timezone.utc).isoformat(),
                )
            )
            await backend.mark_raw_validated(raw_id, status=status, provider="chatgpt", mode="strict")

        plan = await planner.build_plan(sources=[Source(name="inbox-a", path=source_dir)], stage="parse")

        assert plan.summary.counts["parse"] == 2
        assert plan.summary.details["backlog_parse"] == 2
        assert set(plan.parse_ready_raw_ids) == {"raw-passed", "raw-skipped"}

    @pytest.mark.parametrize(("stage", "count_key"), [("render", "render"), ("index", "index")])
    async def test_build_plan_uses_count_query_for_render_and_index(self, tmp_path: Path, stage: str, count_key: str):
        backend = MagicMock(spec=SQLiteBackend)
        backend.count_conversation_ids = AsyncMock(return_value=7)
        backend.iter_conversation_ids.side_effect = AssertionError(
            "build_plan should count render/index scope without materializing IDs"
        )
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        planner = PlanningService(backend=backend, config=config)

        plan = await planner.build_plan(sources=[], stage=stage)

        assert plan.summary.counts == {count_key: 7}
        backend.count_conversation_ids.assert_awaited_once_with(source_names=None)
