"""Focused tests for source selection and preview planning."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from polylogue.config import Config, Source
from polylogue.pipeline.runner import _select_sources, plan_sources
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import PlanResult


def _write_chatgpt_preview_export(path: Path, conversation_id: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": conversation_id,
                    "title": "Test",
                    "create_time": 1704067200,
                    "update_time": 1704067200,
                    "mapping": {
                        "root": {"id": "root", "message": None, "children": ["m1"]},
                        "m1": {
                            "id": "m1",
                            "message": {
                                "id": "m1",
                                "author": {"role": "user"},
                                "content": {"parts": ["Hi"]},
                                "create_time": 1704067200,
                            },
                            "parent": "root",
                            "children": ["m2"],
                        },
                        "m2": {
                            "id": "m2",
                            "message": {
                                "id": "m2",
                                "author": {"role": "assistant"},
                                "content": {"parts": ["Hello"]},
                                "create_time": 1704067201,
                            },
                            "parent": "m1",
                            "children": [],
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )


class TestSelectSources:
    def test_select_all_sources_when_no_filter(self, tmp_path: Path):
        sources = [
            Source(name="source-a", path=tmp_path / "a"),
            Source(name="source-b", path=tmp_path / "b"),
            Source(name="source-c", path=tmp_path / "c"),
        ]
        config = Config(sources=sources, archive_root=tmp_path / "archive", render_root=tmp_path / "render")

        assert _select_sources(config, None) == sources
        assert _select_sources(config, []) == sources

    def test_select_filtered_sources(self, tmp_path: Path):
        sources = [
            Source(name="chatgpt-export", path=tmp_path / "a"),
            Source(name="claude-export", path=tmp_path / "b"),
            Source(name="codex-export", path=tmp_path / "c"),
        ]
        config = Config(sources=sources, archive_root=tmp_path / "archive", render_root=tmp_path / "render")

        assert [source.name for source in _select_sources(config, ["claude-export"])] == ["claude-export"]
        assert {source.name for source in _select_sources(config, ["chatgpt-export", "codex-export"])} == {
            "chatgpt-export",
            "codex-export",
        }

    def test_select_empty_when_no_match(self, tmp_path: Path):
        config = Config(
            sources=[Source(name="source-a", path=tmp_path / "a")],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )
        assert _select_sources(config, ["nonexistent-source"]) == []


class TestPlanSources:
    def test_plan_empty_config(self, tmp_path: Path):
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        backend = SQLiteBackend(db_path=tmp_path / "preview.db")
        try:
            result = plan_sources(config, backend=backend)
        finally:
            asyncio.run(backend.close())

        assert isinstance(result, PlanResult)
        assert result.stage == "all"
        assert result.counts == {}
        assert result.details == {}
        assert result.sources == []
        assert result.cursors == {}

    def test_plan_single_source(self, tmp_path: Path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        _write_chatgpt_preview_export(inbox / "conversations.json", "conv-1")

        config = Config(
            sources=[Source(name="test-source", path=inbox)],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )
        backend = SQLiteBackend(db_path=tmp_path / "preview.db")
        try:
            result = plan_sources(config, backend=backend)
        finally:
            asyncio.run(backend.close())

        assert result.counts["scan"] == 1
        assert result.counts["store_raw"] == 1
        assert result.counts["validate"] == 1
        assert result.counts["parse"] == 1
        assert result.sources == ["test-source"]

    async def test_plan_inside_running_event_loop(self, tmp_path: Path):
        config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
        backend = SQLiteBackend(db_path=tmp_path / "preview.db")
        try:
            result = plan_sources(config, backend=backend)
        finally:
            await backend.close()
        assert result.counts == {}
        assert result.sources == []
