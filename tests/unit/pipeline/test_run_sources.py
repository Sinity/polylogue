"""Focused tests for end-to-end runner stage execution."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.runner import latest_run, run_sources
from polylogue.storage.backends import create_backend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import PlanResult
from tests.infra.storage_records import make_conversation, make_message, store_records


def _seed_conversations(workspace_env, *conversation_ids: str, with_message: bool = False) -> None:
    db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open_connection(db_path) as conn:
        for conversation_id in conversation_ids:
            conversation = make_conversation(conversation_id, title=conversation_id)
            messages = [make_message(f"{conversation_id}:msg-1", conversation_id, text="hello")] if with_message else []
            store_records(conversation=conversation, messages=messages, attachments=[], conn=conn)


def _write_chatgpt_export(path: Path, conversation_id: str, *, text: str = "Test") -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": conversation_id,
                    "title": conversation_id,
                    "create_time": 1704067200,
                    "update_time": 1704067200,
                    "mapping": {
                        "root": {"id": "root", "message": None, "children": ["m1"]},
                        "m1": {
                            "id": "m1",
                            "message": {
                                "id": "m1",
                                "author": {"role": "user"},
                                "content": {"parts": [text]},
                                "create_time": 1704067200,
                            },
                            "parent": "root",
                            "children": [],
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )


class TestRunSourcesRenderFailures:
    def test_render_failure_tracked_in_result(self, workspace_env):
        from polylogue.storage.store import RunResult

        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)
        _seed_conversations(workspace_env, "test:success-conv", "test:fail-conv")
        config = Config(sources=[], archive_root=archive_root, render_root=archive_root / "render")

        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render", new_callable=AsyncMock) as mock_render:
            def render_side_effect(conversation_id, output_path):
                if "fail-conv" in conversation_id:
                    raise ValueError("Render failed for testing")
                return MagicMock()

            mock_render.side_effect = render_side_effect
            result = asyncio.run(run_sources(config=config, stage="render", source_names=None))

        assert isinstance(result, RunResult)
        assert isinstance(result.render_failures, list)
        assert result.render_failures
        failure = result.render_failures[0]
        assert failure["conversation_id"] == "test:fail-conv"
        assert "error" in failure

    def test_render_continues_after_failure(self, workspace_env):
        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)
        _seed_conversations(workspace_env, "test:first", "test:second", "test:third")
        config = Config(sources=[], archive_root=archive_root, render_root=archive_root / "render")
        render_attempts: list[str] = []

        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render", new_callable=AsyncMock) as mock_render:
            def render_side_effect(conversation_id, output_path):
                render_attempts.append(conversation_id)
                if "second" in conversation_id:
                    raise ValueError("Failed on purpose")
                return MagicMock()

            mock_render.side_effect = render_side_effect
            asyncio.run(run_sources(config=config, stage="render"))

        assert set(render_attempts) >= {"test:first", "test:second", "test:third"}

    def test_render_failure_count_in_counts(self, workspace_env):
        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)
        _seed_conversations(workspace_env, "test:success", "test:fail1", "test:fail2")
        config = Config(sources=[], archive_root=archive_root, render_root=archive_root / "render")

        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render", new_callable=AsyncMock) as mock_render:
            def render_side_effect(conversation_id, output_path):
                if conversation_id in ["test:fail1", "test:fail2"]:
                    raise ValueError("Render failed")
                return MagicMock()

            mock_render.side_effect = render_side_effect
            result = asyncio.run(run_sources(config=config, stage="render"))

        assert result.counts["render_failures"] == 2
        assert result.counts["rendered"] == 1

    def test_render_stage_uses_configured_render_root(self, workspace_env, tmp_path: Path):
        custom_render_root = tmp_path / "custom-render-root"
        _seed_conversations(workspace_env, "test:custom-render-root", with_message=True)
        config = Config(sources=[], archive_root=workspace_env["archive_root"], render_root=custom_render_root)

        result = asyncio.run(run_sources(config=config, stage="render"))
        assert result.counts["rendered"] == 1
        assert custom_render_root.exists()
        assert any(custom_render_root.rglob("conversation.html"))


class TestRunSourcesIntegration:
    @pytest.mark.parametrize(
        ("stage", "with_source_data"),
        [("validate", True), ("parse", True), ("render", False), ("index", False), ("all", True)],
    )
    def test_stage_matrix(self, workspace_env, tmp_path: Path, stage: str, with_source_data: bool, monkeypatch):
        monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "strict")
        sources = []
        if with_source_data:
            inbox = tmp_path / f"inbox-{stage}"
            inbox.mkdir(parents=True, exist_ok=True)
            _write_chatgpt_export(inbox / "conversations.json", f"conv-{stage}")
            sources = [Source(name=f"test-{stage}", path=inbox)]

        config = Config(
            sources=sources,
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )
        result = asyncio.run(run_sources(config=config, stage=stage))

        if stage == "validate":
            assert result.counts.get("validated", 0) >= 1
            assert result.counts["conversations"] == 0
            assert result.indexed is False
        elif stage == "parse":
            assert result.counts["conversations"] >= 1
            assert result.counts.get("rendered", 0) == 0
            assert result.indexed is False
        elif stage == "render":
            assert result.counts["conversations"] == 0
            assert result.counts["messages"] == 0
            assert result.counts.get("rendered", 0) == 0
            assert result.indexed is False
        elif stage == "index":
            assert result.counts["conversations"] == 0
            assert result.indexed is True
            assert result.index_error is None
        else:
            assert result.counts["conversations"] >= 1
            assert result.counts.get("rendered", 0) >= 1
            if result.indexed:
                assert result.index_error is None
            else:
                assert result.index_error is not None

    def test_plan_snapshot_is_persisted_without_affecting_drift(self, workspace_env):
        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )
        plan = PlanResult(
            timestamp=123,
            counts={"conversations": 10, "messages": 50, "attachments": 5},
            sources=["test"],
            cursors={},
        )

        result = asyncio.run(run_sources(config=config, stage="parse", plan=plan))
        latest = asyncio.run(latest_run())

        assert latest is not None
        assert latest.plan_snapshot is not None
        assert latest.plan_snapshot["counts"] == plan.counts
        assert result.drift["new"]["conversations"] == 0
        assert result.drift["removed"]["conversations"] == 0

    def test_drift_calculation_without_plan(self, workspace_env, tmp_path: Path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        _write_chatgpt_export(inbox / "conversations.json", "conv-new")
        config = Config(
            sources=[Source(name="test-drift", path=inbox)],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        result = asyncio.run(run_sources(config=config, stage="parse", plan=None))
        assert result.drift["new"]["conversations"] == result.counts["conversations"]

    def test_run_json_written(self, workspace_env):
        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )
        result = asyncio.run(run_sources(config=config, stage="parse"))
        runs_dir = workspace_env["archive_root"] / "runs"
        json_files = list(runs_dir.glob("run-*.json"))
        assert runs_dir.exists()
        assert json_files
        assert json.loads(json_files[0].read_text())["run_id"] == result.run_id

    def test_index_error_captured(self, workspace_env):
        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )
        with patch("polylogue.pipeline.services.indexing.IndexService.rebuild_index", new_callable=AsyncMock) as mock_rebuild:
            mock_rebuild.side_effect = Exception("Index rebuild failed")
            result = asyncio.run(run_sources(config=config, stage="index"))

        assert result.indexed is False
        assert result.index_error is not None
        assert "Index rebuild failed" in result.index_error

    def test_parse_stage_reuses_persisted_validation_status(self, workspace_env):
        from polylogue.storage.store import RawConversationRecord

        backend = create_backend(workspace_env["data_root"] / "polylogue" / "polylogue.db")
        raw_content = json.dumps(
            [
                {
                    "id": "conv-prevalidated",
                    "title": "Prevalidated",
                    "create_time": 1704067200,
                    "update_time": 1704067200,
                    "mapping": {
                        "root": {"id": "root", "message": None, "children": ["m1"]},
                        "m1": {
                            "id": "m1",
                            "message": {
                                "id": "m1",
                                "author": {"role": "user"},
                                "content": {"parts": ["hello"]},
                                "create_time": 1704067200,
                            },
                            "parent": "root",
                            "children": [],
                        },
                    },
                }
            ]
        ).encode("utf-8")
        raw_id = "raw-prevalidated"
        asyncio.run(
            backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=raw_id,
                    provider_name="chatgpt",
                    source_name="seeded",
                    source_path="/tmp/prevalidated.json",
                    raw_content=raw_content,
                    acquired_at="2026-03-05T00:00:00Z",
                )
            )
        )
        asyncio.run(backend.mark_raw_validated(raw_id, status="passed", provider="chatgpt", mode="strict"))

        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )
        result = asyncio.run(run_sources(config=config, stage="parse"))
        asyncio.run(backend.close())
        assert result.counts["conversations"] >= 1
