"""Focused tests for end-to-end runner stage execution."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import Config, Source
from polylogue.pipeline.runner import _select_sources, latest_run, plan_sources, run_sources
from polylogue.sources.parsers.base import RawConversationData
from polylogue.storage.backends import create_backend
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.state_views import PlanResult
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
        from polylogue.storage.state_views import RunResult

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

        # Stages are now independent: validate/parse don't re-run predecessors.
        # Pre-populate the pipeline backlog so each stage has work to find.
        if stage == "validate" and sources:
            asyncio.run(run_sources(config=config, stage="acquire"))
        elif stage == "parse" and sources:
            asyncio.run(run_sources(config=config, stage="acquire"))
            asyncio.run(run_sources(config=config, stage="validate"))

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

        # Stages are independent: populate pipeline backlog before testing parse stage
        asyncio.run(run_sources(config=config, stage="acquire"))
        asyncio.run(run_sources(config=config, stage="validate"))
        result = asyncio.run(run_sources(config=config, stage="parse", plan=None))
        assert result.drift["new"]["conversations"] == result.counts["new_conversations"]
        assert result.counts["conversations"] == (
            result.counts["new_conversations"] + result.counts["changed_conversations"]
        )

    def test_changed_conversation_is_not_reported_as_new(self, workspace_env, tmp_path: Path):
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        export_path = inbox / "conversations.json"
        _write_chatgpt_export(export_path, "conv-update")
        config = Config(
            sources=[Source(name="test-drift", path=inbox)],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        asyncio.run(run_sources(config=config, stage="all"))

        payload = json.loads(export_path.read_text())
        payload[0]["title"] = "Updated title"
        export_path.write_text(json.dumps(payload))

        second = asyncio.run(run_sources(config=config, stage="all"))

        assert second.counts["conversations"] == 1
        assert second.counts["new_conversations"] == 0
        assert second.counts["changed_conversations"] == 1
        assert second.drift["new"]["conversations"] == 0
        assert second.drift["changed"]["conversations"] == 1

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
                    blob_size=len(raw_content),
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


# =====================================================================
# Merged from test_acquisition_service.py (source acquisition/running)
# =====================================================================


class TestAcquisitionServiceAcquireSources:
    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(db_path=tmp_path / "test.db")

    async def test_acquire_empty_sources(self, backend: SQLiteBackend):
        from polylogue.pipeline.services.acquisition import AcquisitionService

        result = await AcquisitionService(backend=backend).acquire_sources([])
        assert all(result.counts[key] == 0 for key in ["acquired", "skipped", "errors"])
        assert result.raw_ids == []

    @patch("polylogue.pipeline.services.acquisition.iter_source_raw_data")
    async def test_progress_callback_called(self, mock_iter, backend: SQLiteBackend):
        from polylogue.pipeline.services.acquisition import AcquisitionService

        raw_data = RawConversationData(
            raw_bytes=b'{"id": "test"}',
            source_path="/tmp/test.json",
            source_index=0,
        )
        mock_iter.return_value = iter([raw_data])
        backend.queries.get_known_source_mtimes = AsyncMock(return_value={})
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
        from polylogue.pipeline.services.acquisition import AcquisitionService

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
        from polylogue.pipeline.services.acquisition import AcquisitionService

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
        from polylogue.storage.blob_store import load_raw_content
        raw_bytes = load_raw_content(result.raw_ids[0])
        data = json.loads(raw_bytes)
        assert stored.provider_name == "chatgpt"
        assert isinstance(data, list)
        assert data[0]["id"] == "conv-1"
        assert data[0]["title"] == "Test Chat"

    async def test_acquire_multiple_json_files(self, tmp_path: Path):
        from polylogue.pipeline.services.acquisition import AcquisitionService

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

    async def test_acquire_claude_code_sidecars_into_artifact_ledger(self, tmp_path: Path):
        from polylogue.pipeline.services.acquisition import AcquisitionService
        from polylogue.schemas.verification_artifacts import list_artifact_observation_rows
        from polylogue.schemas.verification_requests import ArtifactObservationQuery

        session_dir = tmp_path / "claude-code" / "project-a" / "session-1"
        subagents_dir = session_dir / "subagents"
        subagents_dir.mkdir(parents=True)

        (session_dir / "session-1.jsonl").write_text(
            '{"type":"session_meta"}\n{"type":"response_item","payload":{"type":"message"}}\n',
            encoding="utf-8",
        )
        (session_dir / "sessions-index.json").write_text(
            json.dumps({"session-1": {"summary": "Session 1"}}),
            encoding="utf-8",
        )
        (session_dir / "bridge-pointer.json").write_text(
            json.dumps({"sessionId": "session-1", "environmentId": "env-1", "source": "project-a"}),
            encoding="utf-8",
        )
        (subagents_dir / "agent-a123.meta.json").write_text(
            json.dumps({"agentType": "general-purpose"}),
            encoding="utf-8",
        )
        (subagents_dir / "agent-a123.jsonl").write_text(
            '{"type":"session_meta"}\n{"type":"response_item","payload":{"type":"message"}}\n',
            encoding="utf-8",
        )

        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = await AcquisitionService(backend=backend).acquire_sources(
            [Source(name="claude-code", path=session_dir)]
        )

        assert result.counts["acquired"] == 4
        observations = list_artifact_observation_rows(
            db_path=backend.db_path,
            request=ArtifactObservationQuery(),
        )
        assert len(observations) == 5
        assert {row.artifact_kind for row in observations} == {
            "conversation_record_stream",
            "session_index",
            "bridge_pointer",
            "agent_sidecar_meta",
            "subagent_conversation_stream",
        }


# =====================================================================
# Merged from test_runner_preview.py (runner/source tests)
# =====================================================================


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


# =====================================================================
# Merged from test_runner_history.py (runner/source tests)
# =====================================================================


class TestWriteRunJson:
    def test_creates_runs_directory(self, tmp_path):
        import time

        from polylogue.pipeline.runner import _write_run_json

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        payload = {"run_id": "test-run-1", "timestamp": int(time.time()), "counts": {"conversations": 1}}

        result = _write_run_json(archive_root, payload)
        assert (archive_root / "runs").exists()
        assert result.exists()

    def test_writes_correct_content(self, tmp_path):
        import time

        from polylogue.pipeline.runner import _write_run_json

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        timestamp = int(time.time())
        payload = {
            "run_id": "abc123",
            "timestamp": timestamp,
            "counts": {"conversations": 10, "messages": 50},
            "indexed": True,
        }

        result_path = _write_run_json(archive_root, payload)
        content = json.loads(result_path.read_text())
        assert content["run_id"] == "abc123"
        assert content["timestamp"] == timestamp
        assert content["counts"] == {"conversations": 10, "messages": 50}
        assert content["indexed"] is True

    def test_filename_contains_timestamp_and_id(self, tmp_path):
        from polylogue.pipeline.runner import _write_run_json

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        result = _write_run_json(archive_root, {"run_id": "myrun", "timestamp": 1704067200})
        assert result.name == "run-1704067200-myrun.json"


class TestLatestRun:
    def test_no_runs_returns_none(self, workspace_env):
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open_connection(db_path):
            pass
        assert asyncio.run(latest_run()) is None

    def test_returns_most_recent(self, workspace_env):
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open_connection(db_path) as conn:
            for index, timestamp in enumerate([1000, 3000, 2000]):
                conn.execute(
                    """
                    INSERT INTO runs
                    (run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"run-{index}",
                        str(timestamp),
                        None,
                        json.dumps({"conversations": index}),
                        None,
                        1 if index == 1 else 0,
                        100 * index,
                    ),
                )
            conn.commit()

        result = asyncio.run(latest_run())
        assert result is not None
        assert result.run_id == "run-1"
        assert result.timestamp == "3000"

    def test_parses_json_columns(self, workspace_env):
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        plan = {"conversations": 5, "messages": 20}
        counts = {"conversations": 4, "messages": 18, "rendered": 4}
        drift = {"new": {"conversations": 1}, "removed": {"conversations": 2}}

        with open_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs
                (run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("run-json", "5000", json.dumps(plan), json.dumps(counts), json.dumps(drift), 1, 500),
            )
            conn.commit()

        result = asyncio.run(latest_run())
        assert result is not None
        assert result.plan_snapshot == plan
        assert result.counts == counts
        assert result.drift == drift
        assert result.indexed is True
        assert result.duration_ms == 500
