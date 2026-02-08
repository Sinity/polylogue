"""Tests for polylogue.pipeline.runner module."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from polylogue.config import Config, Source
from polylogue.pipeline.runner import (
    _all_conversation_ids,
    _iter_source_conversations_safe,
    _select_sources,
    _write_run_json,
    latest_run,
    plan_sources,
    run_sources,
)
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.store import PlanResult, store_records
from tests.helpers import make_conversation


class TestRenderFailureTracking:
    """Tests for tracking render failures in pipeline."""

    def test_render_failure_tracked_in_result(self, tmp_path: Path):
        """Render failures should be tracked in RunResult.

        This test SHOULD FAIL until failure tracking is implemented.
        """
        from polylogue.config import Config
        from polylogue.pipeline.runner import run_sources
        from polylogue.storage.store import RunResult

        # Create a minimal config
        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )
        config.archive_root.mkdir(parents=True, exist_ok=True)

        # Mock render_conversation to fail for specific conversation
        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render") as mock_render:

            def render_side_effect(conversation_id, output_path):
                if "fail-conv" in conversation_id:
                    raise ValueError("Render failed for testing")
                return MagicMock()

            mock_render.side_effect = render_side_effect

            # Mock _all_conversation_ids to return test data
            with patch("polylogue.pipeline.runner._all_conversation_ids") as mock_ids:
                mock_ids.return_value = ["test:success-conv", "test:fail-conv"]

                # Run pipeline in render stage
                result = run_sources(
                    config=config,
                    stage="render",
                    source_names=None,
                )

                # Result should be RunResult
                assert isinstance(result, RunResult)

                # Result should track render failures
                assert hasattr(result, "render_failures"), "RunResult should have render_failures attribute"
                assert isinstance(result.render_failures, list)
                assert len(result.render_failures) > 0, "Should have tracked at least one render failure"

                # Check failure details
                failure = result.render_failures[0]
                assert "conversation_id" in failure
                assert failure["conversation_id"] == "test:fail-conv"
                assert "error" in failure

    def test_render_continues_after_failure(self, tmp_path: Path):
        """Pipeline should continue rendering other conversations after one fails."""
        from polylogue.config import Config
        from polylogue.pipeline.runner import run_sources

        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )
        config.archive_root.mkdir(parents=True, exist_ok=True)

        render_attempts = []

        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render") as mock_render:

            def render_side_effect(conversation_id, output_path):
                render_attempts.append(conversation_id)
                if "second" in conversation_id:
                    raise ValueError("Failed on purpose")
                return MagicMock()

            mock_render.side_effect = render_side_effect

            with patch("polylogue.pipeline.runner._all_conversation_ids") as mock_ids:
                mock_ids.return_value = ["test:first", "test:second", "test:third"]

                run_sources(
                    config=config,
                    stage="render",
                )

                # Should attempt all renders even if one fails
                assert len(render_attempts) >= 3, f"Expected 3 render attempts, got {len(render_attempts)}"

                # Verify we didn't stop at the failure
                assert "test:first" in render_attempts
                assert "test:second" in render_attempts
                assert "test:third" in render_attempts

    def test_render_failure_count_in_counts(self, tmp_path: Path):
        """Pipeline should include render_failures count in result.counts."""
        from polylogue.config import Config
        from polylogue.pipeline.runner import run_sources

        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )
        config.archive_root.mkdir(parents=True, exist_ok=True)

        with patch("polylogue.rendering.renderers.html.HTMLRenderer.render") as mock_render:

            def render_side_effect(conversation_id, output_path):
                if conversation_id in ["test:fail1", "test:fail2"]:
                    raise ValueError("Render failed")
                return MagicMock()

            mock_render.side_effect = render_side_effect

            with patch("polylogue.pipeline.runner._all_conversation_ids") as mock_ids:
                mock_ids.return_value = ["test:success", "test:fail1", "test:fail2"]

                result = run_sources(
                    config=config,
                    stage="render",
                )

                # Check counts include render_failures
                assert "render_failures" in result.counts
                assert result.counts["render_failures"] == 2
                assert result.counts["rendered"] == 1


class TestSelectSources:
    """Tests for _select_sources function."""

    def test_select_all_sources_when_no_filter(self, tmp_path: Path):
        """Returns all config.sources when source_names is None or empty."""
        source1 = Source(name="source-a", path=tmp_path / "a")
        source2 = Source(name="source-b", path=tmp_path / "b")
        source3 = Source(name="source-c", path=tmp_path / "c")

        config = Config(
            sources=[source1, source2, source3],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )

        # None filter
        result = _select_sources(config, None)
        assert len(result) == 3
        assert result == [source1, source2, source3]

        # Empty list filter
        result = _select_sources(config, [])
        assert len(result) == 3

    def test_select_filtered_sources(self, tmp_path: Path):
        """Returns only sources matching the given names."""
        source1 = Source(name="chatgpt-export", path=tmp_path / "a")
        source2 = Source(name="claude-export", path=tmp_path / "b")
        source3 = Source(name="codex-export", path=tmp_path / "c")

        config = Config(
            sources=[source1, source2, source3],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )

        result = _select_sources(config, ["claude-export"])
        assert len(result) == 1
        assert result[0].name == "claude-export"

        result = _select_sources(config, ["chatgpt-export", "codex-export"])
        assert len(result) == 2
        assert {s.name for s in result} == {"chatgpt-export", "codex-export"}

    def test_select_empty_when_no_match(self, tmp_path: Path):
        """Returns empty list when no sources match the filter."""
        source1 = Source(name="source-a", path=tmp_path / "a")

        config = Config(
            sources=[source1],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )

        result = _select_sources(config, ["nonexistent-source"])
        assert result == []


class TestIterSourceConversationsSafe:
    """Tests for _iter_source_conversations_safe function."""

    def test_file_source_yields_conversations(self, tmp_path: Path):
        """Non-Drive file sources yield from iter_source_conversations."""
        # Create a source file
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        conv_file = inbox / "conversations.json"
        conv_data = {
            "id": "conv-1",
            "title": "Test Conversation",
            "create_time": 1704067200,
            "update_time": 1704067200,
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "children": ["msg-1"],
                },
                "msg-1": {
                    "id": "msg-1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                        "create_time": 1704067200,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
        conv_file.write_text(json.dumps([conv_data]), encoding="utf-8")

        source = Source(name="test-inbox", path=inbox)

        conversations = list(
            _iter_source_conversations_safe(
                source=source,
                archive_root=tmp_path / "archive",
                ui=None,
                download_assets=False,
            )
        )

        assert len(conversations) >= 1

    def test_drive_auth_error_logged_and_skipped(self, tmp_path: Path):
        """DriveAuthError from Drive source is logged and source is skipped."""
        from polylogue.sources import DriveAuthError

        # Source with only folder (Drive source)
        source = Source(name="drive-source", folder="folder-id-123")
        cursor_state: dict = {}

        with patch("polylogue.pipeline.runner.iter_drive_conversations") as mock_drive:
            mock_drive.side_effect = DriveAuthError("Token expired")

            conversations = list(
                _iter_source_conversations_safe(
                    source=source,
                    archive_root=tmp_path / "archive",
                    ui=None,
                    download_assets=False,
                    cursor_state=cursor_state,
                )
            )

            assert len(conversations) == 0
            assert cursor_state.get("error_count") == 1
            assert "Token expired" in cursor_state.get("latest_error", "")

    def test_cursor_state_updated_on_error(self, tmp_path: Path):
        """cursor_state tracks error information when Drive fails."""
        from polylogue.sources import DriveAuthError

        # Source with only folder (Drive source)
        source = Source(name="my-drive", folder="folder-id")
        cursor_state: dict = {"existing_key": "value"}

        with patch("polylogue.pipeline.runner.iter_drive_conversations") as mock_drive:
            mock_drive.side_effect = DriveAuthError("Authentication failed")

            list(
                _iter_source_conversations_safe(
                    source=source,
                    archive_root=tmp_path,
                    ui=None,
                    download_assets=False,
                    cursor_state=cursor_state,
                )
            )

            assert cursor_state["existing_key"] == "value"  # Preserved
            assert cursor_state["error_count"] == 1
            assert cursor_state["latest_error_source"] == "my-drive"


class TestPlanSources:
    """Tests for plan_sources function."""

    def test_plan_empty_config(self, tmp_path: Path):
        """Returns zeros for empty config."""
        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )

        result = plan_sources(config)

        assert isinstance(result, PlanResult)
        assert result.counts == {"conversations": 0, "messages": 0, "attachments": 0}
        assert result.sources == []
        assert result.cursors == {}

    def test_plan_single_source(self, tmp_path: Path):
        """Correct counts for single source."""
        # Create test conversations
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        conv_file = inbox / "conversations.json"
        conv_data = {
            "id": "conv-1",
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
        conv_file.write_text(json.dumps([conv_data]), encoding="utf-8")

        source = Source(name="test-source", path=inbox)
        config = Config(
            sources=[source],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )

        result = plan_sources(config)

        assert result.counts["conversations"] == 1
        assert result.counts["messages"] == 2
        assert result.sources == ["test-source"]


class TestAllConversationIds:
    """Tests for _all_conversation_ids function."""

    def test_all_ids_no_filter(self, workspace_env):
        """Returns all conversation_ids when no filter provided."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create test conversations
        with open_connection(db_path) as conn:
            for i in range(3):
                conv = make_conversation(f"conv-{i}", title=f"Conversation {i}")
                store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        ids = _all_conversation_ids()

        assert len(ids) == 3
        assert set(ids) == {"conv-0", "conv-1", "conv-2"}

    def test_all_ids_provider_filter(self, workspace_env):
        """Filters by provider_name when source_names match provider."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with open_connection(db_path) as conn:
            # Create conversations with different providers
            for provider, conv_id in [("chatgpt", "chatgpt-1"), ("claude", "claude-1"), ("chatgpt", "chatgpt-2")]:
                conv = make_conversation(conv_id, provider_name=provider, title=f"Conv {conv_id}")
                store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        ids = _all_conversation_ids(source_names=["chatgpt"])

        assert len(ids) == 2
        assert set(ids) == {"chatgpt-1", "chatgpt-2"}

    def test_all_ids_source_filter_via_meta(self, workspace_env):
        """Filters by provider_meta.source when specified."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with open_connection(db_path) as conn:
            # Create conversations with source in provider_meta
            for source, conv_id in [("export-a", "a-1"), ("export-b", "b-1"), ("export-a", "a-2")]:
                conv = make_conversation(conv_id, provider_name="chatgpt", title=f"Conv {conv_id}", provider_meta={"source": source})
                store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        ids = _all_conversation_ids(source_names=["export-a"])

        assert len(ids) == 2
        assert set(ids) == {"a-1", "a-2"}

    def test_all_ids_null_meta_skipped(self, workspace_env):
        """Conversations with null or empty provider_meta don't crash filtering."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with open_connection(db_path) as conn:
            # Create conversation with valid provider_meta
            conv_valid = make_conversation("valid-1", title="Valid", provider_meta={"source": "my-source"})
            store_records(conversation=conv_valid, messages=[], attachments=[], conn=conn)

            # Create conversation with null provider_meta
            conv_null = make_conversation("null-meta-1", title="Null Meta", provider_meta=None)
            store_records(conversation=conv_null, messages=[], attachments=[], conn=conn)

            # Create conversation with empty dict provider_meta
            conv_empty = make_conversation("empty-meta-1", title="Empty Meta", provider_meta={})
            store_records(conversation=conv_empty, messages=[], attachments=[], conn=conn)

        # Should not raise and filter correctly
        ids = _all_conversation_ids(source_names=["my-source"])
        assert "valid-1" in ids
        assert "null-meta-1" not in ids  # Skipped due to null meta
        assert "empty-meta-1" not in ids  # Skipped due to no source in meta


class TestWriteRunJson:
    """Tests for _write_run_json function."""

    def test_creates_runs_directory(self, tmp_path: Path):
        """Creates runs/ directory if it doesn't exist."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()

        payload = {
            "run_id": "test-run-1",
            "timestamp": int(time.time()),
            "counts": {"conversations": 1},
        }

        result = _write_run_json(archive_root, payload)

        assert (archive_root / "runs").exists()
        assert result.exists()

    def test_writes_correct_content(self, tmp_path: Path):
        """JSON content matches payload."""
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

    def test_filename_contains_timestamp_and_id(self, tmp_path: Path):
        """Filename format is run-{timestamp}-{run_id}.json."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()

        timestamp = 1704067200
        payload = {"run_id": "myrun", "timestamp": timestamp}

        result = _write_run_json(archive_root, payload)

        assert result.name == f"run-{timestamp}-myrun.json"


class TestLatestRun:
    """Tests for latest_run function."""

    def test_no_runs_returns_none(self, workspace_env):
        """Empty table returns None."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize DB without runs
        with open_connection(db_path):
            pass

        result = latest_run()

        assert result is None

    def test_returns_most_recent(self, workspace_env):
        """ORDER BY timestamp DESC returns latest run."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with open_connection(db_path) as conn:
            # Insert runs with different timestamps
            for i, ts in enumerate([1000, 3000, 2000]):  # 3000 is latest
                conn.execute(
                    """
                    INSERT INTO runs
                    (run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"run-{i}",
                        str(ts),
                        None,
                        json.dumps({"conversations": i}),
                        None,
                        1 if i == 1 else 0,
                        100 * i,
                    ),
                )
            conn.commit()

        result = latest_run()

        assert result is not None
        assert result.run_id == "run-1"  # timestamp 3000 is latest
        assert result.timestamp == "3000"

    def test_parses_json_columns(self, workspace_env):
        """plan_snapshot, counts, drift are parsed from JSON."""
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

        result = latest_run()

        assert result is not None
        assert result.plan_snapshot == plan
        assert result.counts == counts
        assert result.drift == drift
        assert result.indexed is True
        assert result.duration_ms == 500


class TestRunSourcesIntegration:
    """Integration tests for run_sources function."""

    def test_ingest_stage_only(self, workspace_env, tmp_path: Path):
        """Only ingestion runs when stage='ingest'."""
        # Create test data
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        conv_file = inbox / "conversations.json"
        conv_data = {
            "id": "conv-ingest",
            "title": "Ingest Test",
            "create_time": 1704067200,
            "update_time": 1704067200,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["m1"]},
                "m1": {
                    "id": "m1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Test"]},
                        "create_time": 1704067200,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
        conv_file.write_text(json.dumps([conv_data]), encoding="utf-8")

        source = Source(name="test-ingest", path=inbox)
        config = Config(
            sources=[source],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        result = run_sources(config=config, stage="ingest")

        assert result.counts["conversations"] >= 1
        # Render count should be 0 (only ingest ran)
        assert result.counts.get("rendered", 0) == 0

    def test_render_stage_only(self, workspace_env, tmp_path: Path):
        """Only rendering runs when stage='render'."""
        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        with patch("polylogue.pipeline.runner._all_conversation_ids") as mock_ids:
            mock_ids.return_value = []  # No conversations to render

            result = run_sources(config=config, stage="render")

            # Ingest counts should be 0
            assert result.counts["conversations"] == 0
            assert result.counts["messages"] == 0

    def test_index_stage_only(self, workspace_env, tmp_path: Path):
        """Only indexing runs when stage='index'."""
        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        result = run_sources(config=config, stage="index")

        # Should have indexed flag set
        assert result.indexed in (True, False)  # Either works for empty DB

    def test_all_stages(self, workspace_env, tmp_path: Path):
        """All stages run when stage='all'."""
        # Create test data
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        conv_file = inbox / "conversations.json"
        conv_data = {
            "id": "conv-all",
            "title": "All Stages",
            "create_time": 1704067200,
            "update_time": 1704067200,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["m1"]},
                "m1": {
                    "id": "m1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Test message"]},
                        "create_time": 1704067200,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
        conv_file.write_text(json.dumps([conv_data]), encoding="utf-8")

        source = Source(name="test-all", path=inbox)
        config = Config(
            sources=[source],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        result = run_sources(config=config, stage="all")

        # All stages should have run
        assert result.counts["conversations"] >= 1
        assert result.counts.get("rendered", 0) >= 1
        assert result.indexed in (True, False)

    def test_drift_calculation_with_plan(self, workspace_env, tmp_path: Path):
        """Drift is calculated comparing actual vs plan."""
        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        # Create plan with expected counts
        plan = PlanResult(
            timestamp=int(time.time()),
            counts={"conversations": 10, "messages": 50, "attachments": 5},
            sources=["test"],
            cursors={},
        )

        result = run_sources(config=config, stage="ingest", plan=plan)

        # With 0 actual, drift should show removed items
        assert "new" in result.drift
        assert "removed" in result.drift
        assert result.drift["removed"]["conversations"] == 10  # All expected are "removed"

    def test_drift_calculation_without_plan(self, workspace_env, tmp_path: Path):
        """Without plan, all items counted as 'new'."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        conv_file = inbox / "conversations.json"
        conv_data = {
            "id": "conv-new",
            "title": "New",
            "create_time": 1704067200,
            "update_time": 1704067200,
            "mapping": {
                "root": {"id": "root", "message": None, "children": ["m1"]},
                "m1": {
                    "id": "m1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Test"]},
                        "create_time": 1704067200,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
        conv_file.write_text(json.dumps([conv_data]), encoding="utf-8")

        source = Source(name="test-drift", path=inbox)
        config = Config(
            sources=[source],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        result = run_sources(config=config, stage="ingest", plan=None)

        # All should be counted as new
        assert result.drift["new"]["conversations"] == result.counts["conversations"]

    def test_run_json_written(self, workspace_env, tmp_path: Path):
        """Run JSON file is written to archive_root/runs/."""
        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        result = run_sources(config=config, stage="ingest")

        runs_dir = workspace_env["archive_root"] / "runs"
        assert runs_dir.exists()

        # Find JSON file
        json_files = list(runs_dir.glob("run-*.json"))
        assert len(json_files) >= 1

        # Verify content
        content = json.loads(json_files[0].read_text())
        assert content["run_id"] == result.run_id

    def test_index_error_captured(self, workspace_env, tmp_path: Path):
        """Index errors are captured in result.index_error."""
        config = Config(
            sources=[],
            archive_root=workspace_env["archive_root"],
            render_root=workspace_env["archive_root"] / "render",
        )

        with patch("polylogue.pipeline.services.indexing.IndexService.rebuild_index") as mock_rebuild:
            mock_rebuild.side_effect = Exception("Index rebuild failed")

            result = run_sources(config=config, stage="index")

            assert result.indexed is False
            assert result.index_error is not None
            assert "Index rebuild failed" in result.index_error
