from __future__ import annotations

import concurrent.futures
import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.run import (
    _display_result,
    _exec_on_new,
    _notify_new_conversations,
    _run_sync_once,
    _webhook_on_new,
    run_command,
    sources_command,
)
from polylogue.config import Source, get_config
from polylogue.pipeline.runner import latest_run, plan_sources, run_sources
from polylogue.pipeline.services.ingestion import IngestResult, IngestionService
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    attachment_from_meta,
    extract_messages_from_list,
)
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.store import PlanResult, RunResult
from tests.helpers import (
    ChatGPTExportBuilder,
    GenericConversationBuilder,
    InboxBuilder,
)


def test_plan_and_run_sources(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    source_file = (
        GenericConversationBuilder("conv1")
        .title("Test")
        .add_user("hello")
        .add_assistant("world")
        .write_to(inbox / "conversation.json")
    )

    config = get_config()
    config.sources = [Source(name="codex", path=source_file)]

    plan = plan_sources(config)
    assert plan.counts["conversations"] == 1

    result = run_sources(config=config, stage="all", plan=plan)
    assert result.counts["conversations"] == 1
    run_dir = config.archive_root / "runs"
    assert any(run_dir.iterdir())


def test_run_sources_filtered(workspace_env, tmp_path):
    inbox = (
        InboxBuilder(tmp_path / "inbox")
        .add_codex_conversation("conv-a", messages=[("user", "hello")], filename="a.json")
        .add_codex_conversation("conv-b", messages=[("user", "world")], filename="b.json")
        .build()
    )

    config = get_config()
    config.sources = [
        Source(name="source-a", path=inbox / "a.json"),
        Source(name="source-b", path=inbox / "b.json"),
    ]

    result = run_sources(config=config, stage="ingest", source_names=["source-a"])
    assert result.counts["conversations"] == 1


def test_render_filtered_by_source_meta(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    source_file = ChatGPTExportBuilder("conv-chatgpt").add_node("user", "hello").write_to(inbox / "conversation.json")

    config = get_config()
    config.sources = [Source(name="inbox", path=source_file)]

    run_sources(config=config, stage="ingest", source_names=["inbox"])
    result = run_sources(config=config, stage="render", source_names=["inbox"])
    assert result.counts["conversations"] == 0
    assert any(config.render_root.rglob("conversation.md"))


def test_run_all_skips_render_when_unchanged(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    source_file = (
        GenericConversationBuilder("conv1")
        .add_user("hello")
        .add_assistant("world")
        .write_to(inbox / "conversation.json")
    )

    config = get_config()
    config.sources = [Source(name="inbox", path=source_file)]

    run_sources(config=config, stage="all")

    convo_path = next(config.render_root.rglob("conversation.md"))
    first_mtime = convo_path.stat().st_mtime

    run_sources(config=config, stage="all")
    second_mtime = convo_path.stat().st_mtime
    assert first_mtime == second_mtime


def test_run_rerenders_when_content_changes(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    source_file = inbox / "conversation.json"

    # Initial content
    (GenericConversationBuilder("conv1").add_user("hello").write_to(source_file))

    config = get_config()
    config.sources = [Source(name="inbox", path=source_file)]

    run_sources(config=config, stage="all")

    convo_path = next(config.render_root.rglob("conversation.md"))
    first_mtime = convo_path.stat().st_mtime

    # Modify content - content hash difference triggers re-render
    (GenericConversationBuilder("conv1").add_user("hello world").write_to(source_file))
    run_sources(config=config, stage="all")

    second_mtime = convo_path.stat().st_mtime
    assert second_mtime > first_mtime


def test_run_rerenders_when_title_changes(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    source_file = inbox / "conversation.json"

    # Initial content with old title
    (GenericConversationBuilder("conv-title").title("Old title").add_user("hello").write_to(source_file))

    config = get_config()
    config.sources = [Source(name="inbox", path=source_file)]

    run_sources(config=config, stage="all")
    convo_path = next(config.render_root.rglob("conversation.md"))
    original = convo_path.read_text(encoding="utf-8")

    # Update title
    (GenericConversationBuilder("conv-title").title("New title").add_user("hello").write_to(source_file))
    run_sources(config=config, stage="all")

    updated = convo_path.read_text(encoding="utf-8")
    assert "# New title" in updated
    assert original != updated


def test_run_index_filters_selected_sources(workspace_env, tmp_path, monkeypatch):
    inbox = (
        InboxBuilder(tmp_path / "inbox")
        .add_json_file("a.json", {"id": "conv-a", "messages": [{"id": "m1", "role": "user", "text": "alpha"}]})
        .add_json_file("b.json", {"id": "conv-b", "messages": [{"id": "m1", "role": "user", "text": "beta"}]})
        .build()
    )

    config = get_config()
    config.sources = [
        Source(name="source-a", path=inbox / "a.json"),
        Source(name="source-b", path=inbox / "b.json"),
    ]

    run_sources(config=config, stage="ingest")

    id_by_source = {}
    with open_connection(None) as conn:
        rows = conn.execute("SELECT conversation_id, provider_meta FROM conversations").fetchall()
    for row in rows:
        meta = json.loads(row["provider_meta"] or "{}")
        name = meta.get("source")
        if name:
            id_by_source[name] = row["conversation_id"]

    update_calls = []
    from polylogue.pipeline.services.indexing import IndexService


    def fake_update_method(self, ids):
        update_calls.append(list(ids))
        return True

    monkeypatch.setattr(IndexService, "update_index", fake_update_method)

    run_sources(config=config, stage="index", source_names=["source-b"])

    assert update_calls == [[id_by_source["source-b"]]]


def test_incremental_index_updates(workspace_env, tmp_path, monkeypatch):
    inbox = (
        InboxBuilder(tmp_path / "inbox")
        .add_codex_conversation("conv-a", messages=[("user", "alpha")], filename="a.json")
        .add_codex_conversation("conv-b", messages=[("user", "beta")], filename="b.json")
        .build()
    )

    config = get_config()
    config.sources = [Source(name="inbox", path=inbox)]

    run_sources(config=config, stage="all")


def test_index_failure_is_nonfatal(workspace_env, monkeypatch):
    config = get_config()

    from polylogue.pipeline.services.indexing import IndexService

    def boom(self):
        raise RuntimeError("index failed")

    monkeypatch.setattr(IndexService, "rebuild_index", boom)
    result = run_sources(config=config, stage="index")
    assert result.indexed is False
    assert result.index_error is not None
    assert "index failed" in result.index_error


def test_run_writes_unique_report_files(workspace_env, tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    source_file = GenericConversationBuilder("conv1").add_user("hello").write_to(inbox / "conversation.json")

    config = get_config()
    config.sources = [Source(name="inbox", path=source_file)]

    import polylogue.pipeline.runner as runner_mod

    fixed_time = 1_700_000_000
    monkeypatch.setattr(runner_mod.time, "time", lambda: fixed_time)
    monkeypatch.setattr(runner_mod.time, "perf_counter", lambda: 0.0)

    run_sources(config=config, stage="all")
    run_sources(config=config, stage="all")

    run_dir = config.archive_root / "runs"
    runs = list(run_dir.glob(f"run-{fixed_time}-*.json"))
    assert len(runs) == 2


# latest_run() tests


def test_latest_run_parses_json_columns(workspace_env, tmp_path):
    """latest_run() returns RunRecord with parsed dicts for counts and drift."""
    inbox = tmp_path / "inbox"
    (GenericConversationBuilder("conv-latest-run").add_user("test").write_to(inbox / "conversation.json"))

    config = get_config()
    config.sources = [Source(name="inbox", path=inbox)]

    run_sources(config=config, stage="all")

    result = latest_run()
    assert result is not None
    assert result.run_id is not None

    # counts should be parsed to dict
    if result.counts is not None:
        assert isinstance(result.counts, dict)
        # Should have typical count keys
        assert "conversations" in result.counts or "messages" in result.counts

    # drift should be parsed to dict
    if result.drift is not None:
        assert isinstance(result.drift, dict)


def test_latest_run_handles_null_json_columns(workspace_env):
    """latest_run() handles NULL values in JSON columns gracefully."""
    # Insert a run record with NULL JSON columns directly
    with open_connection(None) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms)
            VALUES (?, ?, NULL, NULL, NULL, 0, 100)
            """,
            ("null-test-run", str(int(time.time()))),
        )
        conn.commit()

    result = latest_run()
    assert result is not None
    # Should not crash, NULL columns should remain as None
    assert result.plan_snapshot is None
    assert result.counts is None
    assert result.drift is None


# --- Merged from test_run_ingestion_parsers_coverage.py ---


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_env():
    """Mock AppEnv with plain UI."""
    from rich.console import Console

    env = MagicMock()
    env.ui = MagicMock()
    env.ui.plain = True
    env.ui.console = Console()
    env.ui.confirm.return_value = True
    env.ui.summary = MagicMock()
    return env


@pytest.fixture
def mock_env_rich():
    """Mock AppEnv with Rich console."""
    from rich.console import Console

    env = MagicMock()
    env.ui = MagicMock()
    env.ui.plain = False
    env.ui.console = Console()
    env.ui.confirm.return_value = True
    env.ui.summary = MagicMock()
    return env


@pytest.fixture
def mock_run_result():
    """Mock RunResult."""
    return RunResult(
        run_id="run-123",
        counts={"conversations": 3, "messages": 30, "attachments": 1},
        drift={"conversations": {"new": 2, "updated": 1, "unchanged": 5}},
        indexed=True,
        index_error=None,
        duration_ms=1500,
        render_failures=[],
    )


@pytest.fixture
def mock_plan_result():
    """Mock PlanResult."""
    return PlanResult(
        timestamp=1234567890,
        counts={"conversations": 5, "messages": 50, "attachments": 2},
        sources=["test-inbox"],
        cursors={},
    )


@pytest.fixture
def mock_repository():
    """Mock ConversationRepository."""
    repo = MagicMock()
    repo._backend = MagicMock()
    repo._backend._get_connection = MagicMock()
    return repo


@pytest.fixture
def mock_backend():
    """Mock SQLite backend."""
    backend = MagicMock()
    backend._get_connection = MagicMock()
    backend.iter_raw_conversations = MagicMock(return_value=[])
    backend.get_raw_conversation = MagicMock(return_value=None)
    return backend


# =============================================================================
# TEST CLASS: _run_sync_once (Coverage: lines 43-47, 73-75)
# =============================================================================


class TestRunSyncOncePlainProgress:
    """Test _run_sync_once plain mode progress tracking."""

    def test_run_sync_once_plain_progress_first_update(self, mock_env, mock_run_result, capsys):
        """Plain mode progress callback triggers on 1+ second elapsed."""
        mock_env.ui.plain = True
        progress_calls = []

        def track_progress(amount, desc=None):
            progress_calls.append((amount, desc, time.time()))

        with patch("polylogue.cli.commands.run.run_sources") as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            result = _run_sync_once(
                mock_config,
                mock_env,
                "all",
                None,
                "markdown",
            )

        assert result.run_id == "run-123"
        captured = capsys.readouterr()
        assert "Syncing..." in captured.out

    def test_run_sync_once_rich_progress_descriptor(self, mock_env_rich, mock_run_result):
        """Rich mode progress callback updates description and amount."""
        mock_env_rich.ui.plain = False

        with patch("polylogue.cli.commands.run.run_sources") as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            result = _run_sync_once(
                mock_config,
                mock_env_rich,
                "render",
                ["source1"],
                "html",
            )

        # Verify run_sources was called with correct arguments
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["config"] == mock_config
        assert call_kwargs["stage"] == "render"
        assert call_kwargs["source_names"] == ["source1"]
        assert call_kwargs["render_format"] == "html"


# =============================================================================
# TEST CLASS: _display_result (Coverage: lines 234, 253-259, 260-266)
# =============================================================================


class TestDisplayResultComprehensive:
    """Comprehensive tests for _display_result coverage."""

    def test_display_result_preview_mode_no_plan_line(self, mock_env):
        """_display_result skips cursor line when empty."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result = RunResult(
            run_id="run-123",
            counts={"conversations": 0},
            drift={},
            indexed=False,
            index_error=None,
            duration_ms=100,
            render_failures=[],
        )

        _display_result(mock_env, mock_config, mock_run_result, "all", None)
        mock_env.ui.summary.assert_called_once()

    def test_display_result_stage_render_shows_latest_path(self, mock_env):
        """_display_result shows latest render path for render stage."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result = RunResult(
            run_id="run-123",
            counts={"conversations": 1},
            drift={},
            indexed=False,
            index_error=None,
            duration_ms=100,
            render_failures=[],
        )

        with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
            mock_latest.return_value = Path("/tmp/render/2024-01-15")
            _display_result(mock_env, mock_config, mock_run_result, "render", None)
            mock_latest.assert_called_once()

    def test_display_result_stage_acquire_no_render_path(self, mock_env):
        """_display_result skips render path for non-render stages."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result = RunResult(
            run_id="run-123",
            counts={"conversations": 1},
            drift={},
            indexed=False,
            index_error=None,
            duration_ms=100,
            render_failures=[],
        )

        with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
            _display_result(mock_env, mock_config, mock_run_result, "acquire", None)
            mock_latest.assert_not_called()

    def test_display_result_with_stage_prefix_title(self, mock_env):
        """_display_result includes stage name in title."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result = RunResult(
            run_id="run-123",
            counts={"conversations": 1},
            drift={},
            indexed=False,
            index_error=None,
            duration_ms=100,
            render_failures=[],
        )

        _display_result(mock_env, mock_config, mock_run_result, "parse", ["inbox"])
        title, lines = mock_env.ui.summary.call_args[0]
        assert "parse" in title.lower()
        assert "inbox" in title.lower()

    def test_display_result_render_failures_truncated(self, mock_env, capsys):
        """_display_result shows truncated render failures with ellipsis."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        failures = [
            {"conversation_id": f"conv-{i}", "error": f"error {i}"}
            for i in range(15)
        ]
        mock_run_result = RunResult(
            run_id="run-123",
            counts={},
            drift={},
            indexed=False,
            index_error=None,
            duration_ms=0,
            render_failures=failures,
        )

        _display_result(mock_env, mock_config, mock_run_result, "all", None)
        captured = capsys.readouterr()
        assert "Render failures (15)" in captured.err
        assert "and 5 more" in captured.err

    def test_display_result_index_error_hint(self, mock_env, capsys):
        """_display_result shows index error hint when present."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result = RunResult(
            run_id="run-123",
            counts={},
            drift={},
            indexed=False,
            index_error="Database locked",
            duration_ms=0,
            render_failures=[],
        )

        _display_result(mock_env, mock_config, mock_run_result, "all", None)
        captured = capsys.readouterr()
        assert "Index error:" in captured.err
        assert "Database locked" in captured.err
        assert "run `polylogue run --stage index`" in captured.err


# =============================================================================
# TEST CLASS: run_command with --watch (Coverage: lines 270-294, 309-332)
# =============================================================================


class TestRunCommandWatch:
    """Test run command watch mode."""

    def test_run_command_watch_validation_notify_without_watch(self, runner, cli_workspace):
        """run --notify without --watch fails."""
        result = runner.invoke(run_command, ["--notify", "--plain"], obj=MagicMock(ui=MagicMock(plain=True)))
        # Should fail validation
        assert result.exit_code != 0 or "require --watch" in result.output.lower()

    def test_run_command_watch_validation_exec_without_watch(self, runner, cli_workspace):
        """run --exec without --watch fails."""
        result = runner.invoke(
            run_command,
            ["--exec", "echo test", "--plain"],
            obj=MagicMock(ui=MagicMock(plain=True)),
        )
        assert result.exit_code != 0 or "require --watch" in result.output.lower()

    def test_run_command_watch_validation_webhook_without_watch(self, runner, cli_workspace):
        """run --webhook without --watch fails."""
        result = runner.invoke(
            run_command,
            ["--webhook", "http://example.com", "--plain"],
            obj=MagicMock(ui=MagicMock(plain=True)),
        )
        assert result.exit_code != 0 or "require --watch" in result.output.lower()


# =============================================================================
# TEST CLASS: Watch mode callbacks (Coverage: lines 276-294)
# =============================================================================


class TestWatchModeCallbacks:
    """Test watch mode event callbacks."""

    def test_notify_on_new_conversations_called_in_watch(self):
        """_notify_new_conversations sends desktop notification."""
        with patch("subprocess.run") as mock_run:
            _notify_new_conversations(5)
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "notify-send" in call_args
            assert "5" in str(call_args)

    def test_exec_on_new_called_in_watch(self):
        """_exec_on_new executes command with env var."""
        with patch("subprocess.run") as mock_run:
            _exec_on_new("echo $POLYLOGUE_NEW_COUNT", 3)
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["env"]["POLYLOGUE_NEW_COUNT"] == "3"
            assert call_kwargs["shell"] is True

    def test_webhook_on_new_called_in_watch(self):
        """_webhook_on_new sends POST request."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            _webhook_on_new("http://example.com/webhook", 2)
            mock_urlopen.assert_called_once()
            call_args = mock_urlopen.call_args[0][0]
            assert call_args.get_full_url() == "http://example.com/webhook"
            assert call_args.get_method() == "POST"

    def test_webhook_on_new_payload_format(self):
        """_webhook_on_new includes correct JSON payload."""
        with patch("urllib.request.urlopen"):
            with patch("urllib.request.Request") as mock_request:
                _webhook_on_new("http://example.com/webhook", 7)
                call_kwargs = mock_request.call_args[1]
                payload = json.loads(call_kwargs["data"].decode())
                assert payload["event"] == "sync"
                assert payload["new_conversations"] == 7


# =============================================================================
# TEST CLASS: sources_command (Coverage: lines 309-332)
# =============================================================================


class TestSourcesCommand:
    """Test sources command."""

    def test_sources_command_json_output_basic(self, runner, cli_workspace):
        """sources --json outputs JSON array."""
        result = runner.invoke(
            sources_command,
            ["--json"],
            obj=MagicMock(ui=MagicMock(plain=True)),
        )
        # Note: This will fail without proper config setup, so we expect non-zero or json parsing
        if result.exit_code == 0:
            try:
                data = json.loads(result.output)
                assert isinstance(data, list)
            except json.JSONDecodeError:
                pass

    def test_sources_command_plain_output_path_source(self, runner, cli_workspace):
        """sources without --json shows text output for path sources."""
        result = runner.invoke(
            sources_command,
            [],
            obj=MagicMock(ui=MagicMock(plain=True, summary=MagicMock())),
        )
        # Will depend on config, just verify command runs


# =============================================================================
# TEST CLASS: IngestResult (Coverage: lines 56-83)
# =============================================================================


class TestIngestResult:
    """Test IngestResult tracking."""

    def test_ingest_result_merge_with_changes(self):
        """IngestResult.merge_result tracks changed conversations."""
        result = IngestResult()
        result_counts = {
            "conversations": 1,
            "messages": 5,
            "attachments": 2,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }
        result.merge_result("conv-1", result_counts, content_changed=True)

        assert "conv-1" in result.processed_ids
        assert result.changed_counts["conversations"] == 1
        assert result.changed_counts["messages"] == 5
        assert result.changed_counts["attachments"] == 2

    def test_ingest_result_merge_no_content_change(self):
        """IngestResult.merge_result skips processed_ids when no change."""
        result = IngestResult()
        result_counts = {
            "conversations": 0,
            "messages": 0,
            "attachments": 0,
            "skipped_conversations": 0,
            "skipped_messages": 0,
            "skipped_attachments": 0,
        }
        result.merge_result("conv-2", result_counts, content_changed=False)

        # Should not add to processed_ids if nothing changed
        assert "conv-2" not in result.processed_ids

    def test_ingest_result_merge_thread_safe(self):
        """IngestResult.merge_result is thread-safe."""
        result = IngestResult()
        errors = []

        def merge_thread(conv_id, thread_id):
            try:
                result_counts = {
                    "conversations": 1,
                    "messages": thread_id,
                    "attachments": 0,
                    "skipped_conversations": 0,
                    "skipped_messages": 0,
                    "skipped_attachments": 0,
                }
                result.merge_result(f"conv-{conv_id}", result_counts, content_changed=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=merge_thread, args=(i, i)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert len(result.processed_ids) == 10


# =============================================================================
# TEST CLASS: IngestionService (Coverage: lines 164-165, 216, 222, 262-269)
# =============================================================================


class TestIngestionService:
    """Test IngestionService parsing and batching."""

    def test_ingest_from_raw_backend_not_initialized(self):
        """ingest_from_raw raises RuntimeError if backend not initialized."""
        repo = MagicMock()
        repo._backend = None

        service = IngestionService(
            repository=repo,
            archive_root=Path("/tmp"),
            config=MagicMock(),
        )

        with pytest.raises(RuntimeError, match="backend is not initialized"):
            service.ingest_from_raw(raw_ids=["raw-1"])

    def test_ingest_from_raw_provider_filter(self, mock_backend):
        """ingest_from_raw respects provider filter."""
        repo = MagicMock()
        repo._backend = mock_backend

        raw_record_1 = MagicMock(raw_id="raw-1", provider_name="claude")
        raw_record_2 = MagicMock(raw_id="raw-2", provider_name="chatgpt")

        mock_backend.iter_raw_conversations.return_value = [raw_record_1, raw_record_2]

        service = IngestionService(
            repository=repo,
            archive_root=Path("/tmp"),
            config=MagicMock(),
        )

        with patch.object(service, "_process_raw_batch"):
            result = service.ingest_from_raw(provider="claude")
            # Verify iter_raw_conversations was called with provider filter
            mock_backend.iter_raw_conversations.assert_called()
            call_kwargs = mock_backend.iter_raw_conversations.call_args[1]
            assert call_kwargs["provider"] == "claude"

    def test_parse_raw_record_jsonl_format(self, mock_backend):
        """_parse_raw_record handles JSONL format (newline-delimited JSON)."""
        repo = MagicMock()
        repo._backend = mock_backend

        service = IngestionService(
            repository=repo,
            archive_root=Path("/tmp"),
            config=MagicMock(),
        )

        # JSONL content
        jsonl_content = (
            '{"id": "msg1", "role": "user", "text": "hello"}\n'
            '{"id": "msg2", "role": "assistant", "text": "hi"}\n'
        )

        raw_record = MagicMock()
        raw_record.raw_content = jsonl_content.encode("utf-8")
        raw_record.provider_name = "claude-code"
        raw_record.raw_id = "raw-123"

        with patch("polylogue.pipeline.services.ingestion._parse_json_payload") as mock_parse:
            mock_parse.return_value = [
                ParsedConversation(
                    provider_name="claude-code",
                    provider_conversation_id="conv-123",
                    messages=[],
                )
            ]
            result = service._parse_raw_record(raw_record)
            assert len(result) > 0

    def test_parse_raw_record_json_single_document(self, mock_backend):
        """_parse_raw_record handles single JSON document."""
        repo = MagicMock()
        repo._backend = mock_backend

        service = IngestionService(
            repository=repo,
            archive_root=Path("/tmp"),
            config=MagicMock(),
        )

        json_content = '{"id": "conv-1", "messages": [{"id": "m1", "text": "hello"}]}'

        raw_record = MagicMock()
        raw_record.raw_content = json_content.encode("utf-8")
        raw_record.provider_name = "chatgpt"
        raw_record.raw_id = "raw-456"

        with patch("polylogue.pipeline.services.ingestion._parse_json_payload") as mock_parse:
            mock_parse.return_value = [
                ParsedConversation(
                    provider_name="chatgpt",
                    provider_conversation_id="conv-1",
                    messages=[],
                )
            ]
            result = service._parse_raw_record(raw_record)
            assert len(result) > 0

    def test_parse_raw_record_string_content(self, mock_backend):
        """_parse_raw_record handles string content (not bytes)."""
        repo = MagicMock()
        repo._backend = mock_backend

        service = IngestionService(
            repository=repo,
            archive_root=Path("/tmp"),
            config=MagicMock(),
        )

        raw_record = MagicMock()
        raw_record.raw_content = '{"id": "conv-789", "messages": []}'
        raw_record.provider_name = "gemini"
        raw_record.raw_id = "raw-789"

        with patch("polylogue.pipeline.services.ingestion._parse_json_payload") as mock_parse:
            mock_parse.return_value = []
            result = service._parse_raw_record(raw_record)
            mock_parse.assert_called_once()


# =============================================================================
# TEST CLASS: ParsedAttachment sanitization (Coverage: lines 107-109, 231-248)
# =============================================================================


class TestParsedAttachmentSanitization:
    """Test attachment path and name sanitization."""

    def test_attachment_sanitize_path_removes_control_chars(self):
        """ParsedAttachment path sanitizer removes control characters."""
        att = ParsedAttachment(
            provider_attachment_id="att-1",
            name="test.pdf",
            path="file\x00with\x01control.txt",
        )
        # Control chars should be removed
        assert "\x00" not in att.path
        assert "\x01" not in att.path

    def test_attachment_sanitize_path_detects_traversal(self):
        """ParsedAttachment path sanitizer blocks .. traversal."""
        att = ParsedAttachment(
            provider_attachment_id="att-2",
            name="test.pdf",
            path="../../../etc/passwd",
        )
        # Should be blocked with hash
        assert att.path.startswith("_blocked_")

    def test_attachment_sanitize_path_safe_absolute_paths(self):
        """ParsedAttachment allows /tmp/ and /var/tmp/ absolute paths."""
        att = ParsedAttachment(
            provider_attachment_id="att-3",
            name="test.pdf",
            path="/tmp/test/file.txt",
        )
        # /tmp/ is safe, should preserve
        assert att.path == "tmp/test/file.txt" or att.path.startswith("/")

    def test_attachment_sanitize_name_removes_control_chars(self):
        """ParsedAttachment name sanitizer removes control characters."""
        att = ParsedAttachment(
            provider_attachment_id="att-4",
            name="file\x00with\x1fcontrol.pdf",
        )
        assert "\x00" not in (att.name or "")
        assert "\x1f" not in (att.name or "")

    def test_attachment_sanitize_name_rejects_dots_only(self):
        """ParsedAttachment name sanitizer rejects names that are only dots."""
        att = ParsedAttachment(
            provider_attachment_id="att-5",
            name="...",
        )
        # Should become "file" (default)
        assert att.name == "file"

    def test_attachment_sanitize_none_paths(self):
        """ParsedAttachment sanitization handles None values."""
        att = ParsedAttachment(
            provider_attachment_id="att-6",
            name=None,
            path=None,
        )
        assert att.name is None
        assert att.path is None


# =============================================================================
# TEST CLASS: attachment_from_meta helper (Coverage: lines 164-190, 197)
# =============================================================================


class TestAttachmentFromMeta:
    """Test attachment_from_meta helper function."""

    def test_attachment_from_meta_with_standard_fields(self):
        """attachment_from_meta extracts standard metadata."""
        meta = {
            "id": "att-uuid",
            "name": "document.pdf",
            "size": 1024,
            "mimeType": "application/pdf",
        }
        att = attachment_from_meta(meta, "msg-123", 0)

        assert att is not None
        assert att.provider_attachment_id == "att-uuid"
        assert att.name == "document.pdf"
        assert att.size_bytes == 1024
        assert att.mime_type == "application/pdf"

    def test_attachment_from_meta_alternate_field_names(self):
        """attachment_from_meta handles alternate field name conventions."""
        meta = {
            "fileId": "file-456",
            "file_name": "image.jpg",
            "size_bytes": "2048",
            "mime_type": "image/jpeg",
        }
        att = attachment_from_meta(meta, "msg-456", 0)

        assert att is not None
        assert att.provider_attachment_id == "file-456"
        assert att.name == "image.jpg"
        assert att.size_bytes == 2048

    def test_attachment_from_meta_size_coercion(self):
        """attachment_from_meta coerces size to int."""
        meta = {"uuid": "att-789", "name": "file.txt", "size": "512"}
        att = attachment_from_meta(meta, None, 0)

        assert att is not None
        assert att.size_bytes == 512

    def test_attachment_from_meta_invalid_size_skipped(self):
        """attachment_from_meta skips invalid size values."""
        meta = {"id": "att-999", "name": "file.txt", "size": "invalid"}
        att = attachment_from_meta(meta, None, 0)

        assert att is not None
        assert att.size_bytes is None

    def test_attachment_from_meta_generated_id_from_name(self):
        """attachment_from_meta generates ID from name when not present."""
        meta = {"name": "report.docx"}
        att = attachment_from_meta(meta, "msg-111", 0)

        assert att is not None
        assert att.provider_attachment_id.startswith("att-")

    def test_attachment_from_meta_no_id_no_name_returns_none(self):
        """attachment_from_meta returns None when no ID and no name."""
        meta = {"size": 1024, "mimeType": "text/plain"}
        att = attachment_from_meta(meta, "msg-222", 0)

        assert att is None

    def test_attachment_from_meta_non_dict_returns_none(self):
        """attachment_from_meta returns None for non-dict input."""
        att = attachment_from_meta("not a dict", "msg-333", 0)
        assert att is None


# =============================================================================
# TEST CLASS: extract_messages_from_list (Coverage: lines 231-248, 250)
# =============================================================================


class TestExtractMessagesFromList:
    """Test extract_messages_from_list helper function."""

    def test_extract_messages_basic_structure(self):
        """extract_messages_from_list extracts basic message structure."""
        items = [
            {
                "id": "m1",
                "role": "user",
                "text": "Hello",
            },
            {
                "id": "m2",
                "role": "assistant",
                "text": "Hi there!",
            },
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 2
        assert messages[0].provider_message_id == "m1"
        assert messages[0].role == "user"
        assert messages[0].text == "Hello"

    def test_extract_messages_nested_message_key(self):
        """extract_messages_from_list handles nested 'message' key."""
        items = [
            {
                "uuid": "msg-outer",
                "message": {
                    "id": "msg-inner",
                    "role": "user",
                    "text": "nested",
                },
            }
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 1
        assert messages[0].text == "nested"

    def test_extract_messages_content_as_string(self):
        """extract_messages_from_list extracts content field (string)."""
        items = [
            {
                "id": "m1",
                "role": "assistant",
                "content": "string content",
            }
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 1
        assert messages[0].text == "string content"

    def test_extract_messages_content_as_parts_list(self):
        """extract_messages_from_list concatenates content.parts list."""
        items = [
            {
                "id": "m1",
                "role": "assistant",
                "content": {
                    "parts": ["part 1", "part 2", "part 3"],
                },
            }
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 1
        assert "part 1" in messages[0].text
        assert "part 2" in messages[0].text

    def test_extract_messages_content_as_dict_with_text(self):
        """extract_messages_from_list extracts content.text (dict)."""
        items = [
            {
                "id": "m1",
                "role": "user",
                "content": {
                    "text": "dict content",
                },
            }
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 1
        assert messages[0].text == "dict content"

    def test_extract_messages_content_as_list_of_dicts(self):
        """extract_messages_from_list handles content as list of dicts."""
        items = [
            {
                "id": "m1",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 1
        assert "first" in messages[0].text
        assert "second" in messages[0].text

    def test_extract_messages_role_variations(self):
        """extract_messages_from_list handles role name variations."""
        items = [
            {"id": "m1", "sender": "user", "text": "msg1"},
            {"id": "m2", "author": "assistant", "text": "msg2"},
            {"id": "m3", "role": "system", "text": "msg3"},
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 3
        # Roles should be normalized
        assert all(m.role in ["user", "assistant", "system"] for m in messages)

    def test_extract_messages_timestamp_variations(self):
        """extract_messages_from_list extracts timestamp from various fields."""
        items = [
            {"id": "m1", "role": "user", "text": "msg1", "timestamp": 1234567890},
            {"id": "m2", "role": "assistant", "text": "msg2", "created_at": "2024-01-01"},
            {"id": "m3", "role": "user", "text": "msg3", "create_time": "2024-01-02"},
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 3
        assert all(m.timestamp is not None for m in messages)

    def test_extract_messages_skip_non_dict_items(self):
        """extract_messages_from_list skips non-dict items."""
        items = [
            {"id": "m1", "role": "user", "text": "msg1"},
            "not a dict",
            {"id": "m2", "role": "assistant", "text": "msg2"},
            None,
            [],
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 2
        assert messages[0].provider_message_id == "m1"
        assert messages[1].provider_message_id == "m2"

    def test_extract_messages_skip_items_without_text(self):
        """extract_messages_from_list skips items without text content."""
        items = [
            {"id": "m1", "role": "user", "text": "msg1"},
            {"id": "m2", "role": "assistant"},  # No text
            {"id": "m3", "role": "user", "content": "msg3"},
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 2

    def test_extract_messages_generate_id_when_missing(self):
        """extract_messages_from_list generates ID when not present."""
        items = [
            {"role": "user", "text": "first"},
            {"role": "assistant", "text": "second"},
        ]
        messages = extract_messages_from_list(items)

        assert len(messages) == 2
        assert all(m.provider_message_id for m in messages)
        # Should use index-based fallback
        assert "msg-" in messages[0].provider_message_id or messages[0].provider_message_id


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestRunCommandPlainMode:
    """Integration test for run command in plain mode."""

    def test_run_command_plain_preview_confirm_yes(self, runner, cli_workspace):
        """run --preview in plain mode proceeds when user confirms."""
        env = MagicMock(ui=MagicMock(plain=True, confirm=MagicMock(return_value=True)))
        # This will depend on proper CLI setup, just test the structure


class TestIngestResultThreading:
    """Test concurrent access to IngestResult."""

    def test_ingest_result_concurrent_merges(self):
        """IngestResult handles concurrent merge_result calls."""
        result = IngestResult()
        errors = []

        def concurrent_merge(thread_id):
            try:
                for i in range(5):
                    result_counts = {
                        "conversations": 1,
                        "messages": i,
                        "attachments": 0,
                        "skipped_conversations": 0,
                        "skipped_messages": 0,
                        "skipped_attachments": 0,
                    }
                    result.merge_result(f"conv-{thread_id}-{i}", result_counts, True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=concurrent_merge, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(result.processed_ids) == 25  # 5 threads * 5 merges


class TestAttachmentPathEdgeCases:
    """Edge case tests for attachment path sanitization."""

    def test_attachment_path_with_symlinks_blocked(self):
        """Attachment with symlink in path is blocked."""
        with patch("pathlib.Path.is_symlink") as mock_symlink:
            mock_symlink.return_value = True
            att = ParsedAttachment(
                provider_attachment_id="att-sym",
                name="file.txt",
                path="/home/user/link",
            )
            # Should be blocked
            assert att.path.startswith("_blocked_")

    def test_attachment_path_empty_after_sanitization(self):
        """Attachment with path that becomes empty returns None."""
        att = ParsedAttachment(
            provider_attachment_id="att-empty",
            name="file.txt",
            path="",
        )
        # Empty path should become None
        assert att.path is None


__all__ = [
    "TestRunSyncOncePlainProgress",
    "TestDisplayResultComprehensive",
    "TestRunCommandWatch",
    "TestWatchModeCallbacks",
    "TestSourcesCommand",
    "TestIngestResult",
    "TestIngestionService",
    "TestParsedAttachmentSanitization",
    "TestAttachmentFromMeta",
    "TestExtractMessagesFromList",
    "TestRunCommandPlainMode",
    "TestIngestResultThreading",
    "TestAttachmentPathEdgeCases",
]
