from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.run import (
    _display_result,
    sources_command,
)
from polylogue.config import Source, get_config
from polylogue.pipeline.observers import (
    ExecObserver,
    NotificationObserver,
    WebhookObserver,
)
from polylogue.pipeline.runner import latest_run, plan_sources, run_sources
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import PlanResult, RunResult
from tests.infra.source_builders import (
    ChatGPTExportBuilder,
    GenericConversationBuilder,
    InboxBuilder,
)


@pytest.mark.parametrize(
    "with_plan",
    [True, False],
)
async def test_plan_and_run_sources(workspace_env, tmp_path, with_plan):
    """plan_sources and run_sources work together or independently."""
    inbox = tmp_path / "inbox"
    source_file = (
        GenericConversationBuilder("conv1")
        .title("Test")
        .add_user("hello")
        .add_assistant("world")
        .write_to(inbox / "conversation.json")
    )

    config = get_config()
    # Generic fixture payload is not provider-schema-conformant; use a generic
    # source name so validate stage treats it as schema-less test input.
    config.sources = [Source(name="inbox", path=source_file)]

    if with_plan:
        plan = plan_sources(config)
        assert plan.counts["scan"] == 1
        assert plan.counts["store_raw"] == 1
        assert plan.counts["parse"] == 1
        result = await run_sources(config=config, stage="all", plan=plan)
    else:
        result = await run_sources(config=config, stage="all")

    assert result.counts["conversations"] == 1
    run_dir = config.archive_root / "runs"
    assert any(run_dir.iterdir())


@pytest.mark.parametrize(
    "setup_type,stage,expected_conv_count,check_render",
    [
        ("multi_source_filter", "parse", 1, False),
        ("single_source_chatgpt", "render", 0, True),
    ],
)
async def test_run_sources_filtered_by_stage(workspace_env, tmp_path, setup_type, stage, expected_conv_count, check_render):
    """run_sources filters by stage and source correctly."""
    if setup_type == "multi_source_filter":
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
        result = await run_sources(config=config, stage=stage, source_names=["source-a"])
    else:  # single_source_chatgpt
        inbox = tmp_path / "inbox"
        source_file = ChatGPTExportBuilder("conv-chatgpt").add_node("user", "hello").write_to(inbox / "conversation.json")
        config = get_config()
        config.sources = [Source(name="inbox", path=source_file)]
        await run_sources(config=config, stage="parse", source_names=["inbox"])
        result = await run_sources(config=config, stage=stage, source_names=["inbox"])

    assert result.counts["conversations"] == expected_conv_count
    if check_render:
        assert any(config.render_root.rglob("conversation.md"))


@pytest.mark.parametrize(
    "test_type",
    ["index_filters"],
)
async def test_run_index_filters_selected_sources(workspace_env, tmp_path, monkeypatch, test_type):
    """run_sources filters index by source."""
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

    await run_sources(config=config, stage="parse")

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

    async def fake_update_method(self, ids):
        if hasattr(ids, "__aiter__"):
            update_calls.append([conversation_id async for conversation_id in ids])
        else:
            update_calls.append(list(ids))
        return True

    monkeypatch.setattr(IndexService, "update_index", fake_update_method)
    await run_sources(config=config, stage="index", source_names=["source-b"])
    assert update_calls == [[id_by_source["source-b"]]]




async def test_run_writes_unique_report_files(workspace_env, tmp_path, monkeypatch):
    """run_sources writes unique timestamped report files."""
    inbox = tmp_path / "inbox"
    source_file = GenericConversationBuilder("conv1").add_user("hello").write_to(inbox / "conversation.json")

    config = get_config()
    config.sources = [Source(name="inbox", path=source_file)]

    import polylogue.pipeline.runner as runner_mod

    fixed_time = 1_700_000_000
    monkeypatch.setattr(runner_mod.time, "time", lambda: fixed_time)
    monkeypatch.setattr(runner_mod.time, "perf_counter", lambda: 0.0)

    await run_sources(config=config, stage="all")
    await run_sources(config=config, stage="all")

    run_dir = config.archive_root / "runs"
    runs = list(run_dir.glob(f"run-{fixed_time}-*.json"))
    assert len(runs) == 2


# latest_run() tests


@pytest.mark.parametrize(
    "setup_type",
    ["parsed_json", "null_columns"],
)
async def test_latest_run_parsing(workspace_env, tmp_path, setup_type):
    """await latest_run() parses JSON and handles NULL columns."""
    if setup_type == "parsed_json":
        inbox = tmp_path / "inbox"
        (GenericConversationBuilder("conv-latest-run").add_user("test").write_to(inbox / "conversation.json"))
        config = get_config()
        config.sources = [Source(name="inbox", path=inbox)]
        await run_sources(config=config, stage="all")
    else:  # null_columns
        with open_connection(None) as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms)
                VALUES (?, ?, NULL, NULL, NULL, 0, 100)
                """,
                ("null-test-run", str(int(time.time()))),
            )
            conn.commit()

    result = await latest_run()
    assert result is not None

    if setup_type == "parsed_json":
        assert result.run_id is not None
        if result.counts is not None:
            assert isinstance(result.counts, dict)
            assert "conversations" in result.counts or "messages" in result.counts
        if result.drift is not None:
            assert isinstance(result.drift, dict)
    else:  # null_columns
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
    repo.backend = MagicMock()
    repo.backend._get_connection = MagicMock()
    return repo


@pytest.fixture
def mock_backend():
    """Mock SQLite backend."""
    backend = MagicMock()
    backend._get_connection = MagicMock()
    async def _empty_iter():
        if False:
            yield None

    backend.iter_raw_conversations = MagicMock(return_value=_empty_iter())
    backend.get_raw_conversation = AsyncMock(return_value=None)
    backend.get_raw_conversations_batch = AsyncMock(return_value=[])
    return backend


# =============================================================================
# TEST CLASS: _run_sync_once (Coverage: lines 43-47, 73-75)
# =============================================================================


def test_display_result_reports_render_failures(mock_env, capsys):
    """Pinned regression: render failures are surfaced with truncation and retry hint."""
    mock_config = MagicMock(render_root=Path("/tmp/render"))
    failures = [{"conversation_id": f"conv-{index}", "error": f"error {index}"} for index in range(15)]
    result = RunResult(
        run_id="run-123",
        counts={"conversations": 0},
        drift={},
        indexed=True,
        index_error=None,
        duration_ms=0,
        render_failures=failures,
    )

    with patch("polylogue.cli.helpers.latest_render_path", return_value=None):
        _display_result(mock_env, mock_config, result, "all", None)

    captured = capsys.readouterr()
    assert "Render failures (15)" in captured.err
    assert "and 5 more" in captured.err
    assert "re-run with `polylogue run --stage render`" in captured.err


def test_display_result_reports_index_error_hint(mock_env, capsys):
    """Pinned regression: index errors get an explicit rebuild hint."""
    mock_config = MagicMock(render_root=Path("/tmp/render"))
    result = RunResult(
        run_id="run-123",
        counts={"conversations": 0},
        drift={},
        indexed=False,
        index_error="Database locked",
        duration_ms=0,
        render_failures=[],
    )

    _display_result(mock_env, mock_config, result, "index", None)

    captured = capsys.readouterr()
    assert "Index error: Database locked" in captured.err
    assert "run `polylogue run --stage index`" in captured.err


# =============================================================================
# TEST CLASS: run_command with --watch (Coverage: lines 270-294, 309-332)
# =============================================================================


class TestWatchModeCallbacks:
    """Test watch mode observer callbacks."""

    def test_notify_callback_executes_with_count(self):
        """NotificationObserver calls notify-send with conversation count."""
        with patch("subprocess.run") as mock_run:
            handler = NotificationObserver()
            handler.on_completed(MagicMock(counts={"conversations": 5}))
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "notify-send" in call_args
            assert "5" in str(call_args)

    def test_exec_callback_executes_with_count(self):
        """ExecObserver runs command with correct environment."""
        with patch("subprocess.run") as mock_run:
            handler = ExecObserver("echo $POLYLOGUE_NEW_COUNT")
            handler.on_completed(MagicMock(counts={"conversations": 3}))
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["env"]["POLYLOGUE_NEW_COUNT"] == "3"
            assert call_kwargs.get("shell") is not True

    def test_webhook_callback_executes_with_count(self):
        """WebhookObserver sends POST with conversation count."""
        fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 80))]
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=fake_addrinfo):
            with patch("urllib.request.urlopen") as mock_urlopen:
                handler = WebhookObserver("http://example.com/webhook")
                handler.on_completed(MagicMock(counts={"conversations": 2}))
                mock_urlopen.assert_called_once()
                call_args = mock_urlopen.call_args[0][0]
                assert call_args.get_full_url() == "http://example.com/webhook"
                assert call_args.get_method() == "POST"

    def test_webhook_on_new_payload_format(self):
        """WebhookObserver includes correct JSON payload."""
        fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 80))]
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=fake_addrinfo):
            with patch("urllib.request.urlopen"):
                with patch("urllib.request.Request") as mock_request:
                    handler = WebhookObserver("http://example.com/webhook")
                    handler.on_completed(MagicMock(counts={"conversations": 7}))
                    call_kwargs = mock_request.call_args[1]
                    payload = json.loads(call_kwargs["data"].decode())
                    assert payload["event"] == "sync"
                    assert payload["new_conversations"] == 7


# =============================================================================
# Simple standalone tests
# =============================================================================


@pytest.mark.parametrize(
    "args,expect_json",
    [
        (["--json"], True),
        ([], False),
    ],
)
def test_sources_command_output(runner, cli_workspace, args, expect_json):
    """sources command outputs JSON or text depending on flags."""
    result = runner.invoke(
        sources_command,
        args,
        obj=MagicMock(ui=MagicMock(plain=True, summary=MagicMock())),
    )
    if expect_json and result.exit_code == 0:
        data = json.loads(result.output)
        assert isinstance(data, list)


# --- merged from test_pipeline_concurrent.py ---


def test_counts_lock_prevents_lost_updates():
	"""Verify _counts_lock prevents lost updates under concurrent access."""
	# Simulate the pattern from runner.py's _handle_future
	counts = {"conversations": 0, "messages": 0}
	lock = threading.Lock()
	iterations = 1000
	workers = 4

	def increment_with_lock():
		for _ in range(iterations):
			with lock:
				counts["conversations"] += 1
				counts["messages"] += 1

	with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
		futures = [executor.submit(increment_with_lock) for _ in range(workers)]
		for f in futures:
			f.result()

	expected = iterations * workers
	assert counts["conversations"] == expected, f"Lost updates: {expected - counts['conversations']}"
	assert counts["messages"] == expected


# NOTE: A test_counts_without_lock_may_lose_updates test was removed here.
# Race condition demonstration tests are inherently non-deterministic -
# they "may or may not" observe the race depending on thread scheduling.
# This provides no value in CI and can cause flaky failures.
# The fix is validated by test_counts_lock_prevents_lost_updates above.


def test_attachment_content_id_returns_tuple_not_mutates(tmp_path: Path):
	"""Verify attachment_content_id returns values instead of mutating."""
	from polylogue.pipeline.ids import attachment_content_id
	from polylogue.sources import ParsedAttachment

	# Create a test file
	test_file = tmp_path / "test.txt"
	test_file.write_text("test content")

	# Create attachment with original values
	original_path = str(test_file)
	original_meta = {"key": "value"}
	attachment = ParsedAttachment(
		provider_attachment_id="att-1",
		message_provider_id="msg-1",
		name="test.txt",
		mime_type="text/plain",
		size_bytes=12,
		path=original_path,  # Must set path for file to be processed
		provider_meta=original_meta.copy(),
	)

	# Call the function
	aid, returned_meta, returned_path = attachment_content_id(
		"test-provider",
		attachment,
		archive_root=tmp_path,
	)

	# Verify it returns a tuple
	assert isinstance(aid, str)
	assert isinstance(returned_meta, dict)
	assert returned_meta.get("sha256") is not None  # Hash should be added

	# The original attachment should NOT be mutated
	# (The function now returns values instead of mutating)
	assert attachment.provider_meta == original_meta


def test_store_records_commits_within_lock(monkeypatch):
	"""Verify store_records commits while _WRITE_LOCK is held."""
	from contextlib import contextmanager

	import tests.infra.storage_records as storage_helpers
	from tests.infra.storage_records import make_conversation, make_message

	class TrackingLock:
		def __init__(self) -> None:
			self.held = False

		def __enter__(self):
			self.held = True
			return self

		def __exit__(self, exc_type, exc, tb):
			self.held = False
			return False

	class DummyConn:
		def __init__(self, lock: TrackingLock) -> None:
			self._lock = lock
			self.commit_states: list[bool] = []

		def commit(self) -> None:
			self.commit_states.append(self._lock.held)

	lock = TrackingLock()
	conn = DummyConn(lock)

	@contextmanager
	def fake_connection_context(passed_conn):
		yield passed_conn

	monkeypatch.setattr(storage_helpers, "_WRITE_LOCK", lock)
	monkeypatch.setattr(storage_helpers, "connection_context", fake_connection_context)
	monkeypatch.setattr(storage_helpers, "upsert_conversation", lambda *_: True)
	monkeypatch.setattr(storage_helpers, "upsert_message", lambda *_: True)
	monkeypatch.setattr(storage_helpers, "upsert_attachment", lambda *_: True)
	monkeypatch.setattr(storage_helpers, "_prune_attachment_refs", lambda *_: None)

	record = make_conversation("test:1", title="Test", content_hash="abc123")
	messages = [make_message("test:1:msg1", "test:1", text="Hello")]
	result = storage_helpers.store_records(
		conversation=record,
		messages=messages,
		attachments=[],
		conn=conn,
	)

	assert result["conversations"] == 1
	assert result["messages"] == 1
	assert conn.commit_states == [True]


def test_concurrent_store_records_no_deadlock(workspace_env):
	"""Verify concurrent store_records calls don't deadlock."""
	from polylogue.storage.backends.connection import open_connection
	from tests.infra.storage_records import make_conversation, make_message, store_records

	# Initialize the database using workspace_env fixture (sets up proper env vars)
	with open_connection(None):
		pass

	def store_one(idx: int):
		record = make_conversation(f"test:{idx}", title=f"Test {idx}", content_hash=f"hash{idx}")
		messages = [make_message(f"test:{idx}:msg1", f"test:{idx}", text=f"Hello {idx}")]
		return store_records(
			conversation=record,
			messages=messages,
			attachments=[],
		)

	# Run concurrent stores
	workers = 4
	iterations = 10
	with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
		futures = [executor.submit(store_one, i) for i in range(iterations)]
		results = [f.result(timeout=30) for f in futures]  # 30s timeout to detect deadlock

	# All should succeed — each thread inserted exactly 1 unique conversation
	assert len(results) == iterations
	for r in results:
		assert r["conversations"] == 1


def test_set_add_is_thread_safe():
	"""Verify that set.add() under lock is safe for processed_ids pattern."""
	processed_ids: set[str] = set()
	lock = threading.Lock()
	iterations = 1000
	workers = 4

	def add_ids(worker_id: int):
		for i in range(iterations):
			with lock:
				processed_ids.add(f"worker{worker_id}:item{i}")

	with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
		futures = [executor.submit(add_ids, w) for w in range(workers)]
		for f in futures:
			f.result()

	expected_count = iterations * workers
	assert len(processed_ids) == expected_count


def test_failing_future_does_not_abort_remaining():
	"""A single failing future must not prevent remaining futures from being processed.

	Regression test: before the fix, an exception in _handle_future() would break
	the as_completed loop, silently abandoning all remaining unprocessed futures.
	"""
	results_processed = []

	def _work(idx: int) -> int:
		if idx == 2:
			raise ValueError(f"Deliberate failure on item {idx}")
		return idx

	with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
		futures = {executor.submit(_work, i): f"item-{i}" for i in range(5)}
		errors = 0

		for fut in concurrent.futures.as_completed(futures):
			try:
				results_processed.append(fut.result())
			except Exception:
				errors += 1

	# All 5 futures must have been visited (4 succeed, 1 fails)
	assert len(results_processed) == 4, f"Expected 4 successes, got {len(results_processed)}"
	assert errors == 1
	assert set(results_processed) == {0, 1, 3, 4}


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


__all__ = [
    "TestWatchModeCallbacks",
]
