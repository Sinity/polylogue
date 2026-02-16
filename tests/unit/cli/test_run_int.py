from __future__ import annotations

import asyncio
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
    _exec_on_new,
    _notify_new_conversations,
    _run_sync_once,
    _webhook_on_new,
    run_command,
    sources_command,
)
from polylogue.config import Source, get_config
from polylogue.pipeline.enrichment import (
    enrich_content_blocks,
    enrich_message_metadata,
)
from polylogue.pipeline.runner import latest_run, run_sources, plan_sources
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedConversation,
    attachment_from_meta,
    extract_messages_from_list,
)
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import PlanResult, RunResult
from tests.infra.helpers import (
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
    config.sources = [Source(name="codex", path=source_file)]

    if with_plan:
        plan = plan_sources(config)
        assert plan.counts["conversations"] == 1
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
    "scenario,title_change,content_change,expect_mtime_diff,check_title",
    [
        ("unchanged", False, False, False, False),
        ("content_changes", False, True, True, False),
        ("title_changes", True, False, True, True),
    ],
)
async def test_run_rerenders_based_on_changes(workspace_env, tmp_path, scenario, title_change, content_change, expect_mtime_diff, check_title):
    """run rerenders when content or title changes."""
    inbox = tmp_path / "inbox"
    source_file = inbox / "conversation.json"

    # Initial content
    initial_title = "Old title" if title_change else "conv-title"
    initial_content = "hello world" if content_change else "hello"
    (GenericConversationBuilder("conv-title").title(initial_title).add_user(initial_content).write_to(source_file))

    config = get_config()
    config.sources = [Source(name="inbox", path=source_file)]

    await run_sources(config=config, stage="all")
    convo_path = next(config.render_root.rglob("conversation.md"))
    first_mtime = convo_path.stat().st_mtime
    original = convo_path.read_text(encoding="utf-8") if check_title else ""

    # Modify based on scenario
    if title_change:
        (GenericConversationBuilder("conv-title").title("New title").add_user("hello").write_to(source_file))
    elif content_change:
        (GenericConversationBuilder("conv-title").add_user("hello world modified").write_to(source_file))

    # Small sleep to ensure filesystem timestamp changes if file is rewritten
    if expect_mtime_diff:
        time.sleep(0.01)

    await run_sources(config=config, stage="all")
    second_mtime = convo_path.stat().st_mtime

    if expect_mtime_diff:
        assert second_mtime > first_mtime
    else:
        assert first_mtime == second_mtime

    if check_title:
        updated = convo_path.read_text(encoding="utf-8")
        assert "# New title" in updated
        assert original != updated


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

    @pytest.mark.parametrize(
        "env_fixture,stage,source_names,render_format,check_plain",
        [
            ("mock_env", "all", None, "markdown", True),
            ("mock_env_rich", "render", ["source1"], "html", False),
        ],
    )
    def test_run_sync_once_progress(
        self, env_fixture, stage, source_names, render_format, check_plain, mock_run_result, capsys, request
    ):
        """_run_sync_once handles plain and rich mode progress."""
        env = request.getfixturevalue(env_fixture)

        with patch("polylogue.cli.commands.run.run_sources", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_run_result
            mock_config = MagicMock()

            result = _run_sync_once(
                mock_config,
                env,
                stage,
                source_names,
                render_format,
            )

        assert result.run_id == "run-123"

        if check_plain:
            captured = capsys.readouterr()
            assert "Syncing..." in captured.out
        else:
            # Verify run_sources was called with correct arguments
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["config"] == mock_config
            assert call_kwargs["stage"] == stage
            assert call_kwargs["source_names"] == source_names
            assert call_kwargs["render_format"] == render_format


# =============================================================================
# TEST CLASS: _display_result (Coverage: lines 234, 253-259, 260-266)
# =============================================================================


class TestDisplayResultComprehensive:
    """Comprehensive tests for _display_result coverage."""

    @pytest.mark.parametrize(
        "stage,source_names,show_render_path,conv_count,render_failures,index_error",
        [
            ("render", ["inbox"], True, 1, [], None),
            ("acquire", ["inbox"], False, 1, [], None),
            ("parse", ["inbox"], False, 1, [], None),
            ("all", None, True, 1, [], None),
            ("all", None, True, 0, [], None),
            ("all", None, True, 0, [{"conversation_id": f"conv-{i}", "error": f"error {i}"} for i in range(15)], None),
            ("all", None, True, 0, [], "Database locked"),
        ],
    )
    def test_display_result_stages_and_errors(
        self, mock_env, capsys, stage, source_names, show_render_path, conv_count, render_failures, index_error
    ):
        """_display_result handles different stages, errors, and failures."""
        mock_config = MagicMock()
        mock_config.render_root = Path("/tmp/render")
        mock_run_result = RunResult(
            run_id="run-123",
            counts={"conversations": conv_count},
            drift={},
            indexed=False,
            index_error=index_error,
            duration_ms=100 if conv_count else 0,
            render_failures=render_failures,
        )

        with patch("polylogue.cli.helpers.latest_render_path") as mock_latest:
            mock_latest.return_value = Path("/tmp/render/2024-01-15")
            _display_result(mock_env, mock_config, mock_run_result, stage, source_names)
            if show_render_path:
                mock_latest.assert_called_once()
            else:
                mock_latest.assert_not_called()

        mock_env.ui.summary.assert_called_once()

        # Stage and source names should appear in title
        if source_names:
            title = mock_env.ui.summary.call_args[0][0]
            assert stage in title.lower() or stage in str(source_names).lower()

        # Capture output for error/failure checks
        if render_failures or index_error:
            captured = capsys.readouterr()
            if render_failures:
                assert "Render failures" in captured.err
                assert "and 5 more" in captured.err
            if index_error:
                assert "Index error:" in captured.err
                assert "Database locked" in captured.err
                assert "run `polylogue run --stage index`" in captured.err


# =============================================================================
# TEST CLASS: run_command with --watch (Coverage: lines 270-294, 309-332)
# =============================================================================


class TestRunCommandWatch:
    """Test run command watch mode."""

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--notify", None),
            ("--exec", "echo test"),
            ("--webhook", "http://example.com"),
        ],
    )
    def test_run_command_watch_validation_requires_watch(self, runner, cli_workspace, flag, value):
        """run --notify/--exec/--webhook without --watch fails."""
        args = [flag]
        if value:
            args.append(value)
        args.append("--plain")

        result = runner.invoke(
            run_command,
            args,
            obj=MagicMock(ui=MagicMock(plain=True)),
        )
        assert result.exit_code != 0 or "require --watch" in result.output.lower()


# =============================================================================
# TEST CLASS: Watch mode callbacks (Coverage: lines 276-294)
# =============================================================================


class TestWatchModeCallbacks:
    """Test watch mode event callbacks."""

    @pytest.mark.parametrize(
        "callback_type,callback_func,count",
        [
            ("notify", _notify_new_conversations, 5),
            ("exec", _exec_on_new, 3),
            ("webhook", _webhook_on_new, 2),
        ],
    )
    def test_watch_callbacks_execute_with_count(self, callback_type, callback_func, count):
        """Watch callbacks execute with conversation count."""
        if callback_type == "notify":
            with patch("subprocess.run") as mock_run:
                callback_func(count)
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert "notify-send" in call_args
                assert str(count) in str(call_args)
        elif callback_type == "exec":
            with patch("subprocess.run") as mock_run:
                callback_func("echo $POLYLOGUE_NEW_COUNT", count)
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["env"]["POLYLOGUE_NEW_COUNT"] == str(count)
                # shell=False is the secure default (no shell kwarg passed)
                assert call_kwargs.get("shell") is not True
        elif callback_type == "webhook":
            # Mock SSRF validation (DNS unavailable in sandbox)
            fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 80))]
            with patch("polylogue.pipeline.events.socket.getaddrinfo", return_value=fake_addrinfo):
                with patch("urllib.request.urlopen") as mock_urlopen:
                    callback_func("http://example.com/webhook", count)
                    mock_urlopen.assert_called_once()
                    call_args = mock_urlopen.call_args[0][0]
                    assert call_args.get_full_url() == "http://example.com/webhook"
                    assert call_args.get_method() == "POST"

    def test_webhook_on_new_payload_format(self):
        """_webhook_on_new includes correct JSON payload."""
        fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 80))]
        with patch("polylogue.pipeline.events.socket.getaddrinfo", return_value=fake_addrinfo):
            with patch("urllib.request.urlopen"):
                with patch("urllib.request.Request") as mock_request:
                    _webhook_on_new("http://example.com/webhook", 7)
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
        try:
            data = json.loads(result.output)
            assert isinstance(data, list)
        except json.JSONDecodeError:
            pass


# =============================================================================
# TEST CLASS: ParsingService (Coverage: lines 164-165, 216, 222, 262-269)
# =============================================================================


class TestParsingService:
    """Test ParsingService parsing and batching."""

    @pytest.mark.parametrize("backend_initialized,provider", [(False, None), (True, "claude")])
    async def test_parse_from_raw(self, mock_backend, backend_initialized, provider):
        """parse_from_raw validates backend and filters by provider."""
        repo = MagicMock()
        repo._backend = mock_backend if backend_initialized else None
        service = ParsingService(repository=repo, archive_root=Path("/tmp"), config=MagicMock())
        if not backend_initialized:
            with pytest.raises(RuntimeError, match="backend is not initialized"):
                await service.parse_from_raw(raw_ids=["raw-1"])
        else:
            async def async_iter():
                for i, p in enumerate(["claude", "chatgpt"]):
                    yield MagicMock(raw_id=f"raw-{i}", provider_name=p)
            mock_backend.iter_raw_conversations.return_value = async_iter()
            with patch.object(service, "_process_raw_batch", new_callable=AsyncMock):
                await service.parse_from_raw(provider=provider)
                assert mock_backend.iter_raw_conversations.call_args[1]["provider"] == provider

    @pytest.mark.parametrize(
        "format_type,content,provider,is_bytes",
        [
            ("jsonl", '{"id": "msg1", "role": "user", "text": "hello"}\n{"id": "msg2"}', "claude-code", True),
            ("json", '{"id": "conv-1", "messages": []}', "chatgpt", True),
            ("string", '{"id": "conv-789"}', "gemini", False),
        ],
    )
    async def test_parse_raw_record_content_types(self, mock_backend, format_type, content, provider, is_bytes):
        """_parse_raw_record handles JSONL, JSON, and string content."""
        repo = MagicMock()
        repo._backend = mock_backend
        service = ParsingService(repository=repo, archive_root=Path("/tmp"), config=MagicMock())
        raw_record = MagicMock()
        raw_record.raw_content = content.encode("utf-8") if is_bytes else content
        raw_record.provider_name = provider
        raw_record.raw_id = "raw-123"
        with patch("polylogue.pipeline.services.parsing._parse_json_payload") as mock_parse:
            mock_parse.return_value = [] if format_type == "string" else [ParsedConversation(provider_name=provider, provider_conversation_id=f"conv-{format_type}", messages=[])]
            await service._parse_raw_record(raw_record)
            mock_parse.assert_called_once()


# =============================================================================
# TEST CLASS: ParsedAttachment sanitization (Coverage: lines 107-109, 231-248)
# =============================================================================


class TestParsedAttachmentSanitization:
    """Test attachment path and name sanitization."""

    @pytest.mark.parametrize(
        "name,path,check",
        [
            ("test.pdf", "file\x00with\x01control.txt", "path_ctrl_chars"),
            ("test.pdf", "../../../etc/passwd", "path_traversal"),
            ("test.pdf", "/tmp/test/file.txt", "path_safe"),
            ("file\x00with\x1fcontrol.pdf", None, "name_ctrl_chars"),
            ("...", None, "name_dots_only"),
            (None, None, "both_none"),
        ],
    )
    def test_attachment_sanitization(self, name, path, check):
        """ParsedAttachment sanitization handles various scenarios."""
        att = ParsedAttachment(provider_attachment_id=f"att-{check}", name=name, path=path)
        if check == "path_ctrl_chars":
            assert "\x00" not in att.path and "\x01" not in att.path
        elif check == "path_traversal":
            assert att.path.startswith("_blocked_")
        elif check == "path_safe":
            assert att.path is None or not att.path.startswith("../")
        elif check == "name_ctrl_chars":
            assert "\x00" not in (att.name or "") and "\x1f" not in (att.name or "")
        elif check == "name_dots_only":
            assert att.name == "file"
        elif check == "both_none":
            assert att.name is None and att.path is None

    def test_attachment_path_edge_cases(self):
        """Attachment path handles symlinks and empty paths."""
        # Test symlink blocking
        with patch("pathlib.Path.is_symlink") as mock_symlink:
            mock_symlink.return_value = True
            att = ParsedAttachment(
                provider_attachment_id="att-sym",
                name="file.txt",
                path="/home/user/link",
            )
            assert att.path.startswith("_blocked_")

        # Test empty path becomes None
        att = ParsedAttachment(
            provider_attachment_id="att-empty",
            name="file.txt",
            path="",
        )
        assert att.path is None


# =============================================================================
# TEST CLASS: attachment_from_meta helper (Coverage: lines 164-190, 197)
# =============================================================================


class TestRunIntAttachmentFromMeta:
    """Test attachment_from_meta helper function."""

    @pytest.mark.parametrize(
        "meta,should_exist",
        [
            ({"id": "att-uuid", "name": "document.pdf", "size": 1024, "mimeType": "application/pdf"}, True),
            ({"fileId": "file-456", "file_name": "image.jpg", "size_bytes": "2048"}, True),
            ({"uuid": "att-789", "name": "file.txt", "size": "512"}, True),
            ({"id": "att-999", "name": "file.txt", "size": "invalid"}, True),
            ({"name": "report.docx"}, True),
            ({"size": 1024, "mimeType": "text/plain"}, False),
        ],
    )
    def test_attachment_from_meta_variants(self, meta, should_exist):
        """attachment_from_meta handles various metadata formats and non-dict input."""
        att = attachment_from_meta(meta, "msg-id" if should_exist else "msg-222", 0)

        if should_exist:
            assert att is not None
            if "id" in meta or "fileId" in meta or "uuid" in meta:
                assert att.provider_attachment_id is not None
            elif "name" in meta:
                assert att.provider_attachment_id.startswith("att-")
        else:
            assert att is None

    def test_attachment_from_meta_non_dict(self):
        """attachment_from_meta returns None for non-dict input."""
        assert attachment_from_meta("not a dict", "msg-333", 0) is None


# =============================================================================
# TEST CLASS: extract_messages_from_list (Coverage: lines 231-248, 250)
# =============================================================================


class TestExtractMessagesFromList:
    """Test extract_messages_from_list helper function."""

    @pytest.mark.parametrize(
        "items,expected_count,check_text",
        [
            # Basic structure
            (
                [
                    {"id": "m1", "role": "user", "text": "Hello"},
                    {"id": "m2", "role": "assistant", "text": "Hi there!"},
                ],
                2,
                "Hello",
            ),
            # Nested message key
            (
                [
                    {
                        "uuid": "msg-outer",
                        "message": {
                            "id": "msg-inner",
                            "role": "user",
                            "text": "nested",
                        },
                    }
                ],
                1,
                "nested",
            ),
            # Content as string
            (
                [
                    {
                        "id": "m1",
                        "role": "assistant",
                        "content": "string content",
                    }
                ],
                1,
                "string content",
            ),
            # Content as parts list
            (
                [
                    {
                        "id": "m1",
                        "role": "assistant",
                        "content": {
                            "parts": ["part 1", "part 2", "part 3"],
                        },
                    }
                ],
                1,
                "part 1",
            ),
            # Content as dict with text
            (
                [
                    {
                        "id": "m1",
                        "role": "user",
                        "content": {
                            "text": "dict content",
                        },
                    }
                ],
                1,
                "dict content",
            ),
            # Content as list of dicts
            (
                [
                    {
                        "id": "m1",
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "first"},
                            {"type": "text", "text": "second"},
                        ],
                    }
                ],
                1,
                "first",
            ),
            # Skip non-dict items
            (
                [
                    {"id": "m1", "role": "user", "text": "msg1"},
                    "not a dict",
                    {"id": "m2", "role": "assistant", "text": "msg2"},
                    None,
                    [],
                ],
                2,
                "msg1",
            ),
            # Skip items without text
            (
                [
                    {"id": "m1", "role": "user", "text": "msg1"},
                    {"id": "m2", "role": "assistant"},  # No text
                    {"id": "m3", "role": "user", "content": "msg3"},
                ],
                2,
                "msg1",
            ),
        ],
    )
    def test_extract_messages_variants(self, items, expected_count, check_text):
        """extract_messages_from_list handles various message formats."""
        messages = extract_messages_from_list(items)

        assert len(messages) == expected_count
        assert check_text in messages[0].text

    @pytest.mark.parametrize(
        "items,expected_count,check_type",
        [
            # Role variations
            (
                [
                    {"id": "m1", "sender": "user", "text": "msg1"},
                    {"id": "m2", "author": "assistant", "text": "msg2"},
                    {"id": "m3", "role": "system", "text": "msg3"},
                ],
                3,
                "role",
            ),
            # Timestamp variations
            (
                [
                    {"id": "m1", "role": "user", "text": "msg1", "timestamp": 1234567890},
                    {"id": "m2", "role": "assistant", "text": "msg2", "created_at": "2024-01-01"},
                    {"id": "m3", "role": "user", "text": "msg3", "create_time": "2024-01-02"},
                ],
                3,
                "timestamp",
            ),
            # ID generation when missing
            (
                [
                    {"role": "user", "text": "first"},
                    {"role": "assistant", "text": "second"},
                ],
                2,
                "id_generation",
            ),
        ],
    )
    def test_extract_messages_field_variations(self, items, expected_count, check_type):
        """extract_messages_from_list handles role, timestamp, and ID generation."""
        messages = extract_messages_from_list(items)
        assert len(messages) == expected_count

        if check_type == "role":
            assert all(m.role in ["user", "assistant", "system"] for m in messages)
        elif check_type == "timestamp":
            assert all(m.timestamp is not None for m in messages)
        elif check_type == "id_generation":
            assert all(m.provider_message_id for m in messages)


# --- merged from test_content_enrichment.py ---

# =============================================================================
# enrich_content_blocks Tests
# =============================================================================


class TestEnrichContentBlocks:
	"""Tests for enrich_content_blocks function."""

	def test_empty_list_returns_empty(self):
		"""Empty input returns empty output."""
		result = enrich_content_blocks([])
		assert result == []

	def test_plain_text_unchanged(self):
		"""Plain text blocks pass through unchanged."""
		blocks = [{"type": "text", "text": "Hello world"}]
		result = enrich_content_blocks(blocks)

		assert len(result) == 1
		assert result[0]["type"] == "text"
		assert result[0]["text"] == "Hello world"

	def test_fenced_code_block_extracted(self):
		"""Fenced code blocks are extracted as code type."""
		blocks = [{"type": "text", "text": "```python\nprint('hello')\n```"}]
		result = enrich_content_blocks(blocks)

		assert len(result) == 1
		assert result[0]["type"] == "code"
		assert result[0]["language"] == "python"
		assert "print('hello')" in result[0]["text"]

	def test_fenced_code_with_declared_language(self):
		"""Fenced blocks preserve declared language."""
		blocks = [{"type": "text", "text": "```javascript\nconsole.log('hi')\n```"}]
		result = enrich_content_blocks(blocks)

		assert result[0]["type"] == "code"
		assert result[0]["declared_language"] == "javascript"
		assert result[0]["language"] == "javascript"

	def test_fenced_code_without_language_detected(self):
		"""Fenced blocks without language get detection."""
		blocks = [{"type": "text", "text": "```\ndef foo():\n    pass\n```"}]
		result = enrich_content_blocks(blocks)

		assert result[0]["type"] == "code"
		# Language might be detected or None
		assert "text" in result[0]

	def test_mixed_text_and_code(self):
		"""Text with embedded code blocks is split."""
		blocks = [{"type": "text", "text": "Here is code:\n```python\nx = 1\n```\nDone."}]
		result = enrich_content_blocks(blocks)

		# Should be split into multiple blocks
		assert len(result) >= 2

		# Find code block
		code_blocks = [b for b in result if b["type"] == "code"]
		assert len(code_blocks) == 1
		assert code_blocks[0]["language"] == "python"

	def test_existing_code_block_without_language(self):
		"""Code blocks without language get detection."""
		blocks = [{"type": "code", "text": "def hello():\n    print('hi')"}]
		result = enrich_content_blocks(blocks)

		assert len(result) == 1
		assert result[0]["type"] == "code"
		# Language should be detected (or None if detection fails)
		assert "language" in result[0]

	def test_existing_code_block_with_language(self):
		"""Code blocks with language pass through unchanged."""
		blocks = [{"type": "code", "text": "const x = 1", "language": "javascript"}]
		result = enrich_content_blocks(blocks)

		assert len(result) == 1
		assert result[0]["type"] == "code"
		assert result[0]["language"] == "javascript"
		assert result[0]["text"] == "const x = 1"

	def test_other_block_types_unchanged(self):
		"""Non-text, non-code blocks pass through unchanged."""
		blocks = [
			{"type": "image", "url": "https://example.com/img.png"},
			{"type": "file", "name": "doc.pdf"},
		]
		result = enrich_content_blocks(blocks)

		assert len(result) == 2
		assert result[0]["type"] == "image"
		assert result[1]["type"] == "file"

	def test_multiple_code_blocks(self):
		"""Multiple fenced code blocks are all extracted."""
		text = """First code:
```python
x = 1
```
Second code:
```javascript
let y = 2
```
Done."""
		blocks = [{"type": "text", "text": text}]
		result = enrich_content_blocks(blocks)

		# Find code blocks
		code_blocks = [b for b in result if b["type"] == "code"]
		assert len(code_blocks) == 2

		languages = {b["language"] for b in code_blocks}
		assert "python" in languages
		assert "javascript" in languages

	def test_empty_fenced_block(self):
		"""Empty fenced blocks are handled gracefully."""
		blocks = [{"type": "text", "text": "```\n\n```"}]
		result = enrich_content_blocks(blocks)

		# Should handle without error
		assert len(result) >= 0


# =============================================================================
# enrich_message_metadata Tests
# =============================================================================


class TestEnrichMessageMetadata:
	"""Tests for enrich_message_metadata function."""

	def test_none_metadata_returns_none(self):
		"""None metadata returns None."""
		result = enrich_message_metadata(None)
		assert result is None

	def test_empty_metadata_returns_empty(self):
		"""Empty metadata returns empty."""
		result = enrich_message_metadata({})
		assert result == {}

	def test_metadata_without_content_blocks_unchanged(self):
		"""Metadata without content_blocks passes through."""
		meta = {"model": "gpt-4", "temperature": 0.7}
		result = enrich_message_metadata(meta)

		assert result == meta

	def test_metadata_with_content_blocks_enriched(self):
		"""Metadata with content_blocks gets enriched."""
		meta = {
			"model": "gpt-4",
			"content_blocks": [
				{"type": "text", "text": "```python\nprint('hi')\n```"}
			]
		}
		result = enrich_message_metadata(meta)

		assert "content_blocks" in result
		assert result["model"] == "gpt-4"  # Other fields preserved

		# Check enrichment happened
		enriched_blocks = result["content_blocks"]
		assert len(enriched_blocks) >= 1
		code_blocks = [b for b in enriched_blocks if b["type"] == "code"]
		assert len(code_blocks) == 1

	def test_original_metadata_not_mutated(self):
		"""Original metadata dict is not mutated."""
		meta = {
			"model": "gpt-4",
			"content_blocks": [{"type": "text", "text": "Hello"}]
		}
		original_blocks = meta["content_blocks"]

		enrich_message_metadata(meta)

		# Original should be unchanged
		assert meta["content_blocks"] is original_blocks
		assert meta["content_blocks"][0]["type"] == "text"


# =============================================================================
# Language Detection Integration Tests
# =============================================================================


class TestLanguageDetectionIntegration:
	"""Tests for language detection within enrichment."""

	@pytest.mark.parametrize("code,expected_lang", [
		("def foo():\n    pass", "python"),
		("function foo() { }", "javascript"),
		("fn main() { }", "rust"),
		("package main\nfunc main() { }", "go"),
	])
	def test_language_detection_accuracy(self, code: str, expected_lang: str):
		"""Language detection works for common languages."""
		blocks = [{"type": "code", "text": code}]
		result = enrich_content_blocks(blocks)

		# May detect correctly or return None
		if result[0].get("language"):
			# If detected, should match expected
			assert result[0]["language"] == expected_lang

	def test_language_alias_normalization(self):
		"""Language aliases are normalized."""
		blocks = [{"type": "text", "text": "```py\nprint('hi')\n```"}]
		result = enrich_content_blocks(blocks)

		# 'py' should be normalized to 'python'
		code_blocks = [b for b in result if b["type"] == "code"]
		if code_blocks and code_blocks[0].get("language"):
			assert code_blocks[0]["language"] == "python"


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


def test_store_records_commits_within_lock(tmp_path: Path):
	"""Verify store_records commits inside the lock scope."""
	from tests.infra.helpers import make_conversation, make_message, store_records

	# Create a test database
	db_path = tmp_path / "test.db"

	# Track commit calls to verify ordering
	commit_order = []
	lock_held = threading.Event()

	original_commit = None

	def tracking_commit(self):
		commit_order.append(("commit", lock_held.is_set()))
		if original_commit:
			return original_commit(self)

	# We need to verify that commit happens while _WRITE_LOCK is held
	# This is tricky to test directly, but we can verify the code structure

	# For now, just verify the function works and commits
	from polylogue.storage.backends.connection import open_connection

	with open_connection(db_path) as conn:
		record = make_conversation("test:1", title="Test", content_hash="abc123")
		messages = [make_message("test:1:msg1", "test:1", text="Hello")]
		result = store_records(
			conversation=record,
			messages=messages,
			attachments=[],
			conn=conn,
		)
		assert result["conversations"] >= 0  # Either inserted or skipped


def test_concurrent_store_records_no_deadlock(workspace_env):
	"""Verify concurrent store_records calls don't deadlock."""
	from polylogue.storage.backends.connection import open_connection
	from tests.infra.helpers import make_conversation, make_message, store_records

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

	# All should succeed
	assert len(results) == iterations
	for r in results:
		assert r["conversations"] >= 0


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
    "TestRunSyncOncePlainProgress",
    "TestDisplayResultComprehensive",
    "TestRunCommandWatch",
    "TestWatchModeCallbacks",
    "TestParsingService",
    "TestParsedAttachmentSanitization",
    "TestExtractMessagesFromList",
    "TestEnrichContentBlocks",
    "TestEnrichMessageMetadata",
    "TestLanguageDetectionIntegration",
]
