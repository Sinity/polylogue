from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.commands.run import _display_result
from polylogue.config import Source, get_config
from polylogue.pipeline.observers import ExecObserver, NotificationObserver, WebhookObserver
from polylogue.pipeline.runner import latest_run, plan_sources, run_sources
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.state_views import RunResult
from tests.infra.source_builders import ChatGPTExportBuilder, GenericConversationBuilder, InboxBuilder


@pytest.fixture
def mock_env():
    from rich.console import Console

    env = MagicMock()
    env.ui = MagicMock()
    env.ui.plain = True
    env.ui.console = Console()
    env.ui.confirm.return_value = True
    env.ui.summary = MagicMock()
    return env


@pytest.mark.parametrize("with_plan", [True, False])
async def test_plan_and_run_sources(workspace_env, tmp_path, with_plan):
    inbox = tmp_path / "inbox"
    source_file = (
        GenericConversationBuilder("conv1")
        .title("Test")
        .add_user("hello")
        .add_assistant("world")
        .write_to(inbox / "conversation.json")
    )

    config = get_config()
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
    assert result.run_id is not None


@pytest.mark.parametrize(
    ("setup_type", "stage", "expected_conv_count", "check_render"),
    [
        ("multi_source_filter", "parse", 1, False),
        ("single_source_chatgpt", "render", 0, True),
    ],
)
async def test_run_sources_filtered_by_stage(
    workspace_env,
    tmp_path,
    setup_type,
    stage,
    expected_conv_count,
    check_render,
):
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
        # Stages are independent: acquire + validate before testing parse
        await run_sources(config=config, stage="acquire", source_names=["source-a"])
        await run_sources(config=config, stage="validate", source_names=["source-a"])
        result = await run_sources(config=config, stage=stage, source_names=["source-a"])
    else:
        inbox = tmp_path / "inbox"
        source_file = ChatGPTExportBuilder("conv-chatgpt").add_node("user", "hello").write_to(inbox / "conversation.json")
        config = get_config()
        config.sources = [Source(name="inbox", path=source_file)]
        await run_sources(config=config, stage="acquire", source_names=["inbox"])
        await run_sources(config=config, stage="validate", source_names=["inbox"])
        await run_sources(config=config, stage="parse", source_names=["inbox"])
        result = await run_sources(config=config, stage=stage, source_names=["inbox"])

    assert result.counts["conversations"] == expected_conv_count
    if check_render:
        assert any(config.render_root.rglob("conversation.md"))


async def test_run_index_filters_selected_sources(workspace_env, tmp_path, monkeypatch):
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
    # Stages are independent: populate pipeline through all three predecessors
    await run_sources(config=config, stage="acquire")
    await run_sources(config=config, stage="validate")
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

    # Runs are stored in DB only (no JSON files).
    # Verify by checking the DB has run records.
    from polylogue.pipeline.run_finalization import latest_run

    run = await latest_run()
    assert run is not None


@pytest.mark.parametrize("setup_type", ["parsed_json", "null_columns"])
async def test_latest_run_parsing(workspace_env, tmp_path, setup_type):
    if setup_type == "parsed_json":
        inbox = tmp_path / "inbox"
        GenericConversationBuilder("conv-latest-run").add_user("test").write_to(inbox / "conversation.json")
        config = get_config()
        config.sources = [Source(name="inbox", path=inbox)]
        await run_sources(config=config, stage="all")
    else:
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
        if result.drift is not None:
            assert isinstance(result.drift, dict)
    else:
        assert result.plan_snapshot is None
        assert result.counts is None
        assert result.drift is None


def test_display_result_reports_render_failures(mock_env, capsys):
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


class TestWatchModeCallbacks:
    def test_notify_callback_executes_with_count(self):
        with patch("subprocess.run") as mock_run:
            NotificationObserver().on_completed(
                MagicMock(
                    counts={"conversations": 5, "new_conversations": 5, "changed_conversations": 0},
                    drift={"new": {"conversations": 5}, "changed": {"conversations": 0}},
                )
            )
        assert "notify-send" in mock_run.call_args[0][0]
        assert "5" in str(mock_run.call_args[0][0])

    def test_notify_zero_is_noop(self):
        with patch("subprocess.run") as mock_run:
            NotificationObserver().on_completed(MagicMock(counts={"conversations": 0}, drift={}))
        mock_run.assert_not_called()

    def test_notify_missing_binary_is_ignored(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            NotificationObserver().on_completed(
                MagicMock(
                    counts={"conversations": 1, "new_conversations": 1, "changed_conversations": 0},
                    drift={"new": {"conversations": 1}, "changed": {"conversations": 0}},
                )
            )

    def test_exec_callback_executes_with_count(self):
        with patch("subprocess.run") as mock_run:
            ExecObserver("echo $POLYLOGUE_ACTIVITY_COUNT").on_completed(
                MagicMock(
                    counts={"conversations": 3, "new_conversations": 3, "changed_conversations": 0},
                    drift={"new": {"conversations": 3}, "changed": {"conversations": 0}},
                )
            )
        assert mock_run.call_args.kwargs["env"]["POLYLOGUE_ACTIVITY_COUNT"] == "3"
        assert mock_run.call_args.kwargs["env"]["POLYLOGUE_NEW_CONVERSATION_COUNT"] == "3"
        assert mock_run.call_args.kwargs["env"]["POLYLOGUE_CHANGED_CONVERSATION_COUNT"] == "0"
        assert mock_run.call_args.kwargs.get("shell") is not True

    @pytest.mark.parametrize(
        "dangerous_command",
        [
            "echo hello; rm -rf /",
            "echo hello && cat /etc/passwd",
            "echo `whoami`",
            "echo $(id)",
        ],
    )
    def test_exec_callback_rejects_dangerous_commands(self, dangerous_command):
        with pytest.raises(ValueError, match="unsafe"):
            ExecObserver(dangerous_command)

    def test_webhook_callback_executes_with_count(self):
        fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 80))]
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=fake_addrinfo), patch(
            "urllib.request.urlopen"
        ) as mock_urlopen:
            WebhookObserver("http://example.com/webhook").on_completed(
                MagicMock(
                    counts={"conversations": 2, "new_conversations": 2, "changed_conversations": 0},
                    drift={"new": {"conversations": 2}, "changed": {"conversations": 0}},
                )
            )
        call_args = mock_urlopen.call_args[0][0]
        assert call_args.get_full_url() == "http://example.com/webhook"
        assert call_args.get_method() == "POST"
        assert mock_urlopen.call_args.kwargs["timeout"] == 10

    def test_webhook_payload_format(self):
        fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 80))]
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=fake_addrinfo), patch(
            "urllib.request.urlopen"
        ), patch("urllib.request.Request") as mock_request:
            WebhookObserver("http://example.com/webhook").on_completed(
                MagicMock(
                    counts={"conversations": 7, "new_conversations": 7, "changed_conversations": 0},
                    drift={"new": {"conversations": 7}, "changed": {"conversations": 0}},
                )
            )
        payload = json.loads(mock_request.call_args.kwargs["data"].decode())
        assert payload == {
            "event": "sync",
            "conversation_activity_count": 7,
            "new_conversations": 7,
            "changed_conversations": 0,
        }

    def test_webhook_errors_do_not_raise(self):
        fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 80))]
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=fake_addrinfo), patch(
            "urllib.request.urlopen", side_effect=ConnectionError("Connection failed")
        ):
            WebhookObserver("http://example.com/webhook").on_completed(
                MagicMock(
                    counts={"conversations": 1, "new_conversations": 1, "changed_conversations": 0},
                    drift={"new": {"conversations": 1}, "changed": {"conversations": 0}},
                )
            )
