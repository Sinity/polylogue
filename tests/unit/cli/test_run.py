"""Focused CLI command contracts for run, tags, and embed."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands.run import maybe_prompt_run_stage
from polylogue.sources import DriveError
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.state_views import PlanResult, RunResult
from tests.infra.storage_records import DbFactory


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_plan_result():
    return PlanResult(
        timestamp=1234567890,
        counts={"conversations": 5, "messages": 50, "attachments": 2},
        sources=["test-inbox"],
        cursors={"test-inbox": {"path": "/tmp/inbox"}},
        details={"new": 2, "existing": 3},
    )


@pytest.fixture
def mock_run_result():
    return RunResult(
        run_id="run-123",
        counts={"conversations": 3, "messages": 30, "attachments": 1},
        drift={"conversations": {"new": 2, "updated": 1, "unchanged": 5}},
        indexed=True,
        index_error=None,
        duration_ms=1500,
        render_failures=[],
    )


RUN_CASES = [
    (
        "preview_default",
        ["run", "--preview"],
        True,
        None,
        "all",
        ("acquire", "parse", "materialize", "render", "site", "index"),
        "html",
        True,
    ),
    (
        "preview_parse_source",
        ["run", "--preview", "--source", "test-inbox", "parse"],
        True,
        ["test-inbox"],
        "parse",
        ("parse",),
        "html",
        False,
    ),
    (
        "run_default",
        ["run"],
        False,
        None,
        "all",
        ("acquire", "parse", "materialize", "render", "site", "index"),
        "html",
        True,
    ),
    (
        "run_render_markdown_source",
        ["run", "--source", "drive", "render", "--format", "markdown"],
        False,
        ["drive"],
        "render",
        ("render",),
        "markdown",
        False,
    ),
    (
        "run_reprocess_source",
        ["run", "--source", "drive", "reprocess"],
        False,
        ["drive"],
        "reprocess",
        ("parse", "materialize", "render", "index"),
        "html",
        False,
    ),
]


EMBED_BATCH_CASES = [
    (["run", "embed", "--rebuild"], {"rebuild": True, "limit": None}, None),
    (["run", "embed", "--limit", "50"], {"rebuild": False, "limit": 50}, None),
    (["run", "embed", "--model", "voyage-4-large"], {"rebuild": False, "limit": None}, "voyage-4-large"),
]


STATUS_ROWS = [
    ("tags", ["important", "5", "review", "3", "3 total"], {"important": 5, "review": 3, "draft": 1}, []),
    ("tags_provider", ["claude-tag"], {"claude-tag": 3}, ["-p", "claude-ai"]),
]


def _seed_tag_counts(
    db_path: Path,
    tag_counts: dict[str, int],
    *,
    provider: str = "chatgpt",
) -> None:
    factory = DbFactory(db_path)
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)

    async def _seed() -> None:
        index = 0
        try:
            for tag, count in tag_counts.items():
                for _ in range(count):
                    conversation_id = factory.create_conversation(
                        id=f"{provider}-{tag}-{index}",
                        provider=provider,
                    )
                    await repository.add_tag(conversation_id, tag)
                    index += 1
        finally:
            await backend.close()

    asyncio.run(_seed())


def _close_coroutine(coro: object) -> None:
    close = getattr(coro, "close", None)
    if callable(close):
        close()


def _invoke_run_direct(
    runner: CliRunner,
    args: list[str],
    *,
    plan_result: PlanResult,
    run_result: RunResult,
    selected_sources: list[str] | None,
    prompted_stage: str,
    plan_side_effect: Exception | None = None,
    run_side_effect: Exception | None = None,
) -> tuple[object, dict[str, object]]:
    with ExitStack() as stack:
        mock_config = MagicMock(sources=[])
        mock_config.render_root = Path("/render")
        stack.enter_context(patch("polylogue.config.get_config", return_value=mock_config))
        mock_plan = stack.enter_context(patch("polylogue.cli.commands.run.plan_sources"))
        mock_run = stack.enter_context(patch("polylogue.cli.run_workflow.run_sources", new_callable=AsyncMock))
        mock_resolve = stack.enter_context(
            patch("polylogue.cli.commands.run.resolve_sources", return_value=selected_sources)
        )
        mock_stage_prompt = stack.enter_context(
            patch("polylogue.cli.commands.run.maybe_prompt_run_stage", return_value=prompted_stage)
        )
        mock_prompt = stack.enter_context(
            patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=selected_sources)
        )
        stack.enter_context(
            patch("polylogue.cli.run_workflow.format_plan_counts", return_value="5 conversations, 50 messages")
        )
        stack.enter_context(patch("polylogue.cli.run_workflow.format_plan_details", return_value="new=2, existing=3"))
        stack.enter_context(patch("polylogue.cli.run_workflow.format_cursors", return_value="cursor snapshot"))
        stack.enter_context(
            patch("polylogue.cli.run_workflow.format_counts", return_value="3 conversations, 30 messages")
        )
        stack.enter_context(patch("polylogue.cli.run_workflow.format_run_details", return_value=["Indexed: yes"]))
        mock_format_index = stack.enter_context(
            patch("polylogue.cli.run_workflow.format_index_status", return_value="Index status: indexed")
        )
        mock_latest = stack.enter_context(
            patch("polylogue.cli.helpers.latest_render_path", return_value=Path("/render/latest/conversation.html"))
        )

        mock_plan.return_value = plan_result
        mock_run.return_value = run_result
        if plan_side_effect is not None:
            mock_plan.side_effect = plan_side_effect
        if run_side_effect is not None:
            mock_run.side_effect = run_side_effect

        result = runner.invoke(cli, args)

    return result, {
        "plan": mock_plan,
        "run": mock_run,
        "resolve": mock_resolve,
        "stage_prompt": mock_stage_prompt,
        "prompt": mock_prompt,
        "format_index": mock_format_index,
        "latest_render": mock_latest,
    }


def _invoke_embed(runner: CliRunner, args: list[str]):
    with patch("polylogue.cli.commands.run._run_embed_standalone") as mock_standalone:
        result = runner.invoke(
            cli,
            args,
            env={"POLYLOGUE_FORCE_PLAIN": "1"},
        )
    return result, mock_standalone


class TestRunCommand:
    @pytest.mark.parametrize(
        (
            "case_name",
            "cli_args",
            "preview",
            "selected_sources",
            "expected_stage",
            "expected_stage_sequence",
            "expected_format",
            "expect_stage_prompt",
        ),
        RUN_CASES,
    )
    def test_run_dispatch_matrix(
        self,
        runner,
        cli_workspace,
        mock_plan_result,
        mock_run_result,
        case_name,
        cli_args,
        preview,
        selected_sources,
        expected_stage,
        expected_stage_sequence,
        expected_format,
        expect_stage_prompt,
    ):
        result, mocks = _invoke_run_direct(
            runner,
            cli_args,
            plan_result=mock_plan_result,
            run_result=mock_run_result,
            selected_sources=selected_sources,
            prompted_stage=expected_stage,
        )

        assert result.exit_code == 0, case_name
        if expect_stage_prompt:
            mocks["stage_prompt"].assert_called_once()
        else:
            mocks["stage_prompt"].assert_not_called()
        if preview:
            mocks["plan"].assert_called_once()
            kwargs = mocks["plan"].call_args.kwargs
            assert kwargs["stage"] == expected_stage
            assert kwargs["stage_sequence"] == expected_stage_sequence
            mocks["run"].assert_not_called()
        else:
            mocks["run"].assert_called_once()
            kwargs = mocks["run"].call_args.kwargs
            assert kwargs["stage"] == expected_stage
            assert kwargs["stage_sequence"] == expected_stage_sequence
            assert kwargs["render_format"] == expected_format
            assert kwargs["source_names"] == selected_sources

    def test_preview_summary_contract(self, runner, cli_workspace, mock_plan_result):
        result, mocks = _invoke_run_direct(
            runner,
            ["run", "--preview", "--source", "test-inbox"],
            plan_result=mock_plan_result,
            run_result=RunResult(run_id="unused", counts={}, drift={}, indexed=False, index_error=None, duration_ms=0),
            selected_sources=["test-inbox"],
            prompted_stage="all",
        )

        assert result.exit_code == 0
        assert "Preview" in result.output
        assert "Sources: test-inbox" in result.output
        assert "Work: 5 conversations, 50 messages" in result.output
        assert "State: new=2, existing=3" in result.output
        assert "Cursors: cursor snapshot" in result.output
        mocks["plan"].assert_called_once()

    @pytest.mark.parametrize(
        ("preview", "stage", "run_result", "expected_tokens", "expect_latest_render", "expect_format_index"),
        [
            (
                False,
                "index",
                RunResult(
                    run_id="run-idx",
                    counts={"conversations": 0},
                    drift={},
                    indexed=True,
                    index_error=None,
                    duration_ms=800,
                ),
                ["Sync (index)", "Index status: indexed", "Duration: 800ms"],
                False,
                True,
            ),
            (
                False,
                "render",
                RunResult(
                    run_id="run-render",
                    counts={"conversations": 3},
                    drift={},
                    indexed=True,
                    index_error=None,
                    duration_ms=1200,
                ),
                ["Sync (render)", "Latest render:"],
                True,
                False,
            ),
            (
                False,
                "all",
                RunResult(
                    run_id="run-all",
                    counts={"conversations": 2},
                    drift={},
                    indexed=False,
                    index_error="Vector database unavailable",
                    duration_ms=900,
                    render_failures=[{"conversation_id": "conv-1", "error": "boom"}],
                ),
                ["Sync", "Render failures (1)", "Index error: Vector database unavailable"],
                True,
                True,
            ),
        ],
        ids=["index_stage", "render_stage", "full_run_warnings"],
    )
    def test_run_output_contract(
        self,
        runner,
        cli_workspace,
        mock_plan_result,
        preview,
        stage,
        run_result,
        expected_tokens,
        expect_latest_render,
        expect_format_index,
    ):
        result, mocks = _invoke_run_direct(
            runner,
            ["run", stage],
            plan_result=mock_plan_result,
            run_result=run_result,
            selected_sources=None,
            prompted_stage=stage,
        )

        assert result.exit_code == 0
        for token in expected_tokens:
            assert token in result.output
        assert bool(mocks["latest_render"].call_count) is expect_latest_render
        assert bool(mocks["format_index"].call_count) is expect_format_index

    def test_run_preview_drive_error_contract(self, runner, cli_workspace, mock_plan_result, mock_run_result):
        result, _ = _invoke_run_direct(
            runner,
            ["run", "--preview"],
            plan_result=mock_plan_result,
            run_result=mock_run_result,
            selected_sources=["google-drive"],
            prompted_stage="all",
            plan_side_effect=DriveError("OAuth token expired"),
        )
        assert result.exit_code != 0
        assert "OAuth token expired" in result.output

    def test_run_execution_drive_error_contract(self, runner, cli_workspace, mock_plan_result, mock_run_result):
        result, _ = _invoke_run_direct(
            runner,
            ["run"],
            plan_result=mock_plan_result,
            run_result=mock_run_result,
            selected_sources=["google-drive"],
            prompted_stage="all",
            run_side_effect=DriveError("Drive API rate limit"),
        )
        assert result.exit_code != 0
        assert "Drive API rate limit" in result.output

    def test_run_reparse_contract(self, runner, cli_workspace, mock_plan_result, mock_run_result):
        run_results = iter([7, mock_run_result])

        def _run_async(coro):
            try:
                return next(run_results)
            finally:
                _close_coroutine(coro)

        with (
            patch("polylogue.cli.commands.run.run_coroutine_sync", side_effect=_run_async) as mock_reset_run,
            patch("polylogue.cli.run_workflow.run_coroutine_sync", side_effect=_run_async) as mock_pipeline_run,
            patch("polylogue.config.get_config", return_value=MagicMock(sources=[], render_root=Path("/render"))),
            patch("polylogue.cli.commands.run.resolve_sources", return_value=None),
            patch("polylogue.cli.commands.run.maybe_prompt_run_stage", return_value="all"),
            patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None),
            patch("polylogue.cli.run_workflow.format_counts", return_value="3 conversations, 30 messages"),
            patch("polylogue.cli.run_workflow.format_run_details", return_value=["Indexed: yes"]),
            patch("polylogue.cli.run_workflow.format_index_status", return_value="Index status: indexed"),
            patch("polylogue.cli.run_workflow.run_sources", new_callable=AsyncMock, return_value=mock_run_result),
        ):
            result = runner.invoke(cli, ["run", "--reparse"])

        assert result.exit_code == 0
        assert "Reset parse status for 7 raw records." in result.output
        assert mock_reset_run.call_count + mock_pipeline_run.call_count == 2

    def test_maybe_prompt_run_stage_skips_prompt_when_not_requested(self):
        env = SimpleNamespace(ui=MagicMock(plain=False))

        result = maybe_prompt_run_stage(env, stage="parse", prompt=False)

        assert result == "parse"
        env.ui.choose.assert_not_called()

    def test_maybe_prompt_run_stage_skips_prompt_in_plain_mode(self):
        env = SimpleNamespace(ui=MagicMock(plain=True))

        result = maybe_prompt_run_stage(env, stage="all", prompt=True)

        assert result == "all"
        env.ui.choose.assert_not_called()

    def test_maybe_prompt_run_stage_prompts_for_workflow_choice(self):
        env = SimpleNamespace(ui=MagicMock(plain=False))
        env.ui.choose.return_value = "reprocess"

        result = maybe_prompt_run_stage(env, stage="all", prompt=True)

        assert result == "reprocess"
        env.ui.choose.assert_called_once_with(
            "Select workflow for run",
            ["all", "reprocess", "acquire", "schema", "parse", "materialize", "render", "site", "index"],
        )

    def test_maybe_prompt_run_stage_rejects_empty_choice(self):
        env = SimpleNamespace(ui=MagicMock(plain=False))
        env.ui.choose.return_value = None

        with pytest.raises(SystemExit, match="run: No workflow selected"):
            maybe_prompt_run_stage(env, stage="all", prompt=True)


class TestTagsCommand:
    @pytest.mark.parametrize(
        ("tag_counts", "extra_args", "expected", "provider"),
        [
            ({"important": 5, "review": 3, "draft": 1}, [], ["important", "5", "review", "3", "3 total"], "chatgpt"),
            ({"claude-tag": 3}, ["-p", "claude-ai"], ["claude-tag"], "claude-ai"),
        ],
    )
    def test_tags_plain_output_matrix(self, runner, cli_workspace, tag_counts, extra_args, expected, provider):
        _seed_tag_counts(cli_workspace["db_path"], tag_counts, provider=provider)
        if extra_args == ["-p", "claude-ai"]:
            _seed_tag_counts(cli_workspace["db_path"], {"chatgpt-tag": 4}, provider="chatgpt")
        result = runner.invoke(cli, ["tags", *extra_args])
        assert result.exit_code == 0
        for token in expected:
            assert token in result.output
        if extra_args:
            assert "chatgpt-tag" not in result.output

    def test_tags_json_output(self, runner, cli_workspace):
        import json

        _seed_tag_counts(cli_workspace["db_path"], {"tag1": 10, "tag2": 2})
        result = runner.invoke(cli, ["tags", "--json"])
        assert result.exit_code == 0
        envelope = json.loads(result.output)
        assert envelope["result"]["tags"] == {"tag1": 10, "tag2": 2}

    def test_tags_limit_and_empty_hints(self, runner, cli_workspace):
        _seed_tag_counts(cli_workspace["db_path"], {"a": 10, "b": 5, "c": 1})
        result = runner.invoke(cli, ["tags", "-n", "2"])
        assert result.exit_code == 0
        assert "a" in result.output and "b" in result.output and "c" not in result.output

        empty_provider = runner.invoke(cli, ["tags", "-p", "gemini"])
        assert empty_provider.exit_code == 0
        assert "No tags found for provider 'gemini'" in empty_provider.output

    def test_tags_empty_hints(self, runner, cli_workspace):
        result = runner.invoke(cli, ["tags"])
        assert result.exit_code == 0
        assert "No tags found" in result.output
        assert "--add-tag" in result.output


class TestEmbedCommand:
    def test_embed_requires_api_key_unless_stats(self, runner, cli_workspace):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": ""}, clear=False):
            result = runner.invoke(cli, ["run", "embed"])
        assert result.exit_code != 0
        assert "VOYAGE_API_KEY" in result.output

    @pytest.mark.parametrize("stats_rows", [[5, 3, 45, 2], [10, 7, 100, 3]])
    def test_embed_stats_contract(self, runner, cli_workspace, stats_rows):
        with (
            patch("polylogue.cli.embed_stats.embedding_status_payload") as mock_payload,
            patch.dict(
                "os.environ",
                {"VOYAGE_API_KEY": ""},
                clear=False,
            ),
        ):
            total_conversations, embedded_conversations, embedded_messages, pending_conversations = stats_rows
            mock_payload.return_value = {
                "status": "partial",
                "total_conversations": total_conversations,
                "embedded_conversations": embedded_conversations,
                "embedded_messages": embedded_messages,
                "pending_conversations": pending_conversations,
                "embedding_coverage_percent": round(embedded_conversations / total_conversations * 100, 1)
                if total_conversations
                else 0.0,
                "retrieval_ready": embedded_messages > 0,
                "freshness_status": "partial",
                "stale_messages": 0,
                "messages_missing_provenance": 0,
                "oldest_embedded_at": None,
                "newest_embedded_at": None,
                "embedding_models": {},
                "embedding_dimensions": {},
                "retrieval_bands": {},
            }
            result = runner.invoke(cli, ["run", "embed", "--stats"])

        assert result.exit_code == 0
        assert "Embedding Statistics" in result.output
        assert str(stats_rows[0]) in result.output
        assert str(stats_rows[1]) in result.output
        assert str(stats_rows[2]) in result.output
        assert str(stats_rows[3]) in result.output

    def test_embed_stats_json_contract(self, runner, cli_workspace):
        with (
            patch("polylogue.cli.embed_stats.embedding_status_payload") as mock_payload,
            patch.dict(
                "os.environ",
                {"VOYAGE_API_KEY": ""},
                clear=False,
            ),
        ):
            mock_payload.return_value = {
                "status": "partial",
                "total_conversations": 5,
                "embedded_conversations": 3,
                "embedded_messages": 45,
                "pending_conversations": 2,
                "embedding_coverage_percent": 60.0,
                "retrieval_ready": True,
            }
            result = runner.invoke(cli, ["run", "embed", "--stats", "--json"])

        assert result.exit_code == 0
        assert '"status": "partial"' in result.output
        assert '"embedding_coverage_percent": 60.0' in result.output

    def test_embed_no_sqlite_vec_contract(self, runner, cli_workspace):
        with patch("polylogue.storage.search_providers.create_vector_provider", return_value=None):
            result = runner.invoke(
                cli, ["run", "embed"], env={"VOYAGE_API_KEY": "test-key", "POLYLOGUE_FORCE_PLAIN": "1"}
            )
        assert result.exit_code != 0
        assert "sqlite-vec" in result.output.lower()

    def test_embed_single_not_found_contract(self, runner, cli_workspace):
        with patch("polylogue.cli.embed_runtime.embed_single") as mock_single:
            import click as _click

            mock_single.side_effect = _click.Abort()

            result = runner.invoke(
                cli,
                ["run", "embed", "--conversation", "nonexistent-id"],
                env={"VOYAGE_API_KEY": "test-key", "POLYLOGUE_FORCE_PLAIN": "1"},
            )

        assert result.exit_code != 0

    @pytest.mark.parametrize(("cli_args", "expected_batch_kwargs", "expected_model"), EMBED_BATCH_CASES)
    def test_embed_batch_dispatch_matrix(self, runner, cli_workspace, cli_args, expected_batch_kwargs, expected_model):
        result, mock_standalone = _invoke_embed(runner, cli_args)
        assert result.exit_code == 0
        mock_standalone.assert_called_once()
        opts = mock_standalone.call_args[0][1]
        assert opts.rebuild is expected_batch_kwargs["rebuild"]
        assert opts.limit == expected_batch_kwargs["limit"]
        if expected_model is not None:
            assert opts.model == expected_model

    def test_embed_api_key_env_contract(self, runner, cli_workspace):
        with (
            patch("polylogue.cli.commands.run._run_embed_standalone") as mock_standalone,
            patch("polylogue.storage.search_providers.create_vector_provider") as mock_create,
        ):
            mock_create.return_value = MagicMock()

            result = runner.invoke(
                cli,
                ["run", "embed"],
                env={"VOYAGE_API_KEY": "test-key", "POLYLOGUE_FORCE_PLAIN": "1"},
            )

        assert result.exit_code == 0
        mock_standalone.assert_called_once()
