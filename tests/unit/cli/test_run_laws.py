"""Generalized contracts for CLI run command helper surfaces."""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.cli.commands.run import (
    _display_result,
    _run_sync_once,
    run_command,
)
from polylogue.cli.run_observers import (
    PlainProgressObserver as _PlainProgressObserver,
)
from polylogue.cli.run_observers import (
    _format_elapsed,
)
from polylogue.config import Config, get_config
from polylogue.storage.state_views import RunResult


def _make_env(*, plain: bool) -> MagicMock:
    env = MagicMock()
    env.ui = MagicMock()
    env.ui.plain = plain
    env.ui.console = MagicMock()
    env.ui.summary = MagicMock()
    env.config = Config(
        archive_root=Path("/tmp/archive"),
        render_root=Path("/tmp/render"),
        sources=[],
    )
    env.backend = MagicMock()
    env.repository = MagicMock()
    return env


def _run_result(*, conversations: int = 1, index_error: str | None = None) -> RunResult:
    return RunResult(
        run_id="run-123",
        counts={"conversations": conversations},
        drift={},
        indexed=index_error is None,
        index_error=index_error,
        duration_ms=150,
        render_failures=[],
    )


@contextmanager
def _workspace_paths() -> tuple[Config, Path]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        archive_root = root / "archive"
        env = {
            "POLYLOGUE_CONFIG": str(root / "config" / "config.json"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_STATE_HOME": str(root / "state"),
            "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
            "POLYLOGUE_RENDER_ROOT": str(archive_root / "render"),
            "POLYLOGUE_SCHEMA_VALIDATION": "off",
        }
        with patch.dict(os.environ, env, clear=False):
            yield get_config(), root


@settings(max_examples=40, deadline=None)
@given(
    plain=st.booleans(),
    stage=st.sampled_from(("all", "render", "parse", "index")),
    source_names=st.one_of(
        st.none(),
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=3, max_size=10),
            min_size=1,
            max_size=2,
            unique=True,
        ),
    ),
    render_format=st.sampled_from(("markdown", "html")),
)
def test_run_sync_once_forwards_arguments_contract(
    plain: bool,
    stage: str,
    source_names: list[str] | None,
    render_format: str,
) -> None:
    env = _make_env(plain=plain)
    cfg = Config(archive_root=Path("/tmp/archive"), render_root=Path("/tmp/render"), sources=[])
    result = _run_result()

    with (
        patch("polylogue.cli.run_workflow.run_sources", new_callable=AsyncMock) as mock_run,
        patch("builtins.print") as mock_print,
    ):
        mock_run.return_value = result
        observed = _run_sync_once(cfg, env, stage, source_names, render_format)

    assert observed == result
    kwargs = mock_run.call_args.kwargs
    assert kwargs["config"] == cfg
    assert kwargs["stage"] == stage
    assert kwargs["source_names"] == source_names
    assert kwargs["render_format"] == render_format

    if plain:
        mock_print.assert_any_call("Syncing...", flush=True)
    else:
        mock_print.assert_not_called()


def test_run_sync_once_forwards_plan_snapshot_contract() -> None:
    env = _make_env(plain=True)
    cfg = Config(archive_root=Path("/tmp/archive"), render_root=Path("/tmp/render"), sources=[])
    result = _run_result()

    with (
        patch("polylogue.cli.run_workflow.run_sources", new_callable=AsyncMock) as mock_run,
        patch("builtins.print"),
    ):
        mock_run.return_value = result
        observed = _run_sync_once(
            cfg,
            env,
            "all",
            None,
            "html",
            plan_snapshot=MagicMock(timestamp=123, counts={"scan": 1}, sources=[], cursors={}),
        )

    assert observed == result
    assert "plan" in mock_run.call_args.kwargs
    assert mock_run.call_args.kwargs["plan"] is not None


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    stage=st.sampled_from(("all", "render", "acquire", "validate", "parse", "index")),
    source_names=st.one_of(
        st.none(),
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=3, max_size=10),
            min_size=1,
            max_size=2,
            unique=True,
        ),
    ),
    conversations=st.integers(min_value=0, max_value=3),
    latest_exists=st.booleans(),
)
def test_display_result_title_and_render_lookup_contract(
    stage: str,
    source_names: list[str] | None,
    conversations: int,
    latest_exists: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        env = _make_env(plain=True)
        cfg = Config(archive_root=tmp_path / "archive", render_root=tmp_path / "render", sources=[])
        result = _run_result(conversations=conversations)
        latest_path = tmp_path / "render" / "latest" if latest_exists else None

        with patch("polylogue.cli.helpers.latest_render_path", return_value=latest_path) as mock_latest:
            _display_result(env, cfg, result, stage, source_names)

        title = env.ui.summary.call_args.args[0]
        expected_parts: list[str] = []
        if stage != "all":
            expected_parts.append(stage)
        if source_names:
            expected_parts.append(", ".join(source_names))
        expected_title = f"Sync ({'; '.join(expected_parts)})" if expected_parts else "Sync"
        assert title == expected_title

        if stage in {"render", "all"}:
            mock_latest.assert_called_once_with(cfg.render_root)
            if latest_exists:
                env.ui.console.print.assert_any_call(f"Latest render: {latest_path}")
        else:
            mock_latest.assert_not_called()


@settings(max_examples=40, deadline=None)
@given(
    notify=st.booleans(),
    exec_cmd=st.booleans(),
    webhook=st.booleans(),
)
def test_run_command_watch_flag_contract(notify: bool, exec_cmd: bool, webhook: bool) -> None:
    runner = CliRunner()
    env = _make_env(plain=True)
    args: list[str] = []
    if notify:
        args.append("--notify")
    if exec_cmd:
        args.extend(["--exec", "echo test"])
    if webhook:
        args.extend(["--webhook", "https://example.com"])

    with patch("polylogue.cli.commands.run.resolve_sources", return_value=None), patch(
        "polylogue.cli.commands.run.maybe_prompt_sources", return_value=None
    ), patch("polylogue.cli.commands.run._run_sync_once", return_value=_run_result()):
        result = runner.invoke(run_command, args, obj=env)

    if notify or exec_cmd or webhook:
        assert result.exit_code != 0
        assert "require --watch mode" in result.output.lower()
    else:
        assert result.exit_code == 0


@settings(max_examples=30, deadline=None)
@given(
    title_changed=st.booleans(),
    content_changed=st.booleans(),
)
@pytest.mark.asyncio
async def test_run_rerenders_when_title_or_content_changes_contract(
    title_changed: bool,
    content_changed: bool,
) -> None:
    from polylogue.config import Source, get_config
    from polylogue.pipeline.runner import run_sources
    from tests.infra.source_builders import GenericConversationBuilder

    with _workspace_paths() as (config, root):
        inbox = root / "inbox"
        source_file = inbox / "conversation.json"

        initial_title = "Initial title"
        initial_content = "hello"
        GenericConversationBuilder("conv-law").title(initial_title).add_user(initial_content).write_to(source_file)

        config = get_config()
        config.sources = [Source(name="inbox", path=source_file)]

        await run_sources(config=config, stage="all")
        conversation_path = next(config.render_root.rglob("conversation.md"))
        first_mtime = conversation_path.stat().st_mtime_ns

        updated_title = "Updated title" if title_changed else initial_title
        updated_content = "changed body" if content_changed else initial_content
        GenericConversationBuilder("conv-law").title(updated_title).add_user(updated_content).write_to(source_file)
        time.sleep(0.02)

        await run_sources(config=config, stage="all")
        second_mtime = conversation_path.stat().st_mtime_ns

        if title_changed or content_changed:
            assert second_mtime > first_mtime
        else:
            assert second_mtime == first_mtime


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (59.4, "59s"),
        (60, "1m00s"),
        (125, "2m05s"),
        (3600, "1h00m00s"),
        (3725, "1h02m05s"),
    ],
)
def test_format_elapsed_contract(seconds: float, expected: str) -> None:
    assert _format_elapsed(seconds) == expected


def test_plain_progress_observer_stage_switch_contract() -> None:
    with patch("builtins.print") as mock_print, patch(
        "polylogue.cli.run_observers.time.time",
        side_effect=[100.0, 101.2, 101.2, 102.8, 102.8],
    ):
        observer = _PlainProgressObserver(banner="Running...")
        observer.on_progress(2, "Scanning: 2")
        observer.on_progress(3, "Validation[batch]: 3")

    assert observer._stage_key("Validation[batch]: 3") == "Validation"
    lines = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert lines == [
        "Running...",
        "  Scanning: 2: 2 [1s total]...",
        "  Scanning: done (2 in 0s)",
        "  Validation[batch]: 3: 3 [1s total]...",
    ]


def test_plain_progress_observer_completion_contract() -> None:
    result = _run_result(conversations=3)

    with patch("builtins.print") as mock_print, patch(
        "polylogue.cli.run_observers.time.time",
        side_effect=[200.0, 205.0],
    ):
        observer = _PlainProgressObserver(banner="Syncing...")
        observer.on_completed(result)

    lines = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert lines == [
        "Syncing...",
        "  Counts: 3 conv (3 new)",
        "  Pipeline complete in 5s",
    ]
