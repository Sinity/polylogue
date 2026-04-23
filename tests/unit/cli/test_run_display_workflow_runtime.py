from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from polylogue.cli.run_display_workflow import (
    display_result,
    handle_drive_error,
    render_preview_summary,
    render_sources,
)
from polylogue.cli.run_watch_workflow import WatchDisplayObserver, WatchStatusObserver
from polylogue.cli.types import AppEnv
from polylogue.config import Config
from polylogue.sources import DriveError
from polylogue.storage.run_state import (
    DriftBucket,
    PlanCounts,
    PlanDetails,
    PlanResult,
    RenderFailurePayload,
    RunCounts,
    RunDrift,
    RunResult,
)


def _env() -> AppEnv:
    ui = MagicMock()
    ui.console = MagicMock()
    return cast(AppEnv, SimpleNamespace(ui=ui))


def _cfg() -> Config:
    return Config(
        archive_root=Path("/tmp/archive"),
        render_root=Path("/tmp/render"),
        sources=[],
    )


def _run_result(
    *,
    index_error: str | None = None,
    render_failures: list[RenderFailurePayload] | None = None,
) -> RunResult:
    return RunResult(
        run_id="run-1",
        counts=RunCounts(conversations=3, messages=4, attachments=1),
        drift=RunDrift(new=DriftBucket(conversations=1), changed=DriftBucket(conversations=1)),
        indexed=True,
        index_error=index_error,
        duration_ms=1500,
        render_failures=render_failures or [],
    )


def test_display_result_renders_summary_latest_render_and_failure_hints() -> None:
    env = _env()
    cfg = _cfg()
    result = _run_result(
        index_error="index unavailable",
        render_failures=[{"conversation_id": "conv-1", "error": "boom"}],
    )

    with (
        patch("polylogue.cli.run_display_workflow.format_counts", return_value="3 conversations"),
        patch("polylogue.cli.run_display_workflow.format_run_details", return_value=["new=1", "changed=1"]),
        patch("polylogue.cli.run_display_workflow.format_index_status", return_value="Index status: failed"),
        patch("polylogue.cli.helpers.latest_render_path", return_value=Path("/tmp/render/latest.html")),
        patch("click.echo") as echo,
    ):
        display_result(
            env,
            cfg,
            result,
            "render",
            ["drive"],
            display_stage="site",
            stage_sequence=("render", "site"),
        )

    summary = cast(MagicMock, env.ui.summary)
    console_print = cast(MagicMock, env.ui.console.print)
    summary.assert_called_once_with(
        "Sync (site; drive)",
        ["Counts: 3 conversations", "Index status: failed", "new=1", "changed=1", "Duration: 1500ms"],
    )
    console_print.assert_called_once_with("Latest render: /tmp/render/latest.html")
    echoed = [call.args[0] for call in echo.call_args_list if call.args]
    assert any("Render failures (1):" in line for line in echoed)
    assert "  conv-1: boom" in echoed
    assert "Hint: re-run with `polylogue run render` to retry rendering." in echoed
    assert "Index error: index unavailable" in echoed
    assert "Hint: run `polylogue run index` to rebuild the index." in echoed


def test_display_result_uses_index_only_status_for_index_stage() -> None:
    env = _env()
    cfg = _cfg()

    with (
        patch("polylogue.cli.run_display_workflow.format_index_status", return_value="Index status: indexed"),
        patch("click.echo"),
    ):
        display_result(
            env,
            cfg,
            _run_result(),
            "index",
            None,
            stage_sequence=("index",),
        )

    summary = cast(MagicMock, env.ui.summary)
    console_print = cast(MagicMock, env.ui.console.print)
    summary.assert_called_once_with(
        "Sync (index)",
        ["Index status: indexed", "Duration: 1500ms"],
    )
    console_print.assert_not_called()


def test_render_sources_emits_json_payload_or_plain_summary() -> None:
    cfg = cast(
        Config,
        SimpleNamespace(
            sources=[
                SimpleNamespace(name="drive", path=None, folder="folder-1"),
                SimpleNamespace(name="local", path=Path("/tmp/inbox"), folder=None),
                SimpleNamespace(name="missing", path=None, folder=None),
            ]
        ),
    )
    env = cast(AppEnv, SimpleNamespace(config=cfg, ui=MagicMock()))

    with patch("polylogue.cli.machine_errors.emit_success") as emit_success:
        render_sources(env, json_output=True)

    emit_success.assert_called_once_with(
        {
            "sources": [
                {"name": "drive", "path": None, "folder": "folder-1", "kind": "drive"},
                {"name": "local", "path": "/tmp/inbox", "folder": None, "kind": "path"},
                {"name": "missing", "path": None, "folder": None, "kind": "path"},
            ]
        }
    )

    render_sources(env, json_output=False)
    summary = cast(MagicMock, env.ui.summary)
    summary.assert_called_once_with(
        "Sources",
        [
            "drive: drive folder 'folder-1'",
            "local: /tmp/inbox",
            "missing: (missing path)",
        ],
    )


def test_handle_drive_error_routes_to_run_failure_surface() -> None:
    with patch("polylogue.cli.run_display_workflow.fail") as fail:
        handle_drive_error(DriveError("bad credentials"))

    fail.assert_called_once_with("run", "bad credentials")


def test_render_preview_summary_formats_plan_snapshot() -> None:
    env = _env()
    plan_result = PlanResult(
        timestamp=1713873000,
        counts=PlanCounts(scan=5, store_raw=2),
        details=PlanDetails(new_raw=2, existing_raw=3),
        sources=["drive"],
        cursors={"drive": {"latest_path": "/tmp/inbox"}},
    )

    with (
        patch("polylogue.cli.run_display_workflow.format_plan_counts", return_value="scan=5"),
        patch("polylogue.cli.run_display_workflow.format_plan_details", return_value="new=2, existing=3"),
        patch("polylogue.cli.run_display_workflow.format_cursors", return_value="latest=/tmp/inbox"),
        patch("polylogue.cli.run_display_workflow.format_timestamp", return_value="2026-04-23 12:00:00"),
    ):
        render_preview_summary(env, selected_sources=["drive"], plan_snapshot=plan_result)

    summary = cast(MagicMock, env.ui.summary)
    summary.assert_called_once_with(
        "Preview",
        [
            "Sources: drive",
            "Work: scan=5",
            "State: new=2, existing=3",
            "Cursors: latest=/tmp/inbox",
            "Snapshot: 2026-04-23 12:00:00",
        ],
    )


def test_watch_display_observer_only_renders_when_activity_is_present() -> None:
    env = _env()
    cfg = _cfg()
    result = _run_result()

    observer = WatchDisplayObserver(env, cfg, "all", ["drive"], display_stage="render", stage_sequence=("render",))

    with (
        patch("polylogue.cli.run_watch_workflow.conversation_activity_counts", return_value=(2, 0, 0)),
        patch("polylogue.cli.run_watch_workflow.display_result") as display,
    ):
        observer.on_completed(result)

    display.assert_called_once_with(
        env,
        cfg,
        result,
        "all",
        ["drive"],
        display_stage="render",
        stage_sequence=("render",),
    )

    with (
        patch("polylogue.cli.run_watch_workflow.conversation_activity_counts", return_value=(0, 0, 0)),
        patch("polylogue.cli.run_watch_workflow.display_result") as display,
    ):
        observer.on_completed(result)

    display.assert_not_called()


def test_watch_status_observer_renders_idle_drive_and_generic_errors() -> None:
    observer = WatchStatusObserver()

    with (
        patch("time.strftime", return_value="12:34:56"),
        patch("click.echo") as echo,
    ):
        observer.on_idle(_run_result())
        observer.on_error(DriveError("drive failed"))
        observer.on_error(RuntimeError("boom"))

    echo.assert_any_call("No conversation changes at 12:34:56")
    echo.assert_any_call("Sync error: drive failed", err=True)
    echo.assert_any_call("Unexpected error during sync: boom", err=True)
