from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.daemon.cli import main
from polylogue.sources.live import WatchSource


def test_polylogued_help_lists_watch_command() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "browser-capture" in result.output
    assert "watch" in result.output
    assert "long-lived Polylogue local services" in result.output


def test_polylogued_browser_capture_help_lists_service_commands() -> None:
    result = CliRunner().invoke(main, ["browser-capture", "--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "status" in result.output


def test_polylogued_watch_uses_default_sources() -> None:
    runner = CliRunner()
    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    with (
        patch("polylogue.daemon.cli.default_sources", return_value=sources) as default_sources,
        patch("polylogue.daemon.cli.asyncio.run") as run,
    ):
        result = runner.invoke(main, ["watch", "--debounce-s", "0.25"])

    assert result.exit_code == 0
    default_sources.assert_called_once_with()
    coroutine = run.call_args.kwargs.get("main") or run.call_args.args[0]
    assert inspect.iscoroutine(coroutine)
    coroutine.close()
    assert "Watching 1 source(s); debounce=0.25s" in result.stderr


def test_polylogued_watch_builds_sources_from_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "claude-code"
    root_b = tmp_path / "codex"

    with patch("polylogue.daemon.cli.asyncio.run") as run:
        result = CliRunner().invoke(
            main,
            [
                "watch",
                "--root",
                str(root_a),
                "--root",
                str(root_b),
            ],
        )

    assert result.exit_code == 0
    coroutine = run.call_args.kwargs.get("main") or run.call_args.args[0]
    assert inspect.iscoroutine(coroutine)
    coroutine.close()
    assert "Watching 2 source(s); debounce=2.0s" in result.stderr


def test_run_live_watcher_stops_on_keyboard_interrupt() -> None:
    from polylogue.daemon import cli as daemon_cli

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    stopped: list[bool] = []

    class FakeWatcher:
        stopped = False

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            raise KeyboardInterrupt

        def stop(self) -> None:
            self.stopped = True
            stopped.append(self.stopped)

    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    with (
        patch.object(daemon_cli, "Polylogue", FakePolylogue),
        patch.object(daemon_cli, "LiveWatcher", FakeWatcher),
    ):
        asyncio.run(daemon_cli.run_live_watcher(sources=sources, debounce_s=1.0))

    assert stopped == [True]
