from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.core.json import JSONDocument, loads
from polylogue.daemon.cli import main
from polylogue.sources.live import WatchSource


def test_polylogued_help_lists_watch_command() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "browser-capture" in result.output
    assert "run" in result.output
    assert "status" in result.output
    assert "watch" in result.output
    assert "long-lived Polylogue local services" in result.output


def test_polylogued_status_json_reports_daemon_components(tmp_path: Path) -> None:
    sources = (
        WatchSource(name="exists", root=tmp_path),
        WatchSource(name="missing", root=tmp_path / "missing"),
    )

    with patch("polylogue.daemon.status.default_sources", return_value=sources):
        result = CliRunner().invoke(
            main,
            [
                "status",
                "--spool",
                str(tmp_path / "captures"),
                "--format",
                "json",
            ],
        )

    assert result.exit_code == 0
    payload = loads(result.output)
    assert isinstance(payload, dict)
    live = cast(JSONDocument, payload["live"])
    browser_capture = cast(JSONDocument, payload["browser_capture"])
    assert payload["daemon"] == "polylogued"
    assert live["source_count"] == 2
    assert live["existing_source_count"] == 1
    assert browser_capture["spool_path"] == str(tmp_path / "captures")


def test_polylogued_status_plain_reports_daemon_components(tmp_path: Path) -> None:
    sources = (WatchSource(name="exists", root=tmp_path),)

    with patch("polylogue.daemon.status.default_sources", return_value=sources):
        result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0
    assert "Polylogue daemon" in result.output
    assert "Live sources: 1/1 available" in result.output
    assert f"exists: {tmp_path} (available)" in result.output
    assert "Browser capture spool:" in result.output


def test_polylogued_browser_capture_help_lists_service_commands() -> None:
    result = CliRunner().invoke(main, ["browser-capture", "--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "status" in result.output


def test_polylogued_run_uses_default_sources() -> None:
    sources = (WatchSource(name="codex", root=Path("/tmp/codex")),)

    with (
        patch("polylogue.daemon.cli.default_sources", return_value=sources) as default_sources,
        patch("polylogue.daemon.cli.asyncio.run") as run,
    ):
        result = CliRunner().invoke(main, ["run", "--no-browser-capture", "--debounce-s", "0.25"])

    assert result.exit_code == 0
    default_sources.assert_called_once_with()
    coroutine = run.call_args.kwargs.get("main") or run.call_args.args[0]
    assert inspect.iscoroutine(coroutine)
    coroutine.close()
    assert "Starting polylogued (watch=1 source(s)). Ctrl-C to stop." in result.stderr


def test_polylogued_run_rejects_empty_component_set() -> None:
    result = CliRunner().invoke(main, ["run", "--no-watch", "--no-browser-capture"])

    assert result.exit_code != 0
    assert "at least one daemon component must be enabled" in result.output


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


def test_run_daemon_services_stops_live_watcher_on_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    class FakePolylogue:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    stopped: list[bool] = []

    class FakeWatcher:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            raise RuntimeError("watch stopped")

        def stop(self) -> None:
            stopped.append(True)

    with (
        patch.object(daemon_cli, "Polylogue", FakePolylogue),
        patch.object(daemon_cli, "LiveWatcher", FakeWatcher),
        pytest.raises(RuntimeError, match="watch stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(WatchSource(name="codex", root=Path("/tmp/codex")),),
                debounce_s=1.0,
                enable_watch=True,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert stopped == [True]


def test_run_daemon_services_closes_browser_capture_server_on_failure() -> None:
    from polylogue.daemon import cli as daemon_cli

    class FakeServer:
        shutdown_called = False
        close_called = False

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            assert poll_interval == 0.5
            raise RuntimeError("server stopped")

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.close_called = True

    server = FakeServer()
    with (
        patch.object(daemon_cli, "make_server", return_value=server),
        pytest.raises(RuntimeError, match="server stopped"),
    ):
        asyncio.run(
            daemon_cli.run_daemon_services(
                sources=(),
                debounce_s=1.0,
                enable_watch=False,
                enable_browser_capture=True,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
                browser_capture_spool_path=None,
            )
        )

    assert server.shutdown_called is True
    assert server.close_called is True
