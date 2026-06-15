"""Tests for polylogue import truthfulness (#869 / #1264)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request


def test_import_command_registered() -> None:
    """import command must be available in the CLI group."""
    from polylogue.cli.click_app import cli

    commands = {name for name in cli.commands if not name.startswith("_")}
    assert "import" in commands, "import command not registered"


def test_import_help_includes_inbox_info() -> None:
    """import --help should document that files are staged for daemon processing."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["import", "--help"])
    assert result.exit_code == 0
    assert "daemon" in result.output.lower() or "polylogued" in result.output.lower(), (
        "import help should reference the daemon"
    )
    assert "--demo" in result.output


class _FakeDaemonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeDaemonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_import_command_stages_local_path_before_daemon_request(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """CLI owns arbitrary local path reads; HTTP daemon receives inbox path."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"session"}\n')

    captured: dict[str, Any] = {}

    def fake_urlopen(req: Request, timeout: int) -> _FakeDaemonResponse:
        captured["request"] = req
        captured["timeout"] = timeout
        assert req.data is not None
        request_data = cast("bytes", req.data)
        staged_path = json.loads(request_data.decode("utf-8"))["path"]
        return _FakeDaemonResponse(
            {
                "ok": True,
                "operation_id": "import-source.jsonl",
                "kind": "import",
                "status": "pending",
                "path": staged_path,
                "message": "scheduled",
            }
        )

    runner = CliRunner()
    with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
        result = runner.invoke(
            cli,
            ["import", str(source), "--daemon-url", "http://127.0.0.1:8766"],
        )

    assert result.exit_code == 0, result.output
    staged = workspace_env["archive_root"] / "inbox" / source.name
    assert staged.read_text() == source.read_text()

    request = cast("Request", captured["request"])
    assert request.data is not None
    request_data = cast("bytes", request.data)
    body = json.loads(request_data.decode("utf-8"))
    assert body == {"path": str(staged)}
    assert body["path"] != str(source)
    assert captured["timeout"] == 5

    # Truthfulness: success output must point at observable state — the
    # staged inbox path AND actionable next-step guidance. The old
    # "polylogue status" message was misleading (status doesn't show
    # recent completed operations); #1679 replaced it with journalctl
    # for live progress and polylogue stats to verify the import landed.
    assert str(staged) in result.output
    assert "polylogue stats" in result.output


def test_import_demo_materializes_fixture_world_before_daemon_request(
    workspace_env: dict[str, Path],
) -> None:
    """--demo writes approved fixture sources and still requires daemon acceptance."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    captured: dict[str, Any] = {}

    def fake_urlopen(req: Request, timeout: int) -> _FakeDaemonResponse:
        captured["request"] = req
        captured["timeout"] = timeout
        assert req.data is not None
        request_data = cast("bytes", req.data)
        staged_path = json.loads(request_data.decode("utf-8"))["path"]
        return _FakeDaemonResponse(
            {
                "ok": True,
                "operation_id": "import-demo-fixture-world",
                "kind": "import",
                "status": "pending",
                "path": staged_path,
                "message": "scheduled",
            }
        )

    runner = CliRunner()
    with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
        result = runner.invoke(cli, ["import", "--demo"])

    assert result.exit_code == 0, result.output
    source_root = workspace_env["archive_root"] / "demo-fixture-world-source"
    staged = workspace_env["archive_root"] / "inbox" / "demo-fixture-world-source"
    assert sorted(path.name for path in source_root.iterdir()) == ["chatgpt", "claude-code", "codex"]
    assert sorted(path.name for path in staged.iterdir()) == ["chatgpt", "claude-code", "codex"]
    assert len(tuple(staged.rglob("demo-*.json*"))) == 3

    request = cast("Request", captured["request"])
    assert request.data is not None
    body = json.loads(cast("bytes", request.data).decode("utf-8"))
    assert body == {"path": str(staged)}
    assert captured["timeout"] == 5
    assert str(staged) in result.output
    assert "polylogue stats" in result.output


def test_import_requires_path_or_demo() -> None:
    """Bare import refuses to claim success without a source selector."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["import"])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "path" in combined
    assert "--demo" in combined


def test_import_rejects_path_with_demo(tmp_path: Path) -> None:
    """PATH and --demo are mutually exclusive source selectors."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"session"}\n')

    runner = CliRunner()
    result = runner.invoke(cli, ["import", str(source), "--demo"])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "either path or --demo" in combined


def test_import_rejects_missing_path(tmp_path: Path) -> None:
    """A path that does not exist is rejected by Click before any daemon call."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    missing = tmp_path / "does-not-exist.jsonl"
    runner = CliRunner()
    result = runner.invoke(cli, ["import", str(missing)])

    assert result.exit_code != 0
    # Click's standard "Path 'X' does not exist" or equivalent.
    assert "does not exist" in result.output.lower() or "no such" in result.output.lower()


def test_import_rejects_when_daemon_unreachable(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """With no daemon running, the command must fail with an actionable error."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"session"}\n')

    def fake_urlopen(req: Request, timeout: int) -> object:
        raise URLError("Connection refused")

    runner = CliRunner()
    with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
        result = runner.invoke(
            cli,
            ["import", str(source), "--daemon-url", "http://127.0.0.1:65535"],
        )

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    # Must name the daemon binary so the user knows what to start.
    assert "polylogued" in combined
    assert "127.0.0.1:65535" in combined


def test_import_surfaces_http_error_with_staged_path(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Daemon HTTP 4xx/5xx is reported truthfully, naming the staged file."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"session"}\n')

    def fake_urlopen(req: Request, timeout: int) -> object:
        raise HTTPError(
            url="http://127.0.0.1:8766/api/ingest",
            code=400,
            msg="invalid_request",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    runner = CliRunner()
    with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    staged = workspace_env["archive_root"] / "inbox" / source.name
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "400" in combined
    assert str(staged).lower() in combined


def test_import_refuses_unrecognized_daemon_status(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """If the daemon returns an unrecognized status, refuse to claim success."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"session"}\n')

    def fake_urlopen(req: Request, timeout: int) -> _FakeDaemonResponse:
        return _FakeDaemonResponse(
            {
                "ok": True,
                "operation_id": "import-source.jsonl",
                "kind": "import",
                "status": "mystery_status",
                "path": "/somewhere",
                "message": "??",
            }
        )

    runner = CliRunner()
    with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "mystery_status" in combined


def test_import_surfaces_daemon_failure_status(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Daemon-reported failure must be surfaced, not swallowed."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"session"}\n')

    def fake_urlopen(req: Request, timeout: int) -> _FakeDaemonResponse:
        return _FakeDaemonResponse(
            {
                "ok": False,
                "operation_id": "import-source.jsonl",
                "kind": "import",
                "status": "failed",
                "error": "inbox locked by another operation",
            }
        )

    runner = CliRunner()
    with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "inbox locked" in combined
