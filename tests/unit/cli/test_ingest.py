"""Tests for polylogue ingest truthfulness (#869)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
from urllib.request import Request


def test_ingest_command_registered() -> None:
    """ingest command must be available in the CLI group."""
    from polylogue.cli.click_app import cli

    commands = {name for name in cli.commands if not name.startswith("_")}
    assert "ingest" in commands, "ingest command not registered"


def test_ingest_help_includes_inbox_info() -> None:
    """ingest --help should document that files are staged for daemon processing."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "daemon" in result.output.lower() or "polylogued" in result.output.lower(), (
        "ingest help should reference the daemon"
    )


class _FakeDaemonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeDaemonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_ingest_command_stages_local_path_before_daemon_request(
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
                "operation_id": "ingest-source.jsonl",
                "kind": "import",
                "status": "pending",
                "path": staged_path,
                "message": "scheduled",
            }
        )

    runner = CliRunner()
    with patch("polylogue.cli.commands.ingest.urlopen", side_effect=fake_urlopen):
        result = runner.invoke(
            cli,
            ["ingest", str(source), "--daemon-url", "http://127.0.0.1:8766"],
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
