"""Tests for polylogue import truthfulness (#869 / #1264)."""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
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
    assert body == {"path": str(staged), "source_path": str(source.resolve())}
    assert body["path"] != str(source)
    assert captured["timeout"] == 5

    # Truthfulness: success output must point at observable state — the
    # staged inbox path AND actionable next-step guidance. The old
    # "polylogue ops status" message was misleading (status doesn't show
    # recent completed operations); #1679 replaced it with journalctl
    # for live progress. Convergence/readiness checks should point at daemon
    # and archive status surfaces, not a generic analyze command.
    assert str(staged) in result.output
    assert "polylogued status" in result.output
    assert "polylogue status --full" in result.output


def test_import_command_snapshots_hermes_state_db_before_daemon_request(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Hermes state.db staging uses SQLite backup instead of a raw file copy."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli
    from polylogue.config import Source
    from polylogue.sources.parsers import hermes_state
    from polylogue.sources.source_parsing import iter_source_sessions_with_raw
    from polylogue.sources.sqlite_snapshot import original_sqlite_source_path

    source = tmp_path / "state.db"
    with sqlite3.connect(source) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, user_id TEXT, session_key TEXT,
                chat_id TEXT, chat_type TEXT, thread_id TEXT, model TEXT,
                model_config TEXT, system_prompt TEXT, parent_session_id TEXT,
                started_at REAL, ended_at REAL, end_reason TEXT,
                message_count INTEGER, tool_call_count INTEGER,
                input_tokens INTEGER, output_tokens INTEGER,
                cache_read_tokens INTEGER, cache_write_tokens INTEGER,
                reasoning_tokens INTEGER, cwd TEXT, git_branch TEXT,
                git_repo_root TEXT, billing_provider TEXT, billing_base_url TEXT,
                billing_mode TEXT, estimated_cost_usd REAL, actual_cost_usd REAL,
                cost_status TEXT, cost_source TEXT, pricing_version TEXT,
                title TEXT, api_call_count INTEGER, handoff_state TEXT,
                handoff_platform TEXT, handoff_error TEXT,
                compression_failure_cooldown_until REAL,
                compression_failure_error TEXT, rewind_count INTEGER, archived INTEGER
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_calls TEXT,
                tool_name TEXT, timestamp REAL NOT NULL, token_count INTEGER,
                finish_reason TEXT, reasoning TEXT, reasoning_content TEXT,
                reasoning_details TEXT, codex_reasoning_items TEXT,
                codex_message_items TEXT, platform_message_id TEXT,
                observed INTEGER, active INTEGER, compacted INTEGER
            );
            INSERT INTO sessions(id, started_at, title) VALUES ('h1', 1.0, 'Hermes');
            INSERT INTO messages(session_id, role, content, timestamp) VALUES ('h1', 'user', 'hello', 1.0);
            """
        )
        conn.execute("PRAGMA journal_mode=WAL")

    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    writer.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES ('h1', 'assistant', 'WAL turn', 2.0)"
    )
    writer.commit()

    captured: dict[str, Any] = {}

    def fake_urlopen(req: Request, timeout: int) -> _FakeDaemonResponse:
        captured["request"] = req
        captured["timeout"] = timeout
        assert req.data is not None
        staged_path = json.loads(cast("bytes", req.data).decode("utf-8"))["path"]
        return _FakeDaemonResponse(
            {
                "ok": True,
                "operation_id": "import-state.db",
                "kind": "import",
                "status": "pending",
                "path": staged_path,
                "message": "scheduled",
            }
        )

    try:
        with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
            result = CliRunner().invoke(cli, ["import", str(source), "--daemon-url", "http://127.0.0.1:8766"])
    finally:
        writer.close()

    assert result.exit_code == 0, result.output
    staged = workspace_env["archive_root"] / "inbox" / "state.db"
    with sqlite3.connect(staged) as conn:
        assert conn.execute("SELECT title FROM sessions WHERE id = 'h1'").fetchone()[0] == "Hermes"
        assert conn.execute("SELECT content FROM messages ORDER BY id DESC LIMIT 1").fetchone()[0] == "WAL turn"
    assert original_sqlite_source_path(staged) == source.resolve()
    assert json.loads(cast("bytes", cast("Request", captured["request"]).data).decode("utf-8")) == {
        "path": str(staged),
        "source_path": str(source.resolve()),
    }

    direct = hermes_state.parse_state_db(source, profile_root=source.parent)[0]
    [(raw, imported)] = list(
        iter_source_sessions_with_raw(
            Source(name="inbox", path=staged),
            capture_raw=True,
            blob_root=tmp_path / "blobs",
        )
    )
    assert raw is not None and raw.source_path == str(source.resolve())
    assert imported.provider_session_id == direct.provider_session_id


def test_import_command_uses_daemon_url_env_by_default(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Dev-loop imports must not stage into one archive and schedule another daemon."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"session"}\n')
    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:9876")

    captured: dict[str, Any] = {}

    def fake_urlopen(req: Request, timeout: int) -> _FakeDaemonResponse:
        captured["request"] = req
        assert req.data is not None
        staged_path = json.loads(cast("bytes", req.data).decode("utf-8"))["path"]
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

    with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
        result = CliRunner().invoke(cli, ["import", str(source)])

    assert result.exit_code == 0, result.output
    request = cast("Request", captured["request"])
    assert request.full_url == "http://127.0.0.1:9876/api/ingest"
    assert "Daemon:       http://127.0.0.1:9876" in result.output
    assert (workspace_env["archive_root"] / "inbox" / source.name).is_file()


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
    assert sorted(path.name for path in source_root.iterdir()) == [
        "browser-capture",
        "chatgpt",
        "claude-ai",
        "claude-code",
        "codex",
        "gemini",
    ]
    assert sorted(path.name for path in staged.iterdir()) == [
        "browser-capture",
        "chatgpt",
        "claude-ai",
        "claude-code",
        "codex",
        "gemini",
    ]
    assert len(tuple(staged.rglob("demo-*.json*"))) == 4

    request = cast("Request", captured["request"])
    assert request.data is not None
    body = json.loads(cast("bytes", request.data).decode("utf-8"))
    assert body == {"path": str(staged), "source_path": str(source_root.resolve())}
    assert captured["timeout"] == 5
    assert str(staged) in result.output
    assert "polylogued status" in result.output
    assert "polylogue status --full" in result.output


def test_import_demo_wait_verifies_after_daemon_acceptance(
    workspace_env: dict[str, Path],
) -> None:
    """--demo --wait blocks on the semantic verifier after daemon scheduling."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    captured: dict[str, Any] = {}

    def fake_urlopen(req: Request, timeout: int) -> _FakeDaemonResponse:
        del timeout
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

    def fake_wait(*, timeout_s: float, require_overlays: bool = False) -> None:
        captured["timeout_s"] = timeout_s
        captured["require_overlays"] = require_overlays

    runner = CliRunner()
    with (
        patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen),
        patch("polylogue.cli.commands.import_command._wait_for_demo_archive_ready", side_effect=fake_wait),
    ):
        result = runner.invoke(cli, ["import", "--demo", "--wait", "--timeout", "12.5"])

    assert result.exit_code == 0, result.output
    assert captured == {"timeout_s": 12.5, "require_overlays": False}
    staged = workspace_env["archive_root"] / "inbox" / "demo-fixture-world-source"
    assert str(staged) in result.output
    assert "Demo archive verified" in result.output
    assert "overlays=no" in result.output


def test_import_demo_wait_with_overlays_seeds_after_convergence() -> None:
    """--with-overlays applies deterministic user overlays after base ingest."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    events: list[str] = []

    def fake_urlopen(req: Request, timeout: int) -> _FakeDaemonResponse:
        del timeout
        assert req.data is not None
        staged_path = json.loads(cast("bytes", req.data).decode("utf-8"))["path"]
        events.append("daemon")
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

    def fake_wait(*, timeout_s: float, require_overlays: bool = False) -> None:
        assert timeout_s == 30.0
        assert require_overlays is False
        events.append("wait-base")

    def fake_seed(archive_root: Path) -> object:
        assert archive_root.exists()
        events.append("seed-overlays")
        return object()

    def fake_verify(*, require_overlays: bool = False) -> None:
        assert require_overlays is True
        events.append("verify-overlays")

    runner = CliRunner()
    with (
        patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen),
        patch("polylogue.cli.commands.import_command._wait_for_demo_archive_ready", side_effect=fake_wait),
        patch("polylogue.scenarios.seed_demo_user_overlays", side_effect=fake_seed),
        patch("polylogue.cli.commands.import_command._verify_demo_now", side_effect=fake_verify),
    ):
        result = runner.invoke(cli, ["import", "--demo", "--wait", "--with-overlays"])

    assert result.exit_code == 0, result.output
    assert events == ["daemon", "wait-base", "seed-overlays", "verify-overlays"]
    assert "overlays=yes" in result.output


def test_import_wait_requires_demo(tmp_path: Path) -> None:
    """Waiting is tied to the deterministic demo verifier, not arbitrary imports."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"session"}\n')

    runner = CliRunner()
    result = runner.invoke(cli, ["import", str(source), "--wait"])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "--wait" in combined
    assert "--demo" in combined


def test_import_demo_with_overlays_requires_wait() -> None:
    """Overlay seeding needs daemon-converged target refs."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["import", "--demo", "--with-overlays"])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "--with-overlays" in combined
    assert "--wait" in combined


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


def test_import_surfaces_structured_daemon_rejection_detail(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Unsupported/degraded daemon preflight details reach the operator."""
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "unknown.json"
    source.write_text('{"not":"an export"}\n')

    def fake_urlopen(req: Request, timeout: int) -> object:
        del req, timeout
        raise HTTPError(
            url="http://127.0.0.1:8766/api/ingest",
            code=415,
            msg="Unsupported Media Type",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(
                json.dumps(
                    {
                        "ok": False,
                        "error": "unsupported_import_source",
                        "detail": "Import source is unsupported: no parseable Polylogue export shape was detected.",
                    }
                ).encode("utf-8")
            ),
        )

    runner = CliRunner()
    with patch("polylogue.cli.commands.import_command.urlopen", side_effect=fake_urlopen):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "415" in combined
    assert "unsupported_import_source" in combined
    assert "no parseable polylogue export shape" in combined
    assert str(workspace_env["archive_root"] / "inbox" / source.name).lower() in combined


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
