"""Tests for polylogue import truthfulness (#869 / #1264).

Since S11 the command no longer speaks the browser HTTP API: it submits the
declared ``ingest`` operation through
:func:`polylogue.cli.operation_kernel.configured_accepted_operation` over the
archive-scoped daemon socket, and runs the import preflight client-side. The
truthfulness laws these tests carry are unchanged — an unreachable daemon, a
refused operation, and an envelope without a durable acceptance reference each
have to fail loudly and name the staged file — only their seam moved.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

from polylogue.cli.operation_kernel import (
    OperationFailedError,
    OperationIndeterminateError,
    OperationUnavailableError,
)

# One record of a shape the import preflight actually admits. The preflight is
# a real admissibility check, so a placeholder like ``{"type": "session"}`` is
# refused before any operation is submitted (see
# ``test_import_refuses_inadmissible_source_before_submitting``).
_SUPPORTED_RECORD = {
    "type": "user",
    "uuid": "u1",
    "sessionId": "s1",
    "timestamp": "2024-01-01T00:00:00Z",
    "message": {"role": "user", "content": "hi"},
}


def _write_supported_source(path: Path) -> Path:
    path.write_text(json.dumps(_SUPPORTED_RECORD) + "\n")
    return path


def _accepted_envelope(request_id: str = "ingest-request") -> dict[str, object]:
    """The envelope shape a daemon returns once ingest is durably admitted."""
    return {
        "operation": "ingest",
        "outcome": "accepted",
        "result": {},
        "request_id": request_id,
        "accepted_reference": {"request_id": request_id},
    }


class _RecordingSubmit:
    """Stand-in for the accepted-operation seam that records its request."""

    def __init__(self, envelope: dict[str, object] | None = None) -> None:
        self.envelope = envelope if envelope is not None else _accepted_envelope()
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config
        self.calls.append((operation, dict(payload)))
        return self.envelope

    @property
    def payload(self) -> dict[str, object]:
        assert self.calls, "no ingest submission was made"
        return self.calls[-1][1]


def _patch_submit(submit: _RecordingSubmit) -> Any:
    return patch("polylogue.cli.operation_kernel.configured_accepted_operation", new=submit)


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


def test_import_command_stages_local_path_before_daemon_request(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """CLI owns arbitrary local path reads; the daemon receives the inbox path.

    Anti-vacuity: submitting ``source_path`` as the operation's ``path`` (the
    pre-staging path, i.e. asking the daemon to read an arbitrary local file)
    makes the payload assertion red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = _write_supported_source(tmp_path / "source.jsonl")
    submit = _RecordingSubmit()

    runner = CliRunner()
    with _patch_submit(submit):
        result = runner.invoke(
            cli,
            ["import", str(source), "--daemon-url", "http://127.0.0.1:8766"],
        )

    assert result.exit_code == 0, result.output
    staged = workspace_env["archive_root"] / "inbox" / source.name
    assert staged.read_text() == source.read_text()

    assert submit.calls[-1][0] == "ingest"
    assert submit.payload == {
        "path": str(staged),
        "source_path": str(source.resolve()),
        "idempotency_key": None,
    }
    assert submit.payload["path"] != str(source)

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
    """Hermes state.db staging uses SQLite backup instead of a raw file copy.

    The shared import preflight admits only Antigravity trajectory stores among
    SQLite sources (``sources/import_preflight.py::_preflight_sqlite``), so a
    Hermes ``state.db`` is refused as ``unsupported_import_source`` today —
    exactly as the daemon's ingest route already refuses it. That gap is not
    this test's subject: staging and the submitted payload are, so the
    admissibility check is stood in for here.

    Anti-vacuity: copying the file byte-for-byte instead of taking a SQLite
    backup loses the WAL-resident row and makes the message assertion red.
    """
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

    submit = _RecordingSubmit()

    def admissible(_path: Path) -> Any:
        from polylogue.sources.import_preflight import ImportPreflightResult, ImportPreflightStatus

        return ImportPreflightResult(
            status=ImportPreflightStatus.SUPPORTED,
            source_path=str(_path),
            candidate_count=1,
            supported_count=1,
        )

    try:
        with (
            _patch_submit(submit),
            patch("polylogue.sources.import_preflight.preflight_import_source", new=admissible),
        ):
            result = CliRunner().invoke(cli, ["import", str(source), "--daemon-url", "http://127.0.0.1:8766"])
    finally:
        writer.close()

    assert result.exit_code == 0, result.output
    staged = workspace_env["archive_root"] / "inbox" / "state.db"
    with sqlite3.connect(staged) as conn:
        assert conn.execute("SELECT title FROM sessions WHERE id = 'h1'").fetchone()[0] == "Hermes"
        assert conn.execute("SELECT content FROM messages ORDER BY id DESC LIMIT 1").fetchone()[0] == "WAL turn"
    assert original_sqlite_source_path(staged) == source.resolve()
    assert submit.payload == {
        "path": str(staged),
        "source_path": str(source.resolve()),
        "idempotency_key": None,
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


def test_stage_for_daemon_removes_stale_sqlite_provenance(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    from polylogue.cli.commands.import_command import _stage_for_daemon
    from polylogue.sources.sqlite_snapshot import sqlite_staging_metadata_path, stage_sqlite_snapshot

    first_root = tmp_path / "hermes"
    first_root.mkdir()
    first = first_root / "state.db"
    with sqlite3.connect(first) as conn:
        conn.execute("CREATE TABLE evidence(value TEXT)")

    staged = workspace_env["archive_root"] / "inbox" / "state.db"
    stage_sqlite_snapshot(first, staged)
    metadata_path = sqlite_staging_metadata_path(staged)
    assert metadata_path.exists()

    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    replacement = replacement_root / "state.db"
    replacement.write_bytes(b"not a Hermes database")

    assert _stage_for_daemon(replacement, replace_existing=True) == staged
    assert staged.read_bytes() == replacement.read_bytes()
    assert not metadata_path.exists()


def test_import_command_uses_daemon_url_env_by_default(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Dev-loop imports must not stage into one archive and schedule another daemon.

    Anti-vacuity: restoring the HTTP transport (submitting to the configured
    ``POLYLOGUE_DAEMON_URL`` and printing it as the endpoint) makes the
    socket-path assertion red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = _write_supported_source(tmp_path / "source.jsonl")
    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:9876")

    submit = _RecordingSubmit()
    with _patch_submit(submit):
        result = CliRunner().invoke(cli, ["import", str(source)])

    assert result.exit_code == 0, result.output
    # The transport is the archive-scoped daemon socket, so that -- not the
    # configured browser API URL -- is what the receipt names.
    from polylogue.daemon.socket_path import daemon_socket_path

    socket_path = str(daemon_socket_path(workspace_env["archive_root"]))
    assert f"Daemon:       {socket_path}" in result.output
    assert "http://127.0.0.1:9876" not in result.output
    assert (workspace_env["archive_root"] / "inbox" / source.name).is_file()


def test_import_demo_materializes_fixture_world_before_daemon_request(
    workspace_env: dict[str, Path],
) -> None:
    """--demo writes approved fixture sources and still requires daemon acceptance.

    Anti-vacuity: returning an envelope with no ``accepted_reference`` (see
    ``test_import_refuses_envelope_without_durable_acceptance``) makes the
    success assertions red, so this cannot pass on a fabricated acceptance.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    submit = _RecordingSubmit()
    runner = CliRunner()
    with _patch_submit(submit):
        result = runner.invoke(cli, ["import", "--demo"])

    assert result.exit_code == 0, result.output
    source_root = workspace_env["archive_root"] / "demo-fixture-world-source"
    staged = workspace_env["archive_root"] / "inbox" / "demo-fixture-world-source"
    assert sorted(path.name for path in source_root.iterdir()) == [
        "antigravity",
        "browser-capture",
        "chatgpt",
        "claude-ai",
        "claude-code",
        "codex",
        "gemini",
        "gemini-cli",
        "hermes",
    ]
    assert sorted(path.name for path in staged.iterdir()) == [
        "antigravity",
        "browser-capture",
        "chatgpt",
        "claude-ai",
        "claude-code",
        "codex",
        "gemini",
        "gemini-cli",
        "hermes",
    ]
    assert len(tuple(staged.rglob("demo-*.json*"))) == 7

    assert submit.payload == {
        "path": str(staged),
        "source_path": str(source_root.resolve()),
        "idempotency_key": None,
    }
    assert str(staged) in result.output
    assert "polylogued status" in result.output
    assert "polylogue status --full" in result.output


def test_import_demo_wait_verifies_after_daemon_acceptance(
    workspace_env: dict[str, Path],
) -> None:
    """--demo --wait blocks on the semantic verifier after daemon scheduling.

    Since #3179 (b473d9256), the CLI requests the demo augmentation and runs
    the real ``_verify_demo_now`` unconditionally after the wait — not only on
    the ``--with-overlays`` path — so this test must stand in for both rather
    than let them hit the (unseeded, in this unit test) archive directly.
    Since S11 the augmentation is the declared ``maintenance.demo.augment``
    operation, not an HTTP POST.

    Anti-vacuity: dropping the augmentation request, or reordering it before
    the convergence wait, makes the ``events`` assertion red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli
    from polylogue.demo import DemoVerifyResult

    captured: dict[str, Any] = {}
    events: list[str] = []

    def fake_submit(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config, payload
        assert operation == "ingest"
        events.append("daemon")
        return _accepted_envelope("import-demo-fixture-world")

    def fake_augment(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config
        assert operation == "maintenance.demo.augment"
        captured["augment_payload"] = dict(payload)
        events.append("augment-daemon")
        return {"outcome": "completed", "effect": "committed", "sequence": 1}

    def fake_wait(*, timeout_s: float, require_overlays: bool = False) -> None:
        captured["timeout_s"] = timeout_s
        captured["require_overlays"] = require_overlays
        events.append("wait-base")

    fake_result = DemoVerifyResult(
        archive_root=workspace_env["archive_root"],
        ok=True,
        session_count=19,
        message_count=31,
        query_hits=(),
        overlays_present=False,
        absolute_path_leaks=(),
    )

    def fake_verify(*, require_overlays: bool = False) -> DemoVerifyResult:
        assert require_overlays is False
        events.append("verify")
        return fake_result

    runner = CliRunner()
    with (
        patch("polylogue.cli.operation_kernel.configured_accepted_operation", new=fake_submit),
        patch("polylogue.cli.operation_kernel.configured_mutation_operation", new=fake_augment),
        patch("polylogue.cli.commands.import_command._wait_for_demo_archive_ready", side_effect=fake_wait),
        patch("polylogue.cli.commands.import_command._verify_demo_now", side_effect=fake_verify),
    ):
        result = runner.invoke(cli, ["import", "--demo", "--wait", "--timeout", "12.5"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "timeout_s": 12.5,
        "require_overlays": False,
        "augment_payload": {"with_overlays": False},
    }
    assert events == ["daemon", "wait-base", "augment-daemon", "verify"]
    staged = workspace_env["archive_root"] / "inbox" / "demo-fixture-world-source"
    assert str(staged) in result.output
    assert "Demo archive verified" in result.output
    assert "sessions=19 messages=31" in result.output
    assert "overlays=no" in result.output


def test_import_demo_wait_with_overlays_seeds_after_convergence(
    workspace_env: dict[str, Path],
) -> None:
    """--with-overlays applies deterministic user overlays after base ingest.

    Also covers the unconditional demo augmentation introduced by #3179
    (b473d9256) between the base wait and overlay seeding, now carried by the
    declared ``maintenance.demo.augment`` operation.

    Anti-vacuity: dropping ``with_overlays`` from the augmentation request
    (so the daemon would seed no overlays) makes the payload assertion red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli
    from polylogue.demo import DemoVerifyResult

    events: list[str] = []
    augment_payloads: list[dict[str, object]] = []

    def fake_submit(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config, payload
        assert operation == "ingest"
        events.append("daemon")
        return _accepted_envelope("import-demo-fixture-world")

    def fake_augment(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config
        assert operation == "maintenance.demo.augment"
        augment_payloads.append(dict(payload))
        events.append("augment-daemon")
        return {"outcome": "completed", "effect": "committed", "sequence": 1}

    def fake_wait(*, timeout_s: float, require_overlays: bool = False) -> None:
        assert timeout_s == 30.0
        assert require_overlays is False
        events.append("wait-base")

    fake_result = DemoVerifyResult(
        archive_root=workspace_env["archive_root"],
        ok=True,
        session_count=19,
        message_count=31,
        query_hits=(),
        overlays_present=True,
        absolute_path_leaks=(),
    )

    def fake_verify(*, require_overlays: bool = False) -> DemoVerifyResult:
        assert require_overlays is True
        events.append("verify-overlays")
        return fake_result

    runner = CliRunner()
    with (
        patch("polylogue.cli.operation_kernel.configured_accepted_operation", new=fake_submit),
        patch("polylogue.cli.operation_kernel.configured_mutation_operation", new=fake_augment),
        patch("polylogue.cli.commands.import_command._wait_for_demo_archive_ready", side_effect=fake_wait),
        patch("polylogue.cli.commands.import_command._verify_demo_now", side_effect=fake_verify),
    ):
        result = runner.invoke(cli, ["import", "--demo", "--wait", "--with-overlays"])

    assert result.exit_code == 0, result.output
    assert events == ["daemon", "wait-base", "augment-daemon", "verify-overlays"]
    assert augment_payloads == [{"with_overlays": True}]
    assert "sessions=19 messages=31" in result.output
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
    """With no daemon running, the command must fail with an actionable error.

    The operator's ruling is that the resident daemon is the standard and
    there is no standalone import mode, so the message has to name
    ``polylogued run`` and the archive whose writes are unowned.

    Anti-vacuity: swallowing ``OperationUnavailableError`` and printing
    "Scheduled" anyway makes the nonzero-exit assertion red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = _write_supported_source(tmp_path / "source.jsonl")

    def unavailable(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config, operation, payload
        raise OperationUnavailableError("daemon is unavailable for operation: ingest")

    runner = CliRunner()
    with patch("polylogue.cli.operation_kernel.configured_accepted_operation", new=unavailable):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "polylogued run" in combined
    assert str(workspace_env["archive_root"]).lower() in combined


def test_import_surfaces_refused_operation_with_staged_path(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A typed operation refusal is reported truthfully, naming the staged file.

    This is the law the old HTTP 4xx/5xx test carried: a daemon that answers
    and rejects is a contract problem, and the staged inbox entry is still
    there for the operator to inspect.

    Anti-vacuity: mapping ``OperationFailedError`` onto the success path, or
    dropping the staged path from the message, makes this red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = _write_supported_source(tmp_path / "source.jsonl")

    def refused(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config, operation, payload
        raise OperationFailedError("invalid_request", "inbox entry could not be resolved")

    runner = CliRunner()
    with patch("polylogue.cli.operation_kernel.configured_accepted_operation", new=refused):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    staged = workspace_env["archive_root"] / "inbox" / source.name
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "invalid_request" in combined
    assert "inbox entry could not be resolved" in combined
    assert str(staged).lower() in combined


def test_import_refuses_indeterminate_submission(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """An indeterminate submission is never reported as scheduled, nor retried.

    Anti-vacuity: treating ``OperationIndeterminateError`` as a transport
    failure the caller may retry makes the "do not re-run" guidance absent
    and the exit-code assertion red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = _write_supported_source(tmp_path / "source.jsonl")

    def indeterminate(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config, operation, payload
        raise OperationIndeterminateError("ingest requires receipt recovery for request abc")

    runner = CliRunner()
    with patch("polylogue.cli.operation_kernel.configured_accepted_operation", new=indeterminate):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    staged = workspace_env["archive_root"] / "inbox" / source.name
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "no receipt came back" in combined
    assert "rather than re-running this command" in combined
    assert str(staged) in combined


def test_import_refuses_inadmissible_source_before_submitting(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Unsupported/degraded preflight detail reaches the operator, client-side.

    This is the law the old HTTP-415 test carried: the operator sees the
    preflight ``error_code`` and its summary, and the staged copy is named.
    Since S11 the CLI runs the same read-only check itself, so no operation is
    submitted at all.

    Anti-vacuity: dropping the client-side preflight makes the "no submission
    happened" assertion red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = tmp_path / "unknown.json"
    source.write_text('{"not":"an export"}\n')

    submit = _RecordingSubmit()
    runner = CliRunner()
    with _patch_submit(submit):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    assert submit.calls == []
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "unsupported_import_source" in combined
    assert "no parseable polylogue export shape" in combined
    assert str(workspace_env["archive_root"] / "inbox" / source.name).lower() in combined


def test_import_refuses_envelope_without_durable_acceptance(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Only a durable acceptance reference establishes that ingest was admitted.

    This is the law the old "unrecognized daemon status" test carried: an
    envelope the command cannot read as acceptance is a failure, never a
    fabricated success.

    Anti-vacuity: deriving acceptance from the outcome alone makes this red,
    because the outcome here is ``completed``.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    source = _write_supported_source(tmp_path / "source.jsonl")
    submit = _RecordingSubmit({"operation": "ingest", "outcome": "completed", "result": {}, "accepted_reference": None})

    runner = CliRunner()
    with _patch_submit(submit):
        result = runner.invoke(cli, ["import", str(source)])

    assert result.exit_code != 0
    combined = (result.output + (result.stderr if result.stderr_bytes else "")).lower()
    assert "durable acceptance reference" in combined


def test_import_demo_wait_refuses_failed_augmentation(
    workspace_env: dict[str, Path],
) -> None:
    """A refused demo augmentation never reports a verified demo archive.

    Anti-vacuity: swallowing the kernel error and continuing to the verifier
    makes the nonzero-exit assertion red.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    def fake_submit(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config, operation, payload
        return _accepted_envelope("import-demo-fixture-world")

    def refused(config: Any, operation: str, payload: dict[str, object]) -> dict[str, object]:
        del config, operation, payload
        raise OperationFailedError("write_coordinator_unavailable", "no writer lease")

    def fake_wait(*, timeout_s: float, require_overlays: bool = False) -> None:
        del timeout_s, require_overlays

    runner = CliRunner()
    with (
        patch("polylogue.cli.operation_kernel.configured_accepted_operation", new=fake_submit),
        patch("polylogue.cli.operation_kernel.configured_mutation_operation", new=refused),
        patch("polylogue.cli.commands.import_command._wait_for_demo_archive_ready", side_effect=fake_wait),
    ):
        result = runner.invoke(cli, ["import", "--demo", "--wait"])

    assert result.exit_code != 0
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "refusing to claim a verified archive" in combined
    assert "Demo archive verified" not in combined
