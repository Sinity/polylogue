"""Golden parity: direct CLI execution vs config-matched daemon-proxied execution.

polylogue-20d.1 acceptance criterion: "`--format json` output is byte-identical
between direct and daemon-proxied execution for every read surface on the demo
corpus." A real production UDS daemon server is started against the same
seeded archive the direct path reads, and the same `find` invocation is run
through :class:`click.testing.CliRunner` twice — once with no daemon socket
present (direct path) and once with the daemon reachable (proxied path) — so
this is an end-to-end regression test, not a mock of the daemon transport.

The two envelopes are compared field-for-field rather than as raw text: the
daemon envelope carries an explicit ``"source": "daemon"`` provenance marker
that the direct envelope does not (`archive_query.py::_emit_daemon_list_payload`
vs `_emit_list`) — that is the one intentional, documented difference. Every
other field (`items`, `total`, `limit`, `offset`, `origin`, `next_offset`,
`next_cursor`) must match exactly.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.infra.storage_records import SessionBuilder


@pytest.fixture
def golden_parity_workspace(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A real seeded archive, reused for both the direct and daemon-proxied runs."""

    monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_workspace["archive_root"]))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)

    index_db = cli_workspace["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "conv1")
        .provider("chatgpt")
        .title("Python Error Handling")
        .git_repository_url("polylogue")
        .add_message("m1", role="user", text="How to handle exceptions in Python?")
        .add_message("m2", role="assistant", text="Use try-except blocks.")
        .save()
    )
    (
        SessionBuilder(index_db, "conv2")
        .provider("claude-code")
        .title("Rust Ownership")
        .git_repository_url("polylogue")
        .add_message("m3", role="user", text="What is ownership in Rust?")
        .add_message("m4", role="assistant", text="Rust ownership ensures memory safety.")
        .save()
    )
    return cli_workspace


@pytest.fixture
def _uds_runtime_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A short-path runtime dir so the AF_UNIX socket path stays under the OS limit."""

    runtime_dir = Path(tempfile.mkdtemp(prefix="plg-golden-uds-"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    try:
        yield runtime_dir
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _run_find_json(args: list[str], *, no_daemon: bool = False) -> dict[str, object]:
    from polylogue.cli import cli

    runner = CliRunner()
    # `--no-daemon` and `--repo` are root options (`click_app.py::cli`), not
    # `find` verb options — they must precede `find` in argv. Passing `--repo`
    # here as the root option (rather than a `repo:polylogue` DSL query token,
    # which routes through a different, older rendering path with a distinct
    # envelope shape — see the module docstring follow-up note) is what
    # actually exercises `_try_emit_daemon_session_page` /
    # `_daemon_session_page_supported`, the code this test targets.
    root_flags = ["--plain", *args, *(["--no-daemon"] if no_daemon else [])]
    result = runner.invoke(cli, [*root_flags, "find", "--format", "json", "--limit", "10"])
    assert result.exit_code == 0, result.output
    return dict(json.loads(result.output))


def _strip_provenance(envelope: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in envelope.items() if key != "source"}


def test_find_list_json_parity_between_direct_and_daemon(
    golden_parity_workspace: dict[str, Path],
    _uds_runtime_dir: Path,
) -> None:
    del golden_parity_workspace
    args = ["--repo", "polylogue"]

    # 1. Direct path: no daemon socket exists at XDG_RUNTIME_DIR, so the probe
    # fails in-process and the CLI falls back to opening SQLite itself.
    direct_payload = _run_find_json(args, no_daemon=True)
    assert "source" not in direct_payload

    # 2. Daemon-proxied path: start the production UDS server against the
    # same archive_root the direct run just read, then reissue the identical
    # query with the daemon reachable.
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer, daemon_socket_path

    socket_path = daemon_socket_path(str(_uds_runtime_dir))
    server = DaemonAPIUnixHTTPServer(socket_path, DaemonAPIHandler)
    server.auth_token = ""
    thread = threading.Thread(target=server.serve_forever, name="golden-parity-uds", daemon=True)
    thread.start()
    try:
        from polylogue.cli.daemon_client import DaemonClient

        client = DaemonClient(socket_path, timeout_s=1.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if client.request_json("GET", "/api/health") is not None:
                break
            time.sleep(0.02)
        else:
            pytest.fail("daemon UDS server did not become ready")

        daemon_payload = _run_find_json(args)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert daemon_payload["source"] == "daemon"
    assert _strip_provenance(daemon_payload) == _strip_provenance(direct_payload)
    assert direct_payload["items"], "fixture query must actually match rows, or parity is vacuous"


def test_facets_json_parity_between_direct_and_daemon(
    golden_parity_workspace: dict[str, Path],
    _uds_runtime_dir: Path,
) -> None:
    """The facets surface has its own daemon fast path (`_fetch_daemon_facets`)."""
    del golden_parity_workspace
    from polylogue.cli import cli

    runner = CliRunner()

    direct_result = runner.invoke(cli, ["--plain", "--no-daemon", "facets", "--format", "json"])
    assert direct_result.exit_code == 0, direct_result.output
    direct_payload = json.loads(direct_result.output)

    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer, daemon_socket_path

    socket_path = daemon_socket_path(str(_uds_runtime_dir))
    server = DaemonAPIUnixHTTPServer(socket_path, DaemonAPIHandler)
    server.auth_token = ""
    thread = threading.Thread(target=server.serve_forever, name="golden-parity-facets-uds", daemon=True)
    thread.start()
    try:
        from polylogue.cli.daemon_client import DaemonClient

        client = DaemonClient(socket_path, timeout_s=1.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if client.request_json("GET", "/api/health") is not None:
                break
            time.sleep(0.02)
        else:
            pytest.fail("daemon UDS server did not become ready")

        daemon_result = runner.invoke(cli, ["--plain", "facets", "--format", "json"])
        assert daemon_result.exit_code == 0, daemon_result.output
        daemon_payload = json.loads(daemon_result.output)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    # `generated_at` is a genuine wall-clock timestamp stamped independently
    # by each call, not a parity signal.
    direct_payload.pop("generated_at", None)
    daemon_payload.pop("generated_at", None)
    assert daemon_payload == direct_payload
