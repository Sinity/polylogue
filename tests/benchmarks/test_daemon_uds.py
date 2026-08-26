"""Daemon operation fast-path benchmark (polylogue-bp12n.4).

Covers the actual archive-scoped operation exchange used by the installed CLI.
The fixture readiness check is itself a status operation; it never calls a
health/probe endpoint.

Run with:
    pytest tests/benchmarks/test_daemon_uds.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.benchmarks.conftest import _seed_realistic_db
from tests.benchmarks.helpers import BenchmarkFixture

pytestmark = pytest.mark.uses_real_clock(
    "polylogue-20d.1 daemon UDS benchmark fixture polls a real background-thread HTTP server's readiness with a bounded wall-clock deadline; frozen_clock cannot substitute for waiting on real socket/thread startup."
)


@pytest.fixture(scope="session")
def bench_daemon_uds_archive_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped archive_root with a seeded index.db (~1K messages).

    Deliberately smaller than ``bench_db_5k`` — this surface benchmarks fixed
    per-request UDS/HTTP-handler overhead, not query cost over a large corpus
    (the ``query``/``reader``/``facets`` surfaces already cover that).
    """
    archive_root = tmp_path_factory.mktemp("bench-daemon-uds") / "archive"
    archive_root.mkdir()
    stats = _seed_realistic_db(archive_root / "index.db", target_messages=1000)
    print(f"\nbench_daemon_uds_archive_root: {stats}")
    return archive_root


@pytest.fixture
def bench_daemon_uds_client(
    bench_daemon_uds_archive_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[object]:
    """A live production UDS daemon server + matching ``DaemonClient``.

    ``AF_UNIX`` paths are capped at ~108 bytes on Linux; pytest's default
    ``tmp_path`` nests deep enough (``.../pytest-.../test-name0/...``) to
    blow that budget, so the runtime dir (and only the runtime dir, which
    holds the socket) lives under a short-path ``tempfile.mkdtemp()`` instead.
    """

    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.daemon.http import DaemonAPIHandler
    from polylogue.daemon.uds import DaemonAPIUnixHTTPServer, daemon_socket_path

    runtime_dir = Path(tempfile.mkdtemp(prefix="plg-bench-uds-"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(bench_daemon_uds_archive_root))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)

    socket_path = daemon_socket_path(bench_daemon_uds_archive_root, runtime_dir=str(runtime_dir))
    server = DaemonAPIUnixHTTPServer(socket_path, DaemonAPIHandler)
    server.auth_token = ""
    thread = threading.Thread(target=server.serve_forever, name="bench-daemon-uds", daemon=True)
    thread.start()
    # Wait for the operation endpoint to accept connections rather than a fixed sleep —
    # ThreadingMixIn.serve_forever binds synchronously in __init__, but give
    # the accept loop a moment to actually start before the first operation.
    deadline = time.monotonic() + 2.0
    client = DaemonClient(socket_path, timeout_s=1.0)
    while time.monotonic() < deadline:
        if client.operation("completion", {"kind": "field", "incomplete": ""}) is not None:
            break
        time.sleep(0.02)
    else:
        shutil.rmtree(runtime_dir, ignore_errors=True)
        pytest.fail("daemon UDS server did not become ready")

    try:
        yield client
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(runtime_dir, ignore_errors=True)


@pytest.mark.benchmark
def test_bench_daemon_uds_cli_query(
    benchmark: BenchmarkFixture,
    bench_daemon_uds_client: object,
) -> None:
    """Benchmark the CLI's typed ``cli.query`` UDS operation (find-mode page).

    Matches: ``_try_emit_daemon_session_page`` -> ``DaemonClient.operation`` ->
    ``DaemonAPIHandler._handle_daemon_operation``.
    """
    from polylogue.cli.daemon_client import DaemonClient

    client = bench_daemon_uds_client
    assert isinstance(client, DaemonClient)

    def _query() -> dict[str, object] | None:
        envelope = client.operation("cli.query", {"params": {"limit": 20}})
        if envelope is None or envelope.get("error") is not None:
            return None
        result = envelope.get("result")
        return result if isinstance(result, dict) else None

    result = benchmark(_query)
    assert result is not None
    items = result.get("items")
    assert isinstance(items, list)
    assert len(items) > 0


@pytest.mark.benchmark
def test_bench_daemon_uds_status_operation(
    benchmark: BenchmarkFixture,
    bench_daemon_uds_client: object,
) -> None:
    """Benchmark the typed status operation, including readiness metadata."""
    from polylogue.cli.daemon_client import DaemonClient

    client = bench_daemon_uds_client
    assert isinstance(client, DaemonClient)

    def _probe() -> dict[str, object] | None:
        envelope = client.operation("status", {})
        return envelope if envelope is not None else None

    result = benchmark(_probe)
    assert result is not None
