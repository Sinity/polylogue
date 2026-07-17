"""Daemon UDS fast-path benchmark (polylogue-20d.1 / polylogue-20d.14).

Covers: CLI-to-daemon round trip over the AF_UNIX transport that
``polylogue.cli.archive_query`` proxies ordinary session-page queries through
when a config-matched daemon is reachable. This is the "interactive" SLO
tier's ``daemon_cli_query`` surface: the whole point of the hot-daemon fast
path is that this round trip stays far below the cold, import-paying direct
CLI path.

Run with:
    pytest tests/benchmarks/test_daemon_uds.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.benchmarks.conftest import _seed_realistic_db
from tests.benchmarks.helpers import BenchmarkFixture


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

    socket_path = daemon_socket_path(str(runtime_dir))
    server = DaemonAPIUnixHTTPServer(socket_path, DaemonAPIHandler)
    server.auth_token = ""
    thread = threading.Thread(target=server.serve_forever, name="bench-daemon-uds", daemon=True)
    thread.start()
    # Wait for the socket to accept connections rather than a fixed sleep —
    # ThreadingMixIn.serve_forever binds synchronously in __init__, but give
    # the accept loop a moment to actually start before the first probe.
    deadline = time.monotonic() + 2.0
    client = DaemonClient(socket_path, timeout_s=1.0)
    while time.monotonic() < deadline:
        if client.request_json("GET", "/api/health") is not None:
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
    """Benchmark the CLI's ``/api/cli/query`` UDS round trip (find-mode page).

    Matches: ``polylogue.cli.archive_query._try_emit_daemon_session_page`` ->
    ``DaemonClient.cli_query`` -> ``DaemonAPIHandler._handle_cli_query``.
    """
    from polylogue.cli.daemon_client import DaemonClient

    client = bench_daemon_uds_client
    assert isinstance(client, DaemonClient)

    def _query() -> dict[str, object] | None:
        return client.cli_query({"limit": 20})

    result = benchmark(_query)
    assert result is not None
    items = result.get("items")
    assert isinstance(items, list)
    assert len(items) > 0


@pytest.mark.benchmark
def test_bench_daemon_uds_health_probe(
    benchmark: BenchmarkFixture,
    bench_daemon_uds_client: object,
) -> None:
    """Benchmark the config-matched health probe every fast-path call pays first."""
    from polylogue.cli.daemon_client import DaemonClient
    from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
    from polylogue.version import POLYLOGUE_VERSION

    client = bench_daemon_uds_client
    assert isinstance(client, DaemonClient)
    archive_root = os.environ["POLYLOGUE_ARCHIVE_ROOT"]

    def _probe() -> dict[str, object] | None:
        return client.probe(
            archive_root=archive_root,
            index_schema_version=INDEX_SCHEMA_VERSION,
            daemon_version=POLYLOGUE_VERSION,
        )

    result = benchmark(_probe)
    assert result is not None
