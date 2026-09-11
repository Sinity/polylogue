"""Daemon operation fast-path benchmark (polylogue-bp12n.4).

Covers the actual archive-scoped operation exchange used by the installed CLI
through the maintained production daemon operation fixture.

Run with:
    pytest tests/benchmarks/test_daemon_uds.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.benchmarks.helpers import BenchmarkFixture
from tests.infra.benchmark_archives import seed_benchmark_archive
from tests.infra.daemon_operations import DaemonOperationStack, running_daemon_operations

pytestmark = pytest.mark.uses_real_clock(
    "polylogue-20d.1 daemon UDS benchmark uses the maintained production daemon operation stack; frozen_clock cannot substitute for its real writer/listener lifecycle."
)


@pytest.fixture(scope="session")
def bench_daemon_uds_archive_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped archive root for a seeded production operation stack.

    Deliberately smaller than ``bench_db_5k`` — this surface benchmarks fixed
    per-request UDS/HTTP-handler overhead, not query cost over a large corpus
    (the ``query``/``reader``/``facets`` surfaces already cover that).
    """
    archive_root = tmp_path_factory.mktemp("bench-daemon-uds") / "archive"
    archive_root.mkdir()
    return archive_root


@pytest.fixture
def bench_daemon_uds_stack(
    bench_daemon_uds_archive_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[DaemonOperationStack]:
    """A maintained production UDS operation stack with a seeded archive."""
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(bench_daemon_uds_archive_root))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")
    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)

    with running_daemon_operations(
        bench_daemon_uds_archive_root,
        seed_archive=lambda root: seed_benchmark_archive(root / "index.db", target_messages=1000),
    ) as stack:
        yield stack


@pytest.fixture
def bench_daemon_uds_client(bench_daemon_uds_stack: DaemonOperationStack) -> object:
    return bench_daemon_uds_stack.client


@pytest.mark.benchmark
def test_bench_daemon_uds_cli_query(
    benchmark: BenchmarkFixture,
    bench_daemon_uds_client: object,
) -> None:
    """Benchmark the CLI's typed ``cli.query`` UDS operation (find-mode page).

    Matches: ``_try_emit_daemon_session_page`` -> ``DaemonClient.operation`` ->
    ``DaemonAPIHandler._handle_daemon_operation``.
    """
    from polylogue.daemon_client import DaemonClient

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
    from polylogue.daemon_client import DaemonClient

    client = bench_daemon_uds_client
    assert isinstance(client, DaemonClient)

    def _probe() -> dict[str, object] | None:
        envelope = client.operation("status", {})
        return envelope if envelope is not None else None

    result = benchmark(_probe)
    assert result is not None
