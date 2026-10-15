"""Benchmark measurements retain their own child's resource record."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.infra.benchmark_process import run_with_peak_rss


def test_peak_rss_is_per_child_after_a_larger_child() -> None:
    import json

    # Linux includes the fork's pre-exec RSS in wait4's high-water record.
    # A fresh measuring process keeps pytest's large imported heap out of
    # the two child launches without assuming an operator-specific size.
    script = """
import json, sys
from tests.infra.benchmark_process import run_with_peak_rss
large, large_rss = run_with_peak_rss([sys.executable, '-c', 'x = bytearray(64 * 1024 * 1024); print(len(x))'])
assert large.returncode == 0
small = []
for _ in range(2):
    result, rss = run_with_peak_rss([sys.executable, '-c', "print('small')"])
    assert result.returncode == 0 and result.stdout == b'small\\n' and result.stderr == b''
    small.append(rss)
print(json.dumps({'large': large_rss, 'small': small}))
"""
    result, _ = run_with_peak_rss([sys.executable, "-c", script])
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert len(observed["small"]) == 2
    assert all(0 < rss < observed["large"] for rss in observed["small"])


@pytest.mark.parametrize("host_importtime", (False, True))
def test_cold_cli_measurements_use_their_own_child_rss_without_profiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_importtime: bool
) -> None:
    import json
    import subprocess
    from unittest.mock import Mock

    import tests.benchmarks.test_cli_cold_start as cold

    if host_importtime:
        monkeypatch.setenv("PYTHONPROFILEIMPORTTIME", "1")
    monkeypatch.setenv("POLYLOGUE_BENCH_IMPORTTIME", "1")

    def observed(argv: object, *, env: dict[str, str], cwd: Path) -> tuple[subprocess.CompletedProcess[bytes], int]:
        assert "PYTHONPROFILEIMPORTTIME" not in env
        assert cwd == tmp_path.parent / "isolated-home"
        return subprocess.CompletedProcess(
            [],
            1,
            json.dumps(
                {
                    "ok": False,
                    "daemon_liveness": False,
                    "status_snapshot": {"state": "unavailable", "reason": "daemon_absent"},
                }
            ).encode(),
            b"",
        ), 1024

    monkeypatch.setattr(cold, "run_with_peak_rss", observed)
    monkeypatch.setattr(cold, "benchmark_repeated", lambda benchmark, function, **kwargs: function())
    benchmark = Mock()
    benchmark.extra_info = {}
    cold.test_bench_cli_status_unavailable_cold(benchmark, tmp_path, "status")
    assert "imported_modules" not in benchmark.extra_info
    assert benchmark.extra_info["peak_rss_kib"] == 1024
    assert benchmark.extra_info["profiling"] is False


def test_resident_cold_samples_keep_each_childs_rss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import subprocess
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import Mock

    import tests.benchmarks.test_cli_cold_start as cold
    from tests.infra.cli_cold_start import EmptyResident

    home = tmp_path / "home"
    home.mkdir()
    resident = cast(
        EmptyResident,
        SimpleNamespace(
            home=home,
            root=tmp_path / "archive",
            daemon_pid=123,
            provenance={},
            client_environment=lambda route: {"PYTHONPYCACHEPREFIX": str(tmp_path / "bytecode")},
        ),
    )
    rss = iter((1001, 1002, 1003))
    recorded: list[int] = []
    real_record = cold.record_metrics

    def measured(argv: object, **kwargs: object) -> tuple[subprocess.CompletedProcess[bytes], int]:
        return subprocess.CompletedProcess(
            [], 0, json.dumps({"view": "status", "peers": [], "projection": {}}).encode(), b""
        ), next(rss)

    def repeat(benchmark: object, function: object, **kwargs: object) -> None:
        from collections.abc import Callable

        run = cast(Callable[[], int], function)
        for _ in range(3):
            assert run() == 0

    def record(benchmark: object, **metrics: object) -> None:
        recorded.append(cast(int, metrics["peak_rss_kib"]))
        real_record(benchmark, **metrics)

    monkeypatch.setattr(cold, "run_with_peak_rss", measured)
    monkeypatch.setattr(cold, "benchmark_repeated", repeat)
    monkeypatch.setattr(cold, "record_metrics", record)
    benchmark = Mock()
    benchmark.extra_info = {}
    cold.test_bench_cli_status_cold(benchmark, resident, "agents")
    assert recorded == [1001, 1002, 1003]
    assert benchmark.extra_info["rounds"] == 3
    assert [
        json.loads((tmp_path / f"agents-attempt-{i}.receipt.json").read_text())["peak_rss_kib"] for i in range(1, 4)
    ] == recorded
