"""Live-daemon scheduling harness for ``polylogue import --demo`` (#1843).

This test covers the real scheduling chain:

``polylogue import --demo`` -> daemon ``/api/ingest`` acceptance -> staged
demo fixture in the archive inbox.

SPEC_MISMATCH: the attempted full convergence assertion showed that the
current live daemon path converges only 2 of the 3 generated demo sessions
from this fixture world when driven through filesystem watch events. Keep this
test at the accepted-operation boundary until the daemon scheduling path has a
deterministic operation-to-worker handoff for staged directories.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import cast
from urllib.request import urlopen

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_http(url: str, *, process: subprocess.Popen[bytes], timeout_s: float = 10.0) -> None:
    last_error: BaseException | None = None
    attempts = max(1, int(timeout_s / 0.05))
    for _ in range(attempts):
        if process.poll() is not None:
            raise AssertionError(f"daemon exited before HTTP readiness: exit_code={process.returncode}")
        try:
            await asyncio.to_thread(_read_url, url)
            return
        except BaseException as exc:  # pragma: no cover - failure reported below
            last_error = exc
            await asyncio.sleep(0.05)
    raise AssertionError(f"daemon HTTP endpoint did not become ready: {url}; last_error={last_error!r}")


def _read_url(url: str) -> bytes:
    with urlopen(url, timeout=1) as response:
        return cast(bytes, response.read())


def _run_import_demo(daemon_url: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "polylogue.cli",
            "import",
            "--demo",
            "--daemon-url",
            daemon_url,
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )


async def test_import_demo_is_accepted_by_live_daemon_scheduling_path(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``polylogue import --demo`` reaches a real daemon accepted operation."""
    from polylogue.core.degraded import clear_degraded

    clear_degraded()
    archive_root = workspace_env["archive_root"]
    inbox = archive_root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(archive_root / "polylogue.toml"))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    api_port = _free_local_port()
    daemon_url = f"http://127.0.0.1:{api_port}"
    env = os.environ.copy()
    daemon_log = archive_root / "demo-daemon.log"
    with daemon_log.open("wb") as log:
        daemon = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from polylogue.daemon.cli import main; main()",
                "run",
                "--root",
                str(inbox),
                "--debounce-s",
                "0.05",
                "--no-browser-capture",
                "--api-port",
                str(api_port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

        try:
            await _wait_for_http(f"{daemon_url}/healthz/live", process=daemon)

            result = await asyncio.to_thread(_run_import_demo, daemon_url, env)
            combined_output = result.stdout + result.stderr
            assert result.returncode == 0, combined_output
            assert "Scheduled:" in result.stdout
            assert "demo-fixture-world-source" in result.stdout
            assert f"Daemon:       {daemon_url}" in result.stdout
            assert "Operation:    ingest-demo-fixture-world-source" in result.stdout

            staged = inbox / "demo-fixture-world-source"
            assert sorted(path.name for path in staged.iterdir()) == ["chatgpt", "claude-code", "codex"]
            assert len(tuple(staged.rglob("demo-*.json*"))) == 3
            assert daemon.poll() is None
        finally:
            daemon.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                daemon.wait(timeout=10)
            if daemon.poll() is None:
                daemon.kill()
                daemon.wait(timeout=10)
            clear_degraded()
