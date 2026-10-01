"""Exercise the production smoke helper's real child lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.infra.installed_smoke import make_installed_smoke_bins

pytestmark = pytest.mark.uses_real_clock
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("failure", ["none", "query", "startup"])
def test_installed_smoke_waits_for_daemon_and_stops_it_on_query_failure(tmp_path: Path, failure: str) -> None:
    """Removing startup or finally cleanup loses the lifecycle events; swallowing failures returns zero."""
    bins = make_installed_smoke_bins(tmp_path)
    receipt = tmp_path / "receipt.jsonl"
    # This tiny socket directory must fit AF_UNIX even when the managed test root does not.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pl-smoke-", dir="/tmp") as runtime:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(ROOT / "packaging/smoke-installed.py"),
                "--python",
                str(bins / "python"),
                "--bin-dir",
                str(bins),
                "--work-dir",
                str(tmp_path / "work"),
            ],
            env={
                **os.environ,
                "SMOKE_SOCKET": str(Path(runtime) / "daemon.sock"),
                "SMOKE_RECEIPT": str(receipt),
                "SMOKE_FAILURE": failure,
            },
            text=True,
            capture_output=True,
            check=False,
        )
    assert (result.returncode == 0) is (failure == "none"), result.stderr
    if failure == "startup":
        assert not receipt.exists()
    else:
        rows = [json.loads(line) for line in receipt.read_text().splitlines()]
        assert rows[0]["event"] == "daemon_started"
        assert rows[-1]["event"] == "daemon_stopped"
        assert next(i for i, row in enumerate(rows) if row["event"] == "readiness_ready") < next(
            i for i, row in enumerate(rows) if row["event"] == "cli"
        )
        assert [row["argv"] for row in rows if row["event"] == "module"] == [["-I", "-m", "polylogue", "--version"]]
        commands = [row["argv"] for row in rows if row["event"] == "cli"]
        assert ["--plain", "analyze", "--count"] in commands
        if failure == "none":
            assert ["--plain", "ops", "diagnostics", "workload", "--json"] in commands
            assert ["--plain", "ops", "diagnostics", "space", "--json"] in commands
