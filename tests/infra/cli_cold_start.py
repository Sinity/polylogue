"""Owned empty resident used by the cold status command benchmark."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from devtools.isolated_environment import isolated_home_environment
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def empty_archive_counts(root: Path) -> dict[str, int]:
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        return {
            "raw": int(archive.source_connection.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone()[0]),
            "sessions": int(index.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]),
        }


def assert_served_empty_status(payload: Mapping[str, Any], *, root: Path, daemon_pid: int) -> None:
    """Serving the real empty frame does not assert materialization readiness."""
    assert payload["daemon_liveness"] is True
    snapshot = payload["status_snapshot"]
    assert snapshot["state"] == "fresh"
    assert snapshot["frame_changed"] is False
    assert snapshot["frame_error"] is None and snapshot["refresh_error"] is None
    assert Path(payload["archive_root"]).resolve() == root.resolve()
    storage = payload.get("archive_storage") or payload.get("storage")
    assert storage["identity"]["process_id"] == daemon_pid
    assert payload["raw_materialization_readiness"]["raw_artifact_count"] == 0
    # Current EMPTY materialization is honestly unmeasured, never healthy.
    assert payload["ok"] is False
    if "source" in payload:
        assert payload["source"] == "daemon"
        assert payload["active_archive_root_matches_configured"] is True
        assert payload["sessions"] == payload["messages"] == 0


@dataclass
class EmptyResident:
    root: Path
    home: Path
    environment: dict[str, str]
    daemon_pid: int
    provenance: dict[str, object]

    def client_environment(self, route: str) -> dict[str, str]:
        # Preserve the original benchmark's owned checkout cache policy.
        # Qualification uses a separate daemon prefix and cannot prewarm it.
        cache = Path(__file__).resolve().parents[2] / ".cache" / "pycache"
        cache.mkdir(parents=True, exist_ok=True)
        return {**self.environment, "PYTHONPYCACHEPREFIX": str(cache)}


@contextmanager
def empty_status_resident(directory: Path) -> Iterator[EmptyResident]:
    """Bootstrap and stop one actual console daemon, with no source discovery."""
    root = directory / "archive"
    home = directory / "home"
    home.mkdir(parents=True, mode=0o700)
    initialize_active_archive_root(root)
    assert empty_archive_counts(root) == {"raw": 0, "sessions": 0}
    env = isolated_home_environment(os.environ, home=home)
    for key in (
        "POLYLOGUE_DAEMON_URL",
        "POLYLOGUE_DAEMON_MODE",
        "PYTHONPROFILEIMPORTTIME",
        "PYTHONDONTWRITEBYTECODE",
        "INVOCATION_ID",
        "NOTIFY_SOCKET",
        "JOURNAL_STREAM",
        "SYSTEMD_EXEC_PID",
    ):
        env.pop(key, None)
    # A short private runtime root avoids the Unix socket path length limit.
    runtime = Path(tempfile.mkdtemp(prefix="plg-cold-", dir="/realm/tmp/work"))
    env.update(
        {
            "POLYLOGUE_ARCHIVE_ROOT": str(root),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "POLYLOGUE_FORCE_PLAIN": "1",
            "XDG_RUNTIME_DIR": str(runtime),
            "POLYLOGUE_API_AUTH_TOKEN": "synthetic-cold-status-benchmark",
            "POLYLOGUE_LOG_FILE": str(directory / "events.jsonl"),
            "POLYLOGUE_LOG_FORMAT": "json",
        }
    )
    listener = directory / "listeners.json"
    executable = Path(sys.executable).parent / "polylogued"
    args = [
        str(executable),
        "run",
        "--no-watch",
        "--no-source-catchup",
        "--no-browser-capture",
        "--api-port",
        "0",
        "--api-auth-token",
        "synthetic-cold-status-benchmark",
        "--listener-info-path",
        str(listener),
    ]
    candidate = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    provenance: dict[str, object] = {
        "candidate_sha": candidate,
        "daemon_command": args,
        "runtime_root": str(runtime),
        "empty_before": {"raw": 0, "sessions": 0},
    }
    with (directory / "daemon.stdout.log").open("wb") as out, (directory / "daemon.stderr.log").open("wb") as err:
        try:
            process = subprocess.Popen(
                args,
                env={**env, "PYTHONPYCACHEPREFIX": str(home / "daemon-pycache")},
                cwd=home,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        except BaseException:
            shutil.rmtree(runtime)
            raise
        try:
            started = time.monotonic()
            deadline = started + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(f"owned daemon exited before qualification: {process.returncode}")
                if listener.exists():
                    info = json.loads(listener.read_text())
                    assert info["pid"] == process.pid
                    api = info["listeners"]["api"]
                    request = Request(
                        f"http://{api['host']}:{api['port']}/api/status",
                        headers={"Authorization": f"Bearer {env['POLYLOGUE_API_AUTH_TOKEN']}"},
                    )
                    try:
                        with urlopen(request, timeout=2) as response:
                            payload = json.load(response)
                    except HTTPError as error:
                        if error.code not in {502, 503, 504}:
                            raise
                        payload = None
                    except (URLError, OSError, TimeoutError):
                        payload = None
                    if isinstance(payload, dict):
                        (directory / "resident-status.json").write_text(json.dumps(payload, indent=2) + "\n")
                    if (
                        isinstance(payload, dict)
                        and payload.get("status_snapshot", {}).get("state") == "fresh"
                        and payload.get("archive_storage", {}).get("identity", {}).get("process_id") is not None
                        and payload.get("catchup", {}).get("cold_build_settlement_state") == "complete"
                    ):
                        assert_served_empty_status(payload, root=root, daemon_pid=process.pid)
                        assert payload["archive_storage"]["identity"]["build_commit"] == candidate
                        provenance.update(
                            {
                                "daemon_pid": process.pid,
                                "resident_ready_s": time.monotonic() - started,
                                "build_commit": payload["archive_storage"]["identity"]["build_commit"],
                                "qualification": "settled cold build; fresh authoritative empty; materialization unmeasured",
                            }
                        )
                        break
                time.sleep(0.1)
            else:
                raise AssertionError(
                    "owned resident did not publish a fresh status within the finite qualification window"
                )
            yield EmptyResident(root, home, env, process.pid, provenance)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                    provenance["forced_cleanup"] = True
            provenance["daemon_returncode"] = process.returncode
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                provenance["owned_process_group_stopped"] = True
            else:
                provenance["owned_process_group_stopped"] = False
            provenance["empty_after"] = empty_archive_counts(root)
            (directory / "resident-proof.json").write_text(json.dumps(provenance, indent=2) + "\n")
            if provenance["owned_process_group_stopped"] is True:
                shutil.rmtree(runtime)
            assert provenance.get("forced_cleanup") is not True
            assert provenance["owned_process_group_stopped"] is True
            assert provenance["empty_after"] == {"raw": 0, "sessions": 0}
