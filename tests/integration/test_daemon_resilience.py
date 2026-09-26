"""Daemon resilience integration tests (#1735).

Exercises ``polylogued run`` under failure conditions: SIGKILL recovery,
WAL checkpoint safety, memory pressure, large-session ingestion, and
concurrent-access locking. All tests use ``subprocess.Popen`` to drive
the real daemon binary (not in-process primitives) so the test closure
includes the full process boundary — pidfile locking, journal-level WAL
durability, and OS-level resource enforcement.

.. rubric:: Patterns

- ``workspace_env`` fixture for isolated XDG/archive roots.
- ``_write_claude_code_session`` mirrors the helper from
  ``tests/integration/test_daemon_convergence_evidence.py``.
- ``_wait_for_messages`` polls ``sqlite3`` directly so the test does
  not depend on the HTTP API being enabled; the SIGKILL test probes its
  daemon-owned inactive generation before promotion.
- Subprocess cleanup: every test uses ``try/finally`` with
  ``.terminate()`` → ``.wait(timeout=10)`` → ``.kill()`` → ``.wait()``
  to guarantee no orphan daemons.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.uses_real_clock(
        "Daemon resilience integration tests (#1735) measure real elapsed wall-clock for "
        "process lifecycle events (SIGKILL delivery, subprocess startup, concurrency timing). "
        "frozen_clock cannot substitute for real time when waiting on OS process state."
    ),
    pytest.mark.slow,
    pytest.mark.integration,
    # Real daemon subprocesses started under a wall-clock deadline: the parallel
    # lane's full worker count starves interpreter startup and flakes the whole
    # gate (failed twice in-lane on 2026-08-18, green standalone at ~2s both
    # times). Wall-clock-bound -> the bounded `load_sensitive` lane owns it,
    # capped at devtools.verify.SERIAL_LANE_MAX_WORKERS.
    pytest.mark.load_sensitive,
    # The default bound is 120s (pyproject: timeout, timeout_func_only). Every
    # deadline in this file is a hang detector sized to survive starvation (see
    # the calibration note above `_wait_for_lifecycle_start`), and several
    # declare more than the default on their own: `_wait_for_sessions(...,
    # timeout_s=300.0)` in the memory-pressure test, `_wait_for_messages(...,
    # timeout_s=300.0)` in the large-session test, and a 120s lifecycle wait
    # plus a 90s `daemon.wait` in the SIGTERM tests. Under the default ceiling
    # those waits could never reach their own deadline -- pytest-timeout always
    # fired first, so a starved run reported "Timeout (>120.0s) from
    # pytest-timeout" instead of the test's own diagnostic. This raises the
    # bound above the largest declared wait chain; it is still well inside the
    # 900s policy maximum, and it does not make any individual wait longer.
    pytest.mark.timeout(600),
]

# Bin-packing for the bounded lane.
#
# The lane runs under `--dist=loadgroup`, which keeps one group on one worker
# and hands whole groups out as workers free up. Left to xdist's dynamic
# scheduling these tests pack badly: the longest one (`test_large_session_file`,
# ~23s of real 50K-message ingest) is declared sixth of seven, so it starts last
# and the lane's makespan becomes "when the longest test happened to begin"
# rather than its duration -- 35.1s against a 23.2s floor when measured at four
# workers.
#
# These four groups are longest-processing-time bins over the measured per-test
# call durations (2026-08-19 receipt 20260818T184401Z-full-1494889-23438ba4):
#
#   a  large_session_file 23.18                                = 23.18
#   b  sigkill_recovery 16.30 + sigterm_with_locked_ops 7.27    = 23.57
#   c  concurrent_access 11.55 + wal_checkpoint 8.05            = 19.60
#   d  memory_pressure 8.80 + sigterm_read_only 1.27            = 10.07
#
# Those measurements predate the candidate-progress SIGKILL workload, which
# now crosses a full intake page; re-measure before claiming the bins are still
# balanced. They remain a scheduling hint, not a correctness contract: every
# test has its own archive root and loopback port.
_BIN_A = pytest.mark.xdist_group("daemon-resilience-a")
_BIN_B = pytest.mark.xdist_group("daemon-resilience-b")
_BIN_C = pytest.mark.xdist_group("daemon-resilience-c")
_BIN_D = pytest.mark.xdist_group("daemon-resilience-d")

# ---------------------------------------------------------------------------
# Session file writer (matches test_daemon_convergence_evidence.py)
# ---------------------------------------------------------------------------


def _write_claude_code_session(path: Path, session_id: str, n_messages: int) -> None:
    """Write a realistic Claude Code session JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n_messages):
            role = "user" if i % 2 == 0 else "assistant"
            record = {
                "parentUuid": None if i == 0 else f"msg-{session_id}-{i - 1:03d}",
                "sessionId": session_id,
                "type": role,
                "message": {
                    "role": role,
                    "content": f"Message {i}: {'The quick brown fox jumps over the lazy dog. ' * 3}",
                },
                "uuid": f"msg-{session_id}-{i:03d}",
                "timestamp": f"2026-05-20T00:{i // 60:02d}:{i % 60:02d}.000Z",
                "cwd": "/realm/project/polylogue",
                "version": "1.0.6",
                "isSidechain": False,
                "userType": "external",
            }
            fh.write(json.dumps(record) + "\n")


def _write_large_session(path: Path, session_id: str, n_messages: int) -> int:
    """Write a large Claude Code session JSONL, returning byte size.

    Each message payload is padded to ~200 bytes so 50K msgs ~ 10 MB.
    Timestamps use valid hours spanning multiple days to avoid minute/hour
    overflow.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    pad = "A" * 160  # padding so each record is ~200 bytes
    base_ts = 12 * 3600  # start at T12:00:00
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n_messages):
            role = "user" if i % 2 == 0 else "assistant"
            total_seconds = base_ts + i  # 1-second spacing
            h = (total_seconds // 3600) % 24
            m = (total_seconds // 60) % 60
            s = total_seconds % 60
            # Day-of-month increments naturally; stays valid ISO.
            day = 20 + total_seconds // 86400
            record = {
                "parentUuid": None if i == 0 else f"msg-{session_id}-{i - 1:06d}",
                "sessionId": session_id,
                "type": role,
                "message": {
                    "role": role,
                    "content": f"Msg {i:06d}: {pad}",
                },
                "uuid": f"msg-{session_id}-{i:06d}",
                "timestamp": f"2026-05-{day:02d}T{h:02d}:{m:02d}:{s:02d}.000Z",
                "cwd": "/realm/project/polylogue",
                "version": "1.0.6",
                "isSidechain": False,
                "userType": "external",
            }
            line = json.dumps(record) + "\n"
            fh.write(line)
            total += len(line)
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_path() -> Path:
    """Return the resolved polylogue database path.

    Prefer the active archive root when the test fixture sets one; otherwise
    fall back to the XDG data root.
    """
    archive_root = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    if archive_root:
        return Path(archive_root) / "index.db"
    xdg_data = os.environ.get("XDG_DATA_HOME")
    data_polylogue = Path(xdg_data) / "polylogue" if xdg_data else Path.home() / ".local" / "share" / "polylogue"
    return data_polylogue / "index.db"


def _assert_daemon_alive(proc: subprocess.Popen[bytes]) -> None:
    """Assert the daemon process is still running.

    If the process has exited, include stderr output in the failure message.
    """
    returncode = proc.poll()
    if returncode is not None:
        stderr_text = _stderr_tail(proc)
        raise AssertionError(f"Daemon exited prematurely with code {returncode}. stderr:\n{stderr_text}")


def _stderr_tail(proc: subprocess.Popen[bytes]) -> str:
    """Drain available pipe bytes without waiting for descendants to close it."""
    if proc.stderr is None:
        return "(no stderr)"
    try:
        fd = proc.stderr.fileno()
        os.set_blocking(fd, False)
        tail = bytearray()
        for _ in range(16):
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            tail.extend(chunk)
            del tail[:-4000]
        return tail.decode(errors="replace")
    except OSError as exc:
        return f"(could not read stderr: {type(exc).__name__})"


def _project_binary(name: str) -> str:
    """Return the console script belonging to the checkout under test.

    The running interpreter's own script directory comes first: PATH may
    resolve a different checkout entirely -- in a worktree it resolves the
    main checkout -- and an integration test that launches another tree's
    daemon is not testing this one.
    """
    from shutil import which

    sibling = Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling)
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / ".direnv/sinnix-scope/bin" / name
    if wrapper.exists():
        return str(wrapper)
    candidate = which(name)
    return candidate if candidate is not None else name


def _polylogued_binary() -> str:
    return _project_binary("polylogued")


def _polylogue_binary() -> str:
    return _project_binary("polylogue")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# Deadline calibration: the managed harness runs every subprocess in an
# idle-scheduled containment slice, so under host load a daemon that needs
# ~1s of CPU can take 20-30s of wall clock (measured 2026-08-18: 1.7s clean
# vs 23.7s inside the slice, and SIGTERM exits of 0.16s CPU photographed
# "stuck" in tier DDL by the stack dump). Deadlines here distinguish
# "eventually completes" from "hangs forever" -- they are hang detectors,
# not latency budgets, and must survive starvation (polylogue-b9oi8).
def _wait_for_lifecycle_start(proc: subprocess.Popen[bytes], ops_db: Path, *, timeout_s: float = 120.0) -> None:
    """Wait for the real daemon entry point to persist its lifecycle start row."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _assert_daemon_alive(proc)
        try:
            with sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True) as conn:
                row = conn.execute("SELECT run_id FROM daemon_lifecycle LIMIT 1").fetchone()
            if row is not None:
                return
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc).lower() and "unable to open" not in str(exc).lower():
                raise
        time.sleep(0.1)
    raise TimeoutError("timed out waiting for daemon_lifecycle start row")


def _wait_for_messages(
    db: Path,
    *,
    min_count: int = 1,
    timeout_s: float = 60.0,
    poll_interval: float = 0.5,
) -> int:
    """Poll the database until at least *min_count* messages are present.

    Returns the final message count. Raises ``TimeoutError`` if the
    count is not reached within *timeout_s*.
    """
    import time

    deadline = time.monotonic() + timeout_s
    last_count = 0
    while time.monotonic() < deadline:
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM messages")
                row = cur.fetchone()
                count = int(row[0]) if row else 0
                last_count = count
                if count >= min_count:
                    return count
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if not any(token in msg for token in ("locked", "no such table", "unable to open database file")):
                raise
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting for {min_count} messages after {timeout_s}s; last observed count: {last_count}"
    )


def _wait_for_cold_build_messages(
    proc: subprocess.Popen[bytes], archive_root: Path, *, min_count: int, timeout_s: float
) -> tuple[Path, int, int, list[tuple[str, bytes]]]:
    """Observe rows in this process's inactive generation before promotion.

    The active index deliberately stays empty during a cold build. Metadata
    binds the candidate to this daemon PID, archive root and exact generation
    path, so a leftover candidate from another run cannot satisfy the probe.
    """
    deadline = time.monotonic() + timeout_s
    last_count = 0
    generations = archive_root / ".index-generations"
    while time.monotonic() < deadline:
        _assert_daemon_alive(proc)
        for metadata_path in generations.glob("gen-*/generation.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            candidate = metadata_path.parent / "index.db"
            if (
                metadata.get("owner_id") != f"cold-build:{proc.pid}"
                or metadata.get("archive_root") != str(archive_root)
                or metadata.get("generation_id") != metadata_path.parent.name
                or metadata.get("index_path") != str(candidate)
                or metadata.get("state") != "inactive"
            ):
                continue
            try:
                with sqlite3.connect(f"file:{candidate}?mode=ro", uri=True, timeout=0.1) as conn:
                    count = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
                    last_count = max(last_count, count)
                    if count >= min_count:
                        session_count = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
                        hashes = [
                            (str(session_id), bytes(content_hash))
                            for session_id, content_hash in conn.execute(
                                "SELECT session_id, content_hash FROM sessions ORDER BY session_id LIMIT 10"
                            ).fetchall()
                        ]
                        # Stop before returning to the test: a later assertion
                        # must not give the next intake page time to complete.
                        os.kill(proc.pid, signal.SIGSTOP)
                        return candidate, count, session_count, hashes
            except sqlite3.OperationalError as exc:
                if not any(token in str(exc).lower() for token in ("locked", "no such table", "unable to open")):
                    raise
        time.sleep(0.01)
    raise TimeoutError(
        f"Timed out waiting for {min_count} messages in daemon-owned inactive generation "
        f"after {timeout_s}s; last observed count: {last_count}"
    )


def _wait_for_api_ready(
    proc: subprocess.Popen[bytes],
    port: int,
    *,
    timeout_s: float = 120.0,
) -> None:
    """Wait until the daemon's HTTP surface answers, i.e. startup has finished.

    ``_wait_for_lifecycle_start`` is NOT a readiness signal. It returns as soon
    as ``DaemonLifecycle.start`` has persisted its row, which happens early in
    ``polylogue.daemon.cli``: signal handlers are installed just after it, and
    the startup lifecycle event, source-root creation and the maintenance loops
    all run later and all write the ops tier. A test that acts on the lifecycle
    row alone is racing the rest of startup, and the race widens exactly when
    the host is busy.

    Two observed consequences, both intermittent and both load-dependent:
    signalling in that window can reach the process before its SIGTERM handler
    exists, and taking an EXCLUSIVE ops-tier lock in that window parks a
    startup write inside a blocking ``sqlite3_step``, where the interpreter
    cannot run the Python signal handler until the busy timeout expires. The
    second is what held a signalled daemon past the 90s ``wait`` bound in a
    strictly serial run on 2026-08-19 (1 failure in 3 repeats under host load).

    Answering on the API port happens after that startup sequence, so it is the
    signal these tests actually want.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _assert_daemon_alive(proc)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for daemon API readiness on port {port}")


@_BIN_D
def test_sigterm_read_only_daemon_records_forensics(
    workspace_env: dict[str, Path],
) -> None:
    """A real read-only daemon records SIGTERM before its process exits."""
    archive_root = workspace_env["archive_root"]
    daemon_log = archive_root / "daemon-sigterm.log"
    api_port = _free_local_port()
    env = os.environ.copy()

    with daemon_log.open("wb") as log:
        daemon = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from polylogue.daemon.cli import main; main()",
                "run",
                "--no-watch",
                "--no-source-catchup",
                "--no-browser-capture",
                "--api-port",
                str(api_port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            _wait_for_lifecycle_start(daemon, archive_root / "ops.db")
            _wait_for_api_ready(daemon, api_port)
            daemon.send_signal(signal.SIGTERM)
            assert daemon.wait(timeout=90) == 128 + signal.SIGTERM
        finally:
            if daemon.poll() is None:
                _cleanup_process(daemon)

    with sqlite3.connect(archive_root / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT signal, exit_kind, stopped_at_ms
            FROM daemon_lifecycle
            ORDER BY started_at_ms DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row[:2] == ("SIGTERM", "signal")
    assert isinstance(row[2], int)
    log_text = daemon_log.read_text(encoding="utf-8", errors="replace")
    # PRs #5072/#5073 retired the prose sentence this used to grep for.
    # ``install_signal_handlers`` now emits the stable structured token
    # ``daemon.lifecycle.signal_received``, rendered to the daemon's stream in
    # either the console or the JSON form, so match the token and the signal
    # it names rather than a sentence.
    #
    # Anti-vacuity: drop the ``emit`` from ``install_signal_handlers`` and no
    # line carries the token; drop the ``faulthandler.dump_traceback`` and the
    # stack dump disappears.
    signal_lines = [line for line in log_text.splitlines() if "daemon.lifecycle.signal_received" in line]
    assert signal_lines, log_text
    assert any("SIGTERM" in line for line in signal_lines), signal_lines
    assert "Current thread" in log_text


@_BIN_B
def test_sigterm_with_locked_ops_exits_without_normal_sqlite_wait(
    workspace_env: dict[str, Path],
) -> None:
    """A contended OPS tier cannot hold a signalled daemon for the normal 30 seconds."""
    archive_root = workspace_env["archive_root"]
    api_port = _free_local_port()
    # A real file avoids a pipe backpressure stall while the signal handler
    # dumps thread stacks. Only a bounded tail is read into a failed assertion.
    with (archive_root.parent / "sigterm-locked-ops.stderr").open("w+b") as stderr_capture:
        daemon = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from polylogue.daemon.cli import main; main()",
                "run",
                "--no-watch",
                "--no-source-catchup",
                "--no-browser-capture",
                "--api-port",
                str(api_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=stderr_capture,
            env=os.environ.copy(),
        )
        lock = sqlite3.connect(archive_root / "ops.db")
        try:
            _wait_for_lifecycle_start(daemon, archive_root / "ops.db")
            _wait_for_api_ready(daemon, api_port)
            lock.execute("BEGIN EXCLUSIVE")
            started = time.monotonic()
            daemon.send_signal(signal.SIGTERM)
            exit_code = daemon.wait(timeout=90)
            elapsed_s = time.monotonic() - started
            stderr_capture.seek(max(0, os.fstat(stderr_capture.fileno()).st_size - 8192))
            stderr_tail = stderr_capture.read(8192).decode(errors="replace")
            assert exit_code == 128 + signal.SIGTERM, stderr_tail
            # The contract is "well under the normal 30s SQLite wait"; the bound
            # leaves headroom for the harness's idle-scheduled containment slice
            # (see the deadline-calibration note at _wait_for_lifecycle_start).
            assert elapsed_s < 25.0, stderr_tail
        finally:
            lock.rollback()
            lock.close()
            if daemon.poll() is None:
                _cleanup_process(daemon)


def _daemon_debug(proc: subprocess.Popen[bytes], *, db: Path, corpus_root: Path, stderr_log: Path | None = None) -> str:
    if proc.poll() is None:
        _cleanup_process(proc)
    if stderr_log is None:
        stderr_text = _stderr_tail(proc)
    else:
        with stderr_log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4000))
            stderr_text = stream.read(4000).decode(errors="replace")
    files = sorted(str(path) for path in corpus_root.glob("**/*") if path.is_file())[:20]
    return (
        f"returncode={proc.poll()} db={db} db_exists={db.exists()} "
        f"source_exists={db.with_name('source.db').exists()} corpus_files={files}\n"
        f"stderr:\n{stderr_text}"
    )


def _sigkill_ingest_diagnostics(archive_root: Path, *, owner_pid: int) -> str:
    """Bounded read-only evidence for either SIGKILL phase timeout."""

    def query(db: Path, sql: str) -> object:
        if not db.is_file():
            return "absent"
        try:
            with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.1)) as conn:
                return conn.execute(sql).fetchall()
        except sqlite3.Error as exc:
            return f"{type(exc).__name__}: {exc}"

    active = archive_root / "index.db"
    source = archive_root / "source.db"
    ops = archive_root / "ops.db"
    evidence: dict[str, object] = {
        "active": query(active, "SELECT (SELECT COUNT(*) FROM sessions), (SELECT COUNT(*) FROM messages)"),
        "source_raws": query(source, "SELECT COUNT(*) FROM raw_sessions"),
        "cursor": query(
            ops,
            "SELECT COUNT(*), SUM(content_fingerprint IS NOT NULL), SUM(failure_count > 0), "
            "SUM(excluded), SUM(next_retry_at IS NOT NULL) FROM ingest_cursor",
        ),
        "attempts": query(
            ops,
            "SELECT status, phase, parsed_raw_count, materialized_count "
            "FROM ingest_attempts ORDER BY started_at_ms DESC LIMIT 3",
        ),
    }
    candidates: list[dict[str, object]] = []
    for metadata_path in sorted((archive_root / ".index-generations").glob("gen-*/generation.json"))[:3]:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            candidates.append({"metadata_error": f"{type(exc).__name__}: {exc}"})
            continue
        if metadata.get("owner_id") != f"cold-build:{owner_pid}":
            continue
        candidates.append(
            {
                "state": metadata.get("state"),
                "generation_id": metadata.get("generation_id"),
                "counts": query(
                    metadata_path.parent / "index.db",
                    "SELECT (SELECT COUNT(*) FROM sessions), (SELECT COUNT(*) FROM messages), "
                    "(SELECT COUNT(*) FROM raw_revision_applications)",
                ),
            }
        )
    evidence["candidates"] = candidates
    event_rows = query(
        ops,
        "SELECT payload_json FROM daemon_events WHERE kind = 'ingestion_batch' ORDER BY id DESC LIMIT 2",
    )
    batches: list[dict[str, object]] = []
    if isinstance(event_rows, list):
        for row in event_rows:
            try:
                payload = json.loads(row[0])
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                batches.append(
                    {
                        key: payload.get(key)
                        for key in (
                            "succeeded_file_count",
                            "failed_file_count",
                            "excluded_file_count",
                            "deferred_file_count",
                            "time_budget_exceeded",
                            "parse_time_s",
                            "convergence_time_s",
                            "total_time_s",
                            "stage_timings_s",
                        )
                    }
                )
    evidence["recent_batches"] = batches if isinstance(event_rows, list) else event_rows
    return json.dumps(evidence, sort_keys=True, default=str)[:6000]


def _wait_for_sessions(
    db: Path,
    *,
    min_count: int = 1,
    timeout_s: float = 60.0,
    poll_interval: float = 0.5,
) -> int:
    """Poll the database until at least *min_count* sessions are present."""
    deadline = time.monotonic() + timeout_s
    last_count = 0
    while time.monotonic() < deadline:
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM sessions")
                row = cur.fetchone()
                count = int(row[0]) if row else 0
                last_count = count
                if count >= min_count:
                    return count
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if not any(token in msg for token in ("locked", "no such table", "unable to open database file")):
                raise
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting for {min_count} sessions after {timeout_s}s; last observed count: {last_count}"
    )


def _wait_for_daemon_ready(proc: subprocess.Popen[bytes], *, timeout_s: float = 120.0) -> bool:
    """Wait for daemon process to be alive and responding.

    Uses ``proc.poll() is None`` to check liveness rather than
    ``os.kill(pid, 0)``, which is vulnerable to PID reuse: on a busy
    system the daemon's PID could be recycled between the daemon
    exiting and the next liveness check.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is None:
            return True
        time.sleep(0.1)
    return False


def _get_fts_triggers(db: Path) -> list[str]:
    """Return the sorted list of FTS trigger names present in the database."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE '%_fts_%' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


def _expected_fts_triggers() -> set[str]:
    """The canonical set of FTS sync triggers."""
    return {
        "messages_fts_ai",
        "messages_fts_au",
        "messages_fts_ad",
    }


def _content_hashes(db: Path, limit: int = 10) -> list[tuple[str, bytes]]:
    """Return (session_id, content_hash) for up to *limit* sessions."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT session_id, content_hash FROM sessions ORDER BY session_id LIMIT ?",
            (limit,),
        ).fetchall()
    return [(str(session_id), bytes(content_hash)) for session_id, content_hash in rows]


def _wal_size(db: Path) -> int:
    """Return WAL file size in bytes (0 if absent)."""
    wal = db.with_suffix(".db-wal")
    if wal.exists():
        return wal.stat().st_size
    return 0


def _cleanup_process(proc: subprocess.Popen[bytes] | None) -> int | None:
    """Terminate, wait, force-kill a subprocess. Returns exit code or None."""
    if proc is None:
        return None
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        return proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None


_HAS_SYSTEMD_SCOPE: bool | None = None


def _has_systemd_scope() -> bool:
    """Check whether ``systemd-run --user --scope`` is available.

    The result is memoised so the subprocess runs at most once per
    test session, avoiding repeated fork+exec at collection time (the
    function is called by ``@pytest.mark.skipif`` at module load).
    """
    global _HAS_SYSTEMD_SCOPE
    if _HAS_SYSTEMD_SCOPE is None:
        try:
            result = subprocess.run(
                ["systemd-run", "--user", "--scope", "--quiet", "--", "true"],
                capture_output=True,
                timeout=5,
            )
            _HAS_SYSTEMD_SCOPE = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            _HAS_SYSTEMD_SCOPE = False
    return _HAS_SYSTEMD_SCOPE


# ---------------------------------------------------------------------------
# SIGKILL recovery test
# ---------------------------------------------------------------------------


@_BIN_B
def test_sigkill_recovery(workspace_env: dict[str, Path]) -> None:
    """Kill the daemon mid-ingest and verify clean recovery on restart.

    Assertions:
    - FTS triggers are present after restart.
    - Candidate sessions are recovered and every source session becomes active.
    - Content hashes unchanged.
    - Daemon reaches ready state within timeout.
    """
    archive_root = workspace_env["archive_root"]
    corpus_root = archive_root / "corpus" / "projects"
    db = archive_root / "index.db"

    # 1. Create source files.
    # More than one fair-intake discovery page, so observing the first
    # candidate page proves there is still source work when SIGKILL lands.
    N_SESSIONS = 33
    MESSAGES_PER_SESSION = 2
    for session_index in range(N_SESSIONS):
        session_id = f"ccccc000-0000-0000-0000-{session_index:012d}"
        _write_claude_code_session(corpus_root / f"{session_id}.jsonl", session_id, MESSAGES_PER_SESSION)

    polylogued = _polylogued_binary()

    # 2. Start daemon.
    env = os.environ.copy()
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    env["POLYLOGUE_SCHEMA_VALIDATION"] = "off"
    env["POLYLOGUE_CONFIG"] = ""  # disable host config
    proc: subprocess.Popen[bytes] | None = None
    # The daemon logs throughout intake. An undrained PIPE fills and blocks
    # its event loop before the next page, turning this recovery probe into a
    # logging backpressure test. Keep the stream on disk and read a bounded
    # tail only when the assertion needs diagnostics.
    stderr_log = archive_root.parent / "sigkill-recovery.stderr"
    stderr_capture = stderr_log.open("w+b")
    try:
        proc = subprocess.Popen(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--no-api",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_capture,
        )
        _assert_daemon_alive(proc)

        # 3. Wait for substantial ingest progress so the SIGKILL reliably
        # lands during active ingestion rather than after the daemon has
        # already finished processing.
        try:
            candidate_db, msg_count, conv_count_before, pre_hashes = _wait_for_cold_build_messages(
                proc,
                archive_root,
                min_count=50,
                timeout_s=60.0,
            )
        except TimeoutError as exc:
            evidence = _sigkill_ingest_diagnostics(archive_root, owner_pid=proc.pid)
            debug = _daemon_debug(proc, db=db, corpus_root=corpus_root, stderr_log=stderr_log)
            raise TimeoutError(f"{exc}\nphase=pre_kill {evidence}\n{debug}") from exc
        # 4. SIGKILL.
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        proc = None
        assert msg_count > 0, "No messages were ingested before SIGKILL"
        assert msg_count < N_SESSIONS * MESSAGES_PER_SESSION, "Ingestion finished before SIGKILL"
        assert conv_count_before > 0
        assert pre_hashes

        # SIGKILL cannot run the daemon's discard/promotion cleanup. The
        # candidate remains inactive and may be corrupt after an abrupt
        # death; recovery must rebuild its observed rows from durable source.
        generation_metadata = json.loads((candidate_db.parent / "generation.json").read_text(encoding="utf-8"))
        assert generation_metadata["state"] == "inactive"
        assert _wait_for_sessions(db, min_count=0, timeout_s=10.0) == 0

        # 5. Restart daemon.
        restart: subprocess.Popen[bytes] = subprocess.Popen(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--no-api",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_capture,
        )
        try:
            _assert_daemon_alive(restart)
            assert _wait_for_daemon_ready(restart, timeout_s=120.0), "Daemon did not reach ready state after restart"

            # Let it catch up.
            try:
                _wait_for_sessions(db, min_count=N_SESSIONS, timeout_s=60.0)
            except TimeoutError as exc:
                evidence = _sigkill_ingest_diagnostics(archive_root, owner_pid=restart.pid)
                debug = _daemon_debug(restart, db=db, corpus_root=corpus_root, stderr_log=stderr_log)
                raise TimeoutError(f"{exc}\nphase=recovery {evidence}\n{debug}") from exc

            # 6. Assertions.
            # FTS triggers present.
            triggers = _get_fts_triggers(db)
            missing = _expected_fts_triggers() - set(triggers)
            assert not missing, f"Missing FTS triggers after restart: {sorted(missing)}"

            # Sessions not lost.
            conv_count_after = (
                sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            )
            assert conv_count_after >= conv_count_before, f"Sessions lost: {conv_count_before} → {conv_count_after}"
            # All N sessions should be present eventually.
            assert conv_count_after == N_SESSIONS, f"Expected {N_SESSIONS} sessions, got {conv_count_after}"

            # Content hashes unchanged (sample the first few that were ingested
            # before SIGKILL).
            post_hashes = dict(_content_hashes(db, limit=N_SESSIONS))
            for conv_id, h in pre_hashes:
                if conv_id in post_hashes:
                    assert post_hashes[conv_id] == h, f"Content hash changed for {conv_id}"

            # Daemon alive.
            assert restart.poll() is None, "Daemon should still be alive after recovery"
        finally:
            _cleanup_process(restart)
    finally:
        if proc is not None:
            _cleanup_process(proc)
        stderr_capture.close()


# ---------------------------------------------------------------------------
# WAL checkpoint test
# ---------------------------------------------------------------------------


@_BIN_C
def test_wal_checkpoint_recovery(workspace_env: dict[str, Path]) -> None:
    """Verify WAL checkpoint succeeds, and large-WAL recovery is clean.

    1. Start daemon, let it ingest.
    2. Run ``PRAGMA wal_checkpoint(TRUNCATE)`` — assert success.
    3. Kill daemon while WAL is large (during active ingest of many files).
    4. Restart; assert WAL is checkpointed and no corruption.
    """
    archive_root = workspace_env["archive_root"]
    corpus_root = archive_root / "corpus" / "projects"
    db = archive_root / "index.db"

    # Write enough sessions to keep the daemon busy.
    N_SESSIONS = 10
    MESSAGES_PER_SESSION = 50
    for session_index in range(N_SESSIONS):
        session_id = f"wal-recv-{session_index:012d}"
        _write_claude_code_session(corpus_root / f"{session_id}.jsonl", session_id, MESSAGES_PER_SESSION)

    polylogued = _polylogued_binary()
    env = os.environ.copy()
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    env["POLYLOGUE_SCHEMA_VALIDATION"] = "off"
    env["POLYLOGUE_CONFIG"] = ""

    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--no-api",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _assert_daemon_alive(proc)

        # 1. Let daemon ingest some sessions.
        _wait_for_messages(db, min_count=50, timeout_s=120.0)

        # 2. Run PRAGMA wal_checkpoint(TRUNCATE).
        from polylogue.storage.sqlite.connection_profile import open_connection

        with open_connection(db, timeout=5.0) as conn:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        # (busy, log, checkpointed)
        assert row is not None, "wal_checkpoint returned None"
        # A busy result is acceptable when the daemon is actively writing;
        # the checkpoint result should not error even under load.
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        proc = None

        wal_after_kill = _wal_size(db)
        # WAL should exist after active-ingest kill.
        if wal_after_kill > 0:
            # 4. Restart.
            restart: subprocess.Popen[bytes] = subprocess.Popen(
                [
                    polylogued,
                    "run",
                    "--root",
                    str(corpus_root),
                    "--no-browser-capture",
                    "--no-api",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                _assert_daemon_alive(restart)
                assert _wait_for_daemon_ready(restart, timeout_s=120.0)
                # Give it a moment to checkpoint.
                time.sleep(3)

                # 5. Assert clean recovery: WAL checkpointed, no corruption.
                wal_after_restart = _wal_size(db)
                # After restart, the WAL should be bounded — recovery should
                # not leave an unbounded WAL from the pre-kill write window.
                assert wal_after_restart < 100 * 1024 * 1024, (
                    f"WAL after restart abnormally large: {wal_after_restart / 1024 / 1024:.1f} MiB"
                )
                from polylogue.storage.sqlite.connection_profile import open_connection as oc2

                with oc2(db, timeout=5.0) as conn:
                    row2 = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                assert row2 is not None
                busy_after = int(row2[0])
                # If not busy, checkpoint should succeed.
                if busy_after == 0:
                    wal_final = _wal_size(db)
                    assert wal_final < 10 * 1024 * 1024, (
                        f"WAL too large after recovery: {wal_final / 1024 / 1024:.1f} MiB"
                    )

                # Verify integrity.
                with oc2(db, timeout=5.0) as conn:
                    result = conn.execute("PRAGMA integrity_check").fetchone()
                assert result is not None
                assert result[0] == "ok", f"Integrity check failed: {result[0]}"
            finally:
                _cleanup_process(restart)
    finally:
        if proc is not None:
            _cleanup_process(proc)


# ---------------------------------------------------------------------------
# Memory pressure test (systemd-run --scope)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_systemd_scope(), reason="systemd-run --user --scope not available")
@_BIN_D
def test_daemon_memory_pressure(workspace_env: dict[str, Path]) -> None:
    """Start daemon under a cgroup memory limit and assert it stays within budget.

    Uses ``systemd-run --user --scope -p MemoryMax=2G``. The test is
    skipped when systemd-run is not available (CI without systemd, macOS).
    """
    archive_root = workspace_env["archive_root"]
    corpus_root = archive_root / "corpus" / "projects"
    db = archive_root / "index.db"

    N_SESSIONS = 8
    MESSAGES_PER_SESSION = 100
    for session_index in range(N_SESSIONS):
        session_id = f"memtest-{session_index:012d}"
        _write_claude_code_session(corpus_root / f"{session_id}.jsonl", session_id, MESSAGES_PER_SESSION)

    polylogued = _polylogued_binary()
    env = os.environ.copy()
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    env["POLYLOGUE_SCHEMA_VALIDATION"] = "off"
    env["POLYLOGUE_CONFIG"] = ""

    proc: subprocess.Popen[bytes] | None = None
    stderr_path = archive_root.parent / "memory-pressure.stderr"
    try:
        # systemd-run with 2 GiB memory limit.
        cmd: list[str] = [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            "MemoryMax=2G",
            "-p",
            "MemorySwapMax=0",
            "--quiet",
            "--",
            polylogued,
            "run",
            "--root",
            str(corpus_root),
            "--no-browser-capture",
            "--no-api",
        ]
        with stderr_path.open("w+b") as stderr_capture:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=stderr_capture,
            )
            _assert_daemon_alive(proc)

            # Wait for all sessions to be ingested.
            try:
                conv_count = _wait_for_sessions(db, min_count=N_SESSIONS, timeout_s=300.0)
            except TimeoutError as exc:

                def session_ids(path: Path) -> tuple[set[str], str | None]:
                    try:
                        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.1) as conn:
                            return {str(row[0]) for row in conn.execute("SELECT native_id FROM sessions")}, None
                    except (OSError, sqlite3.Error) as error:
                        return set(), type(error).__name__

                active_ids, active_error = session_ids(db)
                inactive: list[tuple[str, set[str] | None, str | None]] = []
                for metadata_path in sorted((archive_root / ".index-generations").glob("gen-*/generation.json"))[:4]:
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata.get("state") != "inactive":
                            continue
                        ids, error = session_ids(metadata_path.parent / "index.db")
                        inactive.append((metadata_path.parent.name, ids, error))
                    except (OSError, ValueError) as error:
                        inactive.append((metadata_path.parent.name, None, type(error).__name__))

                cursor_rows: dict[str, tuple[int, str | None, int, int] | None] = {}
                debt_rows: list[tuple[str, str, str]] = []
                ops_error: str | None = None
                try:
                    with sqlite3.connect(f"file:{archive_root / 'ops.db'}?mode=ro", uri=True, timeout=0.1) as conn:
                        for path in sorted(corpus_root.glob("memtest-*.jsonl"))[:N_SESSIONS]:
                            row = conn.execute(
                                "SELECT failure_count, next_retry_at, excluded, content_fingerprint IS NOT NULL "
                                "FROM ingest_cursor WHERE source_path = ?",
                                (str(path),),
                            ).fetchone()
                            cursor_rows[path.stem] = tuple(row) if row is not None else None
                        debt_rows = [
                            (str(stage), str(status), Path(str(target_id)).stem)
                            for stage, status, target_id in conn.execute(
                                "SELECT stage, status, target_id FROM convergence_debt ORDER BY stage, target_id LIMIT 32"
                            )
                        ]
                except (OSError, sqlite3.Error) as error:
                    ops_error = type(error).__name__

                path_states = []
                for path in sorted(corpus_root.glob("memtest-*.jsonl"))[:N_SESSIONS]:
                    row = cursor_rows.get(path.stem)
                    cursor_state = (
                        "missing"
                        if row is None
                        else "excluded"
                        if row[2]
                        else "failed"
                        if row[0]
                        else "deferred"
                        if row[1] is not None and not row[3]
                        else "accepted"
                        if row[3]
                        else "unresolved"
                    )
                    path_states.append(
                        f"{path.stem}: active={path.stem in active_ids} "
                        f"inactive={[name for name, ids, _error in inactive if ids is not None and path.stem in ids]} "
                        f"cursor_state={cursor_state} cursor={row}"
                    )
                stderr_capture.flush()
                stderr_capture.seek(max(0, os.fstat(stderr_capture.fileno()).st_size - 8192))
                stderr_tail = stderr_capture.read(8192).decode(errors="replace")
                # Keep event names, numeric counters and error types only. A daemon
                # log line can contain source text, so raw stderr is not a safe
                # assertion payload even with this synthetic fixture.
                stderr_events = [
                    " ".join(
                        re.findall(r"\b(?:daemon|live)\.[A-Za-z0-9_.]+\b", line)[:3]
                        + re.findall(r"\b[A-Za-z_]+(?:Error|Exception)\b", line)[:2]
                        + re.findall(r"\b(?:files|succeeded|failed|retried|deferred|refused)=\d+\b", line)[:6]
                    )
                    for line in stderr_tail.splitlines()[-40:]
                ]
                stderr_events = [line for line in stderr_events if line]
                raise TimeoutError(
                    f"{exc}; active_sessions={len(active_ids)} active_error={active_error}; "
                    f"inactive_generations={[(name, len(ids) if ids is not None else None, error) for name, ids, error in inactive]}; "
                    f"ops_error={ops_error}; paths={path_states}; debt={debt_rows}; "
                    f"stderr_events={stderr_events[-20:]}"
                ) from exc
        assert conv_count == N_SESSIONS, f"Expected {N_SESSIONS} sessions, got {conv_count}"

        # Assert daemon is still alive (did not OOM).
        assert proc.poll() is None, f"Daemon exited prematurely with code {proc.returncode} — likely OOM"

        # Verify messages ingested.
        msg_count = _wait_for_messages(db, min_count=N_SESSIONS * MESSAGES_PER_SESSION, timeout_s=60.0)
        assert msg_count >= N_SESSIONS * MESSAGES_PER_SESSION, (
            f"Message ingestion incomplete: {msg_count} < {N_SESSIONS * MESSAGES_PER_SESSION}"
        )
    finally:
        if proc is not None:
            _cleanup_process(proc)


# ---------------------------------------------------------------------------
# Large session file test
# ---------------------------------------------------------------------------


@_BIN_A
def test_large_session_file(workspace_env: dict[str, Path]) -> None:
    """Generate a 50K-message JSONL file and verify the daemon ingests it.

    Asserts:
    - Daemon processes the file without exceeding 2 GB RSS.
    - All 50K messages are ingested correctly.
    - FTS triggers intact.
    """
    archive_root = workspace_env["archive_root"]
    corpus_root = archive_root / "corpus" / "projects"
    db = archive_root / "index.db"

    session_id = "large-session-000000000000"
    n_messages = 50_000
    total_bytes = _write_large_session(corpus_root / f"{session_id}.jsonl", session_id, n_messages)
    assert total_bytes > 5_000_000, f"Large session too small: {total_bytes} bytes"

    polylogued = _polylogued_binary()
    env = os.environ.copy()
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    env["POLYLOGUE_SCHEMA_VALIDATION"] = "off"
    env["POLYLOGUE_CONFIG"] = ""

    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--no-api",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _assert_daemon_alive(proc)

        # Wait for ingestion to complete.
        _wait_for_messages(db, min_count=n_messages, timeout_s=300.0)

        # Verify daemon is still alive.
        assert proc.poll() is None, f"Daemon exited prematurely with code {proc.returncode}"

        # Peak RESIDENT memory (VmHWM), not peak virtual address space (VmPeak).
        # VmRSS sampled after ingestion reports the post-cleanup state rather than
        # the high-water mark, so a peak field is required -- but VmPeak measures
        # reservation, not use. A free-threaded interpreter with SQLite mmap and
        # per-thread arenas reserves multiple GiB of address space while staying
        # well under a GiB resident (measured here: VmPeak 3591 MiB against VmHWM
        # 590 MiB), so bounding VmPeak fails for reasons unrelated to memory
        # pressure and says nothing about whether the daemon actually ballooned.
        pid = proc.pid
        peak_rss_bytes = 0
        peak_vsize_bytes = 0
        try:
            status_path = Path(f"/proc/{pid}/status")
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                if line.startswith("VmHWM:"):
                    peak_rss_bytes = int(parts[1]) * 1024  # kB → bytes
                elif line.startswith("VmPeak:"):
                    peak_vsize_bytes = int(parts[1]) * 1024
        except OSError:
            pass

        peak_rss_mb = peak_rss_bytes / (1024 * 1024) if peak_rss_bytes > 0 else 0
        if peak_rss_mb > 0:
            assert peak_rss_bytes <= 2 * 1024 * 1024 * 1024, (
                f"Daemon peak RSS exceeds 2 GB: {peak_rss_mb:.0f} MiB "
                f"(VmPeak/address space {peak_vsize_bytes / (1024 * 1024):.0f} MiB, informational)"
            )

        # All messages ingested.
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count >= n_messages, f"Expected {n_messages} messages, got {count}"

        # FTS triggers intact.
        triggers = _get_fts_triggers(db)
        missing = _expected_fts_triggers() - set(triggers)
        assert not missing, f"Missing FTS triggers: {sorted(missing)}"

        # FTS index may still be converging (triggers suspended during bulk
        # ingest are restored + rebuilt by the convergence stage). Poll for
        # the expected row count with a grace window.
        fts_deadline = time.monotonic() + 30.0
        fts_count = 0
        while time.monotonic() < fts_deadline:
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
                if fts_count >= n_messages:
                    break
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if not any(token in msg for token in ("locked", "no such table", "unable to open database file")):
                    raise
            time.sleep(0.5)
        assert fts_count >= n_messages, f"FTS index under-populated after 30s: {fts_count} < {n_messages}"
    finally:
        if proc is not None:
            _cleanup_process(proc)


# ---------------------------------------------------------------------------
# Concurrent access safety test
# ---------------------------------------------------------------------------


@_BIN_C
def test_concurrent_access_safety(workspace_env: dict[str, Path]) -> None:
    """Verify public reads alongside the resident writer and second-daemon refusal.

    The CLI's read operations use the daemon's UDS transport. The production
    API must be serving before either CLI call can succeed, and an isolated
    HTTP port keeps this daemon separate from other resilience workers.

    1. Start daemon, let it begin ingesting.
    2. Start a second daemon process — assert the archive owner refuses it.
    3. Run ``polylogue --plain status`` while the first daemon owns the archive.
    4. Run ``polylogue --plain analyze --count`` through the same public route.
    """
    archive_root = workspace_env["archive_root"]
    corpus_root = archive_root / "corpus" / "projects"
    db = archive_root / "index.db"

    # Write sessions so the daemon stays busy.
    N_SESSIONS = 5
    MESSAGES_PER_SESSION = 30
    for session_index in range(N_SESSIONS):
        session_id = f"concurrent-{session_index:012d}"
        _write_claude_code_session(corpus_root / f"{session_id}.jsonl", session_id, MESSAGES_PER_SESSION)

    polylogued = _polylogued_binary()
    polylogue = _polylogue_binary()
    api_port = _free_local_port()
    env = os.environ.copy()
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    env["POLYLOGUE_SCHEMA_VALIDATION"] = "off"
    env["POLYLOGUE_CONFIG"] = ""
    env["POLYLOGUE_DAEMON_URL"] = f"http://127.0.0.1:{api_port}"

    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--api-port",
                str(api_port),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _assert_daemon_alive(proc)
        _wait_for_api_ready(proc, api_port)

        # Wait for ingest to begin.
        try:
            _wait_for_messages(db, min_count=1, timeout_s=60.0)
        except TimeoutError as exc:
            raise TimeoutError(f"{exc}\n{_daemon_debug(proc, db=db, corpus_root=corpus_root)}") from exc

        # 1. A second daemon cannot acquire this archive's write ownership.
        result = subprocess.run(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--no-api",
            ],
            env=env,
            capture_output=True,
            timeout=90,
        )
        assert result.returncode != 0, (
            f"Second daemon should exit non-zero; got {result.returncode}\n"
            f"stderr: {result.stderr.decode(errors='replace')[:500]}"
        )

        # 2. CLI status through the running daemon.
        env["POLYLOGUE_FORCE_PLAIN"] = "1"
        status_result = subprocess.run(
            [polylogue, "--plain", "status", "--format", "json"],
            env=env,
            capture_output=True,
            timeout=120,
        )
        # A live status read may truthfully exit 1 while ingestion leaves
        # archive readiness degraded. The JSON envelope proves this was the
        # daemon's answer for this archive, not an unreachable/direct fallback.
        assert status_result.returncode in {0, 1}, status_result.stderr.decode(errors="replace")[:500]
        status_payload = json.loads(status_result.stdout)
        assert status_payload["source"] == "daemon", status_payload
        assert status_payload["daemon_liveness"] is True, status_payload
        assert status_payload["archive_root"] == str(archive_root), status_payload
        assert status_payload["active_archive_root_matches_configured"] is True, status_payload

        # 3. CLI analyze --count through the running daemon.
        count_result = subprocess.run(
            [polylogue, "--plain", "analyze", "--count"],
            env=env,
            capture_output=True,
            timeout=120,
        )
        assert count_result.returncode == 0, (
            f"polylogue analyze --count failed: {count_result.returncode}\n"
            f"stderr: {count_result.stderr.decode(errors='replace')[:500]}"
        )
        # Output should contain a message count.
        stdout = count_result.stdout.decode(errors="replace")
        assert stdout.strip(), "polylogue analyze --count produced empty output"

    finally:
        if proc is not None:
            _cleanup_process(proc)
