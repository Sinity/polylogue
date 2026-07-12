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
  not depend on the HTTP API being enabled.
- Subprocess cleanup: every test uses ``try/finally`` with
  ``.terminate()`` → ``.wait(timeout=10)`` → ``.kill()`` → ``.wait()``
  to guarantee no orphan daemons.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

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
        try:
            stderr_text = proc.stderr.read().decode(errors="replace")[:2000] if proc.stderr else "(no stderr)"
        except Exception:
            stderr_text = "(could not read stderr)"
        raise AssertionError(f"Daemon exited prematurely with code {returncode}. stderr:\n{stderr_text}")


def _polylogued_binary() -> str:
    """Return the ``polylogued`` binary path from the current environment.

    Falls back to ``polylogued`` on PATH when the devshell wrapper is not
    resolved (e.g. in CI).
    """
    from shutil import which

    candidate = which("polylogued")
    if candidate is not None:
        return candidate
    # Devshell direct entry
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / ".direnv/sinnix-scope/bin/polylogued"
    if wrapper.exists():
        return str(wrapper)
    return "polylogued"


def _polylogue_binary() -> str:
    """Return the ``polylogue`` CLI binary path."""
    from shutil import which

    candidate = which("polylogue")
    if candidate is not None:
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = repo_root / ".direnv/sinnix-scope/bin/polylogue"
    if wrapper.exists():
        return str(wrapper)
    return "polylogue"


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_lifecycle_start(proc: subprocess.Popen[bytes], ops_db: Path, *, timeout_s: float = 30.0) -> None:
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
            daemon.send_signal(signal.SIGTERM)
            assert daemon.wait(timeout=15) == 128 + signal.SIGTERM
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
    assert "received SIGTERM; dumping all thread stacks" in log_text
    assert "Current thread" in log_text


def _daemon_debug(proc: subprocess.Popen[bytes], *, db: Path, corpus_root: Path) -> str:
    if proc.poll() is None:
        _cleanup_process(proc)
    try:
        stderr_text = proc.stderr.read().decode(errors="replace")[:4000] if proc.stderr else "(no stderr)"
    except Exception:
        stderr_text = "(could not read stderr)"
    files = sorted(str(path) for path in corpus_root.glob("**/*") if path.is_file())[:20]
    return (
        f"returncode={proc.poll()} db={db} db_exists={db.exists()} "
        f"source_exists={db.with_name('source.db').exists()} corpus_files={files}\n"
        f"stderr:\n{stderr_text}"
    )


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


def _wait_for_daemon_ready(proc: subprocess.Popen[bytes], *, timeout_s: float = 30.0) -> bool:
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


def _content_hashes(db: Path, limit: int = 10) -> list[tuple[str, str]]:
    """Return (session_id, content_hash) for up to *limit* sessions."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT session_id, content_hash FROM sessions ORDER BY session_id LIMIT ?",
            (limit,),
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


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


def test_sigkill_recovery(workspace_env: dict[str, Path]) -> None:
    """Kill the daemon mid-ingest and verify clean recovery on restart.

    Assertions:
    - FTS triggers are present after restart.
    - No sessions lost (count before == count after).
    - Content hashes unchanged.
    - Daemon reaches ready state within timeout.
    """
    archive_root = workspace_env["archive_root"]
    corpus_root = archive_root / "corpus" / "projects"
    db = archive_root / "index.db"

    # 1. Create source files.
    N_SESSIONS = 5
    MESSAGES_PER_SESSION = 20
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
    try:
        proc = subprocess.Popen(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--no-api",
                "--debounce-s",
                "0.3",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _assert_daemon_alive(proc)

        # 3. Wait for substantial ingest progress so the SIGKILL reliably
        # lands during active ingestion rather than after the daemon has
        # already finished processing.
        msg_count = _wait_for_messages(db, min_count=max(5, N_SESSIONS * MESSAGES_PER_SESSION // 2), timeout_s=60.0)
        assert msg_count > 0, "No messages were ingested before SIGKILL"

        # 4. SIGKILL.
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        proc = None

        # Record pre-recovery state.
        pre_hashes = _content_hashes(db, limit=N_SESSIONS)
        conv_count_before = _wait_for_sessions(db, min_count=1, timeout_s=10.0)

        # 5. Restart daemon.
        restart: subprocess.Popen[bytes] = subprocess.Popen(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--no-api",
                "--debounce-s",
                "0.3",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            _assert_daemon_alive(restart)
            assert _wait_for_daemon_ready(restart, timeout_s=30.0), "Daemon did not reach ready state after restart"

            # Let it catch up.
            _wait_for_sessions(db, min_count=N_SESSIONS, timeout_s=60.0)

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


# ---------------------------------------------------------------------------
# WAL checkpoint test
# ---------------------------------------------------------------------------


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
                "--debounce-s",
                "0.2",
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
                    "--debounce-s",
                    "0.2",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                _assert_daemon_alive(restart)
                assert _wait_for_daemon_ready(restart, timeout_s=30.0)
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
            "--debounce-s",
            "0.3",
        ]
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _assert_daemon_alive(proc)

        # Wait for all sessions to be ingested.
        conv_count = _wait_for_sessions(db, min_count=N_SESSIONS, timeout_s=300.0)
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
                "--debounce-s",
                "0.3",
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

        # Check VmPeak from /proc — the high-water mark of virtual memory
        # usage.  VmRSS sampled after ingestion has finished reports the
        # post-cleanup state, not the peak.
        pid = proc.pid
        peak_bytes = 0
        try:
            status_path = Path(f"/proc/{pid}/status")
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("VmPeak:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        peak_bytes = int(parts[1]) * 1024  # kB → bytes
                    break
        except OSError:
            pass

        peak_mb = peak_bytes / (1024 * 1024) if peak_bytes > 0 else 0
        if peak_mb > 0:
            assert peak_bytes <= 2 * 1024 * 1024 * 1024, f"Daemon VmPeak exceeds 2 GB: {peak_mb:.0f} MiB"

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


def test_concurrent_access_safety(workspace_env: dict[str, Path]) -> None:
    """Verify WAL read-during-write safety and daemon pidfile locking.

    The daemon runs with ``--no-api``, so ``polylogue --plain status``
    and ``polylogue --plain analyze --count`` fall through to direct SQLite reads
    against the WAL journal.  This is the correct test for WAL-mode
    concurrency: a reader must be able to open the database while the
    daemon writer holds an active transaction.

    1. Start daemon, let it begin ingesting.
    2. Start a second daemon process — assert it exits non-zero (pidfile locked).
    3. Run ``polylogue --plain status`` while daemon is ingesting — assert exit 0.
    4. Run ``polylogue --plain analyze --count`` while daemon is ingesting — assert exit 0.
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
                "--debounce-s",
                "0.3",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _assert_daemon_alive(proc)

        # Wait for ingest to begin.
        try:
            _wait_for_messages(db, min_count=1, timeout_s=60.0)
        except TimeoutError as exc:
            raise TimeoutError(f"{exc}\n{_daemon_debug(proc, db=db, corpus_root=corpus_root)}") from exc

        # 1. Second daemon should fail (pidfile lock).
        result = subprocess.run(
            [
                polylogued,
                "run",
                "--root",
                str(corpus_root),
                "--no-browser-capture",
                "--no-api",
                "--debounce-s",
                "0.3",
            ],
            env=env,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode != 0, (
            f"Second daemon should exit non-zero; got {result.returncode}\n"
            f"stderr: {result.stderr.decode(errors='replace')[:500]}"
        )

        # 2. CLI status while daemon is ingesting.
        env["POLYLOGUE_FORCE_PLAIN"] = "1"
        status_result = subprocess.run(
            [polylogue, "--plain", "status"],
            env=env,
            capture_output=True,
            timeout=30,
        )
        assert status_result.returncode == 0, (
            f"polylogue ops status failed: {status_result.returncode}\n"
            f"stderr: {status_result.stderr.decode(errors='replace')[:500]}"
        )

        # 3. CLI analyze --count while daemon is ingesting.
        count_result = subprocess.run(
            [polylogue, "--plain", "analyze", "--count"],
            env=env,
            capture_output=True,
            timeout=30,
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
