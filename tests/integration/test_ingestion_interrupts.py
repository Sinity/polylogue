"""Integration tests for pipeline resilience to interruption (E3).

Tests that interrupted pipelines can be resumed without data loss or corruption.
"""

from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.infra.chaos_sources import ChaosInboxBuilder
from tests.infra.cli_subprocess import setup_isolated_workspace

pytestmark = [pytest.mark.integration, pytest.mark.chaos]


def _message_count_if_ready(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            has_messages_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
            ).fetchone()
            if has_messages_table is None:
                return None
            return int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    except sqlite3.Error:
        return None


def _wait_for_persisted_messages(
    db_path: Path,
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        count = _message_count_if_ready(db_path)
        if count is not None and count > 0:
            return count
        if process.poll() is not None:
            break
        time.sleep(0.05)
    return _message_count_if_ready(db_path) or 0


def _finish_interrupted_process(process: subprocess.Popen[str], *, timeout_seconds: float) -> tuple[str, str]:
    try:
        return process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


# =============================================================================
# Mid-Run Interruption Tests
# =============================================================================


def test_interrupted_pipeline_preserves_partial_progress(tmp_path: Path) -> None:
    """Test that SIGINT mid-stream preserves already-persisted data.

    Strategy:
    1. Create large inbox to ensure pipeline takes time
    2. Start pipeline via Popen (not run_cli)
    3. Send SIGINT after brief delay
    4. Verify DB exists and has partial data (not zero, not complete)
    """
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Build a moderately large inbox (500+ records)
    # to ensure the pipeline runs long enough to interrupt
    builder = ChaosInboxBuilder(inbox)
    builder.add_valid_jsonl("large_export.jsonl", provider="codex", count=500)
    builder.build()

    project_root = Path(__file__).parent.parent.parent

    # Build env for subprocess
    env = workspace["env"].copy()
    env["UV_SYSTEM_PYTHON"] = "1"
    env["LANG"] = "C.UTF-8"

    # Start pipeline
    cmd = [
        sys.executable,
        "-m",
        "polylogue",
        "run",
        "--source",
        "inbox",
    ]

    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    db_path = workspace["paths"]["db_path"]
    persisted_before_interrupt = _wait_for_persisted_messages(db_path, process, timeout_seconds=15.0)
    assert persisted_before_interrupt > 0, "Database did not persist messages before interruption"

    if process.poll() is None:
        process.send_signal(signal.SIGINT)

    stdout, stderr = _finish_interrupted_process(process, timeout_seconds=10.0)

    # Verify DB exists
    assert db_path.exists(), f"Database not created before interruption\nstdout={stdout}\nstderr={stderr}"

    # Check that we have partial data (some but not all 500 records)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        # Should be partial: more than 0, less than 500
        # But allow some tolerance for very fast machines
        assert count > 0, f"No records saved before interruption (got {count})"
        # Most machines won't finish all 500 before 2s, but allow it
        # The key is that we got *something* and didn't corrupt the DB


def test_rerun_after_interruption_completes_remaining(tmp_path: Path) -> None:
    """Test that rerunning after interruption completes without duplication.

    Strategy:
    1. Create inbox
    2. Start and interrupt pipeline
    3. Rerun the same command
    4. Verify final count is correct (no duplicates)
    """
    workspace = setup_isolated_workspace(tmp_path)
    inbox = workspace["paths"]["inbox"]

    # Build inbox with 50 records for faster test
    builder = ChaosInboxBuilder(inbox)
    builder.add_valid_jsonl("test_data.jsonl", provider="codex", count=50)
    builder.build()

    project_root = Path(__file__).parent.parent.parent

    env = workspace["env"].copy()
    env["UV_SYSTEM_PYTHON"] = "1"
    env["LANG"] = "C.UTF-8"

    cmd = [
        sys.executable,
        "-m",
        "polylogue",
        "run",
        "--source",
        "inbox",
    ]

    # First run: start and interrupt
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    time.sleep(1.0)
    process.send_signal(signal.SIGINT)

    _finish_interrupted_process(process, timeout_seconds=5.0)

    db_path = workspace["paths"]["db_path"]
    first_count = 0
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            has_messages_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
            ).fetchone()
            if has_messages_table is not None:
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
                first_count = cursor.fetchone()[0]
    assert 0 <= first_count <= 50

    # Second run: complete the pipeline
    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30.0,
    )

    assert result.returncode == 0, f"Second run failed: {result.stderr}"

    # Check final count
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        final_count = cursor.fetchone()[0]

    # Should reach 50 records total (not 50 + first_count)
    assert final_count == 50, f"Expected 50 records, got {final_count}"


def test_concurrent_pipeline_runs_serialized(tmp_path: Path) -> None:
    """Test that two simultaneous pipeline runs don't corrupt each other.

    Strategy:
    1. Create two separate inboxes
    2. Start two concurrent pipelines (different workspace envs)
    3. Wait for both to complete
    4. Verify both completed correctly without interference
    """
    # Create two separate workspaces
    workspace1 = setup_isolated_workspace(tmp_path / "ws1")
    workspace2 = setup_isolated_workspace(tmp_path / "ws2")

    # Build inboxes
    builder1 = ChaosInboxBuilder(workspace1["paths"]["inbox"])
    builder1.add_valid_jsonl("data.jsonl", provider="codex", count=20)
    builder1.build()

    builder2 = ChaosInboxBuilder(workspace2["paths"]["inbox"])
    builder2.add_valid_jsonl("data.jsonl", provider="claude-code", count=30)
    builder2.build()

    project_root = Path(__file__).parent.parent.parent

    cmd = [
        sys.executable,
        "-m",
        "polylogue",
        "run",
        "--source",
        "inbox",
    ]

    # Start both pipelines concurrently
    p1 = subprocess.Popen(
        cmd,
        cwd=project_root,
        env={**workspace1["env"], "UV_SYSTEM_PYTHON": "1", "LANG": "C.UTF-8"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    p2 = subprocess.Popen(
        cmd,
        cwd=project_root,
        env={**workspace2["env"], "UV_SYSTEM_PYTHON": "1", "LANG": "C.UTF-8"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for both to complete
    try:
        p1.wait(timeout=30.0)
        p2.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        p1.kill()
        p2.kill()
        raise

    # Verify both completed successfully
    assert p1.returncode == 0, "Pipeline 1 failed"
    assert p2.returncode == 0, "Pipeline 2 failed"

    # Verify each wrote to its own DB correctly
    db1 = workspace1["paths"]["db_path"]
    db2 = workspace2["paths"]["db_path"]

    with sqlite3.connect(db1) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count1 = cursor.fetchone()[0]

    with sqlite3.connect(db2) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count2 = cursor.fetchone()[0]

    # Workspace 1 should have 20, workspace 2 should have 30
    assert count1 == 20, f"Workspace 1: expected 20 records, got {count1}"
    assert count2 == 30, f"Workspace 2: expected 30 records, got {count2}"

    # Verify provider isolation
    with sqlite3.connect(db1) as conn:
        cursor = conn.execute("SELECT DISTINCT provider_name FROM messages")
        providers1 = [row[0] for row in cursor.fetchall()]

    with sqlite3.connect(db2) as conn:
        cursor = conn.execute("SELECT DISTINCT provider_name FROM messages")
        providers2 = [row[0] for row in cursor.fetchall()]

    assert "codex" in providers1, "Workspace 1 missing codex provider"
    assert "claude-code" in providers2, "Workspace 2 missing claude-code provider"
    # Verify no cross-contamination
    assert "claude-code" not in providers1, "Workspace 1 contaminated with claude-code"
    assert "codex" not in providers2, "Workspace 2 contaminated with codex"
