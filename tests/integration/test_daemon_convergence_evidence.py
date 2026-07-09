"""Production convergence evidence for the daemon (#1245, slice B of #845).

Drives a real-shape synthetic corpus through ``LiveBatchProcessor`` +
``DaemonConverger`` (the same primitives ``polylogued run`` wires up at
:func:`polylogue.daemon.cli.run_daemon_services`) and asserts the
convergence shape using before/after snapshots from
``devtools.daemon_workload_probe.probe`` and its arithmetic
``compare()``.

This is the integration-level closure for slice B: prior runs produced
useful evidence about memory pressure, FK index gaps, and resumability
but never landed as a final acceptance test. The shape proved here:

1. ``daemon_workload_probe`` baseline before any ingest is well-formed
   (``ok=True``, expected FTS triggers present on a fresh schema).
2. After convergence over an N-session real-shape corpus:

   - ``sessions`` and ``messages`` rows grew by the expected
     deltas (no silently dropped sessions);
   - zero ``failed`` / ``running`` live-ingest attempts remain;
   - zero ``live_convergence_debt`` rows remain;
   - all six FTS sync triggers are present.

3. The ``compare()`` diff between the two probe snapshots is the
   structured evidence artifact requested by the issue and is written
   to ``.local/convergence-evidence/`` for later auditing.

The test runs the same primitives that ``polylogued run`` instantiates,
not a subprocess: ``polylogued`` adds watchfiles + HTTP surfaces on top
of these calls. Driving them in-process keeps the test deterministic
while still exercising the production write/convergence path.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from devtools.daemon_workload_probe import REPORT_VERSION, compare, probe
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_default_convergence_stages
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import WatchSource

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def _write_claude_code_session(path: Path, session_id: str, n_messages: int) -> None:
    """Write a realistic Claude Code session JSONL.

    The shape mirrors what production parsers see — parentUuid chains,
    sessionId, timestamps, message envelopes. Matches
    ``tests/unit/daemon/test_convergence_final_state.py``.
    """
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
                    "content": f"Message {i}: {'The quick brown fox jumps. ' * 4}",
                },
                "uuid": f"msg-{session_id}-{i:03d}",
                "timestamp": f"2026-05-19T00:{i // 60:02d}:{i % 60:02d}.000Z",
                "cwd": "/realm/project/polylogue",
                "version": "1.0.6",
                "isSidechain": False,
                "userType": "external",
            }
            fh.write(json.dumps(record) + "\n")


class _MinimalPolylogue:
    """Minimal polylogue double sufficient for LiveBatchProcessor.

    Matches the shape used by ``tests/unit/daemon/test_convergence_final_state.py``
    — ``archive_root`` and ``backend.db_path`` are the only fields the
    batch processor and converger touch at this scope.
    """

    def __init__(self, archive_root: Path, db_path: Path) -> None:
        self.archive_root = archive_root
        self.backend = SimpleNamespace(db_path=db_path)
        self.config = None


# ---------------------------------------------------------------------------
# Probe artifact persistence (the issue's "evidence committed" requirement)
# ---------------------------------------------------------------------------


def _write_evidence_artifact(
    before: dict[str, Any],
    after: dict[str, Any],
    diff: dict[str, Any],
) -> Path | None:
    """Persist the before/after/diff snapshots to .local/convergence-evidence/.

    The issue requires the evidence to be "committed as a
    ``.local/convergence-evidence/`` artifact (gitignored, but referenced in
    PR)". When running outside a polylogue ops doctorout (or in a sandbox
    that forbids writes there) we silently skip — the assertions still
    run on the in-memory snapshots.
    """
    try:
        # Repo root is two levels up from tests/integration/.
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = repo_root / ".local" / "convergence-evidence"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "before.json").write_text(json.dumps(before, indent=2))
        (out_dir / "after.json").write_text(json.dumps(after, indent=2))
        (out_dir / "diff.json").write_text(json.dumps(diff, indent=2))
        return out_dir
    except OSError:
        return None


# ---------------------------------------------------------------------------
# The evidence
# ---------------------------------------------------------------------------


SESSION_COUNT = 5
MESSAGES_PER_SESSION = 12


def test_daemon_convergence_evidence_full_archive_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end convergence evidence against a real-shape Claude Code corpus.

    Drives ``LiveBatchProcessor`` + ``DaemonConverger`` (the same primitives
    ``polylogued run`` wires up at
    :func:`polylogue.daemon.cli.run_daemon_services`) over N sessions and
    asserts the resulting archive state via ``daemon_workload_probe``.
    """
    corpus_root = tmp_path / "corpus" / "projects"
    db_path = tmp_path / "index.db"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")

    # ── Seed N real-shape Claude Code session files ──────────────────
    files: list[Path] = []
    for session_index in range(SESSION_COUNT):
        session_id = f"bbbb0000-0000-0000-0000-{session_index:012d}"
        path = corpus_root / f"{session_id}.jsonl"
        _write_claude_code_session(path, session_id, MESSAGES_PER_SESSION)
        files.append(path)

    # ── Bootstrap the archive tiers so the BEFORE probe sees a real DB ──────
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(tmp_path):
        pass

    # ── BEFORE snapshot ──────────────────────────────────────────────
    before = probe(db_path, exact_table_counts=True)
    assert before["ok"] is True, before.get("error")
    assert before["report_version"] == REPORT_VERSION

    before_counts = before["boundary_table_counts"]
    assert before_counts["sessions"] == 0, before_counts
    assert before_counts["messages"] == 0, before_counts
    assert before["fts_trigger_state"]["all_present"] is True, before["fts_trigger_state"]

    # ── Drive convergence: same primitives as polylogued run ─────────
    converger = DaemonConverger(
        stages=make_default_convergence_stages(db_path),
        max_workers=2,
    )
    polylogue = _MinimalPolylogue(tmp_path, db_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="claude-code", root=corpus_root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="convergence-evidence-v1",
        converger=converger,
    )

    async def _run_convergence() -> Any:
        await converger.start()
        try:
            metrics = await processor.ingest_files(files, emit_event=False)
        finally:
            await converger.stop()
        return metrics

    metrics = asyncio.run(_run_convergence())

    # ── Ingest completeness ─────────────────────────────────────────
    assert metrics.failed_file_count == 0, (
        f"convergence evidence failed: {metrics.failed_file_count} file(s) failed ingest"
    )
    assert metrics.succeeded_file_count == len(files), (
        f"convergence evidence: expected {len(files)} successes, got {metrics.succeeded_file_count}"
    )

    # ── AFTER snapshot ──────────────────────────────────────────────
    after = probe(db_path, exact_table_counts=True)
    assert after["ok"] is True, after.get("error")
    assert after["report_version"] == REPORT_VERSION

    diff = compare(before, after)
    assert diff["ok"] is True, diff.get("error")

    # Persist the structured evidence artifact (issue requirement).
    _write_evidence_artifact(before, after, diff)

    after_counts = after["boundary_table_counts"]

    # ── Convergence shape — strict acceptance criteria ──────────────
    # Sessions grew by exactly SESSION_COUNT. The corpus is fresh
    # (no pre-existing rows) so this is a hard equality, not a lower bound.
    assert after_counts["sessions"] == SESSION_COUNT, (
        f"expected {SESSION_COUNT} sessions after convergence, "
        f"got {after_counts['sessions']}; "
        f"diff={diff['boundary_table_counts'].get('sessions')}"
    )

    # Each session contributes MESSAGES_PER_SESSION messages; provider
    # parsing may collapse adjacent records but never expand them, so
    # use a lower bound on per-session contribution while keeping the
    # total bounded above by the raw record count.
    expected_messages = SESSION_COUNT * MESSAGES_PER_SESSION
    assert after_counts["messages"] >= expected_messages, (
        f"expected >= {expected_messages} messages, got {after_counts['messages']}"
    )
    assert after_counts["messages"] <= expected_messages, (
        f"unexpected message expansion: produced {after_counts['messages']} "
        f"messages from {expected_messages} input records"
    )

    # raw_sessions is the ingest landing table in source.db — one per source file.
    raw_count = after["archive_tiers"]["tiers"]["source"]["table_counts"]["raw_sessions"]
    assert raw_count == SESSION_COUNT, f"expected {SESSION_COUNT} raw_sessions, got {raw_count}"

    # ── No stuck or failed live-ingest attempts ─────────────────────
    attempt_counts = after["attempt_counts"]
    assert attempt_counts["running"] == 0, (
        f"convergence left {attempt_counts['running']} live_ingest_attempt rows in 'running'"
    )
    assert attempt_counts["failed"] == 0, f"convergence left {attempt_counts['failed']} failed live_ingest_attempt rows"

    # ── Zero convergence debt ───────────────────────────────────────
    debt = after["convergence_debt"]
    assert debt["failed_count"] == 0, f"convergence left {debt['failed_count']} unresolved debt rows: {debt}"

    # ── FTS sync triggers intact ────────────────────────────────────
    fts_state = after["fts_trigger_state"]
    assert fts_state["all_present"] is True, f"FTS trigger drift after convergence: {fts_state}"
    assert not fts_state["missing"], f"missing FTS triggers post-convergence: {fts_state['missing']}"

    # ── FTS sanity: searchable content lands in the index ──────────
    # The strongest end-to-end signal that the write path stayed
    # consistent with the FTS triggers throughout convergence.
    with sqlite3.connect(db_path) as conn:
        (fts_rows,) = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()
    assert fts_rows >= expected_messages, (
        f"FTS index under-populated: {fts_rows} rows vs {expected_messages} expected messages"
    )
