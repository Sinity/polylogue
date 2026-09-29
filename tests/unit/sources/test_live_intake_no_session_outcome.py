"""A source file that yields no session is excluded, not admitted.

A transcript whose records carry no conversational evidence is acquired and
parsed, and its raw row records the typed terminal outcome. The intake
outcome for that file must say so: reporting ``ADMITTED`` counted a file that
produced nothing among the operator's admitted files (polylogue-xf8qp).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue import Polylogue
from polylogue.daemon.intake import AdmissionOutcome
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.operations.operation_context import open_operation_read
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore

_MAX_DEFERRED_PAGES = 20


async def _admit(archive_root: Path, source_root: Path, metrics_sink: list[Any] | None = None) -> dict[str, Any]:
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    watcher = LiveWatcher(
        archive,
        (WatchSource(name="claude-code", root=source_root),),
        cursor=CursorStore(archive_root / "index.db"),
        read_snapshot=open_operation_read,
    )
    if metrics_sink is not None:
        ingest_files = watcher._ingest_files

        async def recording_ingest(*args: Any, **kwargs: Any) -> Any:
            metrics = await ingest_files(*args, **kwargs)
            metrics_sink.append(metrics)
            return metrics

        watcher._ingest_files = recording_ingest  # type: ignore[method-assign]
    try:
        outcomes: dict[str, Any] = {}
        for _ in range(_MAX_DEFERRED_PAGES):
            adapter = FileIntakeAdapter(
                DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=watcher._sources),
                watcher._sources[0],
            )
            outcomes = dict(await adapter.admit_page(await adapter.discover(limit=8)))
            if {result.outcome for result in outcomes.values()} != {AdmissionOutcome.DEFERRED}:
                return outcomes
        raise AssertionError(f"the source stayed deferred for {_MAX_DEFERRED_PAGES} pages: {outcomes}")
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_a_file_without_sessions_is_excluded_not_admitted(workspace_env: dict[str, Path]) -> None:
    """Anti-vacuity: the pre-fix route put the no-session raw into the
    succeeded set, so the outcome was ``ADMITTED`` with no session written."""
    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "claude-projects"
    source_root.mkdir(parents=True)
    source_path = source_root / "silent.jsonl"
    # A well-formed Claude Code record with no conversational content: it is
    # acquired and parsed, and yields no session with positive evidence.
    source_path.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": ""},
                "uuid": "u1",
                "sessionId": "silent",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    batches: list[Any] = []
    outcomes = await _admit(archive_root, source_root, batches)

    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.EXCLUDED}, outcomes
    # The batch receipt agrees with the intake outcome (Codex): it once
    # counted the same file as a success with its bytes ingested.
    metrics = batches[-1]
    assert metrics.succeeded_file_count == 0 and not metrics.succeeded_paths
    assert metrics.excluded_file_count == 1
    assert metrics.excluded_paths == {str(source_path): "no_sessions"}
    assert metrics.ingested_bytes == 0
    # The durable attempt row carries the same typed refusal, not SUCCESS.
    from polylogue.core.enums import IngestOutcome

    with sqlite3.connect(archive_root / "ops.db") as ops:
        (outcome_code,) = ops.execute("SELECT outcome_code FROM ingest_attempts ORDER BY rowid DESC LIMIT 1").fetchone()
    assert outcome_code == IngestOutcome.UNSUPPORTED_SHAPE.value
    (result,) = outcomes.values()
    assert result.reason is not None and "no sessions" in result.reason
    with sqlite3.connect(archive_root / "index.db") as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
