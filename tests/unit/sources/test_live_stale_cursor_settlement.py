"""A stale cursor retries only its own item through production intake."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue import Polylogue
from polylogue.daemon.intake import AdmissionOutcome
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.source_layout import export_drop_layout
from tests.infra.raw_owner_routes import live_owner_set


def _claude_record(uuid: str, parent: str | None, role: str, text: str) -> bytes:
    content: object = text if role == "user" else [{"type": "text", "text": text}]
    return (
        json.dumps(
            {
                "type": role,
                "message": {"role": role, "content": content},
                "uuid": uuid,
                "parentUuid": parent,
                "sessionId": "stale-settlement",
                "timestamp": "2026-10-07T00:00:00Z",
            }
        ).encode()
        + b"\n"
    )


@pytest.mark.asyncio
async def test_stale_cursor_retries_only_its_sibling_in_production_intake(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "claude-code"
    source_root.mkdir(parents=True)
    stable = source_root / "stable.jsonl"
    stale = source_root / "stale.jsonl"
    for path, prefix in ((stable, "stable"), (stale, "stale")):
        path.write_bytes(
            _claude_record(f"{prefix}-user", None, "user", f"{prefix} question")
            + _claude_record(f"{prefix}-assistant", f"{prefix}-user", "assistant", f"{prefix} answer")
        )
    original_observations = {
        path: (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in (stable, stale)
    }

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    async with live_owner_set(archive_root) as owners:
        cursor = CursorStore(archive_root / "index.db")
        watcher = LiveWatcher(
            archive,
            (WatchSource(name="claude-code", root=source_root, layout=export_drop_layout((".jsonl",))),),
            cursor=cursor,
            **owners.watcher_kwargs(),
        )
        real_record = watcher._batch_processor._record_full_cursor

        def lose_one_cursor_write(path: Path, **kwargs: Any) -> int:
            if path == stale:
                watcher._batch_processor._last_cursor_write_stale = True
                return 0
            return real_record(path, **kwargs)

        monkeypatch.setattr(watcher._batch_processor, "_record_full_cursor", lose_one_cursor_write)
        batches: list[Any] = []
        real_ingest = watcher._ingest_files

        async def recording_ingest(*args: Any, **kwargs: Any) -> Any:
            metrics = await real_ingest(*args, **kwargs)
            batches.append(metrics)
            return metrics

        watcher._ingest_files = recording_ingest  # type: ignore[method-assign]
        try:
            adapter = FileIntakeAdapter(
                DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=watcher._sources),
                watcher._sources[0],
            )
            outcomes = await adapter.admit_page(await adapter.discover(limit=8))
        finally:
            watcher.stop()
    await archive.close()

    assert outcomes[f"file:{stable}"].outcome is AdmissionOutcome.ADMITTED, outcomes
    assert outcomes[f"file:{stale}"].outcome is AdmissionOutcome.RETRYABLE
    assert outcomes[f"file:{stale}"].reason == "source cursor write was stale"
    metrics = batches[-1]
    assert metrics.stale_cursor_write_count == 1
    assert metrics.stale_cursor_paths == (str(stale),)
    assert stable in metrics.succeeded_paths
    assert str(stable) not in metrics.failed_paths
    with sqlite3.connect(archive_root / "source.db") as source_db:
        assert source_db.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE source_path = ?",
            (str(stable),),
        ).fetchone() == (1,)
    assert {
        path: (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in (stable, stale)
    } == original_observations
