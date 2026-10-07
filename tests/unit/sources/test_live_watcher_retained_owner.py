"""The production watcher preserves supplied retained publication ownership."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage
from polylogue.sources.live.watcher import LiveWatcher, WatchSource
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import prepared_live_convergence_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("derived_blocked", [False, True])
async def test_watcher_routes_acquired_raw_to_its_supplied_retained_owner(
    tmp_path: Path, derived_blocked: bool
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    source_root = tmp_path / "source"
    source_root.mkdir()
    path = source_root / "conversation.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "watcher-retained",
                    "title": "Watcher retained owner",
                    "create_time": 1,
                    "current_node": "m",
                    "mapping": {
                        "m": {
                            "id": "m",
                            "parent": None,
                            "children": [],
                            "message": {
                                "id": "m",
                                "author": {"role": "user"},
                                "create_time": 1,
                                "content": {"content_type": "text", "parts": ["retained message"]},
                            },
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    async with prepared_live_convergence_owner(root) as owner:
        coordinator = owner._write_coordinator
        await coordinator.run_sync("fixture.watcher.bootstrap", lambda: bootstrap_archive_root(root))
        cursor = await coordinator.run_sync(
            "fixture.watcher.cursor", lambda: CursorStore(root / "index.db", ops_db_path=root / "ops.db")
        )
        archive = Polylogue(archive_root=root)
        watcher = LiveWatcher(
            archive,
            [WatchSource(name="chatgpt", root=source_root, layout=export_drop_layout((".json",)))],
            cursor=cursor,
            write_coordinator=coordinator,
            sqlite_capture_stage=LiveSQLiteCaptureStage(compute_adapter=owner._compute_adapter),
            append_runner=owner.ingest_append_plans,
            convergence_runner=owner.run_convergence_sync,
            retained_runner=None if derived_blocked else owner.ingest_retained_raw_ids,
        )
        if derived_blocked:
            set_degraded(DegradedReason(code="schema_skew", message="Index unavailable", derived_only=True))
        try:
            metrics = await watcher._ingest_files([path])
            assert metrics.failed_file_count == 0
            assert metrics.ingested_session_count == (0 if derived_blocked else 1)

            def read_published() -> tuple[int, int]:
                with ArchiveStore.open_existing(root, read_only=True) as stored:
                    index = stored.index_connection
                    assert index is not None
                    raw_count = int(stored.source_connection.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0])
                    session_count = int(index.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
                    if not derived_blocked:
                        row = index.execute(
                            "SELECT session_id FROM sessions WHERE session_id = ?", ("chatgpt-export:watcher-retained",)
                        ).fetchone()
                        assert row is not None
                    return raw_count, session_count

            assert await owner.run_convergence_sync("fixture.watcher.readback", read_published) == (
                1,
                0 if derived_blocked else 1,
            )
        finally:
            clear_degraded()
            watcher.stop()
            await archive.close()
