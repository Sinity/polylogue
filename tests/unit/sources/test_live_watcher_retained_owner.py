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
from tests.infra.raw_owner_routes import live_owner_set


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


@pytest.mark.asyncio
async def test_live_blank_only_append_advances_raw_evidence_without_adding_messages(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    source_root = tmp_path / "sessions"
    source_root.mkdir()
    path = source_root / "stable-session.jsonl"
    original = (
        b'{"type":"session_meta","payload":{"id":"stable-session"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m1",'
        b'"role":"user","content":[{"type":"input_text","text":"stable prompt"}]}}\n'
    )
    path.write_bytes(original)
    async with live_owner_set(root) as owners:
        await owners.coordinator.run_sync("fixture.blank-append.bootstrap", lambda: bootstrap_archive_root(root))
        cursor = await owners.coordinator.run_sync(
            "fixture.blank-append.cursor", lambda: CursorStore(root / "index.db", ops_db_path=root / "ops.db")
        )
        archive = Polylogue(archive_root=root)
        watcher = LiveWatcher(
            archive,
            [WatchSource(name="codex", root=source_root)],
            cursor=cursor,
            **owners.watcher_kwargs(),
        )

        def read_state() -> tuple[str, int, tuple[tuple[str, bytes], ...], tuple[tuple[str, str], ...]]:
            with ArchiveStore.open_existing(root, read_only=True) as stored:
                source = stored.source_connection
                index = stored.index_connection
                assert source is not None and index is not None
                row = index.execute(
                    "SELECT raw_id, message_count FROM sessions WHERE native_id='stable-session'"
                ).fetchone()
                assert row is not None
                raws = tuple(
                    (str(raw[0]), stored.raw_revision_material(str(raw[0]))[1])
                    for raw in source.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid")
                )
                applications = tuple(
                    (str(application[0]), str(application[1]))
                    for application in index.execute("SELECT raw_id, decision FROM raw_revision_applications")
                )
                return str(row[0]), int(row[1]), raws, applications

        try:
            baseline = await watcher._ingest_files([path])
            assert baseline.succeeded_file_count == 1
            assert baseline.failed_file_count == 0
            before = await owners.raw_owner.run_convergence_sync("fixture.blank-append.before", read_state)
            assert before[1] == 1
            assert before[2] == ((before[0], original),)

            with path.open("ab") as handle:
                handle.write(b"\n\n")
            appended = await watcher._ingest_files([path])
            after = await owners.raw_owner.run_convergence_sync("fixture.blank-append.after", read_state)
            assert appended.append_file_count == 1
            assert appended.succeeded_file_count == 1
            assert appended.failed_file_count == 0
            assert after[0] != before[0]
            assert after[1] == before[1] == 1
            assert len(after[2]) == 2
            assert after[2][0] == before[2][0]
            assert after[2][1] == (after[0], b"\n\n")
            assert (after[0], "applied_append") in after[3]
        finally:
            watcher.stop()
            await archive.close()
