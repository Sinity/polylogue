"""An empty Claude input must not hold a productive cold candidate open."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue import Polylogue
from polylogue.daemon.drive_catchup import DriveCatchupExecution
from polylogue.daemon.intake import AdmissionOutcome
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.source_layout import SYNTHETIC_CLAUDE_CODE_PROJECT
from tests.infra.raw_owner_routes import live_owner_set


def _sidechain_record(role: str, uuid: str, parent: str | None, *, sidechain: bool = True) -> bytes:
    content: object = "neutral" if role == "user" else [{"type": "text", "text": "neutral reply"}]
    return (
        json.dumps(
            {
                "type": role,
                "message": {"role": role, "content": content},
                "uuid": uuid,
                "parentUuid": parent,
                "sessionId": "neutral-parent" if sidechain else "neutral-grown",
                "agentId": "abcdef0" if sidechain else None,
                "isSidechain": sidechain,
                "timestamp": "2026-10-07T00:00:00Z",
            }
        ).encode()
        + b"\n"
    )


@pytest.mark.asyncio
async def test_empty_claude_sibling_settles_before_productive_cold_promotion(
    workspace_env: dict[str, Path],
) -> None:
    """Rejecting the empty cursor prevents intake from draining and turns red."""
    root = workspace_env["archive_root"]
    source_root = workspace_env["home_dir"] / ".claude" / "projects"
    project = source_root / SYNTHETIC_CLAUDE_CODE_PROJECT
    project.mkdir(parents=True)
    empty = project / "00000000-0000-4000-8000-000000000001.jsonl"
    positive = project / "agent-abcdef0.jsonl"
    empty.write_bytes(b"")
    positive.write_bytes(_sidechain_record("user", "u", None) + _sidechain_record("assistant", "a", "u"))
    source = WatchSource("claude-code", source_root)
    archive = Polylogue(archive_root=root, db_path=root / "index.db")
    async with live_owner_set(root) as owners:
        cursor = CursorStore(root / "index.db")
        watcher = LiveWatcher(archive, (source,), cursor=cursor, **owners.watcher_kwargs())
        observed = ColdBuildGeneration.observe_source_baseline((source,))
        cold = await owners.coordinator.run_sync(
            "fixture.empty-cold.begin",
            ColdBuildGeneration.begin,
            root,
            reason="neutral-empty-cold",
            observed=observed,
        )
        register_cold_build_generation(cold)
        try:
            adapter = FileIntakeAdapter(
                DaemonIntakeContext(archive_root=root, watcher=watcher, sources=(source,)), source
            )
            outcomes = await adapter.admit_page(await adapter.discover(limit=8))
            assert outcomes[f"file:{empty}"].outcome is AdmissionOutcome.EXCLUDED, outcomes
            assert outcomes[f"file:{positive}"].outcome is AdmissionOutcome.ADMITTED, outcomes
            empty_cursor = cursor.get_record(empty)
            assert empty_cursor is not None
            assert empty_cursor.failure_count == 0
            with sqlite3.connect(root / "source.db") as conn:
                raws = dict(conn.execute("SELECT source_path,raw_id FROM raw_sessions"))
                assert conn.execute(
                    "SELECT status FROM raw_membership_census WHERE raw_id=?", (raws[str(empty)],)
                ).fetchone() == ("non_session",)
                assert conn.execute(
                    "SELECT parsed_at_ms FROM raw_sessions WHERE raw_id=?", (raws[str(positive)],)
                ).fetchone() == (None,)
            with sqlite3.connect(cold.generation.index_path) as conn:
                assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
                assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (2,)
            await owners.coordinator.run_sync("fixture.empty-cold.readiness", cold.prepare_promotion_candidate)
            promotion = DriveCatchupExecution(owners.coordinator, compute_adapter=owners.compute)
            await promotion.publish_prepared_sync(
                "fixture.empty-cold.promote", cold.prepare_promotion_proof, cold.promote_prepared
            )
            clear_cold_build_generation()
            (await owners.raw_owner.replay_retained_raw_ids(tuple(raws.values()))).require_complete()
            with sqlite3.connect(root / "source.db") as conn:
                assert conn.execute(
                    "SELECT parsed_at_ms IS NOT NULL FROM raw_sessions WHERE raw_id=?", (raws[str(positive)],)
                ).fetchone() == (1,)
            batches: list[Any] = []
            real_ingest = watcher._ingest_files

            async def observed_ingest(*args: Any, **kwargs: Any) -> Any:
                metrics = await real_ingest(*args, **kwargs)
                batches.append(metrics)
                return metrics

            watcher._ingest_files = observed_ingest  # type: ignore[method-assign]
            empty.write_bytes(
                _sidechain_record("user", "grown-u", None, sidechain=False)
                + _sidechain_record("assistant", "grown-a", "grown-u", sidechain=False)
            )
            watcher._note_intake_hint(empty)
            grown = await adapter.admit_page(await adapter.discover(limit=8))
            assert grown[f"file:{empty}"].outcome is AdmissionOutcome.ADMITTED, grown
            assert len(batches) == 1
            assert batches[0].full_file_count == 1 and batches[0].append_file_count == 0
            with sqlite3.connect(root / "index.db") as conn:
                assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (2,)
        finally:
            watcher.stop()
            clear_cold_build_generation()
            if not cold.promoted:
                await owners.coordinator.run_sync("fixture.empty-cold.discard", cold.discard)
    await archive.close()
