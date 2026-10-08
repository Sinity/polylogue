"""Original batch visibility and BaseException rollback on retained publication."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.enums import Provider
from polylogue.storage.fts.fts_lifecycle import FTS_TRIGGER_NAMES
from polylogue.storage.fts.sql import FTS_BULK_SESSION_WRITE_GUARD
from polylogue.storage.sqlite.archive_tiers import write
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.source_builders import ChatGPTExportBuilder


def _visible(root: Path) -> list[str]:
    with closing(sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)) as conn:
        return [row[0] for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id")]


def _triggers(root: Path) -> set[str]:
    with closing(sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)) as conn:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
            if row[0] in FTS_TRIGGER_NAMES
        }


async def _acquire(root: Path) -> tuple[str, str]:
    def acquire() -> tuple[str, str]:
        bootstrap_archive_root(root)
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            ids = []
            for name in ("boundary-one", "boundary-two"):
                payload = json.dumps(
                    ChatGPTExportBuilder(name).add_node("user", f"batch transaction boundary {name}").build()
                ).encode()
                ids.append(
                    archive.write_raw_payload(
                        provider=Provider.CHATGPT,
                        payload=payload,
                        source_path=f"{name}.json",
                        canonical_source_path=f"{name}.json",
                        acquired_at_ms=1,
                    )
                )
        return ids[0], ids[1]

    return await run_archive_fixture_write(root, acquire)


@pytest.mark.asyncio
async def test_retained_component_is_not_committed_per_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "archive"
    raw_ids = await _acquire(root)
    observed: list[list[str]] = []
    original = write._write_messages

    def observe(conn: sqlite3.Connection, *args: Any, **kwargs: Any) -> None:
        original(conn, *args, **kwargs)
        observed.append(_visible(root))

    monkeypatch.setattr(write, "_write_messages", observe)
    async with prepared_live_convergence_owner(root) as owner:
        receipts = (
            await owner.replay_retained_raw_ids(raw_ids, select_retained_raw_ids=lambda reader: raw_ids)
        ).require_complete()
    expected = ["chatgpt-export:boundary-one", "chatgpt-export:boundary-two"]
    assert len(observed) == 2, observed
    assert observed == [[], []]
    assert sorted({key for receipt in receipts for key in receipt.changed_session_ids}) == expected
    assert _visible(root) == expected


def test_retained_component_interrupt_preserves_fts_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # asyncio lets KeyboardInterrupt escape the event loop instead of
    # delivering it to the awaiting coroutine, so the genuine interrupt is
    # observed where the loop returns it: outside ``asyncio.run``.
    root = tmp_path / "archive"
    raw_ids = asyncio.run(_acquire(root))
    baseline = _triggers(root)
    assert baseline == set(FTS_TRIGGER_NAMES)
    original = write._write_messages
    calls = 0
    primary = KeyboardInterrupt()

    def interrupt(conn: sqlite3.Connection, *args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert (
                conn.execute(
                    "SELECT 1 FROM derived_refresh_guard WHERE guard_name=?",
                    (FTS_BULK_SESSION_WRITE_GUARD,),
                ).fetchone()
                is not None
            )
            raise primary
        original(conn, *args, **kwargs)

    monkeypatch.setattr(write, "_write_messages", interrupt)

    async def replay() -> None:
        async with prepared_live_convergence_owner(root) as owner:
            (
                await owner.replay_retained_raw_ids(raw_ids, select_retained_raw_ids=lambda reader: raw_ids)
            ).require_complete()

    with pytest.raises(KeyboardInterrupt) as caught:
        asyncio.run(replay())
    assert caught.value is primary
    assert calls == 2
    assert _triggers(root) == baseline
    assert _visible(root) == []
    with closing(sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM derived_refresh_guard WHERE guard_name=?",
                (FTS_BULK_SESSION_WRITE_GUARD,),
            ).fetchone()
            is None
        )
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 0
