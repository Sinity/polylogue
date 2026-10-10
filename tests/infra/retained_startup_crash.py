"""Abrupt exits on the actual retained-startup writer and checkpoint route."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import pytest

from polylogue.daemon import cli
from polylogue.operations import index_reconvergence_startup as startup
from polylogue.storage import index_generation
from polylogue.storage.sqlite.archive_tiers import archive, revision_governance


def crash_on_next_page(phase: str) -> None:
    startup._RAW_PAGE = 1
    checkpoint = index_generation.IndexGenerationStore.checkpoint_reconstruction
    apply = revision_governance.apply_raw_revision_replay
    completed = False

    def checkpoint_then_exit(
        self: index_generation.IndexGenerationStore,
        generation: index_generation.IndexGeneration,
        *,
        raw_id: str,
        source_sequence: int,
        rewind: bool = False,
    ) -> index_generation.IndexGeneration:
        nonlocal completed
        if completed and phase == "committed":
            os._exit(72)
        result = checkpoint(self, generation, raw_id=raw_id, source_sequence=source_sequence, rewind=rewind)
        completed = True
        return result

    def apply_then_exit(store: archive.ArchiveStore, *args: Any, **kwargs: Any) -> Any:
        if completed and phase == "uncommitted":
            conn = store._conn
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            # Damage an already-completed row without committing. Only the
            # WAL writer profile makes this interrupted transaction disposable.
            assert conn.execute("UPDATE sessions SET title='uncommitted damage'").rowcount > 0
            os._exit(71)
        return apply(store, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(index_generation.IndexGenerationStore, "checkpoint_reconstruction", checkpoint_then_exit)
        patch.setattr(archive, "apply_raw_revision_replay", apply_then_exit)
        asyncio.run(
            cli.run_daemon_services(
                sources=(),
                enable_watch=False,
                enable_browser_capture=False,
                browser_capture_host="127.0.0.1",
                browser_capture_port=8765,
            )
        )
    raise AssertionError("startup did not reach the next page crash")


if __name__ == "__main__":
    crash_on_next_page(sys.argv[1])
