"""Timing budgets for the archive batch-read and FTS surfaces.

Budgets are conservative (10-20x typical times on a modern workstation) and
catch a growth-shape regression such as a per-session connection storm, not
small latency drift. They measure real wall-clock, so they live with the rest
of the benchmark suite rather than in the unit corpus.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder, _record_to_parsed_session, db_setup

pytestmark = [
    pytest.mark.uses_real_clock("Archive read budgets measure real wall-clock to enforce growth-shape SLOs."),
]


def _seed_archive(
    workspace_env: dict[str, Path],
    count: int,
    *,
    msgs_per_conv: int = 3,
    id_prefix: str = "scale-conv",
    providers: tuple[str, ...] = ("chatgpt", "claude-ai"),
) -> list[str]:
    """Seed ``count`` sessions through a single archive ArchiveStore writer.

    Returns the list of archive session ids in seed order.
    """
    db_path = db_setup(workspace_env)
    ids: list[str] = []
    with ArchiveStore(workspace_env["archive_root"]) as archive:
        for i in range(count):
            provider = providers[i % len(providers)]
            builder = (
                SessionBuilder(db_path, f"{id_prefix}-{i:04d}").provider(provider).title(f"Scale Test Session {i}")
            )
            for j in range(msgs_per_conv):
                builder.add_message(
                    role="user" if j % 2 == 0 else "assistant",
                    text=f"Message {j} in session {i}",
                )
            parsed = _record_to_parsed_session(builder.conv, builder.messages, builder.attachments)
            archive.write_parsed(parsed)
            ids.append(builder.native_session_id())
    return ids


@pytest.mark.slow
@pytest.mark.load_sensitive
class TestPerformanceBudget:
    """Performance budget tests — each asserts a timing SLA.

    Budgets are conservative (10–20x typical times on a modern workstation).
    """

    @pytest.mark.asyncio
    async def test_list_performance_budget(self, workspace_env: dict[str, Path]) -> None:
        """list_sessions(limit=50) on a 500-session DB must finish in <500ms."""
        _seed_archive(workspace_env, 500, msgs_per_conv=5)
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            t0 = time.monotonic()
            results = await archive.list_sessions(limit=50)
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert len(results) == 50
        assert elapsed_ms < 500, f"list_sessions took {elapsed_ms:.0f}ms (budget: 500ms)"

    @pytest.mark.asyncio
    async def test_get_sessions_performance_budget(self, workspace_env: dict[str, Path]) -> None:
        """get_sessions(100 ids) on a 500-session DB must finish in <2s."""
        ids = _seed_archive(workspace_env, 500, msgs_per_conv=5)
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            sample = ids[:100]
            t0 = time.monotonic()
            results = await archive.get_sessions(sample)
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert len(results) == 100
        assert elapsed_ms < 2000, f"get_sessions(100) took {elapsed_ms:.0f}ms (budget: 2000ms)"

    @pytest.mark.asyncio
    async def test_fts_search_budget(self, workspace_env: dict[str, Path]) -> None:
        """FTS search for a common term on a 500-session DB must finish in <1s."""
        _seed_archive(workspace_env, 500, msgs_per_conv=5)
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            t0 = time.monotonic()
            results = await archive.search("Message", limit=20)
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 1000, f"FTS search took {elapsed_ms:.0f}ms (budget: 1000ms)"
        _ = results  # exercised
