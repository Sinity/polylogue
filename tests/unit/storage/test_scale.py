"""Native scale tests for the archive read surfaces.

These tests verify correct behavior with 200+ sessions to catch:
- N+1 query patterns (per-session connection storms)
- Memory issues with large result sets
- Correct data association across batch reads

The original production failure was a connection storm with thousands of
concurrent SQLite connections. These tests ensure the archive batch read path
(``Polylogue.get_sessions`` / ``list_sessions``) works at moderate
scale.

Performance budget tests (TestPerformanceBudget) assert that key operations
finish within fixed timing budgets. They use ``@pytest.mark.slow`` and are
excluded from the normal fast unit run.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder, _record_to_parsed_session, db_setup

# 200 is enough to expose N+1 patterns while keeping tests reasonably fast.
SCALE_COUNT = 200


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


class TestBatchReadScale:
    """Native batch reads at 200+ sessions."""

    @pytest.mark.asyncio
    async def test_get_sessions_returns_all(self, workspace_env: dict[str, Path]) -> None:
        ids = _seed_archive(workspace_env, SCALE_COUNT)
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            convos = await archive.get_sessions(ids)
        assert len(convos) == SCALE_COUNT
        for convo in convos:
            assert len(convo.messages) == 3, f"Conv {convo.id} has {len(convo.messages)} messages, expected 3"

    @pytest.mark.asyncio
    async def test_get_sessions_preserves_input_order(self, workspace_env: dict[str, Path]) -> None:
        ids = _seed_archive(workspace_env, 50)
        reversed_ids = list(reversed(ids))
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            convos = await archive.get_sessions(reversed_ids)
        assert [str(c.id) for c in convos] == reversed_ids

    @pytest.mark.asyncio
    async def test_messages_belong_to_their_session(self, workspace_env: dict[str, Path]) -> None:
        ids = _seed_archive(workspace_env, SCALE_COUNT)
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            convos = await archive.get_sessions(ids)
        for convo in convos:
            for msg in convo.messages:
                assert str(msg.id).startswith(str(convo.id) + ":"), (
                    f"Message {msg.id} does not belong to session {convo.id}"
                )

    @pytest.mark.asyncio
    async def test_varying_message_counts(self, workspace_env: dict[str, Path]) -> None:
        """Sessions with different message counts read back correctly."""
        db_path = db_setup(workspace_env)
        msg_counts = [1, 5, 10]
        ids: list[str] = []
        expected: dict[str, int] = {}
        with ArchiveStore(workspace_env["archive_root"]) as archive:
            for msgs in msg_counts:
                for i in range(20):
                    builder = SessionBuilder(db_path, f"var-{msgs}msg-{i:03d}").provider("chatgpt")
                    for j in range(msgs):
                        builder.add_message(role="user", text=f"msg {j}")
                    parsed = _record_to_parsed_session(builder.conv, builder.messages, builder.attachments)
                    archive.write_parsed(parsed)
                    sid = builder.native_session_id()
                    ids.append(sid)
                    expected[sid] = msgs

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as archive:
            convos = await archive.get_sessions(ids)
        assert len(convos) == 60  # 20 * 3
        for convo in convos:
            assert len(convo.messages) == expected[str(convo.id)]

    @pytest.mark.asyncio
    async def test_get_sessions_empty_input(self, workspace_env: dict[str, Path]) -> None:
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            assert await archive.get_sessions([]) == []


class TestListScale:
    """Archive list at scale, with and without provider filter."""

    @pytest.mark.asyncio
    async def test_list_returns_all(self, workspace_env: dict[str, Path]) -> None:
        _seed_archive(workspace_env, SCALE_COUNT)
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            # limit=None defaults to 50; pass an explicit ceiling to read the whole archive.
            convos = await archive.list_sessions(limit=SCALE_COUNT)
        assert len(convos) == SCALE_COUNT

    @pytest.mark.asyncio
    async def test_list_with_origin_filter(self, workspace_env: dict[str, Path]) -> None:
        _seed_archive(workspace_env, SCALE_COUNT)  # alternates chatgpt / claude-ai
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            chatgpt = await archive.list_sessions(origin="chatgpt-export", limit=SCALE_COUNT)
        assert len(chatgpt) == SCALE_COUNT // 2
        for convo in chatgpt:
            assert str(convo.origin) == "chatgpt-export"

    @pytest.mark.asyncio
    async def test_list_partitions_disjointly_at_scale(self, workspace_env: dict[str, Path]) -> None:
        _seed_archive(workspace_env, SCALE_COUNT)
        async with Polylogue(db_path=db_setup(workspace_env), archive_root=workspace_env["archive_root"]) as archive:
            all_ids = {str(c.id) for c in await archive.list_sessions(limit=SCALE_COUNT)}
            chatgpt = {str(c.id) for c in await archive.list_sessions(origin="chatgpt-export", limit=SCALE_COUNT)}
            claude = {str(c.id) for c in await archive.list_sessions(origin="claude-ai-export", limit=SCALE_COUNT)}
        assert chatgpt.isdisjoint(claude)
        assert chatgpt | claude == all_ids


class TestLargeInputRoundTrip:
    """Property: oversized message text round-trips through archive storage and FTS."""

    @pytest.mark.asyncio
    async def test_large_message_round_trips(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        large_text = "word " * 50_000  # ~250KB, 50k words

        builder = SessionBuilder(db_path, "large-conv-1").provider("chatgpt").title("Large Message Test")
        builder.add_message(role="user", text=large_text)
        builder.save()
        session_id = builder.native_session_id()

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as archive:
            result = await archive.get_session(session_id)
            assert result is not None
            assert len(result.messages) == 1
            assert result.messages.to_list()[0].text == large_text

            # FTS over the large text must work.
            hits = await archive.search("word", limit=5)
            assert any(str(hit.session_id) == session_id for hit in hits.hits)


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
