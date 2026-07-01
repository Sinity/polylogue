"""Tests for current archive session-link resolution (#1258 / #866).

Covers all four ACs from #1258:

1. Claude Code subagent fixture: parent absent -> ``unresolved`` row with
   ``link_type=subagent``.
2. Claude Code sidechain fixture: parent absent -> ``unresolved`` row with
   ``link_type=sidechain``.
3. Codex continuation fixture: parent absent -> ``unresolved`` row with
   ``link_type=continuation``.
4. ChatGPT branched session referencing an unimported parent
   session -> ``unresolved`` row.

Plus the structural invariants the issue calls out:

- Fast-path preservation: when the parent IS already ingested, the
  session's ``parent_session_id`` is still set AND a corresponding
  ``session_links`` row exists with ``status=resolved``.
- Out-of-order resolve: ingest child first, then parent -> link flips to
  ``resolved`` and ``dst_session_id`` is populated.
- Idempotency: re-ingesting the same child twice produces exactly one
  ``session_links`` row.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.topology.edge import TopologyEdgeStatus, TopologyEdgeType
from polylogue.core.enums import Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.sources import origin_from_provider
from polylogue.sources import iter_source_sessions
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.live_ingest import ingest_session
from tests.infra.storage_records import db_setup

WorkspaceEnv = dict[str, Path]


def _make_repository(db_path: Path) -> SessionRepository:
    return SessionRepository(backend=SQLiteBackend(db_path=db_path))


def _write_payload(tmp_path: Path, filename: str, payload: object) -> Path:
    path = tmp_path / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parse_single(source_name: str, source_path: Path) -> ParsedSession:
    from polylogue.config import Source

    sessions = list(iter_source_sessions(Source(name=source_name, path=source_path)))
    assert len(sessions) == 1
    return sessions[0]


def _fetch_edges(db_path: Path) -> list[sqlite3.Row]:
    with open_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT src_session_id,
                   dst_native_id AS dst_session_native_id,
                   link_type,
                   resolved_dst_session_id AS dst_session_id,
                   CASE
                       WHEN status IS NOT NULL THEN status
                       WHEN resolved_dst_session_id IS NOT NULL THEN 'resolved'
                       ELSE 'unresolved'
                   END AS status
              FROM session_links
             ORDER BY src_session_id, link_type
            """
        )
        return list(cursor.fetchall())


def _archive_session_id(provider: Provider, native_id: str) -> str:
    return archive_session_id(origin_from_provider(provider).value, native_id)


async def _ingest_synthetic_child(
    *,
    repo: SessionRepository,
    provider: Provider,
    child_id: str,
    parent_id: str,
    branch_type: BranchType | None,
) -> str:
    parsed = ParsedSession(
        source_name=provider,
        provider_session_id=child_id,
        title=f"Child {child_id}",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hi")],
        parent_session_provider_id=parent_id,
        branch_type=branch_type,
    )
    await ingest_session(
        parsed,
        backend=repo.backend,
    )
    return _archive_session_id(provider, child_id)


class TestTopologyEdgeUnresolvedAC:
    """The four acceptance-criteria fixtures from #1258."""

    @pytest.mark.asyncio
    async def test_claude_code_subagent_parent_absent_unresolved(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CLAUDE_CODE,
                child_id="subagent-sess",
                parent_id="missing-parent-sess",
                branch_type=BranchType.SUBAGENT,
            )

        edges = _fetch_edges(db_path)
        assert len(edges) == 1
        edge = edges[0]
        assert edge["dst_session_native_id"] == "missing-parent-sess"
        assert edge["link_type"] == TopologyEdgeType.SUBAGENT.value
        assert edge["status"] == TopologyEdgeStatus.UNRESOLVED.value
        assert edge["dst_session_id"] is None

    @pytest.mark.asyncio
    async def test_claude_code_sidechain_parent_absent_unresolved(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CLAUDE_CODE,
                child_id="side-child",
                parent_id="missing-main-sess",
                branch_type=BranchType.SIDECHAIN,
            )

        edges = _fetch_edges(db_path)
        assert len(edges) == 1
        assert edges[0]["link_type"] == TopologyEdgeType.SIDECHAIN.value
        assert edges[0]["status"] == TopologyEdgeStatus.UNRESOLVED.value

    @pytest.mark.asyncio
    async def test_codex_continuation_parent_absent_unresolved(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="cont-child",
                parent_id="missing-codex-parent",
                branch_type=BranchType.CONTINUATION,
            )

        edges = _fetch_edges(db_path)
        assert len(edges) == 1
        assert edges[0]["link_type"] == TopologyEdgeType.CONTINUATION.value
        assert edges[0]["status"] == TopologyEdgeStatus.UNRESOLVED.value
        assert edges[0]["dst_session_native_id"] == "missing-codex-parent"

    @pytest.mark.asyncio
    async def test_codex_unclassified_parent_uses_generic_link(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        """Codex forked_from_id proves a parent, not fork-vs-resume."""

        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            child_id = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="unclassified-child",
                parent_id="unclassified-parent",
                branch_type=None,
            )

            with open_connection(db_path) as conn:
                child_before = conn.execute(
                    "SELECT parent_session_id, branch_type FROM sessions WHERE session_id = ?",
                    (child_id,),
                ).fetchone()
            assert child_before["parent_session_id"] is None
            assert child_before["branch_type"] is None

            parent_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="unclassified-parent",
                title="Unclassified Parent",
                messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="parent")],
            )
            await ingest_session(parent_parsed, backend=repo.backend)
            parent_id = _archive_session_id(Provider.CODEX, "unclassified-parent")

        edges = _fetch_edges(db_path)
        assert len(edges) == 1
        assert edges[0]["link_type"] == TopologyEdgeType.BRANCH.value
        assert edges[0]["status"] == TopologyEdgeStatus.RESOLVED.value
        assert edges[0]["dst_session_id"] == parent_id

        with open_connection(db_path) as conn:
            child_after = conn.execute(
                "SELECT parent_session_id, branch_type FROM sessions WHERE session_id = ?",
                (child_id,),
            ).fetchone()
        assert child_after["parent_session_id"] == parent_id
        assert child_after["branch_type"] is None

    @pytest.mark.asyncio
    async def test_chatgpt_branch_parent_session_absent_unresolved(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parsed = ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="forked-chat",
                title="Forked Chat",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hi")],
                parent_session_provider_id="orig-chat",
                branch_type=BranchType.FORK,
            )
            await ingest_session(
                parsed,
                backend=repo.backend,
            )

        edges = _fetch_edges(db_path)
        assert len(edges) == 1
        assert edges[0]["link_type"] == TopologyEdgeType.FORK.value
        assert edges[0]["status"] == TopologyEdgeStatus.UNRESOLVED.value


class TestTopologyEdgeFastPathPreserved:
    @pytest.mark.asyncio
    async def test_parent_first_ingest_fast_path_and_resolved_edge(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parent_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="parent-id",
                title="Parent",
                messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="p")],
            )
            await ingest_session(
                parent_parsed,
                backend=repo.backend,
            )
            parent_session_id = _archive_session_id(Provider.CODEX, "parent-id")
            child_cid = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="child-id",
                parent_id="parent-id",
                branch_type=BranchType.CONTINUATION,
            )

        # Fast-path: parent_session_id on the sessions row is set.
        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE session_id = ?",
                (child_cid,),
            ).fetchone()
        assert row["parent_session_id"] == parent_session_id

        edges = _fetch_edges(db_path)
        assert len(edges) == 1
        edge = edges[0]
        assert edge["status"] == TopologyEdgeStatus.RESOLVED.value
        assert edge["dst_session_id"] == parent_session_id


class TestTopologyEdgeOutOfOrderResolve:
    @pytest.mark.asyncio
    async def test_child_first_then_parent_flips_to_resolved(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            # Child ingested before parent → unresolved.
            child_cid = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="ooo-child",
                parent_id="ooo-parent",
                branch_type=BranchType.CONTINUATION,
            )

            edges_before = _fetch_edges(db_path)
            assert len(edges_before) == 1
            assert edges_before[0]["status"] == TopologyEdgeStatus.UNRESOLVED.value
            assert edges_before[0]["dst_session_id"] is None

            # Now the parent lands.
            parent_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="ooo-parent",
                title="Parent (late)",
                messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="p")],
            )
            await ingest_session(
                parent_parsed,
                backend=repo.backend,
            )
            parent_session_id = _archive_session_id(Provider.CODEX, "ooo-parent")

        edges_after = _fetch_edges(db_path)
        # The edge should flip; there should still be one row for the child.
        child_edges = [e for e in edges_after if e["src_session_id"] == child_cid]
        assert len(child_edges) == 1
        assert child_edges[0]["status"] == TopologyEdgeStatus.RESOLVED.value
        assert child_edges[0]["dst_session_id"] == parent_session_id

        # Slice B (#1259 / #866): the late-arriving parent now backfills
        # the child's ``sessions.parent_session_id`` and
        # ``branch_type`` columns, so the fast-path ancestry walk benefits
        # without requiring re-ingest of the child.
        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT parent_session_id, branch_type FROM sessions WHERE session_id = ?",
                (child_cid,),
            ).fetchone()
        assert row["parent_session_id"] == parent_session_id
        assert row["branch_type"] == BranchType.CONTINUATION.value


class TestTopologyEdgeIdempotency:
    @pytest.mark.asyncio
    async def test_reingesting_child_produces_one_row(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            for _ in range(2):
                await _ingest_synthetic_child(
                    repo=repo,
                    provider=Provider.CODEX,
                    child_id="idem-child",
                    parent_id="idem-parent",
                    branch_type=BranchType.CONTINUATION,
                )

        edges = _fetch_edges(db_path)
        assert len(edges) == 1
        assert edges[0]["status"] == TopologyEdgeStatus.UNRESOLVED.value


class TestTopologyLateParentRepair:
    """Slice B (#1259 / #866): late-parent arrival deterministic edge repair.

    Covers the slice B acceptance criteria:

    - On insert of the parent session, the previously-unresolved edge is
              repaired (dst_session_id set, status=resolved) AND the child session's
      ``parent_session_id`` + ``branch_type`` columns are backfilled.
    - Running the resolver twice produces the same result (idempotent).
    - When the parent never arrives, the edge stays unresolved and the
      child's parent_session_id stays NULL.
    - Multiple unresolved children pointing at the same absent parent are
      all repaired by a single parent ingest.
    - When the child's ``parent_session_id`` was already populated
      (parent-first fast path), the repair pass does not overwrite it on
      a subsequent re-ingest of the parent.
    """

    @pytest.mark.asyncio
    async def test_late_parent_arrival_backfills_fast_path(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            child_cid = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="late-child",
                parent_id="late-parent",
                branch_type=BranchType.SIDECHAIN,
            )

            # Before parent arrives: fast-path NULL, edge unresolved. The
            # ``branch_type`` is set on the child at its original write time
            # from the parser-asserted classification — it does not depend on
            # the parent being present.
            with open_connection(db_path) as conn:
                row = conn.execute(
                    "SELECT parent_session_id, branch_type FROM sessions WHERE session_id = ?",
                    (child_cid,),
                ).fetchone()
            assert row["parent_session_id"] is None
            assert row["branch_type"] == BranchType.SIDECHAIN.value

            parent_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="late-parent",
                title="Late parent",
                messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="p")],
            )
            await ingest_session(
                parent_parsed,
                backend=repo.backend,
            )
            parent_session_id = _archive_session_id(Provider.CODEX, "late-parent")

        # After parent arrives: fast-path AND branch_type backfilled.
        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT parent_session_id, branch_type FROM sessions WHERE session_id = ?",
                (child_cid,),
            ).fetchone()
        assert row["parent_session_id"] == parent_session_id
        assert row["branch_type"] == BranchType.SIDECHAIN.value

    @pytest.mark.asyncio
    async def test_repair_is_idempotent_across_parent_reingest(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            child_cid = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="idem-repair-child",
                parent_id="idem-repair-parent",
                branch_type=BranchType.SUBAGENT,
            )

            parent_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="idem-repair-parent",
                title="Idem parent",
                messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="p")],
            )
            # Ingest the parent twice.
            await ingest_session(
                parent_parsed,
                backend=repo.backend,
            )
            await ingest_session(
                parent_parsed,
                backend=repo.backend,
            )
            parent_session_id = _archive_session_id(Provider.CODEX, "idem-repair-parent")

        edges_after = _fetch_edges(db_path)
        child_edges = [e for e in edges_after if e["src_session_id"] == child_cid]
        assert len(child_edges) == 1
        assert child_edges[0]["status"] == TopologyEdgeStatus.RESOLVED.value
        assert child_edges[0]["dst_session_id"] == parent_session_id

        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT parent_session_id, branch_type FROM sessions WHERE session_id = ?",
                (child_cid,),
            ).fetchone()
        assert row["parent_session_id"] == parent_session_id
        assert row["branch_type"] == BranchType.SUBAGENT.value

    @pytest.mark.asyncio
    async def test_parent_never_arrives_keeps_edge_unresolved(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            child_cid = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="orphan-child",
                parent_id="never-arrives",
                branch_type=BranchType.CONTINUATION,
            )

            # Ingest a different session — its arrival must not
            # spuriously repair the unrelated unresolved edge.
            unrelated = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="unrelated-parent",
                title="Unrelated",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="x")],
            )
            await ingest_session(
                unrelated,
                backend=repo.backend,
            )

        edges = _fetch_edges(db_path)
        child_edges = [e for e in edges if e["src_session_id"] == child_cid]
        assert len(child_edges) == 1
        assert child_edges[0]["status"] == TopologyEdgeStatus.UNRESOLVED.value
        assert child_edges[0]["dst_session_id"] is None

        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE session_id = ?",
                (child_cid,),
            ).fetchone()
        assert row["parent_session_id"] is None

    @pytest.mark.asyncio
    async def test_multiple_pending_children_repaired_by_single_parent(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            child_a = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="shared-parent-child-a",
                parent_id="shared-parent",
                branch_type=BranchType.SIDECHAIN,
            )
            child_b = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="shared-parent-child-b",
                parent_id="shared-parent",
                branch_type=BranchType.SUBAGENT,
            )

            parent_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="shared-parent",
                title="Shared parent",
                messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="p")],
            )
            await ingest_session(
                parent_parsed,
                backend=repo.backend,
            )
            parent_session_id = _archive_session_id(Provider.CODEX, "shared-parent")

        edges = _fetch_edges(db_path)
        repaired = [e for e in edges if e["src_session_id"] in (child_a, child_b)]
        assert len(repaired) == 2
        assert all(e["status"] == TopologyEdgeStatus.RESOLVED.value for e in repaired)
        assert all(e["dst_session_id"] == parent_session_id for e in repaired)

        with open_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, parent_session_id, branch_type "
                "FROM sessions WHERE session_id IN (?, ?) "
                "ORDER BY session_id",
                (child_a, child_b),
            ).fetchall()

        by_id = {row["session_id"]: row for row in rows}
        assert by_id[child_a]["parent_session_id"] == parent_session_id
        assert by_id[child_a]["branch_type"] == BranchType.SIDECHAIN.value
        assert by_id[child_b]["parent_session_id"] == parent_session_id
        assert by_id[child_b]["branch_type"] == BranchType.SUBAGENT.value

    @pytest.mark.asyncio
    async def test_repair_does_not_overwrite_existing_fast_path(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        """Parent-first fast path is preserved when the parent is re-ingested."""
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parent_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="preserve-parent",
                title="Parent",
                messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="p")],
            )
            await ingest_session(
                parent_parsed,
                backend=repo.backend,
            )
            parent_session_id = _archive_session_id(Provider.CODEX, "preserve-parent")
            child_cid = await _ingest_synthetic_child(
                repo=repo,
                provider=Provider.CODEX,
                child_id="preserve-child",
                parent_id="preserve-parent",
                branch_type=BranchType.FORK,
            )

            # Re-ingest the parent — repair pass must be a no-op on the
            # already-resolved child.
            await ingest_session(
                parent_parsed,
                backend=repo.backend,
            )

        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT parent_session_id, branch_type FROM sessions WHERE session_id = ?",
                (child_cid,),
            ).fetchone()
        assert row["parent_session_id"] == parent_session_id
        assert row["branch_type"] == BranchType.FORK.value

    @pytest.mark.asyncio
    async def test_concurrent_parent_and_child_ingest_resolves(
        self,
        workspace_env: WorkspaceEnv,
        tmp_path: Path,
    ) -> None:
        """Concurrent ingest of parent + child must converge to a resolved edge.

        The two ``ingest_session`` coroutines below open independent connections, so
        coroutines below execute in some interleaved order. Regardless of which
        one wins the race, the post-condition is the same: the topology edge
        is resolved, ``dst_session_id`` points at the parent,
        and the child's ``parent_session_id`` is backfilled.
        """
        import asyncio

        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            child_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="race-child",
                title="Race child",
                messages=[ParsedMessage(provider_message_id="cm1", role=Role.USER, text="c")],
                parent_session_provider_id="race-parent",
                branch_type=BranchType.CONTINUATION,
            )
            parent_parsed = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="race-parent",
                title="Race parent",
                messages=[ParsedMessage(provider_message_id="pm1", role=Role.USER, text="p")],
            )

            child_task = ingest_session(
                child_parsed,
                backend=repo.backend,
            )
            parent_task = ingest_session(
                parent_parsed,
                backend=repo.backend,
            )
            await asyncio.gather(child_task, parent_task)
            child_session_id = _archive_session_id(Provider.CODEX, "race-child")
            parent_session_id = _archive_session_id(Provider.CODEX, "race-parent")

        edges = _fetch_edges(db_path)
        child_edges = [e for e in edges if e["src_session_id"] == child_session_id]
        assert len(child_edges) == 1
        assert child_edges[0]["status"] == TopologyEdgeStatus.RESOLVED.value
        assert child_edges[0]["dst_session_id"] == parent_session_id

        with open_connection(db_path) as conn:
            row = conn.execute(
                "SELECT parent_session_id, branch_type FROM sessions WHERE session_id = ?",
                (child_session_id,),
            ).fetchone()
        assert row["parent_session_id"] == parent_session_id
        assert row["branch_type"] == BranchType.CONTINUATION.value


__all__: tuple[str, ...] = ()
