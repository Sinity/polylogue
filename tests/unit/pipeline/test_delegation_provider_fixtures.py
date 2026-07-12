"""polylogue-lph4: ingest-shaped delegation fixtures.

The `delegations` view (polylogue-y964, `polylogue/storage/sqlite/archive_tiers/
index.py`) spines on `actions.semantic_type='subagent'` (a parent-side Task
tool_use) corroborated against resolved `session_links(link_type='subagent')`
edges. `tests/unit/storage/test_delegations_view.py` proves the view's SQL
contract by hand-inserting rows directly into `actions`/`session_links` --
legitimate for exercising the view's join logic in isolation, but it does not
prove that a REAL provider parser ever produces those rows in that shape.

These tests drive real JSONL/dict-shaped payloads through the actual parser
dispatch (`iter_source_sessions`, matching `tests/unit/pipeline/
test_branching.py`'s pattern) and the real writer (`ingest_session` ->
`write_parsed_session_to_archive`), then read the resulting `delegations`
view back. Three things must hold under real ingestion, none of which the
direct-SQL fixtures can prove on their own:

  1. A Claude Code Task dispatch + its `agent-*.jsonl` subagent transcript
     resolve to mapping_state='resolved' with real child-to-parent lineage
     direction (not the pre-y964 reversed direction).
  2. A Codex subagent spawn (`session_meta.source.subagent`, no parent-side
     Task action observable) resolves to mapping_state='edge_only' -- never
     a fabricated instruction.
  3. Auto-compaction (`agent-acompact-*.jsonl`, BranchType.CONTINUATION) and
     a plain Codex continuation never appear in `delegations` at all --
     `link_type != 'subagent'` is excluded at the view's own WHERE clause,
     and only a real parser round-trip proves the classifier actually
     assigns CONTINUATION (not SUBAGENT) to these shapes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.sources import origin_from_provider
from polylogue.sources import iter_source_sessions
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
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
    sessions = list(iter_source_sessions(Source(name=source_name, path=source_path)))
    assert len(sessions) == 1
    return sessions[0]


def _archive_session_id(provider: Provider, native_id: str) -> str:
    return archive_session_id(origin_from_provider(provider).value, native_id)


def _delegations_rows(db_path: Path, *, parent_session_id: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return list(
            conn.execute(
                "SELECT * FROM delegations WHERE parent_session_id = ? ORDER BY instruction_tool_use_block_id",
                (parent_session_id,),
            ).fetchall()
        )
    finally:
        conn.close()


def _claude_code_task_dispatch_payload(*, session_id: str) -> list[dict[str, object]]:
    """Real Claude Code JSONL shape: an assistant Task tool_use record
    followed by the user tool_result record for the same tool_use id."""

    return [
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": None,
            "sessionId": session_id,
            "timestamp": "2025-01-01T10:00:00Z",
            "cwd": "/repo",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_task1",
                        "name": "Task",
                        "input": {"prompt": "Investigate X", "subagent_type": "general-purpose"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "a1",
            "sessionId": session_id,
            "timestamp": "2025-01-01T10:00:05Z",
            "cwd": "/repo",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_task1", "content": "Subagent finished."}],
            },
        },
    ]


def _claude_code_agent_child_payload(*, parent_session_id: str) -> list[dict[str, object]]:
    """A subagent transcript's own records carry the PARENT's native session
    id in `sessionId` -- `code_parser.py` reads `parent_session_id = session_id`
    when the file's fallback_id (filename stem) starts with `agent-`."""

    return [
        {
            "type": "user",
            "uuid": "cu1",
            "sessionId": parent_session_id,
            "timestamp": "2025-01-01T10:00:01Z",
            "message": {"role": "user", "content": "Subagent task"},
        },
        {
            "type": "assistant",
            "uuid": "ca1",
            "parentUuid": "cu1",
            "sessionId": parent_session_id,
            "timestamp": "2025-01-01T10:00:04Z",
            "message": {"role": "assistant", "content": "Subagent reply"},
        },
    ]


def _claude_code_plain_payload(*, session_id: str) -> list[dict[str, object]]:
    return [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": session_id,
            "timestamp": "2025-01-01T10:00:00Z",
            "message": {"role": "user", "content": "hello"},
        },
    ]


def _codex_subagent_spawn_payload(*, child_id: str, parent_id: str) -> list[dict[str, object]]:
    """Real Codex shape for an edge-only subagent: the child's own
    session_meta carries `source.subagent` and `forked_from_id`, with no
    parent-side dispatch action observable at all (codex.py:766-770,
    840-846)."""

    return [
        {
            "type": "session_meta",
            "payload": {
                "id": child_id,
                "timestamp": "2025-01-02T10:00:00Z",
                "forked_from_id": parent_id,
                "source": {"subagent": {"thread_spawn": True}},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Working on it"}],
            },
        },
    ]


class TestDelegationIngestShapedFixtures:
    @pytest.mark.asyncio
    async def test_claude_code_task_dispatch_and_agent_child_resolve_through_real_parser(
        self, workspace_env: WorkspaceEnv, tmp_path: Path
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parent_path = _write_payload(
                tmp_path, "claude_code_parent.json", _claude_code_task_dispatch_payload(session_id="delegation-parent")
            )
            parent_parsed = _parse_single("claude-code", parent_path)
            parent_session_id = await ingest_session(parent_parsed, backend=repo.backend)
            assert parent_session_id == _archive_session_id(Provider.CLAUDE_CODE, "delegation-parent")

            # Real subagent filename shape: fallback_id (filename stem) must
            # start with "agent-" for code_parser.py to classify it as a
            # subagent transcript rather than a generic sidechain.
            child_path = _write_payload(
                tmp_path,
                "agent-delegation-child.json",
                _claude_code_agent_child_payload(parent_session_id="delegation-parent"),
            )
            child_parsed = _parse_single("claude-code", child_path)
            await ingest_session(child_parsed, backend=repo.backend)

        rows = _delegations_rows(db_path, parent_session_id=parent_session_id)
        assert len(rows) == 1
        row = rows[0]
        # The load-bearing direction assertion (polylogue-y964): parent_session_id
        # is the session that DISPATCHED, not the one dispatched to.
        assert row["parent_session_id"] == parent_session_id
        assert row["mapping_state"] == "resolved"
        assert row["child_session_id"] is not None
        assert row["instruction_tool_use_block_id"] is not None
        assert "Investigate X" in (row["instruction_payload"] or "")
        assert row["artifact_text"] == "Subagent finished."

    @pytest.mark.asyncio
    async def test_codex_subagent_spawn_resolves_edge_only_through_real_parser(
        self, workspace_env: WorkspaceEnv, tmp_path: Path
    ) -> None:
        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parent_path = _write_payload(
                tmp_path,
                "codex_parent.json",
                [
                    {"type": "session_meta", "payload": {"id": "codex-parent", "timestamp": "2025-01-01T10:00:00Z"}},
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "id": "p-msg-1",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Parent question"}],
                        },
                    },
                ],
            )
            parent_session_id = await ingest_session(_parse_single("codex", parent_path), backend=repo.backend)
            assert parent_session_id == _archive_session_id(Provider.CODEX, "codex-parent")

            child_path = _write_payload(
                tmp_path,
                "codex_subagent_child.json",
                _codex_subagent_spawn_payload(child_id="codex-subagent-child", parent_id="codex-parent"),
            )
            await ingest_session(_parse_single("codex", child_path), backend=repo.backend)

        rows = _delegations_rows(db_path, parent_session_id=parent_session_id)
        assert len(rows) == 1
        row = rows[0]
        assert row["mapping_state"] == "edge_only"
        assert row["child_session_id"] is not None
        # No parent-side dispatch action exists for this pair -- never fabricate one.
        assert row["instruction_tool_use_block_id"] is None
        assert row["instruction_payload"] is None

    @pytest.mark.asyncio
    async def test_claude_code_auto_compaction_child_excluded_from_delegations(
        self, workspace_env: WorkspaceEnv, tmp_path: Path
    ) -> None:
        """agent-acompact-*.jsonl classifies as BranchType.CONTINUATION, not
        SUBAGENT (code_parser.py's is_acompact check) -- it must never appear
        in `delegations`, real dispatch action or not."""

        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parent_path = _write_payload(
                tmp_path,
                "claude_code_acompact_parent.json",
                _claude_code_task_dispatch_payload(session_id="acompact-parent"),
            )
            parent_session_id = await ingest_session(_parse_single("claude-code", parent_path), backend=repo.backend)

            acompact_path = _write_payload(
                tmp_path,
                "agent-acompact-1.json",
                _claude_code_plain_payload(session_id="acompact-parent"),
            )
            acompact_parsed = _parse_single("claude-code", acompact_path)
            assert acompact_parsed.branch_type == "continuation"
            await ingest_session(acompact_parsed, backend=repo.backend)

        rows = _delegations_rows(db_path, parent_session_id=parent_session_id)
        # The real Task dispatch is still an unresolved attempt (its own row);
        # the auto-compaction child must NOT be attached to it as a resolved
        # subagent, and must not appear as any kind of delegation row itself.
        assert len(rows) == 1
        assert rows[0]["mapping_state"] == "unresolved"
        assert rows[0]["child_session_id"] is None

    @pytest.mark.asyncio
    async def test_codex_continuation_excluded_from_delegations(
        self, workspace_env: WorkspaceEnv, tmp_path: Path
    ) -> None:
        """A plain Codex continuation (`forked_from_id` with no `source.subagent`
        marker) is `link_type='continuation'`/unclassified branch_type, not a
        delegation, even though the topology link resolves cleanly."""

        db_path = db_setup(workspace_env)
        async with _make_repository(db_path) as repo:
            parent_path = _write_payload(
                tmp_path,
                "codex_continuation_parent.json",
                [
                    {
                        "type": "session_meta",
                        "payload": {"id": "codex-cont-parent", "timestamp": "2025-01-01T10:00:00Z"},
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "id": "p-msg-1",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Parent question"}],
                        },
                    },
                ],
            )
            parent_session_id = await ingest_session(_parse_single("codex", parent_path), backend=repo.backend)

            child_path = _write_payload(
                tmp_path,
                "codex_continuation_child.json",
                [
                    {
                        "type": "session_meta",
                        "payload": {"id": "codex-cont-child", "timestamp": "2025-01-02T10:00:00Z"},
                    },
                    {
                        "type": "session_meta",
                        "payload": {"id": "codex-cont-parent", "timestamp": "2025-01-01T10:00:00Z"},
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "id": "msg-1",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Continue from parent"}],
                        },
                    },
                ],
            )
            child_parsed = _parse_single("codex", child_path)
            await ingest_session(child_parsed, backend=repo.backend)

        rows = _delegations_rows(db_path, parent_session_id=parent_session_id)
        assert rows == []
