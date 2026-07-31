"""MCP ``query`` tool resolves ``from query:<hash>|result-set:<id>`` pipelines (polylogue-rxdo.6).

Before this, ``ReferenceQueryPipeline``/``RefOperand``/``parse_reference_query_pipeline``
and the real ``DurableRefResolver``/``ArchiveCanonicalPlanEvaluator`` planner
seam (both landed as production implementations in PR #2899) had zero callers
anywhere in ``polylogue/cli/*.py``, ``polylogue/mcp/*.py``, or
``polylogue/daemon/*.py`` -- the compatibility selector in
``archive/query/expression.py:compile_expression`` unconditionally hard-erred
on any ``from <ref>`` pipeline. These tests exercise the real production
route wired into the MCP ``query`` tool (``_resolve_reference_query_pipeline``
in ``polylogue/mcp/server_cutover.py``): a real archive, a real durable
``query:<hash>`` object, and the actual planner/resolver classes -- no test
double stands in for either.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.core.query_identity import JsonValue
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.query_objects import QueryObject, put_query
from tests.infra.mcp import MCPServerUnderTest, invoke_surface
from tests.unit.mcp.test_contract_evidence import _seeded_runtime_services


def _seed_archive(archive_root: Path) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    with ArchiveStore(archive_root) as archive:
        for provider, native_id, title in (
            (Provider.CODEX, "codex-1", "codex session"),
            (Provider.CLAUDE_CODE, "claude-1", "claude session"),
        ):
            archive.write_parsed(
                ParsedSession(
                    source_name=provider,
                    provider_session_id=native_id,
                    title=title,
                    created_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:01:00+00:00",
                    messages=[
                        ParsedMessage(
                            provider_message_id=f"{native_id}-m1",
                            role=Role.USER,
                            text="hello",
                            timestamp="2026-01-01T00:00:00+00:00",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello")],
                        )
                    ],
                )
            )
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    initialize_archive_database(archive_root / "ops.db", ArchiveTier.OPS)


def _origin_query(conn: sqlite3.Connection, *, origin: str) -> QueryObject:
    ast: dict[str, JsonValue] = {
        "kind": "field",
        "field": "origin",
        "op": "=",
        "values": [origin],
    }
    return put_query(conn, ast, grain="session", lane="dialogue", rank_policy="mixed", created_at_ms=1)


def test_mcp_query_resolves_from_query_reference(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        query = _origin_query(conn, origin="codex-session")
        conn.commit()

    with _seeded_runtime_services(archive_root):
        result = invoke_surface(
            mcp_server._tool_manager._tools["query"].fn,
            expression=f"from query:{query.query_hash}",
        )

    body = json.loads(result)
    assert body["source"] == f"query:{query.query_hash}"
    assert body["grain"] == "session"
    assert body["member_count"] == 1
    assert len(body["members"]) == 1
    assert body["members"][0].startswith("session:codex-session:")
    assert body["truncated"] is False


def test_mcp_query_from_unknown_query_hash_returns_typed_not_found(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)

    with _seeded_runtime_services(archive_root):
        result = invoke_surface(
            mcp_server._tool_manager._tools["query"].fn,
            expression="from query:0000000000000000000000000000000000000000000000000000000000000000",
        )

    body = json.loads(result)
    assert body.get("code") == "not_found", body


def test_mcp_query_from_reference_with_stages_returns_typed_not_implemented(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """Stage composition after the root operand is honestly unimplemented, not silently dropped or crashed."""
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        query = _origin_query(conn, origin="codex-session")
        conn.commit()

    with _seeded_runtime_services(archive_root):
        result = invoke_surface(
            mcp_server._tool_manager._tools["query"].fn,
            expression=f"from query:{query.query_hash} | count",
        )

    body = json.loads(result)
    assert body.get("code") == "not_implemented", body
