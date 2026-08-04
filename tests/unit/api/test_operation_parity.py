"""Semantic-operation parity tests for the documented Python facade."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from devtools.render_api_operation_parity import (
    build_parity_payload,
    render_library_api_section,
    validate_library_api_section,
)
from polylogue.api import Polylogue
from polylogue.api.operation_parity import API_OPERATIONS, ApiOperation, declared_python_bindings, validate_live_facade
from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

ROUTE_PROOF_BINDINGS = {
    "lifecycle": "Polylogue.open",
    "source-index-write": "Polylogue.parse_file",
    "source-read": "Polylogue.get_raw_artifacts_for_session",
    "index-read": "Polylogue.stats",
    "index-write": "Polylogue.rebuild_index",
    "user-read": "Polylogue.list_tags",
    "user-write": "Polylogue.add_tag",
    "cross-tier": "Polylogue.compile_and_record_context",
    "embedding-status": "Polylogue.embedding_status",
    "embedding-preflight": "Polylogue.embedding_preflight",
    "embedding-read": "Polylogue.search_similar_sessions",
}


def _archive(root: Path) -> Polylogue:
    with ArchiveStore(root):
        pass
    return Polylogue(archive_root=root, db_path=root / "index.db")


def _operation_for(binding: str) -> ApiOperation:
    declaration = declared_python_bindings()[binding]
    assert isinstance(declaration, ApiOperation)
    return declaration


def _assert_route(operation: ApiOperation, *, expected: str) -> None:
    """Keep route proofs tied to declaration authority, not method spelling."""

    assert operation.route_class == expected


def test_every_live_public_callable_has_semantic_operation_or_exclusion() -> None:
    validate_live_facade()
    operation_ids = [operation.operation_id for operation in API_OPERATIONS]
    assert len(operation_ids) == len(set(operation_ids))


def test_every_declared_route_class_has_a_real_route_proof_binding() -> None:
    assert {operation.route_class for operation in API_OPERATIONS} == set(ROUTE_PROOF_BINDINGS)
    for route_class, binding in ROUTE_PROOF_BINDINGS.items():
        _assert_route(_operation_for(binding), expected=route_class)


def test_rendered_matrix_preserves_stable_operation_bindings() -> None:
    payload = cast(dict[str, Any], build_parity_payload())
    operations = cast(list[dict[str, Any]], payload["operations"])
    assert payload["operation_count"] == len(API_OPERATIONS)
    assert {row["operation_id"] for row in operations} == {row.operation_id for row in API_OPERATIONS}
    embedding = next(row for row in operations if row["operation_id"] == "api.embedding.preflight")
    assert embedding["python"][0]["binding"] == "Polylogue.embedding_preflight"
    assert embedding["route_class"] == "embedding-preflight"


def test_generated_library_api_section_is_signature_asyncness_and_section_aware() -> None:
    section = render_library_api_section()
    assert "#### `api.assertion.review`" in section
    assert "`Polylogue.import_annotation_batch`" in section
    assert "`async (self, request:" in section
    validate_library_api_section(section)
    with pytest.raises(ValueError, match="signatures"):
        validate_library_api_section(section.replace("`async ", "`", 1))


@pytest.mark.asyncio
async def test_real_route_lifecycle_constructs_and_closes_temporary_archive(tmp_path: Path) -> None:
    operation = _operation_for("Polylogue.open")
    _assert_route(operation, expected="lifecycle")
    archive = Polylogue.open(archive_root=tmp_path, db_path=tmp_path / "index.db")
    assert archive.archive_root == tmp_path.resolve()
    await archive.close()


@pytest.mark.asyncio
async def test_real_route_parse_file_writes_declared_source_and_index_tiers(tmp_path: Path) -> None:
    operation = _operation_for("Polylogue.parse_file")
    _assert_route(operation, expected="source-index-write")
    archive = _archive(tmp_path)
    payload = {
        "sessionId": "api-parity-parse",
        "projectHash": "api-parity",
        "startTime": "2026-08-04T00:00:00.000Z",
        "lastUpdated": "2026-08-04T00:01:00.000Z",
        "kind": "chat",
        "summary": "API parity parse",
        "messages": [
            {"id": "u1", "timestamp": "2026-08-04T00:00:01.000Z", "type": "user", "content": ["parity needle"]},
            {"id": "a1", "timestamp": "2026-08-04T00:00:02.000Z", "type": "gemini", "content": "done"},
        ],
    }
    source = tmp_path / "session.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    try:
        result = await archive.parse_file(source, source_name=Provider.GEMINI_CLI.value)
        assert result.changed_counts["sessions"] == 1
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (1,)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_real_route_reads_source_evidence_from_temporary_archive(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    operation = _operation_for("Polylogue.get_raw_artifacts_for_session")
    _assert_route(operation, expected="source-read")
    source = tmp_path / "source-evidence.json"
    source.write_text(
        json.dumps(
            {
                "sessionId": "source-read",
                "projectHash": "api-parity",
                "startTime": "2026-08-04T00:00:00.000Z",
                "lastUpdated": "2026-08-04T00:01:00.000Z",
                "kind": "chat",
                "summary": "source evidence",
                "messages": [
                    {"id": "m1", "timestamp": "2026-08-04T00:00:01.000Z", "type": "user", "content": ["evidence"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        await archive.parse_file(source, source_name=Provider.GEMINI_CLI.value)
        session_id = "gemini-cli-session:source-read"
        rows, total = await archive.get_raw_artifacts_for_session(session_id)
        assert total == 1
        assert rows[0]["source_path"] == str(source)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_real_route_reads_index_state_from_temporary_archive(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    operation = _operation_for("Polylogue.stats")
    _assert_route(operation, expected="index-read")
    try:
        with ArchiveStore(tmp_path) as store:
            store.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="index-read",
                    title="index state",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="index")],
                        )
                    ],
                )
            )
        assert (await archive.stats()).session_count == 1
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_real_route_rebuilds_index_through_declared_maintenance_class(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    operation = _operation_for("Polylogue.rebuild_index")
    _assert_route(operation, expected="index-write")
    try:
        assert await archive.rebuild_index() is True
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_real_route_writes_and_reads_declared_user_tier(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    write_operation = _operation_for("Polylogue.add_tag")
    read_operation = _operation_for("Polylogue.list_tags")
    _assert_route(write_operation, expected="user-write")
    _assert_route(read_operation, expected="user-read")
    try:
        with ArchiveStore(tmp_path) as store:
            session_id = store.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="user-write",
                    title="user state",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="user")],
                        )
                    ],
                )
            )
        assert (await archive.add_tag(session_id, "parity")).outcome == "added"
        assert await archive.list_tags() == {"parity": 1}
        with sqlite3.connect(tmp_path / "user.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'tag'").fetchone() == (1,)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_real_route_compiles_and_records_declared_cross_tier_context(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    operation = _operation_for("Polylogue.compile_and_record_context")
    _assert_route(operation, expected="cross-tier")
    try:
        with ArchiveStore(tmp_path) as store:
            store.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="cross-tier-context",
                    title="cross-tier context",
                    created_at="2026-08-04T00:00:00+00:00",
                    updated_at="2026-08-04T00:01:00+00:00",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="cross-tier archival evidence")],
                        )
                    ],
                )
            )
        receipt = await archive.compile_and_record_context(
            recipient_ref="agent:api-parity",
            delivered_by_ref="user:local",
            boundary="api-parity",
            query="cross-tier archival",
            max_sessions=1,
        )
        assert receipt.outcome == "recorded"
        assert await archive.get_context_delivery(receipt.snapshot_ref, recipient_ref="agent:api-parity") is not None
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_real_route_embedding_status_and_preflight_are_no_spend_temporary_archive_routes(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    status_operation = _operation_for("Polylogue.embedding_status")
    preflight_operation = _operation_for("Polylogue.embedding_preflight")
    _assert_route(status_operation, expected="embedding-status")
    _assert_route(preflight_operation, expected="embedding-preflight")
    try:
        status = archive.embedding_status()
        preflight = archive.embedding_preflight(max_sessions=1)
        assert status["status"] in {"empty", "none", "disabled", "ready", "stale", "pending"}
        assert "pending_sessions" in preflight
        assert (tmp_path / "embeddings.db").exists()
        assert status["retrieval_ready"] is False
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_real_route_embedding_search_is_declared_as_embedding_read(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    operation = _operation_for("Polylogue.search_similar_sessions")
    _assert_route(operation, expected="embedding-read")
    try:
        with pytest.raises(ValueError, match="No vector provider configured"):
            await archive.search_similar_sessions("missing-session")
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_route_proof_rejects_a_deliberately_misrouted_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path)
    operation = _operation_for("Polylogue.add_tag")
    _assert_route(operation, expected="user-write")

    async def misrouted_add_tag(self: Polylogue, session_id: str, tag: str, **kwargs: object) -> object:
        del tag, kwargs
        return await self.get_metadata(session_id)

    monkeypatch.setattr(Polylogue, "add_tag", misrouted_add_tag)
    try:
        with ArchiveStore(tmp_path) as store:
            session_id = store.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="misrouted-user-write",
                    title="misrouted user state",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="user")],
                        )
                    ],
                )
            )
        await archive.add_tag(session_id, "should-not-write")
        with sqlite3.connect(tmp_path / "user.db") as conn:
            with pytest.raises(AssertionError):
                assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'tag'").fetchone() == (1,)
    finally:
        await archive.close()
