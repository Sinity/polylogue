"""MCP durable writes obey the shared embedded writer boundary."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.mcp.declarations.models import MCPCapabilities
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
from tests.infra.live_ingest import write_index_session
from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async


def _user_archive(root: Path) -> None:
    root.mkdir()
    initialize_archive_database(root / "user.db", ArchiveTier.USER)


def _seed_candidate(root: Path, assertion_id: str) -> None:
    with sqlite3.connect(root / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id=assertion_id,
            target_ref="session:embedded-boundary",
            kind=AssertionKind.LESSON,
            body_text="A synthetic candidate for the embedded MCP writer boundary.",
            author_ref="agent:test",
            author_kind="agent",
            evidence_refs=["session:embedded-boundary"],
            status="candidate",
            now_ms=1_700_000_000_000,
        )


def _annotation_import_fields(session_id: str, batch_id: str) -> dict[str, object]:
    return {
        "jsonl": json.dumps(
            {
                "row_key": "writer-boundary-row",
                "value": {"activity": "research", "confidence": 0.9},
                "evidence_refs": [session_id],
            }
        ),
        "batch_id": batch_id,
        "schema_id": "seed.activity",
        "schema_version": 1,
        "target_ref": f"session:{session_id}",
        "source_result_ref": f"result-set:{batch_id}",
        "actor_ref": "agent:writer-boundary-test",
        "model_ref": "agent:model",
        "prompt_ref": "block:prompt:0",
    }


@pytest.mark.asyncio
async def test_mcp_import_without_daemon_refuses_before_user_commit(tmp_path: Path) -> None:
    """The MCP surface cannot fall back to its own writable user tier."""
    from polylogue.mcp.server import build_server

    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="annotation-writer-boundary",
                title="Writer boundary fixture",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
            ),
        )
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
    write_fn = server._tool_manager._tools["write"].fn
    with installed_runtime_services(archive_root):
        result = json.loads(
            await invoke_surface_async(
                write_fn,
                operation="import_annotation_batch",
                fields=_annotation_import_fields(session_id, "writer-boundary-batch"),
            )
        )
    assert result.get("is_error") is True, result
    assert result.get("detail") == "FacadeDaemonRequiredError", result
    with sqlite3.connect(archive_root / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE key = 'writer-boundary-row'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_mcp_import_roundtrips_through_real_daemon_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted MCP import reaches the daemon writer and returns its receipt."""
    from polylogue.daemon.socket_path import daemon_socket_path
    from polylogue.mcp.server import build_server
    from tests.infra.daemon_operations import running_daemon_operations

    archive_root = tmp_path / "archive"
    session_ids: list[str] = []

    def seed(root: Path) -> None:
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            session_ids.append(
                write_index_session(
                    archive,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id="annotation-wire-roundtrip",
                        title="Wire roundtrip fixture",
                        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
                    ),
                )
            )

    monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_args, **_kwargs: None)
    with running_daemon_operations(archive_root, seed_archive=seed, socket_path=daemon_socket_path(archive_root)):
        server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(write=True)))
        write_fn = server._tool_manager._tools["write"].fn
        with installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="import_annotation_batch",
                    fields=_annotation_import_fields(session_ids[0], "writer-boundary-wire"),
                )
            )
        assert result.get("is_error") is not True, result
        with sqlite3.connect(archive_root / "user.db") as conn:
            assert conn.execute("SELECT status FROM assertions WHERE key = 'writer-boundary-row'").fetchone() == (
                "candidate",
            )


@pytest.mark.asyncio
async def test_judge_only_mcp_without_daemon_refuses_before_user_write(tmp_path: Path) -> None:
    """Judgment writes use the same daemon-required public route."""
    from polylogue.mcp.server import build_server

    archive_root = tmp_path / "archive"
    _user_archive(archive_root)
    _seed_candidate(archive_root, "offline-boundary-candidate")
    server = cast(MCPServerUnderTest, build_server(capabilities=MCPCapabilities(judge=True)))
    judge_fn = server._tool_manager._tools["judge"].fn
    with installed_runtime_services(archive_root):
        result = json.loads(
            await invoke_surface_async(
                judge_fn,
                candidate_ref="assertion:offline-boundary-candidate",
                decision="accept",
            )
        )
    assert result.get("is_error") is True, result
    assert result.get("detail") == "FacadeDaemonRequiredError", result
    with sqlite3.connect(archive_root / "user.db") as conn:
        assert conn.execute(
            "SELECT status FROM assertions WHERE assertion_id = ?", ("offline-boundary-candidate",)
        ).fetchone() == ("candidate",)


def test_mcp_capability_inventory_is_declaration_derived() -> None:
    """Every capability whose handlers can write is enumerated with its tools.

    ``write`` routes through facade mutations, ``judge`` reaches the direct
    user-tier judgment route, and ``maintenance`` dispatches daemon-owned
    maintenance operations. Keeping the declared tool mapping here makes a
    newly added capability or privileged handler fail until its route is
    audited against the shared ownership boundary.
    """

    from polylogue.mcp.declarations.registry import MCP_TOOL_DECLARATIONS

    declared_capabilities = {
        declaration.required_capability
        for declaration in MCP_TOOL_DECLARATIONS
        if declaration.required_capability is not None
    }
    assert declared_capabilities == {"write", "judge", "maintenance"}
    tools_by_capability = {
        capability: {
            declaration.name for declaration in MCP_TOOL_DECLARATIONS if declaration.required_capability == capability
        }
        for capability in ("write", "judge", "maintenance")
    }
    assert tools_by_capability == {
        "write": {"write", "record_work_event", "emit_decision", "run"},
        "judge": {"judge"},
        "maintenance": {"maintenance"},
    }
