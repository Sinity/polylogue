"""Tests for the multi-session ``read --view context-pack`` capability.

The standalone ``context-pack`` command was absorbed into the read-view surface
(#1842): ``read --view context-pack``. The pack logic lives in
``polylogue.context.pack.run_context_pack_view``; the MCP
``build_context_pack`` tool exposes the same capability programmatically.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from polylogue.archive.message.roles import Role
from polylogue.cli.shared.types import AppEnv
from polylogue.context.pack import run_context_pack_view
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion


def _archive_env(archive_root: Path) -> AppEnv:
    services: Any = MagicMock()
    services.get_config.return_value = SimpleNamespace(
        archive_root=archive_root,
        db_path=archive_root / "index.db",
    )
    services.get_repository.side_effect = AssertionError("context pack must not open the unsupported repository path")
    return AppEnv(services=services)


def test_context_pack_view_reads_archive_from_archive_tiers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="context-pack-v1",
                title="Archive context pack",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="hello archive pack",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello archive pack")],
                    )
                ],
            )
        )

    project_path = "/realm/project/polylogue"
    run_context_pack_view(
        _archive_env(archive_root),
        query="hello archive",
        project_path=project_path,
        max_sessions=1,
        max_messages=1,
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["total_sessions"] == 1
    assert payload["total_messages"] == 1
    assert payload["selection_strategy"] == "relaxed_project_term_recall"
    assert payload["scope"]["read_views"] == ["context-pack"]
    assert payload["scope"]["project_path"] == "<redacted-path>/polylogue"
    assert payload["evidence_refs"] == ["codex-session:context-pack-v1"]
    assert payload["redaction_policy"] == "public_refs_and_redacted_paths"
    assert payload["token_estimate"] > 0
    assert payload["size_estimate"]["json_bytes"] > 0
    assert payload["size_estimate"]["message_text_bytes"] > 0
    assert any(omission["reason"] == "redacted" for omission in payload["omissions"])
    assert payload["provenance"]["archive_runtime"] == "archive_file_set"
    assert payload["provenance"]["redacted"] is True
    assert payload["provenance"]["archive_root"] is None
    assert payload["provenance"]["active_db_path"] is None
    assert payload["provenance"]["redaction_policy"] == "public_refs_and_redacted_paths"
    assert payload["query_context"]["query"] == "hello archive"
    assert payload["query_context"]["sessions_included"] == 1
    session = payload["sessions"][0]
    assert session["session_id"] == "codex-session:context-pack-v1"
    assert session["origin"] == "codex-session"
    assert "provider" not in session
    assert "source" not in session
    assert session["messages"][0]["role"] == "user"
    assert session["messages"][0]["text"] == "hello archive pack"

    run_context_pack_view(
        _archive_env(archive_root),
        query="hello archive",
        project_path=project_path,
        max_sessions=1,
        max_messages=1,
        no_redact=True,
    )
    raw_payload = json.loads(capsys.readouterr().out)
    assert raw_payload["redaction_policy"] == "raw_paths_explicit_opt_in"
    assert raw_payload["provenance"]["redacted"] is False
    assert raw_payload["scope"]["project_path"] == project_path
    assert raw_payload["provenance"]["archive_root"] == str(archive_root)
    assert raw_payload["provenance"]["active_db_path"] == str(archive_root / "index.db")


def test_context_pack_view_reads_injectable_assertions_from_user_tier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="context-pack-assertions",
                title="Archive context pack assertions",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="assertion context pack",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="assertion context pack")],
                    )
                ],
            )
        )
        with sqlite3.connect(archive.user_db_path) as conn:
            upsert_assertion(
                conn,
                assertion_id="inject-decision",
                target_ref="session:codex-session:context-pack-assertions",
                scope_ref="repo:polylogue",
                kind=AssertionKind.DECISION,
                body_text="Use the shared assertion facade in context surfaces.",
                status="active",
                context_policy={"inject": True},
                now_ms=1_700_000_000_000,
            )
            upsert_assertion(
                conn,
                assertion_id="private-caveat",
                target_ref="session:codex-session:context-pack-assertions",
                scope_ref="repo:polylogue",
                kind=AssertionKind.CAVEAT,
                body_text="This private caveat should stay out of context.",
                status="active",
                context_policy={"inject": False},
                now_ms=1_700_000_000_100,
            )
            conn.commit()

    run_context_pack_view(_archive_env(archive_root), query="assertion", max_sessions=1, max_messages=1)

    payload = json.loads(capsys.readouterr().out)
    assert payload["decisions"]["items"] == [
        "decision: Use the shared assertion facade in context surfaces. [session:codex-session:context-pack-assertions]"
    ]
