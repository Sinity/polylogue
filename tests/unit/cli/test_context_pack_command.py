"""Tests for the ``polylogue context-pack`` command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from click.testing import CliRunner

from polylogue.archive.message.roles import Role
from polylogue.cli.commands.context_pack import context_pack_command
from polylogue.cli.shared.types import AppEnv
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.types import ContentBlockType, Provider


def _archive_env(archive_root: Path) -> AppEnv:
    services: Any = MagicMock()
    services.get_config.return_value = SimpleNamespace(
        archive_root=archive_root,
        db_path=archive_root / "index.db",
    )
    services.get_repository.side_effect = AssertionError("context pack must not open the unsupported repository path")
    return AppEnv(services=services)


def test_context_pack_reads_archive_from_archive_tiers(tmp_path: Path) -> None:
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
                        content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text="hello archive pack")],
                    )
                ],
            )
        )

    result = CliRunner().invoke(
        context_pack_command,
        ["--query", "hello", "--max-sessions", "1", "--max-messages", "1"],
        obj=_archive_env(archive_root),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_sessions"] == 1
    assert payload["total_messages"] == 1
    assert payload["provenance"]["archive_runtime"] == "archive_file_set"
    assert payload["provenance"]["archive_root"] == str(archive_root)
    assert payload["provenance"]["active_db_path"] == str(archive_root / "index.db")
    assert payload["query_context"]["query"] == "hello"
    assert payload["query_context"]["sessions_included"] == 1
    session = payload["sessions"][0]
    assert session["session_id"] == "codex-session:context-pack-v1"
    assert session["origin"] == "codex-session"
    assert "provider" not in session
    assert "source" not in session
    assert session["messages"][0]["role"] == "user"
    assert session["messages"][0]["text"] == "hello archive pack"
