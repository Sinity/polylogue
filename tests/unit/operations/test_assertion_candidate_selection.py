"""Assertion target selection reads identities without unused summary metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.operations.mutation_actuators import resolve_assertion_candidate_refs
from polylogue.operations.operation_context import open_operation_read
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import write_index_session


def test_annotation_last_selects_identity_without_summary_metadata(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    repository = tmp_path / "repository"
    nested = repository / "nested"
    nested.mkdir(parents=True)
    (repository / ".git").mkdir()
    bootstrap_archive_root(root)
    selected_id = ""
    with ArchiveStore(root) as writer:
        for native_id, directory, date in (
            ("older", repository / "older", "2026-01-01T00:00:00+00:00"),
            ("selected", nested, "2026-01-02T00:00:00+00:00"),
            ("outside", tmp_path / "outside", "2026-01-03T00:00:00+00:00"),
        ):
            session_id = write_index_session(
                writer,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=native_id,
                    created_at=date,
                    updated_at=date,
                    working_directories=[str(directory / f"dir-{index}") for index in range(101)],
                    messages=[ParsedMessage(provider_message_id="one", role=Role.USER, text="neutral prose")],
                ),
            )
            writer.add_user_tags((session_id,), tuple(f"tag-{index}" for index in range(101)))
            if native_id == "selected":
                selected_id = session_id
    assert selected_id
    with open_operation_read(root) as pinned:
        connection = pinned.archive.index_connection
        assert connection is not None
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        assert resolve_assertion_candidate_refs(pinned.archive, ["last"], cwd=nested) == [f"session:{selected_id}"]
        assert not any("json_group_array" in sql.lower() for sql in statements)
        assert any("SELECT s.session_id,s.origin" in sql for sql in statements)


def test_annotation_last_retains_empty_scope_refusal(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    with open_operation_read(root) as pinned:
        with pytest.raises(ValueError, match="found no archived session"):
            resolve_assertion_candidate_refs(pinned.archive, ["last"], cwd=tmp_path)
