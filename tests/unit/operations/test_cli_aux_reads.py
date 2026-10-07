"""Resident execution for small CLI reads outside the query grammar."""

from __future__ import annotations

from pathlib import Path

from polylogue.operations.daemon_protocol import validate_operation_result
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder


def test_identity_reset_targets_resolve_on_the_pinned_archive(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "reset-targets").provider("codex").title("Target").save()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        session_id = archive.list_summaries(limit=1)[0].session_id
        result = execute_read_operation(
            "session.identity-reset.targets",
            {"session": session_id},
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("session.identity-reset.targets", result)
    assert result["session_ids"] == [session_id]


def test_assertion_export_keeps_a_missing_user_tier_empty(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    SessionBuilder(root / "index.db", "assertion-export").provider("codex").title("Export").save()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        result = execute_read_operation(
            "user.assertions.export",
            {},
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("user.assertions.export", result)
    assert result["items"] == []
    assert result["total"] == 0
    outcome = result["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["state"] == "empty"
