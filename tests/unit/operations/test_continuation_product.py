"""Continuation routes share the resident pin and original product owners."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.query.execution_control import QueryCancelledError
from polylogue.operations.continuation import execute_continuation_read
from polylogue.operations.daemon_protocol import validate_operation_result
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.derived.session.rebuild import rebuild_archive_session_insights
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.storage_records import SessionBuilder


def _seed(root: Path) -> None:
    for native, cwd in (("resume-a", "/neutral/repo"), ("resume-b", "/neutral/other")):
        (
            SessionBuilder(root / "index.db", native)
            .provider("codex")
            .title(native)
            .working_directories([cwd])
            .add_message("user", role="user", text=f"Continue {native}")
            .save()
        )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        rebuild_archive_session_insights(archive)


def test_real_uds_continuation_keeps_route_and_ranked_context_options(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed) as stack:
        session_id = native_session_id_for("codex", "resume-a")
        envelope = stack.client.operation_to_completion(
            "continuation.route", {"session_id": session_id}, archive_root=str(stack.archive_root)
        )
        assert envelope and envelope["outcome"] == "completed", envelope
        route = cast("dict[str, Any]", envelope["result"])
        assert route["argv"] == ["codex", "resume", "ext-resume-a"]
        assert route["cwd"] == "/neutral/repo"
        assert route["command"] == "cd /neutral/repo && codex resume ext-resume-a"
        for limit, expected in ((0, 0), (1, 1), (10, 1)):
            candidates = stack.client.operation_to_completion(
                "continuation.candidates",
                {
                    "repo_path": "/neutral/repo",
                    "cwd": "/neutral/repo",
                    "recent_files": ["/neutral/repo/a.py"],
                    "limit": limit,
                },
                archive_root=str(stack.archive_root),
            )
            assert candidates and candidates["outcome"] == "completed", candidates
            window = cast("dict[str, Any]", candidates["result"])
            assert window["returned"] == expected
            assert window["limit"] == limit
            assert "total" not in window
            if expected:
                assert window["candidates"][0]["logical_session_id"] == session_id
                assert window["candidates"][0]["score_breakdown"]["cwd_match"] == 1.0


def test_continuation_route_reads_metadata_only_and_context_uses_original_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed) as stack:
        session_id = native_session_id_for("codex", "resume-a")
        with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as pinned:
            original = pinned.archive.read_session

            def no_transcript(_session_id: str) -> object:
                raise AssertionError("resume routing must not read the transcript")

            monkeypatch.setattr(pinned.archive, "read_session", no_transcript)
            route = execute_read_operation(
                "continuation.route", {"session_id": session_id}, archive=pinned.archive, serving_identity="test"
            )
            assert route["status"] == "supported"
            monkeypatch.setattr(pinned.archive, "read_session", original)
            with sqlite3.connect(stack.archive_root / "index.db") as writer:
                writer.execute("UPDATE sessions SET title='later-title' WHERE session_id=?", (session_id,))
            assert pinned.archive.read_summary(session_id).title == "resume-a"
            ops_before = hashlib.sha256((stack.archive_root / "ops.db").read_bytes()).digest()
            result = execute_read_operation(
                "continuation.context",
                {"session_id": session_id, "observed_at_ms": 1000},
                archive=pinned.archive,
                serving_identity="test",
            )
            image = cast("dict[str, Any]", result["payload"])
            assert image["spec"]["purpose"] == "continue"
            assert any(
                "Continue resume-a" in segment["markdown"]
                for segment in image["segments"]
                if segment["payload_kind"] == "messages"
            )
            validate_operation_result("continuation.context", result)
            assert hashlib.sha256((stack.archive_root / "ops.db").read_bytes()).digest() == ops_before
        with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as fresh:
            assert fresh.archive.read_summary(session_id).title == "later-title"


def test_continuation_cancellation_reaches_ranked_population_and_closes_reader(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed) as stack:
        calls = 0

        def cancel() -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise QueryCancelledError("cancel ranking")

        with pytest.raises(QueryCancelledError):
            with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as pinned:
                connection = pinned.archive.index_connection
                execute_continuation_read(
                    "continuation.candidates",
                    {"repo_path": "/neutral/repo"},
                    archive=pinned.archive,
                    serving_identity="test",
                    checkpoint=cancel,
                )
        assert calls == 3
        assert connection is not None
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_bound_continuation_refuses_changed_selection_frame(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed) as stack:
        session_id = native_session_id_for("codex", "resume-a")
        envelope = stack.client.operation_to_completion(
            "cli.query", {"params": {"session_id": session_id}}, archive_root=str(stack.archive_root)
        )
        assert envelope and envelope["outcome"] == "completed", envelope
        epoch = cast("dict[str, Any]", envelope["result"])["snapshot_epoch"]
        with sqlite3.connect(stack.archive_root / "user.db") as writer:
            writer.execute(
                "INSERT INTO assertions(assertion_id,target_ref,key,kind,value_json,created_at_ms,updated_at_ms) "
                "VALUES (?,?,?,?,?,?,?)",
                ("continuation-pin", f"session:{session_id}", "neutral", "tag", "{}", 1000, 1000),
            )
        stale = stack.client.operation_to_completion(
            "continuation.route",
            {"session_id": session_id, "selection_epoch": epoch},
            archive_root=str(stack.archive_root),
        )
        assert stale and stale["outcome"] == "rejected", stale
        assert cast("dict[str, Any]", stale["error"])["code"] == "query_continuation_stale"
