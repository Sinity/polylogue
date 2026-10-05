"""Working-directory completion uses the original resident Index snapshot."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.query.execution_control import QueryCancelledError
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from tests.infra.daemon_operations import running_daemon_operations


def _seed_directories(root: Path) -> None:
    # Exact read-model fixture rows. These claim no acquired Source evidence.
    with sqlite3.connect(root / "index.db") as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for native_id, paths in (
            (
                "cwd-a",
                ("/neutral/repo", "/neutral/repo/child", "/neutral/repo_%", "/neutral/my folder", "/neutral/myths"),
            ),
            ("cwd-b", ("/neutral/repo", "/neutral/repository")),
            ("cwd-c", ("C:\\neutral\\project",)),
        ):
            connection.execute(
                "INSERT INTO sessions(native_id, origin, content_hash) VALUES (?, 'claude-code-session', ?)",
                (native_id, bytes(32)),
            )
            connection.executemany(
                "INSERT INTO session_working_dirs(session_id, path, position) VALUES (?, ?, ?)",
                [(f"claude-code-session:{native_id}", path, number) for number, path in enumerate(paths)],
            )


def test_resident_cwd_completion_filters_distinct_paths_before_the_window(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_directories) as stack:
        for prefix, limit, expected in (
            ("/neutral/rep", 1, [("/neutral/repo", "2 sessions")]),
            ("/neutral/repo_", 3, [("/neutral/repo_%", "1 sessions")]),
            ("/neutral/repo/", 3, [("/neutral/repo/child", "1 sessions")]),
            ("C:\\neutral\\", 3, [("C:/neutral/project", "1 sessions")]),
            ("/neutral/my ", 3, [("/neutral/my folder", "1 sessions")]),
            ("/absent", 3, []),
        ):
            envelope = stack.client.operation_to_completion(
                "completion",
                {"source": "cwd_prefix", "incomplete": prefix, "limit": limit},
                archive_root=str(stack.archive_root),
            )
            assert envelope is not None and envelope["outcome"] == "completed", envelope
            result = cast("dict[str, Any]", envelope["result"])["value_completions"]
            assert result["source"] == "cwd_prefix"
            assert [(row["value"], row["help"]) for row in result["values"]] == expected


def test_cwd_completion_keeps_the_original_pin_after_directory_changes(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_directories) as stack:
        with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as pinned:
            original = pinned.archive.list_working_directory_completions("/neutral/repo", limit=10)

            def remove_directories() -> None:
                with sqlite3.connect(stack.archive_root / "index.db") as connection:
                    connection.execute("DELETE FROM session_working_dirs")

            stack.write_bridge.run_sync("fixture.remove-cwd", remove_directories)
            assert pinned.archive.list_working_directory_completions("/neutral/repo", limit=10) == original
        with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as current:
            assert current.archive.list_working_directory_completions("/neutral/repo", limit=10) == []


def test_cwd_completion_checks_cancellation_and_physically_closes_reader(tmp_path: Path) -> None:
    cancellation = QueryCancelledError("neutral-cwd-cancellation")
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_directories) as stack:
        with pytest.raises(QueryCancelledError) as caught:
            with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as pinned:
                connection = pinned.archive.index_connection
                checks = 0

                def checkpoint() -> None:
                    nonlocal checks
                    checks += 1
                    if checks == 2:
                        raise cancellation

                pinned.archive.set_read_progress_guard(lambda: 0, n_opcodes=1000, check_cancelled=checkpoint)
                execute_read_operation(
                    "completion",
                    {"source": "cwd_prefix", "limit": 10},
                    archive=pinned.archive,
                    serving_identity="daemon",
                )
        assert caught.value is cancellation and checks == 2
        assert connection is not None
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
