"""Production-route evidence for durable MCP call telemetry (polylogue-7s57)."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.mcp.call_log import flush_mcp_call_log
from polylogue.mcp.server import build_server
from polylogue.mcp.server_support import _set_runtime_services
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    MCP_CALL_LOG_RETENTION_MS,
    ArchiveMcpCallLogEntry,
    list_mcp_calls,
    record_mcp_call,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.mcp import MCPServerUnderTest, invoke_surface


@contextmanager
def _running_daemon() -> Iterator[str]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    thread = threading.Thread(target=server.serve_forever, name="mcp-call-log-test-daemon", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _read_calls(archive_root: Path, **filters: object) -> tuple[ArchiveMcpCallLogEntry, ...]:
    conn = open_readonly_connection(archive_root / "ops.db")
    try:
        return list_mcp_calls(conn, **filters)  # type: ignore[arg-type]
    finally:
        conn.close()


def test_registered_tools_persist_success_and_typed_failure_through_daemon(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise FastMCP registration, background HTTP, writer gate, and ops SQL."""
    with _running_daemon() as daemon_url:
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", daemon_url)
        _set_runtime_services(None)
        try:
            server = cast(MCPServerUnderTest, build_server(role="read"))

            stats = invoke_surface(server._tool_manager._tools["stats"].fn)
            missing = invoke_surface(
                server._tool_manager._tools["get_session_summary"].fn,
                id="codex-session:missing",
            )
            assert '"total_sessions"' in stats
            assert '"error": "not_found"' in missing
            assert flush_mcp_call_log(timeout=5.0)
        finally:
            _set_runtime_services(None)

    calls = _read_calls(workspace_env["archive_root"])
    by_tool = {entry.tool_name: entry for entry in calls}
    assert {"stats", "get_session_summary"} <= set(by_tool), tuple(by_tool)
    assert by_tool["stats"].success is True
    assert by_tool["stats"].error_detail is None
    assert by_tool["get_session_summary"].success is False
    assert by_tool["get_session_summary"].error_detail == "not_found"
    session_calls = _read_calls(workspace_env["archive_root"], session_id="codex-session:missing")
    assert [entry.tool_name for entry in session_calls] == ["get_session_summary"]


def test_existing_same_version_ops_database_receives_additive_call_log(tmp_path: Path) -> None:
    ops_db = tmp_path / "ops.db"
    with sqlite3.connect(ops_db) as conn:
        conn.execute("PRAGMA user_version = 1")
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "mcp_call_log" in tables


def test_record_mcp_call_prunes_expired_rows_in_same_transaction(tmp_path: Path) -> None:
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        record_mcp_call(
            conn,
            call_id="expired",
            tool_name="old_tool",
            started_at_ms=1,
            finished_at_ms=2,
            success=True,
        )
        current_start = MCP_CALL_LOG_RETENTION_MS + 10
        record_mcp_call(
            conn,
            call_id="current",
            tool_name="new_tool",
            started_at_ms=current_start,
            finished_at_ms=current_start + 1,
            success=True,
        )
        remaining = list_mcp_calls(conn)
    assert [entry.call_id for entry in remaining] == ["current"]


def test_call_log_enqueue_never_changes_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.mcp import call_log
    from polylogue.mcp.server_support import _safe_call

    def reject(_delivery: object) -> None:
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(call_log._DISPATCHER, "submit", reject)
    assert _safe_call("stats", lambda: '{"ok": true}') == '{"ok": true}'
