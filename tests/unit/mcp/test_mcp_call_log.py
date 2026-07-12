"""Production-route evidence for durable MCP call telemetry (polylogue-7s57)."""

from __future__ import annotations

import ast
import json
import queue
import socket
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast
from urllib.error import HTTPError

import pytest

from polylogue.config import load_polylogue_config
from polylogue.mcp.call_log import (
    McpCallLogEvent,
    _Delivery,
    _McpCallLogDispatcher,
    _persist_delivery,
    _post_call_log,
    flush_mcp_call_log,
)
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
def _running_daemon(port: int = 0) -> Iterator[str]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    server = DaemonAPIHTTPServer(("127.0.0.1", port), DaemonAPIHandler)
    thread = threading.Thread(target=server.serve_forever, name="mcp-call-log-test-daemon", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _reserve_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until(predicate: object, *, timeout: float = 3.0) -> None:
    assert callable(predicate)
    for _attempt in range(max(1, int(timeout / 0.01))):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def _event(call_id: str, *, session_id: str = "codex-session:resume-target") -> McpCallLogEvent:
    return McpCallLogEvent(
        call_id=call_id,
        tool_name="compose_context_preamble",
        session_id=session_id,
        session_ids=(),
        started_at_ms=1_700_000_000_000,
        finished_at_ms=1_700_000_000_025,
        success=True,
        error_detail=None,
    )


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


def test_session_tools_and_successor_preamble_are_queryable_by_session(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real FastMCP wrappers for the previously uncorrelated tools."""
    missing_session = "codex-session:missing-correlation"
    successor_session = "claude-code-session:new-successor"
    neighbor_session = "codex-session:neighbor-seed"
    comparison_sessions = ("codex-session:compare-a", "claude-code-session:compare-b")
    with _running_daemon() as daemon_url:
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", daemon_url)
        _set_runtime_services(None)
        try:
            server = cast(MCPServerUnderTest, build_server(role="read"))
            tools = server._tool_manager._tools
            invoke_surface(tools["get_messages"].fn, session_id=missing_session)
            invoke_surface(tools["raw_artifacts"].fn, session_id=missing_session)
            invoke_surface(
                tools["compose_context_preamble"].fn,
                cwd=str(workspace_env["archive_root"]),
                successor_session_id=successor_session,
            )
            invoke_surface(tools["neighbor_candidates"].fn, id=neighbor_session)
            invoke_surface(tools["compare_sessions"].fn, session_ids=",".join(comparison_sessions))
            assert flush_mcp_call_log(timeout=5.0)
        finally:
            _set_runtime_services(None)

    missing_calls = _read_calls(workspace_env["archive_root"], session_id=missing_session)
    assert {entry.tool_name for entry in missing_calls} == {"get_messages", "raw_artifacts"}
    successor_calls = _read_calls(workspace_env["archive_root"], session_id=successor_session)
    assert [entry.tool_name for entry in successor_calls] == ["compose_context_preamble"]
    neighbor_calls = _read_calls(workspace_env["archive_root"], session_id=neighbor_session)
    assert [entry.tool_name for entry in neighbor_calls] == ["neighbor_candidates"]
    for session_id in comparison_sessions:
        comparison_calls = _read_calls(workspace_env["archive_root"], session_id=session_id)
        assert [entry.tool_name for entry in comparison_calls] == ["compare_sessions"]


def test_readiness_surface_exposes_outbox_pressure(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    del workspace_env
    with _running_daemon() as daemon_url:
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", daemon_url)
        _set_runtime_services(None)
        try:
            server = cast(MCPServerUnderTest, build_server(role="read"))
            payload = json.loads(invoke_surface(server._tool_manager._tools["readiness_check"].fn))
            delivery = payload["mcp_call_delivery"]
            assert {
                "pending_count",
                "pending_bytes",
                "quarantined_count",
                "quarantined_bytes",
                "oldest_started_at_ms",
                "wake_queue_depth",
                "wakeups_dropped",
                "delivery_failures",
            } == set(delivery)
            assert flush_mcp_call_log(timeout=5.0)
        finally:
            _set_runtime_services(None)


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

    def reject(_config: object, _event: object) -> None:
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(call_log._DISPATCHER, "submit", reject)
    assert _safe_call("stats", lambda: '{"ok": true}') == '{"ok": true}'


def test_daemon_outage_and_dispatcher_restart_drain_durable_outbox(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise fsync+replace, failed HTTP, restart scan, daemon ACK, and SQL."""
    port = _reserve_port()
    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", f"http://127.0.0.1:{port}")
    config = load_polylogue_config()
    first = _McpCallLogDispatcher()
    first.submit(config, _event("restart-call"))
    _wait_until(lambda: first.status(config).delivery_failures > 0)
    first.shutdown()
    assert first.status(config).pending_count == 1

    with _running_daemon() as restarted_daemon_url:
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", restarted_daemon_url)
        restarted_config = load_polylogue_config()
        restarted = _McpCallLogDispatcher()
        try:
            restarted.register(restarted_config)
            assert restarted.flush(restarted_config, timeout=5.0)
        finally:
            restarted.shutdown()

    assert first.status(config).pending_count == 0
    calls = _read_calls(workspace_env["archive_root"], session_id="codex-session:resume-target")
    assert [entry.call_id for entry in calls] == ["restart-call"]


def test_wake_queue_pressure_never_discards_spooled_events(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded queue is only a wake hint; the filesystem is the debt ledger."""
    del workspace_env
    config = load_polylogue_config()
    dispatcher = _McpCallLogDispatcher()
    dispatcher._queue = queue.Queue(maxsize=1)
    monkeypatch.setattr(dispatcher, "_ensure_started", lambda: None)

    for index in range(3):
        dispatcher.submit(config, _event(f"pressure-{index}"))

    status = dispatcher.status(config)
    assert status.pending_count == 3
    assert status.pending_bytes > 0
    assert status.oldest_started_at_ms == 1_700_000_000_000
    assert status.wake_queue_depth == 1
    assert status.wakeups_dropped == 2


def test_duplicate_daemon_delivery_is_idempotent(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with _running_daemon() as daemon_url:
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", daemon_url)
        event = replace(
            _event("duplicate-call"),
            session_id=None,
            session_ids=("codex-session:member-a", "codex-session:member-b"),
        )
        delivery = _Delivery(event=event, daemon_url=daemon_url, auth_token=None)
        _post_call_log(delivery)
        _post_call_log(delivery)

    calls = _read_calls(workspace_env["archive_root"])
    assert [entry.call_id for entry in calls] == ["duplicate-call"]
    for session_id in event.session_ids:
        assert [entry.call_id for entry in _read_calls(workspace_env["archive_root"], session_id=session_id)] == [
            "duplicate-call"
        ]


def test_conflicting_duplicate_is_rejected_and_original_remains(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with _running_daemon() as daemon_url:
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", daemon_url)
        original = _Delivery(event=_event("conflict-call"), daemon_url=daemon_url, auth_token=None)
        _post_call_log(original)
        conflicting = replace(original, event=replace(original.event, tool_name="get_messages"))
        with pytest.raises(HTTPError) as caught:
            _post_call_log(conflicting)
        assert caught.value.code == 409

    [stored] = _read_calls(workspace_env["archive_root"])
    assert stored.call_id == "conflict-call"
    assert stored.tool_name == "compose_context_preamble"


def test_conflict_is_quarantined_without_blocking_later_delivery(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with _running_daemon() as daemon_url:
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", daemon_url)
        original = _Delivery(event=_event("a-conflict"), daemon_url=daemon_url, auth_token=None)
        _post_call_log(original)
        config = load_polylogue_config()
        dispatcher = _McpCallLogDispatcher()
        try:
            dispatcher.submit(config, replace(original.event, tool_name="get_messages"))
            dispatcher.submit(config, _event("z-valid"))
            assert dispatcher.flush(config, timeout=5.0)
            status = dispatcher.status(config)
            assert status.pending_count == 0
            assert status.quarantined_count == 1
            assert status.quarantined_bytes > 0
        finally:
            dispatcher.shutdown()

    calls = _read_calls(workspace_env["archive_root"])
    assert {entry.call_id for entry in calls} == {"a-conflict", "z-valid"}


def test_two_dispatchers_can_quarantine_the_same_conflict(
    workspace_env: dict[str, Path],
) -> None:
    del workspace_env
    config = load_polylogue_config()
    path = _persist_delivery(config, _event("shared-conflict"))
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def quarantine(dispatcher: _McpCallLogDispatcher) -> None:
        try:
            barrier.wait(timeout=2.0)
            dispatcher._quarantine_conflict(path)
        except BaseException as exc:
            failures.append(exc)

    first = _McpCallLogDispatcher()
    second = _McpCallLogDispatcher()
    threads = [
        threading.Thread(target=quarantine, args=(first,)),
        threading.Thread(target=quarantine, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)

    assert not failures
    assert not path.exists()
    assert (path.parent.parent / "quarantine" / path.name).is_file()


def test_every_session_tool_forwards_telemetry_identity() -> None:
    """Signature-driven inventory guard for singular and plural session tools."""
    mcp_root = Path(__file__).parents[3] / "polylogue" / "mcp"
    checked: set[str] = set()
    aliases = {
        "blackboard_post": ("scope_session", "session_id"),
        "compose_context_preamble": ("successor_session_id", "session_id"),
        "get_session_summary": ("id", "session_id"),
        "neighbor_candidates": ("id", "session_id"),
    }
    for path in sorted(mcp_root.glob("server_*tools.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            is_tool = any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "tool"
                for decorator in node.decorator_list
            )
            argument_names = {argument.arg for argument in (*node.args.args, *node.args.kwonlyargs)}
            source_argument: str | None = None
            telemetry_keyword: str | None = None
            if "session_id" in argument_names:
                source_argument, telemetry_keyword = "session_id", "session_id"
            elif "session_ids" in argument_names:
                source_argument, telemetry_keyword = "session_ids", "session_ids"
            elif node.name in aliases:
                source_argument, telemetry_keyword = aliases[node.name]
            if not is_tool or source_argument is None or telemetry_keyword is None:
                continue
            safe_calls = [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"safe_call", "async_safe_call"}
            ]
            assert len(safe_calls) == 1, (path, node.name)
            assert any(keyword.arg == telemetry_keyword for keyword in safe_calls[0].keywords), (
                path,
                node.name,
            )
            checked.add(node.name)

    assert {
        "bulk_tag_sessions",
        "compare_sessions",
        "get_messages",
        "maintenance_execute",
        "neighbor_candidates",
        "raw_artifacts",
        "record_correction",
        "update_index",
    } <= checked


def test_context_preamble_declares_successor_session_correlation() -> None:
    path = Path(__file__).parents[3] / "polylogue" / "mcp" / "server_context_tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    [function] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "compose_context_preamble"
    ]
    assert "successor_session_id" in {argument.arg for argument in (*function.args.args, *function.args.kwonlyargs)}
    [safe_call] = [
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "async_safe_call"
    ]
    session_keyword = next(keyword for keyword in safe_call.keywords if keyword.arg == "session_id")
    assert isinstance(session_keyword.value, ast.Name)
    assert session_keyword.value.id == "successor_session_id"


@pytest.mark.parametrize(
    ("relative_path", "function_name", "argument_name"),
    [
        ("server_tools.py", "get_session_summary", "id"),
        ("server_mutation_tools.py", "blackboard_post", "scope_session"),
    ],
)
def test_session_alias_tools_forward_telemetry_identity(
    relative_path: str,
    function_name: str,
    argument_name: str,
) -> None:
    path = Path(__file__).parents[3] / "polylogue" / "mcp" / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    [function] = [
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name
    ]
    [safe_call] = [
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "async_safe_call"
    ]
    session_keyword = next(keyword for keyword in safe_call.keywords if keyword.arg == "session_id")
    assert isinstance(session_keyword.value, ast.Name)
    assert session_keyword.value.id == argument_name
