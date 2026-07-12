"""Durable MCP call-log write path (polylogue-7s57).

Pins that ``_safe_call``/``_async_safe_call`` — the shared wrappers every
registered MCP tool body routes through — persist a best-effort row to the
disposable ``ops.db`` ``mcp_call_log`` table for every call, capturing tool
name, the optional caller-supplied ``session_id``, timing, and
success/failure. This is the durable-call-log prerequisite for the
resume-efficacy analysis (polylogue-9e5.10): without it there is no way to
know whether ``get_resume_brief``/``compose_context_preamble`` were ever
invoked for a given session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.mcp.server_support import _async_safe_call, _safe_call
from polylogue.storage.sqlite.archive_tiers.ops_write import list_mcp_calls
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def _read_calls(archive_root: Path, **filters: object) -> tuple[object, ...]:
    ops_db = archive_root / "ops.db"
    conn = open_readonly_connection(ops_db)
    try:
        return list_mcp_calls(conn, **filters)  # type: ignore[arg-type]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_async_safe_call_logs_success_with_session_id(workspace_env: dict[str, Path]) -> None:
    async def run() -> str:
        return '{"ok": true}'

    result = await _async_safe_call("get_resume_brief", run, session_id="codex-session:abc")
    assert '"ok"' in result

    calls = _read_calls(workspace_env["archive_root"], session_id="codex-session:abc")
    assert len(calls) == 1
    entry = calls[0]
    assert entry.tool_name == "get_resume_brief"  # type: ignore[attr-defined]
    assert entry.session_id == "codex-session:abc"  # type: ignore[attr-defined]
    assert entry.success is True  # type: ignore[attr-defined]
    assert entry.error_detail is None  # type: ignore[attr-defined]
    assert entry.finished_at_ms >= entry.started_at_ms  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_async_safe_call_logs_failure_without_session_id(workspace_env: dict[str, Path]) -> None:
    async def boom() -> str:
        raise RuntimeError("trigger")

    await _async_safe_call("compose_context_preamble", boom)

    calls = _read_calls(workspace_env["archive_root"], tool_name="compose_context_preamble")
    assert len(calls) == 1
    entry = calls[0]
    assert entry.session_id is None  # type: ignore[attr-defined]
    assert entry.success is False  # type: ignore[attr-defined]
    assert entry.error_detail == "RuntimeError"  # type: ignore[attr-defined]


def test_safe_call_logs_sync_success_with_session_id(workspace_env: dict[str, Path]) -> None:
    def run() -> str:
        return '{"ok": true}'

    _safe_call("session_profile", run, session_id="claude-code-session:def")

    calls = _read_calls(workspace_env["archive_root"], session_id="claude-code-session:def")
    assert len(calls) == 1
    assert calls[0].tool_name == "session_profile"  # type: ignore[attr-defined]
    assert calls[0].success is True  # type: ignore[attr-defined]


def test_safe_call_without_workspace_env_never_raises() -> None:
    """The call-log write is best-effort: a tool call must still return its
    result even when the archive/config resolution behind it is unusual
    (no ``workspace_env`` fixture here means the autouse-isolated tmp XDG
    dirs are used instead, exercising the "no archive yet" path)."""

    def run() -> str:
        return '{"ok": true}'

    assert _safe_call("stats", run) == '{"ok": true}'
