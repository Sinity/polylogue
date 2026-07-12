"""Production-route tests for durable Claude Code/Codex hook capture."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.sources.hooks import (
    acknowledged_hook_spool_dir,
    drain_hook_event_spool,
    enqueue_hook_event,
    pending_hook_spool_dir,
)


@pytest.mark.parametrize(
    ("provider", "event_type", "session_id", "expected_origin"),
    [
        ("claude-code", "PostToolUse", "claude-session", "claude-code-session"),
        ("codex", "PostToolUse", "codex-session", "codex-session"),
    ],
)
def test_hook_spool_acknowledges_only_after_source_tier_materialization(
    tmp_path: Path,
    provider: str,
    event_type: str,
    session_id: str,
    expected_origin: str,
) -> None:
    """The real source-tier row is the receipt before the file is acknowledged."""

    spool_root = tmp_path / "hooks"
    archive_root = tmp_path / "archive"
    event_path = enqueue_hook_event(
        event_id=f"{provider}-event",
        provider=provider,
        event_type=event_type,
        session_id=session_id,
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_name": "exec", "tool_call_id": "call-1"},
        root=spool_root,
    )

    result = drain_hook_event_spool(archive_root, root=spool_root)

    assert result.acknowledged == 1
    assert result.failed == 0
    assert event_path.exists() is False
    assert (acknowledged_hook_spool_dir(spool_root) / event_path.name).exists()
    with sqlite3.connect(archive_root / "source.db") as conn:
        rows = conn.execute("SELECT origin, session_native_id, event_type FROM raw_hook_events").fetchall()
    assert rows == [(expected_origin, session_id, event_type)]


def test_hook_spool_keeps_event_pending_when_source_tier_write_fails(tmp_path: Path) -> None:
    """A failed receiver write cannot falsely acknowledge a hook event."""

    spool_root = tmp_path / "hooks"
    event_path = enqueue_hook_event(
        event_id="retry-me",
        provider="claude-code",
        event_type="PreToolUse",
        session_id="session-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_name": "Bash"},
        root=spool_root,
    )
    blocked_root = tmp_path / "not-a-directory"
    blocked_root.write_text("not a directory", encoding="utf-8")

    result = drain_hook_event_spool(blocked_root, root=spool_root)

    assert result == type(result)(acknowledged=0, failed=1)
    assert event_path.exists()
    assert list(acknowledged_hook_spool_dir(spool_root).glob("*.json")) == []
    assert list(pending_hook_spool_dir(spool_root).glob("*.json")) == [event_path]


def test_hook_spool_replay_is_idempotent_after_interrupted_acknowledgement(tmp_path: Path) -> None:
    """A crash after persistence but before the rename cannot duplicate evidence."""

    spool_root = tmp_path / "hooks"
    archive_root = tmp_path / "archive"
    event_path = enqueue_hook_event(
        event_id="stable-event-id",
        provider="codex",
        event_type="PostToolUse",
        session_id="session-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_name": "exec"},
        root=spool_root,
    )
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1

    replay_path = pending_hook_spool_dir(spool_root) / event_path.name
    (acknowledged_hook_spool_dir(spool_root) / event_path.name).replace(replay_path)
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1

    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone() == (1,)
