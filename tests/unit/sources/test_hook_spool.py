"""Production-route tests for durable Claude Code/Codex hook capture."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import watchfiles
from watchfiles import Change

from polylogue.hooks import hook_main
from polylogue.sources.hooks import (
    acknowledged_hook_spool_dir,
    drain_hook_event_spool,
    enqueue_hook_event,
    hook_spool_root,
    pending_hook_spool_dir,
)
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore


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


@pytest.mark.asyncio
async def test_live_watcher_drains_the_configured_hook_spool_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented producer override reaches the daemon's real drain path."""

    spool_root = tmp_path / "configured-hooks"
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    monkeypatch.setenv("POLYLOGUE_HOOK_SIDECAR_DIR", str(spool_root))
    event_path = enqueue_hook_event(
        event_id="configured-root-event",
        provider="claude-code",
        event_type="SessionStart",
        session_id="session-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"cwd": "/workspace"},
    )
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=None)),
        (WatchSource(name="hooks", root=pending_hook_spool_dir(), suffixes=(".json",)),),
        cursor=CursorStore(archive_root / "ops.db"),
    )

    await watcher._catch_up([pending_hook_spool_dir()])

    assert hook_spool_root() == spool_root
    assert event_path.exists() is False
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT session_native_id FROM raw_hook_events").fetchone() == ("session-1",)


@pytest.mark.asyncio
async def test_live_watcher_observes_a_spool_created_after_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon started before the first hook still drains the later envelope."""

    spool_root = tmp_path / "hooks"
    pending = pending_hook_spool_dir(spool_root)
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=None)),
        (WatchSource(name="hooks", root=pending, suffixes=(".json",)),),
        cursor=CursorStore(archive_root / "ops.db"),
    )

    async def emit_first_hook(*roots: Path, **_kwargs: object) -> AsyncIterator[set[tuple[Change, str]]]:
        assert roots == (pending,)
        event_path = enqueue_hook_event(
            event_id="first-after-startup",
            provider="codex",
            event_type="SessionStart",
            session_id="session-1",
            timestamp="2026-07-12T10:00:00Z",
            payload={"cwd": "/workspace"},
            root=spool_root,
        )
        yield {(Change.added, str(event_path))}
        watcher.stop()

    monkeypatch.setattr(watchfiles, "awatch", emit_first_hook)

    await watcher.run()

    assert (acknowledged_hook_spool_dir(spool_root) / "first-after-startup.json").exists()
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT session_native_id FROM raw_hook_events").fetchone() == ("session-1",)


def test_hook_spool_retains_sqlite_failures_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite receiver faults leave the real envelope pending for a future drain."""

    spool_root = tmp_path / "hooks"
    event_path = enqueue_hook_event(
        event_id="sqlite-retry",
        provider="codex",
        event_type="PostToolUse",
        session_id="session-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_name": "exec"},
        root=spool_root,
    )

    def fail_persistence(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("polylogue.sources.hooks._persist_record", fail_persistence)
    result = drain_hook_event_spool(tmp_path / "archive", root=spool_root)

    assert result.acknowledged == 0
    assert result.failed == 1
    assert event_path.exists()


@pytest.mark.parametrize(
    ("provider", "session_id"),
    [("claude-code", "claude-session"), ("codex", "codex-session")],
)
def test_hook_entrypoint_spools_and_materializes_configured_runtime_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    session_id: str,
) -> None:
    """The installed hook command reaches the durable source-tier receipt path."""

    spool_root = tmp_path / "hooks"
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_HOOK_SIDECAR_DIR", str(spool_root))
    monkeypatch.setattr("sys.stdin", StringIO(f'{{"session_id":"{session_id}","tool_name":"exec"}}'))

    assert hook_main(["PostToolUse", "--provider", provider]) == 0

    pending = list(pending_hook_spool_dir(spool_root).glob("*.json"))
    assert len(pending) == 1
    record = json.loads(pending[0].read_text(encoding="utf-8"))
    assert record["provider"] == provider
    assert record["session_id"] == session_id
    assert record["event_type"] == "PostToolUse"
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT session_native_id FROM raw_hook_events").fetchone() == (session_id,)
