"""Hook-flow liveness and partial-coverage evidence tests."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.daemon.health import HealthSeverity, _check_hook_flow_fast
from polylogue.daemon.metrics import format_metrics
from polylogue.hooks import hook_status, plan_hook_change, resolve_events
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

NOW_MS = 2_000_000_000_000


@pytest.fixture
def hook_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    archive = tmp_path / "archive"
    initialize_archive_database(archive / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(archive / "index.db", ArchiveTier.INDEX)
    plan_hook_change(
        "install",
        "claude-code",
        resolve_events("claude-code", "recommended"),
        dry_run=False,
    )
    settings = home / ".claude" / "settings.json"
    os.utime(settings, ns=((NOW_MS - 10_000) * 1_000_000,) * 2)
    monkeypatch.setattr("polylogue.hooks.shutil.which", lambda _name: "/usr/bin/polylogue-hook")
    monkeypatch.setattr("polylogue.daemon.health.archive_root", lambda: archive)
    return archive


def _insert_session(
    archive: Path,
    native_id: str,
    *,
    authored_prompts: int = 1,
    tool_uses: int = 1,
) -> None:
    with sqlite3.connect(archive / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, content_hash, updated_at_ms,
                authored_user_message_count, tool_use_count
            ) VALUES (?, 'claude-code-session', ?, ?, ?, ?)
            """,
            (native_id, bytes.fromhex("11" * 32), NOW_MS, authored_prompts, tool_uses),
        )


def _insert_hook_event(archive: Path, native_id: str, event: str, *, suffix: str = "1") -> None:
    payload = {"event_type": event, "session_id": native_id, "provider": "claude-code"}
    with sqlite3.connect(archive / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id,
                source_path, event_type, payload_json, observed_at_ms
            ) VALUES (?, 'claude-code-session', ?, ?, ?, ?, ?, ?)
            """,
            (
                f"hook-{native_id}-{event}-{suffix}",
                native_id,
                native_id,
                f"/hooks/{native_id}.jsonl",
                event,
                json.dumps(payload),
                NOW_MS,
            ),
        )


def test_wired_harness_with_session_and_no_events_is_gap(hook_archive: Path) -> None:
    _insert_session(hook_archive, "session-gap")

    status = hook_status(
        "claude-code",
        archive_root_path=hook_archive,
        now_ms=NOW_MS,
    )

    assert status.flow_state == "gap"
    assert status.eligible_session_count == 1
    assert status.sessions_without_hook_events == 1


def test_partial_event_coverage_uses_defensible_session_opportunities(hook_archive: Path) -> None:
    _insert_session(hook_archive, "session-partial", authored_prompts=1, tool_uses=1)
    _insert_hook_event(hook_archive, "session-partial", "SessionStart")
    _insert_hook_event(hook_archive, "session-partial", "UserPromptSubmit")

    status = hook_status(
        "claude-code",
        archive_root_path=hook_archive,
        now_ms=NOW_MS,
    )

    assert status.flow_state == "partial"
    by_event = {row.event: row for row in status.coverage}
    assert by_event["SessionStart"].missing_expected_count == 0
    assert by_event["UserPromptSubmit"].expected_session_count == 1
    assert by_event["PreToolUse"].expected_session_count == 1
    assert by_event["PreToolUse"].missing_expected_count == 1
    assert by_event["PostToolUse"].expected_session_count is None
    assert by_event["Stop"].expected_session_count is None


def test_health_alerts_within_first_session_without_hook_events(hook_archive: Path) -> None:
    _insert_session(hook_archive, "session-broken-script")

    alert = _check_hook_flow_fast()

    assert alert.check_name == "hook_flow"
    assert alert.severity == HealthSeverity.ERROR
    assert "1/1 recent session(s) have no hook events" in alert.message


def test_complete_recommended_flow_is_healthy(hook_archive: Path) -> None:
    _insert_session(hook_archive, "session-ok")
    for index, event in enumerate(("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop")):
        _insert_hook_event(hook_archive, "session-ok", event, suffix=str(index))

    status = hook_status(
        "claude-code",
        archive_root_path=hook_archive,
        now_ms=NOW_MS,
    )

    assert status.flow_state == "healthy"
    assert status.sessions_without_hook_events == 0


def test_metrics_project_hook_flow_gap(hook_archive: Path) -> None:
    _insert_session(hook_archive, "session-metric-gap")

    body = format_metrics(hook_archive / "index.db")

    assert 'polylogue_hook_flow_healthy{harness="claude-code"} 0' in body
    assert 'polylogue_hook_flow_state{harness="claude-code",state="gap"} 1' in body
    assert 'polylogue_hook_sessions{harness="claude-code",state="without_events"} 1' in body
