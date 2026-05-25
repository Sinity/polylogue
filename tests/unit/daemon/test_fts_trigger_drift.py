"""FTS sync-trigger drift detection and auto-restore (#1229).

The daemon's FAST tier inspects the canonical FTS triggers for active
archive and insight search surfaces every health
cycle. A SIGKILL during the bulk-write suspension window (see
``docs/internals.md`` "FTS5 Model") leaves these triggers dropped and
silently corrupts search.

This module pins three behaviors:

1. Healthy archive → check is OK.
2. Missing trigger → check is CRITICAL with the missing-trigger name in
   the alert message and a restore command suggestion.
3. With ``[health] fts_auto_restore = true`` configured, the missing
   triggers are restored in place and the FTS index is rebuilt; the
   alert is downgraded to WARNING ("auto-recovered") so the operator
   is still notified that drift happened, and a follow-up check
   returns to OK.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.daemon import health as health_module
from polylogue.daemon.health import (
    _EXPECTED_FTS_TRIGGERS,
    HealthSeverity,
    HealthTier,
    _check_fts_trigger_drift_fast,
    _find_missing_fts_triggers,
)
from polylogue.paths import db_path
from polylogue.storage.fts.fts_lifecycle import (
    ensure_fts_index_sync,
    restore_fts_triggers_sync,
)

# Minimal table shapes needed for the FTS rebuild path to execute. We do
# not need to mirror the full schema; the auto-restore touches only the
# FTS5 control surface and the source tables it reads from.
_MESSAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    text TEXT
)
"""

_CONTENT_BLOCKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS content_blocks (
    block_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    block_index INTEGER NOT NULL,
    type TEXT NOT NULL,
    text TEXT,
    tool_name TEXT,
    tool_id TEXT,
    tool_input TEXT,
    metadata TEXT,
    semantic_type TEXT,
    UNIQUE (message_id, block_index)
)
"""

_ACTION_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS action_events (
    event_id TEXT PRIMARY KEY,
    message_id TEXT,
    conversation_id TEXT NOT NULL,
    action_kind TEXT NOT NULL,
    normalized_tool_name TEXT,
    search_text TEXT
)
"""

_SESSION_WORK_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_work_events (
    event_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    heuristic_label TEXT NOT NULL,
    search_text TEXT NOT NULL
)
"""

_SESSION_WORK_EVENTS_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS session_work_events_fts USING fts5(
    event_id UNINDEXED,
    conversation_id UNINDEXED,
    source_name UNINDEXED,
    heuristic_label UNINDEXED,
    text,
    tokenize='unicode61'
)
"""

_WORK_THREADS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS work_threads (
    thread_id TEXT PRIMARY KEY,
    root_id TEXT NOT NULL,
    search_text TEXT NOT NULL
)
"""

_WORK_THREADS_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS work_threads_fts USING fts5(
    thread_id UNINDEXED,
    root_id UNINDEXED,
    text,
    tokenize='unicode61'
)
"""


@pytest.fixture(autouse=True)
def _reset_failure_counts() -> Iterator[None]:
    """Each test starts with an empty consecutive-failure map."""
    health_module._failure_counts.clear()
    yield
    health_module._failure_counts.clear()


def _seed_archive_with_triggers(path: Path) -> None:
    """Bootstrap the minimal table set + the canonical FTS triggers.

    The check is schema-agnostic — it only reads ``sqlite_schema`` — but
    the auto-restore path needs the active source and FTS tables to
    exist because repair recreates triggers and rebuilds the indexes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_MESSAGES_TABLE_SQL)
        conn.execute(_CONTENT_BLOCKS_TABLE_SQL)
        conn.execute(_ACTION_EVENTS_TABLE_SQL)
        conn.execute(_SESSION_WORK_EVENTS_TABLE_SQL)
        conn.execute(_SESSION_WORK_EVENTS_FTS_SQL)
        conn.execute(_WORK_THREADS_TABLE_SQL)
        conn.execute(_WORK_THREADS_FTS_SQL)
        ensure_fts_index_sync(conn)
        conn.commit()
    finally:
        conn.close()


def _drop_trigger(path: Path, name: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Healthy / inventory contract
# ---------------------------------------------------------------------------


def test_expected_fts_trigger_inventory_covers_archive_and_insight_search() -> None:
    """The contract is fifteen triggers over archive and insight FTS tables."""
    assert len(_EXPECTED_FTS_TRIGGERS) == 15
    assert set(_EXPECTED_FTS_TRIGGERS) == {
        "messages_fts_ai",
        "messages_fts_ad",
        "messages_fts_au",
        "content_blocks_fts_ai",
        "content_blocks_fts_ad",
        "content_blocks_fts_au",
        "action_events_fts_ai",
        "action_events_fts_ad",
        "action_events_fts_au",
        "session_work_events_fts_ai",
        "session_work_events_fts_ad",
        "session_work_events_fts_au",
        "work_threads_fts_ai",
        "work_threads_fts_ad",
        "work_threads_fts_au",
    }


def test_check_returns_ok_when_database_absent(
    workspace_env: dict[str, Path],
) -> None:
    """Fresh-install path: no DB yet → check passes."""
    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.OK
    assert alert.tier == HealthTier.FAST
    assert "no database yet" in alert.message
    assert alert.consecutive_failures == 0


def test_check_returns_ok_when_all_triggers_present(
    workspace_env: dict[str, Path],
) -> None:
    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.OK
    assert alert.tier == HealthTier.FAST
    assert "all 15 active FTS triggers present" in alert.message
    assert alert.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", list(_EXPECTED_FTS_TRIGGERS))
def test_check_detects_any_missing_trigger(
    workspace_env: dict[str, Path],
    dropped: str,
) -> None:
    """Dropping any active canonical trigger must surface as CRITICAL."""
    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    _drop_trigger(dbf, dropped)

    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.CRITICAL
    assert alert.tier == HealthTier.FAST
    assert dropped in alert.message
    assert "restore" in alert.message.lower()
    assert "polylogue check --repair-fts" in alert.message
    assert alert.consecutive_failures == 1


def test_check_lists_all_missing_triggers_when_several_drop(
    workspace_env: dict[str, Path],
) -> None:
    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    _drop_trigger(dbf, "messages_fts_ai")
    _drop_trigger(dbf, "action_events_fts_ad")

    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.CRITICAL
    assert "2/15 missing" in alert.message
    assert "messages_fts_ai" in alert.message
    assert "action_events_fts_ad" in alert.message


def test_find_missing_fts_triggers_helper_returns_canonical_order(
    workspace_env: dict[str, Path],
) -> None:
    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    _drop_trigger(dbf, "action_events_fts_au")
    _drop_trigger(dbf, "messages_fts_ad")

    conn = sqlite3.connect(str(dbf))
    try:
        missing = _find_missing_fts_triggers(conn)
    finally:
        conn.close()
    # Order is from _EXPECTED_FTS_TRIGGERS, not from insertion order.
    assert missing == ["messages_fts_ad", "action_events_fts_au"]


# ---------------------------------------------------------------------------
# Auto-restore (#1229 ambitious expansion)
# ---------------------------------------------------------------------------


def _enable_auto_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the health-loop config to flip ``fts_auto_restore`` on.

    Avoids writing a TOML file in tests — the config layer already
    exposes the flag via ``POLYLOGUE_HEALTH_FTS_AUTO_RESTORE``, which
    ``load_polylogue_config`` consumes through the env-override layer.
    """
    monkeypatch.setenv("POLYLOGUE_HEALTH_FTS_AUTO_RESTORE", "1")


def test_auto_restore_repairs_missing_triggers_and_warns(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With auto-restore on, drift self-heals and the alert is WARNING."""
    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    _drop_trigger(dbf, "messages_fts_au")
    _drop_trigger(dbf, "action_events_fts_ai")

    _enable_auto_restore(monkeypatch)

    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.WARNING
    assert alert.tier == HealthTier.FAST
    assert "auto-recovered" in alert.message
    assert "messages_fts_au" in alert.message
    assert "action_events_fts_ai" in alert.message
    # WARNING is a healthy outcome for the consecutive-failure counter:
    # the check ultimately succeeded, so the counter resets.
    assert alert.consecutive_failures == 0

    # The triggers must actually be back on disk.
    conn = sqlite3.connect(str(dbf))
    try:
        missing = _find_missing_fts_triggers(conn)
    finally:
        conn.close()
    assert missing == []

    # The next health cycle returns to OK.
    next_alert = _check_fts_trigger_drift_fast()
    assert next_alert.severity == HealthSeverity.OK
    assert "all 15 active FTS triggers present" in next_alert.message


def test_auto_restore_disabled_by_default_keeps_alert_critical(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the opt-in flag, drift surfaces as CRITICAL even on a healable archive."""
    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    _drop_trigger(dbf, "messages_fts_ai")
    # Ensure the env var is not leaking in.
    monkeypatch.delenv("POLYLOGUE_HEALTH_FTS_AUTO_RESTORE", raising=False)

    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.CRITICAL
    # The trigger is still missing — no repair was attempted.
    conn = sqlite3.connect(str(dbf))
    try:
        missing = _find_missing_fts_triggers(conn)
    finally:
        conn.close()
    assert "messages_fts_ai" in missing


def test_auto_restore_failure_falls_back_to_critical(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the auto-restore raises, the alert remains CRITICAL with the cause."""
    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    _drop_trigger(dbf, "messages_fts_ai")
    _enable_auto_restore(monkeypatch)

    def _boom(_db: object) -> tuple[list[str], bool, str]:
        raise RuntimeError("simulated restore failure")

    monkeypatch.setattr(health_module, "_auto_restore_fts_triggers", _boom)

    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.CRITICAL
    assert "auto-restore failed" in alert.message
    assert "simulated restore failure" in alert.message
    assert "polylogue check --repair-fts" in alert.message
    assert alert.consecutive_failures == 1


# ---------------------------------------------------------------------------
# Notification-backend integration
# ---------------------------------------------------------------------------


def test_drift_alert_routes_through_notification_backend(
    workspace_env: dict[str, Path],
) -> None:
    """The CRITICAL drift alert flows through send_notifications uniformly."""
    from polylogue.daemon.health import HealthAlert
    from polylogue.daemon.notifications import send_notifications

    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    _drop_trigger(dbf, "messages_fts_ai")
    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.CRITICAL

    seen: list[HealthAlert] = []

    class _Backend:
        def notify(
            self,
            alerts: list[HealthAlert],
            *,
            config: dict[str, object] | None = None,
        ) -> None:
            seen.extend(alerts)

    send_notifications([alert], backend=_Backend())
    assert len(seen) == 1
    assert seen[0].check_name == "fts_trigger_drift"
    assert seen[0].severity == HealthSeverity.CRITICAL


def test_auto_recovery_alert_routes_through_notification_backend(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WARNING auto-recovery alert is still forwarded so operators see the self-heal."""
    from polylogue.daemon.health import HealthAlert
    from polylogue.daemon.notifications import send_notifications

    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    _drop_trigger(dbf, "messages_fts_ai")
    _enable_auto_restore(monkeypatch)

    alert = _check_fts_trigger_drift_fast()
    assert alert.severity == HealthSeverity.WARNING

    seen: list[HealthAlert] = []

    class _Backend:
        def notify(
            self,
            alerts: list[HealthAlert],
            *,
            config: dict[str, object] | None = None,
        ) -> None:
            seen.extend(alerts)

    send_notifications([alert], backend=_Backend())
    assert len(seen) == 1
    assert seen[0].check_name == "fts_trigger_drift"
    assert "auto-recovered" in seen[0].message


# ---------------------------------------------------------------------------
# Helper: the schema-bootstrap path is left untouched by our minimal seed
# ---------------------------------------------------------------------------


def test_restore_fts_triggers_sync_is_idempotent_on_already_healthy_db(
    workspace_env: dict[str, Path],
) -> None:
    """Sanity check on the underlying repair helper used by auto-restore."""
    dbf = db_path()
    _seed_archive_with_triggers(dbf)
    conn = sqlite3.connect(str(dbf))
    try:
        restore_fts_triggers_sync(conn)
        conn.commit()
        missing = _find_missing_fts_triggers(conn)
    finally:
        conn.close()
    assert missing == []
