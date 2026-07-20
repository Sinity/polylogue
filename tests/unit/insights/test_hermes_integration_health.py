"""Fixture-backed tests for the bounded Hermes integration health rollup (fs1.15).

fs1.15 asks for one composed, read-only view of Hermes integration liveness
that renders explicit degraded states -- never a crash or a silent zero --
for a stale producer, a malformed event, watcher lag, and an unavailable
archive. This module proves each of those scenarios plus the healthy path,
using the same real production primitives the composer wires together
(``explain_import_path``, ``project_named_source_freshness``,
``convergence_debt_summary_info``, lifecycle reconciliation, delivery
correlation) rather than a parallel test-only mechanism.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.insights.hermes_integration_health import build_hermes_integration_health
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore

_STATE_DB_SCHEMA = """
CREATE TABLE schema_version(version INTEGER NOT NULL);
INSERT INTO schema_version(version) VALUES (19);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    model_config TEXT,
    parent_session_id TEXT,
    started_at REAL,
    ended_at REAL,
    end_reason TEXT,
    title TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL NOT NULL,
    tool_calls TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1,
    compacted INTEGER DEFAULT 0
);
"""


def _make_processor(
    workspace_env: dict[str, Path], root_name: str, db_name: str
) -> tuple[Polylogue, LiveBatchProcessor, Path]:
    root = workspace_env["data_root"] / root_name
    root.mkdir(parents=True)
    db_path = workspace_env["data_root"] / db_name
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="hermes", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    return archive, processor, root


def test_disabled_when_hermes_root_is_absent(tmp_path: Path) -> None:
    """No Hermes runtime root on this host is an explicit disabled state, not a crash."""

    health = build_hermes_integration_health(tmp_path / "archive", hermes_root=tmp_path / "no-such-hermes-root")
    assert health.enabled is False
    assert health.verdict == "disabled"
    assert health.sources == ()
    assert health.parser_failures == ()


def test_unavailable_when_archive_root_is_absent(tmp_path: Path) -> None:
    """A present Hermes root but no archive at all renders unavailable, never a crash."""

    hermes_root = tmp_path / "hermes-home"
    hermes_root.mkdir(parents=True)
    (hermes_root / "state.db").write_bytes(b"not-a-real-db")

    health = build_hermes_integration_health(tmp_path / "no-such-archive", hermes_root=hermes_root)
    assert health.enabled is True
    assert health.verdict == "unavailable"
    assert any("archive root does not exist" in caveat for caveat in health.caveats)


def test_malformed_event_renders_explicit_parser_failure(workspace_env: dict[str, Path]) -> None:
    """A non-JSON file under the Hermes root is surfaced as a named parser failure, not dropped."""

    hermes_root = workspace_env["data_root"] / "hermes-malformed"
    hermes_root.mkdir(parents=True)
    (hermes_root / "session.json").write_text("{not-valid-json-at-all", encoding="utf-8")

    health = build_hermes_integration_health(workspace_env["archive_root"], hermes_root=hermes_root)

    assert health.enabled is True
    assert health.verdict == "degraded"
    assert len(health.parser_failures) == 1
    failure = health.parser_failures[0]
    assert failure.source_ref == "session.json"
    assert "decode failure" in failure.reason
    # No raw content or absolute path leaks into the response.
    assert str(hermes_root) not in failure.reason
    assert str(hermes_root) not in failure.source_ref


def test_unpaired_atof_scope_is_surfaced_as_fidelity_debt(workspace_env: dict[str, Path]) -> None:
    """A crashed/truncated ATOF scope (start with no matching end) is visible debt, not silence."""

    hermes_root = workspace_env["data_root"] / "hermes-unpaired"
    hermes_root.mkdir(parents=True)
    record = {
        "atof_version": "0.1",
        "kind": "scope",
        "category": "tool",
        "scope_category": "start",
        "uuid": "tool-crashed-1",
        "timestamp": "2026-07-18T09:00:00Z",
        "name": "terminal",
        "metadata": {"session_id": "unpaired-session-1", "tool_call_id": "call-1"},
    }
    (hermes_root / "events.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    health = build_hermes_integration_health(workspace_env["archive_root"], hermes_root=hermes_root)

    assert health.enabled is True
    assert health.parser_failures == ()
    capability_names = {cap.capability for cap in health.fidelity_capabilities}
    assert "unpaired_scope_debt" in capability_names
    unpaired_capability = next(cap for cap in health.fidelity_capabilities if cap.capability == "unpaired_scope_debt")
    assert unpaired_capability.status == "degraded"
    assert unpaired_capability.observed == 1
    assert unpaired_capability.source_refs == ("events.jsonl",)


@pytest.mark.asyncio
async def test_healthy_state_db_reaches_healthy_verdict_through_named_freshness(
    workspace_env: dict[str, Path],
) -> None:
    """A real state.db ingested end-to-end reaches an explicit ``healthy`` verdict.

    Anti-vacuity: removing the ``project_named_source_freshness`` call (or
    feeding it the wrong source path) from the composer makes ``stage``
    report ``"unknown"``/``"unseen"`` here instead of an indexed/searchable
    stage, and the assertion below fails.
    """

    archive, processor, root = _make_processor(workspace_env, "hermes-home-healthy", "hermes-state-healthy.db")
    source_path = root / "state.db"
    try:
        with sqlite3.connect(source_path) as conn:
            conn.executescript(_STATE_DB_SCHEMA)
            conn.execute(
                "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
                "VALUES ('root', 'cli', '{}', 1.0, 8.0, 'completed', 'root')"
            )
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (1, 'root', 'user', 'hi', 2.0)"
            )

        metrics = await processor.ingest_files([source_path], emit_event=False)
        assert metrics.failed_file_count == 0
        assert metrics.ingested_session_count == 1

        health = build_hermes_integration_health(workspace_env["archive_root"], hermes_root=root)
        assert health.enabled is True
        assert health.parser_failures == ()
        assert len(health.sources) == 1
        source = health.sources[0]
        assert source.source_ref == "state.db"
        assert source.source_class == "state_db"
        assert source.stage in {"indexed-unconverged", "searchable"}
        assert health.verdict == "healthy"
        # No raw filesystem path leaks into the response.
        assert str(root) not in source.source_ref
        assert all(str(root) not in caveat for caveat in health.caveats)
    finally:
        await archive.close()


def test_watcher_lag_is_visible_as_a_non_searchable_stage(workspace_env: dict[str, Path]) -> None:
    """A source file present on disk but never ingested reports its real unseen/lagging stage.

    This is the "watcher lag" degraded scenario: the file exists, but no
    cursor/raw-revision evidence exists yet because nothing has drained it --
    the rollup must report that honestly (an unseen/pending stage) rather
    than a stage that implies convergence.
    """

    hermes_root = workspace_env["data_root"] / "hermes-lagging"
    hermes_root.mkdir(parents=True)
    with sqlite3.connect(hermes_root / "state.db") as conn:
        conn.executescript(_STATE_DB_SCHEMA)
        conn.execute(
            "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
            "VALUES ('root', 'cli', '{}', 1.0, 8.0, 'completed', 'root')"
        )

    health = build_hermes_integration_health(workspace_env["archive_root"], hermes_root=hermes_root)

    assert health.enabled is True
    assert len(health.sources) == 1
    source = health.sources[0]
    assert source.source_ref == "state.db"
    # Anti-vacuity: a real named-source freshness projection over an
    # ingested-but-never-drained file reports a real pipeline stage short of
    # "searchable" -- swapping in a stub/constant projection instead of the
    # real ``project_named_source_freshness`` call would make this stage
    # report "unknown" and fail this assertion.
    assert source.stage in {"unseen", "acquired-unparsed", "parsed-unindexed", "indexed-unconverged"}
    assert source.parse_state != "parsed"
