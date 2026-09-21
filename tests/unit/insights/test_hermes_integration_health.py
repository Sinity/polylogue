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

import polylogue.analysis.hermes_integration_health as hermes_health
import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.analysis.hermes_integration_health import build_hermes_integration_health
from polylogue.sources.hooks import append_hook_event
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.hermes_lifecycle import TOOL_START
from tests.infra.hook_carriers import materialize_hook_carriers

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


def test_deferred_convergence_debt_is_reported_without_degrading(workspace_env: dict[str, Path]) -> None:
    hermes_root = workspace_env["data_root"] / "hermes-deferred-debt"
    hermes_root.mkdir(parents=True)

    health = build_hermes_integration_health(
        workspace_env["archive_root"],
        hermes_root=hermes_root,
        convergence_debt_deferred_count=2,
    )

    assert health.convergence_debt_failed_count == 0
    assert health.convergence_debt_deferred_count == 2
    assert health.verdict == "healthy"


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


def _seed_hermes_state_db(root: Path) -> Path:
    """Write one minimal Hermes ``state.db`` under ``root`` and return its path."""

    source_path = root / "state.db"
    with sqlite3.connect(source_path) as conn:
        conn.executescript(_STATE_DB_SCHEMA)
        conn.execute(
            "INSERT INTO sessions (id, source, model_config, started_at, ended_at, end_reason, title) "
            "VALUES ('root', 'cli', '{}', 1.0, 8.0, 'completed', 'root')"
        )
    return source_path


def test_unreadable_source_tier_is_unavailable_not_healthy(workspace_env: dict[str, Path]) -> None:
    """A durable spool that cannot be read is reported unavailable, never healthy.

    The Hermes root is deliberately empty, so no freshness projection runs and
    nothing else in the rollup can notice the damage: the *only* probe that
    touches the corrupted ``source.db`` is the recent-session sample. Compare
    with ``test_deferred_convergence_debt_is_reported_without_degrading``,
    which uses the same empty root over an intact archive and must stay
    ``healthy`` -- the corrupted tier is the single differing input.

    Anti-vacuity: delete the ``sample_unmeasured is not None`` branch in
    ``build_hermes_integration_health`` (or make
    ``_recent_hermes_session_native_ids`` return a bare empty tuple again) and
    the unreadable tier becomes indistinguishable from an empty one, the
    verdict falls through to ``healthy``, and this test is red.
    """

    hermes_root = workspace_env["data_root"] / "hermes-unreadable-source"
    hermes_root.mkdir(parents=True)
    archive_root = workspace_env["archive_root"]
    for suffix in ("-wal", "-shm"):
        sidecar = archive_root / f"source.db{suffix}"
        if sidecar.exists():
            sidecar.unlink()
    (archive_root / "source.db").write_bytes(b"this is not a sqlite database")

    health = build_hermes_integration_health(archive_root, hermes_root=hermes_root)

    assert health.enabled is True
    assert health.verdict == "unavailable"
    assert health.measurement_coverage.complete is False
    assert any("source tier" in reason for reason in health.measurement_coverage.unmeasured_reasons)
    # The gap reaches the operator-facing caveat list too, not just the record.
    assert any("source tier" in caveat for caveat in health.caveats)


def test_absent_index_tier_is_unavailable_not_healthy(workspace_env: dict[str, Path]) -> None:
    """Lifecycle debt "reconciled against an empty snapshot" is not measured debt.

    With real Hermes hook events in the durable spool there is a session
    sample to reconcile, but no ``index.db`` to reconcile it against, so
    ``sessions_checked`` stays 0 and ``unpaired_event_count`` stays 0. Those
    zeros previously read as clean.

    Anti-vacuity: delete the ``lifecycle_unmeasured``/``lifecycle_unsampled``
    bookkeeping in the ``else`` branch of ``_sample_session_debt``'s index-tier
    probe -- so the absent tier contributes only a caveat string and the
    verdict falls through to ``healthy`` -- and this test is red.
    """

    hermes_root = workspace_env["data_root"] / "hermes-absent-index"
    hermes_root.mkdir(parents=True)
    archive_root = workspace_env["archive_root"]

    append_hook_event(
        event_id="e1",
        provider="hermes",
        event_type=TOOL_START,
        session_id="conv-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_call_id": "call-1", "message_id": "m1"},
        root=archive_root / "hooks",
    )
    assert materialize_hook_carriers(archive_root) == 1

    for suffix in ("", "-wal", "-shm"):
        tier = archive_root / f"index.db{suffix}"
        if tier.exists():
            tier.unlink()

    health = build_hermes_integration_health(archive_root, hermes_root=hermes_root)

    assert health.enabled is True
    assert health.verdict == "unavailable"
    # The zeros are still reported -- they are just no longer read as clean.
    assert health.lifecycle_debt.sessions_checked == 0
    assert health.lifecycle_debt.unpaired_event_count == 0
    assert health.measurement_coverage.lifecycle_sessions_sampled == 0
    assert health.measurement_coverage.lifecycle_sessions_unsampled == 1
    assert any("index tier" in reason for reason in health.measurement_coverage.unmeasured_reasons)


def test_raising_freshness_projection_is_unavailable_not_healthy(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A freshness projection that raises leaves a source unmeasured, not fine.

    On the exception path the source row keeps its pre-probe placeholders --
    ``operational_state == "unknown"`` and ``projection_error_count == 0`` --
    and the verdict's tests are for ``"degraded"`` and a nonzero error count,
    so neither could ever see the failure.

    Anti-vacuity: delete the ``sources_unprojected``/``unmeasured_reasons``
    bookkeeping from that ``except`` clause in
    ``build_hermes_integration_health``, leaving only the caveat, and the
    verdict falls through to ``healthy`` -- this test is red.
    """

    hermes_root = workspace_env["data_root"] / "hermes-freshness-raises"
    hermes_root.mkdir(parents=True)
    _seed_hermes_state_db(hermes_root)

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError("freshness projection unavailable")

    monkeypatch.setattr(hermes_health, "project_named_source_freshness", _raise)

    health = build_hermes_integration_health(workspace_env["archive_root"], hermes_root=hermes_root)

    assert health.enabled is True
    assert len(health.sources) == 1
    source = health.sources[0]
    # These are exactly the placeholders the verdict cannot read as a failure.
    assert source.operational_state == "unknown"
    assert source.projection_error_count == 0
    assert health.verdict == "unavailable"
    assert health.measurement_coverage.sources_projected == 0
    assert health.measurement_coverage.sources_unprojected == 1
    assert any("freshness projection failed" in reason for reason in health.measurement_coverage.unmeasured_reasons)


def test_fully_measured_archive_still_reaches_healthy(workspace_env: dict[str, Path]) -> None:
    """The totality guard must not turn every quiet archive into unavailable.

    An intact archive with an empty Hermes root measured everything it
    attempted and found nothing: coverage is complete and the verdict stays
    ``healthy``. This is the control for the three tests above -- without it
    they would also pass under a verdict that always answered ``unavailable``.
    """

    hermes_root = workspace_env["data_root"] / "hermes-quiet-but-intact"
    hermes_root.mkdir(parents=True)

    health = build_hermes_integration_health(workspace_env["archive_root"], hermes_root=hermes_root)

    assert health.verdict == "healthy"
    assert health.measurement_coverage.complete is True
    assert health.measurement_coverage.unmeasured_reasons == ()
