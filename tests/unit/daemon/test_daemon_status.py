from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from polylogue.browser_capture.receiver import BrowserCaptureReceiverConfig
from polylogue.core.json import JSONDocument
from polylogue.daemon import status as status_module
from polylogue.daemon.status import (
    _insight_freshness_info,
    browser_capture_status_payload,
    build_daemon_status,
    daemon_status_payload,
    format_daemon_status_lines,
)
from polylogue.daemon.status_snapshot import (
    configure_runtime_components,
    get_status_snapshot_payload,
    refresh_status_snapshot,
)
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    record_daemon_stage_event,
    record_ingest_attempt,
    upsert_ingest_cursor,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.frozen_clock import FrozenClock


def test_status_snapshot_serves_cached_payload_without_rebuilding_status(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: JSONDocument = {"ok": True, "daemon_liveness": True, "checked_at": "cached"}
    refresh_status_snapshot(payload=payload)

    monkeypatch.setattr(
        "polylogue.daemon.status.daemon_status_payload",
        lambda: (_ for _ in ()).throw(AssertionError("request path must not rebuild rich status")),
    )

    result = get_status_snapshot_payload()

    assert result["checked_at"] == "cached"
    snapshot = result["status_snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["state"] == "fresh"


def test_status_snapshot_minimal_refresh_stays_request_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "index.db"
    db.touch()
    monkeypatch.setattr("polylogue.daemon.status_snapshot.active_index_db_path", lambda: db)
    monkeypatch.setattr(
        "polylogue.daemon.status.daemon_status_payload",
        lambda: (_ for _ in ()).throw(AssertionError("minimal snapshot must not build rich status")),
    )

    snapshot = refresh_status_snapshot(rich=False)

    status_snapshot = snapshot.payload["status_snapshot"]
    assert isinstance(status_snapshot, dict)
    assert status_snapshot["state"] == "minimal"
    assert snapshot.payload["db_path"] == str(db)


def test_status_snapshot_minimal_refresh_prefers_archive(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_home = workspace_env["data_root"] / "polylogue"
    archive_root = workspace_env["archive_root"]
    data_home.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    db_anchor = data_home / "index.db"
    archive_db = archive_root / "index.db"
    db_anchor.touch()
    archive_db.touch()
    monkeypatch.setattr(
        "polylogue.daemon.status.daemon_status_payload",
        lambda: (_ for _ in ()).throw(AssertionError("minimal snapshot must not build rich status")),
    )

    snapshot = refresh_status_snapshot(rich=False)

    assert snapshot.payload["db_path"] == str(archive_db)


def test_status_snapshot_uses_runtime_browser_capture_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "index.db"
    db.touch()
    spool = tmp_path / "browser-capture"
    monkeypatch.setattr("polylogue.daemon.status_snapshot.active_index_db_path", lambda: db)

    configure_runtime_components(
        api_enabled=True,
        watcher_enabled=True,
        watcher_roots=("/watch/a", "/watch/b"),
        browser_capture_enabled=True,
        browser_capture_spool_path=spool,
    )

    snapshot = refresh_status_snapshot(rich=False)

    assert snapshot.payload["browser_capture_active"] is True
    assert snapshot.payload["watcher_roots"] == ["/watch/a", "/watch/b"]
    component_state = snapshot.payload["component_state"]
    assert isinstance(component_state, dict)
    assert component_state["browser_capture"] == "running"
    browser_capture = snapshot.payload["browser_capture"]
    assert isinstance(browser_capture, dict)
    assert browser_capture["active"] is True
    readiness = snapshot.payload["component_readiness"]
    assert isinstance(readiness, dict)
    browser_readiness = readiness["browser_capture"]
    assert isinstance(browser_readiness, dict)
    assert browser_readiness["state"] == "ready"


def test_status_snapshot_reports_disk_free_for_archive_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "missing-archive" / "index.db"
    monkeypatch.setattr("polylogue.daemon.status_snapshot.active_index_db_path", lambda: db)

    snapshot = refresh_status_snapshot(rich=False)

    assert cast(int, snapshot.payload["disk_free_bytes"]) > 0


def test_status_snapshot_marks_disabled_browser_capture_inactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "index.db"
    db.touch()
    monkeypatch.setattr("polylogue.daemon.status_snapshot.active_index_db_path", lambda: db)

    configure_runtime_components(
        api_enabled=True,
        watcher_enabled=True,
        browser_capture_enabled=False,
    )

    snapshot = refresh_status_snapshot(rich=False)

    assert snapshot.payload["browser_capture_active"] is False
    component_state = snapshot.payload["component_state"]
    assert isinstance(component_state, dict)
    assert component_state["browser_capture"] == "stopped"
    browser_capture = snapshot.payload["browser_capture"]
    assert isinstance(browser_capture, dict)
    assert browser_capture["active"] is False
    readiness = snapshot.payload["component_readiness"]
    assert isinstance(readiness, dict)
    browser_readiness = readiness["browser_capture"]
    assert isinstance(browser_readiness, dict)
    assert browser_readiness["state"] == "missing"


def test_status_snapshot_refresh_default_builds_rich_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: JSONDocument = {
        "ok": True,
        "daemon_liveness": True,
        "checked_at": "rich",
        "raw_materialization_readiness": {"total": 2},
        "component_readiness": {
            "raw_materialization": {
                "component": "raw_materialization",
                "scope": "archive",
                "state": "stale",
                "summary": "raw evidence pending materialization",
                "counts": {"total": 2},
                "caveats": [],
                "repair_hint": "polylogued run",
                "evidence_refs": [],
            }
        },
    }
    build = Mock(return_value=payload)
    monkeypatch.setattr("polylogue.daemon.status.daemon_status_payload", build)

    snapshot = refresh_status_snapshot()

    assert build.call_count == 1
    assert snapshot.payload["checked_at"] == "rich"
    assert snapshot.payload["raw_materialization_readiness"] == {"total": 2}


def test_build_daemon_status_reports_failed_live_cursor_files(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    failed = tmp_path / "failed.jsonl"
    failed.write_text('{"bad":true}\n')
    cursor = CursorStore(db)
    cursor.mark_failed(failed)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        status = build_daemon_status(sources=())

    assert status.failing_files == [str(failed)]
    assert status.live_cursor.tracked_file_count == 1
    assert status.live_cursor.failed_file_count == 1
    assert status.live_cursor.in_backoff_file_count == 1
    assert status.live_cursor.failing_files[0].source_path == str(failed)
    assert status.live_cursor.failing_files[0].failure_count == 1
    assert status.live_cursor.failing_files[0].next_retry_at is not None


def test_daemon_status_redacts_default_browser_capture_spool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_spool = tmp_path / "browser-capture"
    monkeypatch.setattr(
        BrowserCaptureReceiverConfig,
        "default",
        classmethod(lambda cls: BrowserCaptureReceiverConfig(spool_path=expected_spool)),
    )

    payload = daemon_status_payload(sources=())

    assert payload["browser_capture_active"] is True
    browser_capture = payload["browser_capture"]
    assert isinstance(browser_capture, dict)
    assert browser_capture["spool_ready"] is True
    assert "spool_path" not in browser_capture
    component_state = payload["component_state"]
    assert isinstance(component_state, dict)
    assert component_state["browser_capture"] == "running"


def test_browser_capture_status_payload_can_include_spool_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_spool = tmp_path / "browser-capture"
    monkeypatch.setattr(
        BrowserCaptureReceiverConfig,
        "default",
        classmethod(lambda cls: BrowserCaptureReceiverConfig(spool_path=expected_spool)),
    )

    payload = browser_capture_status_payload(include_spool_path=True)

    assert payload["spool_path"] == str(expected_spool)


def test_daemon_status_honors_explicit_disabled_browser_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_spool = tmp_path / "browser-capture"
    monkeypatch.setattr(
        BrowserCaptureReceiverConfig,
        "default",
        classmethod(lambda cls: BrowserCaptureReceiverConfig(spool_path=expected_spool)),
    )

    status = build_daemon_status(sources=(), browser_capture_enabled=False)

    assert status.browser_capture_active is False
    assert status.component_state.browser_capture == "stopped"


def test_daemon_status_payload_and_plain_output_include_failed_files(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    failed = tmp_path / "failed.jsonl"
    failed.write_text('{"bad":true}\n')
    cursor = CursorStore(db)
    cursor.mark_failed(failed)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    assert payload["failing_files"] == [str(failed)]
    live_cursor = payload["live_cursor"]
    assert isinstance(live_cursor, dict)
    assert live_cursor["tracked_file_count"] == 1
    assert live_cursor["failed_file_count"] == 1
    failing_files = live_cursor["failing_files"]
    assert isinstance(failing_files, list)
    first_failure = failing_files[0]
    assert isinstance(first_failure, dict)
    assert first_failure["source_path"] == str(failed)
    lines = format_daemon_status_lines(payload)
    assert "Live cursor: 1 tracked, 1 failed, 0 excluded, 0 retry due, 1 in backoff" in lines
    assert "Failing files: 1" in lines
    assert f"  {failed}" in lines


def test_daemon_status_payload_links_unified_archive_debt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.surfaces.payloads import (
        ArchiveDebtListPayload,
        ArchiveDebtRowPayload,
        ArchiveDebtTotalsPayload,
    )

    payload = ArchiveDebtListPayload(
        generated_at="2026-06-20T00:00:00+00:00",
        archive_root=str(tmp_path),
        rows=(
            ArchiveDebtRowPayload(
                debt_ref="debt:embedding:catchup:backlog",
                kind="embedding",
                stage="catchup",
                subject_ref="embedding:pending",
                severity="warning",
                status="actionable",
                owner="daemon",
                summary="3 session(s) pending embedding catch-up",
            ),
        ),
        totals=ArchiveDebtTotalsPayload(total=1, warning=1, actionable=1),
    )

    monkeypatch.setattr("polylogue.daemon.status.archive_root", lambda: tmp_path)
    monkeypatch.setattr("polylogue.operations.archive_debt.archive_debt_list", lambda **_kwargs: payload)

    with (
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        status_payload = daemon_status_payload(sources=())

    archive_debt = cast(dict[str, object], status_payload["archive_debt"])
    assert archive_debt["endpoint"] == "/api/archive-debt"
    assert archive_debt["available"] is True
    totals = cast(dict[str, object], archive_debt["totals"])
    assert {key: totals[key] for key in ("total", "critical", "warning", "info", "actionable", "blocked")} == {
        "total": 1,
        "critical": 0,
        "warning": 1,
        "info": 0,
        "actionable": 1,
        "blocked": 0,
    }
    rows = cast(list[dict[str, object]], archive_debt["rows"])
    assert rows[0]["debt_ref"] == "debt:embedding:catchup:backlog"


def test_daemon_status_marks_raw_materialization_debt_not_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("polylogue.daemon.status.archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        "polylogue.daemon.status.raw_materialization_readiness_snapshot",
        lambda _root: {
            "available": True,
            "classification": "not_run",
            "precision": "raw_id_join_gap",
            "raw_artifact_count": 300,
            "materialized_raw_artifact_count": 62,
            "archive_session_count": 80,
            "join_gap_count": 238,
            "total": 238,
            "critical": 0,
            "warning": 0,
            "actionable": 0,
            "blocked": 0,
            "classified": 0,
            "unchecked": 238,
            "affected_total": 238,
            "affected_actionable": 0,
            "affected_blocked": 0,
            "affected_open": 0,
            "affected_classified": 0,
            "affected_unchecked": 238,
            "category_counts": {"raw_id_join_gap": 238},
            "source_family_counts": {"aistudio-drive": 238},
        },
    )

    with (
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={"messages_ready": True}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        status_payload = daemon_status_payload(sources=())

    materialization = cast(dict[str, object], status_payload["raw_materialization_readiness"])
    assert materialization["total"] == 238
    assert materialization["raw_artifact_count"] == 300
    assert materialization["materialized_raw_artifact_count"] == 62
    assert materialization["warning"] == 0
    assert materialization["affected_total"] == 238
    assert materialization["affected_actionable"] == 0
    assert materialization["affected_unchecked"] == 238
    assert materialization["affected_open"] == 0
    assert materialization["category_counts"] == {"raw_id_join_gap": 238}
    assert materialization["source_family_counts"] == {"aistudio-drive": 238}

    readiness = cast(dict[str, object], status_payload["component_readiness"])
    raw_component = cast(dict[str, object], readiness["raw_materialization"])
    assert raw_component["state"] == "degraded"
    assert raw_component["summary"] == "raw/index join gaps need classification"
    assert raw_component["repair_hint"] == "polylogued run"
    counts = cast(dict[str, object], raw_component["counts"])
    assert counts["total"] == 238
    assert counts["affected_total"] == 238
    assert counts["affected_unchecked"] == 238
    metadata = cast(dict[str, object], raw_component["metadata"])
    assert metadata["category_counts"] == {"raw_id_join_gap": 238}

    lines = format_daemon_status_lines(status_payload)
    assert "Raw materialization: 62/300 materialized; 238 raw/index join gap(s) need classification" in lines


def test_daemon_status_preserves_lost_source_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sample = {
        "session_id": "codex-session:native-1",
        "origin": "codex-session",
        "native_id": "native-1",
        "missing_raw_id": "raw-missing",
        "evidence_status": "lost_source_evidence",
    }
    monkeypatch.setattr("polylogue.daemon.status.archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        "polylogue.daemon.status.raw_materialization_readiness_snapshot",
        lambda _root: {
            "available": True,
            "classification": "cheap_projection",
            "precision": "raw_id_join_gap",
            "total": 0,
            "lost_source_evidence_count": 1,
            "lost_source_evidence_samples": [sample],
        },
    )

    with (
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={"messages_ready": True}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        status_payload = daemon_status_payload(sources=())

    materialization = cast(dict[str, object], status_payload["raw_materialization_readiness"])
    assert materialization["lost_source_evidence_count"] == 1
    assert materialization["lost_source_evidence_samples"] == [sample]
    raw_component = cast(dict[str, Any], status_payload["component_readiness"])["raw_materialization"]
    assert raw_component["state"] == "blocked"
    assert raw_component["summary"] == "source evidence missing"
    assert raw_component["repair_hint"] == "restore exact raw artifact"


def test_daemon_status_payload_maps_component_readiness(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    db.touch()

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status.index_db_path", return_value=db),
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=True),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch(
            "polylogue.daemon.status._fts_readiness_info",
            return_value={
                "indexed_surface": "messages_fts",
                "messages_ready": False,
                "message_indexed_count": 4,
                "message_indexable_count": 10,
                "coverage_pct": 40.0,
            },
        ),
        patch(
            "polylogue.daemon.status._insight_freshness_info",
            return_value={"sessions_with_profiles": 7, "total_sessions": 10},
        ),
        patch(
            "polylogue.daemon.status.embedding_readiness_info",
            return_value={
                "embedding_config_enabled": True,
                "embedding_has_voyage_key": True,
                "embedding_status": "partial",
                "embedding_freshness_status": "stale",
                "embedding_retrieval_ready": True,
                "embedding_pending_count": 2,
                "embedding_pending_message_count": 12,
                "embedding_pending_message_count_exact": True,
                "embedding_stale_count": 3,
                "embedding_coverage_percent": 70.0,
                "embedding_failure_count": 0,
            },
        ),
    ):
        payload = daemon_status_payload(sources=())

    readiness = payload["component_readiness"]
    assert isinstance(readiness, dict)
    search = readiness["search"]
    session_profiles = readiness["session_profiles"]
    embeddings = readiness["embeddings"]
    api = readiness["daemon_api"]
    ingest = readiness["daemon_ingest"]
    assert isinstance(search, dict)
    assert isinstance(session_profiles, dict)
    assert isinstance(embeddings, dict)
    assert isinstance(api, dict)
    assert isinstance(ingest, dict)
    search_counts = cast(dict[str, object], search["counts"])
    assert search["state"] == "stale"
    assert search_counts["message_indexed_count"] == 4
    assert search["repair_hint"] == "polylogued run"
    profile_counts = cast(dict[str, object], session_profiles["counts"])
    assert session_profiles["state"] == "degraded"
    assert session_profiles["scope"] == "insights"
    assert profile_counts["sessions_with_profiles"] == 7
    assert profile_counts["missing_profiles"] == 3
    assert session_profiles["repair_hint"] == "polylogued run"
    assert embeddings["state"] == "stale"
    assert embeddings["scope"] == "semantic"
    embedding_counts = cast(dict[str, object], embeddings["counts"])
    assert embedding_counts["pending_messages"] == 12
    assert embedding_counts["pending_messages_exact"] is True
    assert embeddings["repair_hint"] == "polylogue ops embed backfill"
    assert api["state"] == "ready"
    assert ingest["state"] == "ready"


def test_daemon_status_prefers_archive_ops_live_cursor(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    failed = tmp_path / "failed.jsonl"
    failed.write_text('{"bad":true}\n')
    cursor = CursorStore(db)
    cursor.mark_failed(failed)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        status = build_daemon_status(sources=())

    assert status.live_cursor.failed_file_count == 1
    assert status.live_cursor.failing_files[0].source_path == str(failed)


def test_plain_daemon_status_reports_bounded_embedding_pending_messages() -> None:
    payload: JSONDocument = {
        "embedding_readiness": {
            "embedding_enabled": False,
            "embedding_has_voyage_key": True,
            "embedding_status": "none",
            "embedding_freshness_status": "none",
            "embedding_retrieval_ready": False,
            "embedding_pending_count": 7,
            "embedding_pending_message_count": None,
            "embedding_pending_message_count_exact": False,
        }
    }

    lines = format_daemon_status_lines(payload)

    assert (
        "Embeddings: disabled (key present; none/none, not ready; 7 pending convs, pending msgs not calculated)"
        in lines
    )


def test_daemon_status_caps_failed_file_samples(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    cursor = CursorStore(db)
    for index in range(55):
        failed = tmp_path / f"failed-{index:02d}.jsonl"
        failed.write_text('{"bad":true}\n')
        cursor.mark_failed(failed)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    failing_files = payload["failing_files"]
    assert isinstance(failing_files, list)
    assert len(failing_files) == 50
    live_cursor = payload["live_cursor"]
    assert isinstance(live_cursor, dict)
    assert live_cursor["failed_file_count"] == 55
    assert live_cursor["sampled_file_count"] == 50
    assert live_cursor["omitted_file_count"] == 5
    lines = format_daemon_status_lines(payload)
    assert "Failing files: 50 shown, 5 omitted" in lines


def test_daemon_status_reports_live_ingest_attempts(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    attempt_id = cursor.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    cursor.update_ingest_attempt(
        attempt_id,
        phase="full_parse",
        succeeded_file_count=0,
        failed_file_count=0,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        parse_time_s=0.0,
        current_source="codex",
        current_path=source,
        rss_current_mb=42.0,
        cgroup_path="/user.slice/test.scope",
        cgroup_memory_current_mb=2048.0,
        cgroup_memory_peak_mb=4096.0,
    )
    cursor.record_ingest_stage_event(
        attempt_id,
        phase="full_parse",
        status="running",
        queued_file_count=1,
        needed_file_count=1,
        skipped_file_count=0,
        succeeded_file_count=0,
        failed_file_count=0,
        input_bytes=source.stat().st_size,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        archive_write_bytes_delta=4096,
        parse_time_s=0.0,
        total_time_s=2.0,
        current_source="codex",
        current_path=source,
        stage_timings_json='{"full_parse": 1.25, "convergence": 0.75}',
    )

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    attempts = payload["live_ingest_attempts"]
    assert isinstance(attempts, dict)
    assert attempts["running_count"] == 1
    recent = attempts["recent"]
    assert isinstance(recent, list)
    latest = recent[0]
    assert isinstance(latest, dict)
    assert latest["phase"] == "full_parse"
    assert latest["current_path"] == str(source)
    assert latest["rss_current_mb"] == 42.0
    assert latest["cgroup_path"] == "/user.slice/test.scope"
    assert latest["cgroup_memory_current_mb"] == 2048.0
    assert latest["cgroup_memory_peak_mb"] == 4096.0
    assert latest["total_read_bytes"] == 0
    assert latest["read_amplification"] == 0.0
    assert latest["files_per_second"] == 0.0
    assert latest["archive_write_bytes_delta"] == 4096
    assert latest["total_time_s"] == 2.0
    assert latest["stage_timings_s"] == {"full_parse": 1.25, "convergence": 0.75}
    catchup = payload["catchup"]
    assert isinstance(catchup, dict)
    assert catchup["mode"] == "catching_up"
    assert catchup["current_phase"] == "full_parse"
    assert catchup["queued_file_count"] == 1
    assert catchup["read_amplification"] == 0.0
    recent_events = catchup["recent_events"]
    assert isinstance(recent_events, list)
    first_event = recent_events[0]
    assert isinstance(first_event, dict)
    assert first_event["current_path"] == str(source)
    lines = format_daemon_status_lines(payload)
    assert "Live ingest attempts: 1 running" in lines
    assert "  latest: running full_parse 0/1 files" in lines
    assert "  workload: read amp 0.00x, 0.00 MiB/s source, 0.00 files/s" in lines
    assert "  memory: cgroup 2048.0 MiB peak 4096.0 MiB" in lines
    assert "Catch-up: catching_up 0/1 files, read amp 0.0x" in lines


def test_daemon_status_reads_ops_tier_from_archive_tiers(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    with sqlite3.connect(ops_db) as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        upsert_ingest_cursor(
            conn,
            source_path="/tmp/v1-source.jsonl",
            updated_at_ms=1_700_000_000_000,
            failure_count=2,
            next_retry_at="2026-01-01T00:00:00+00:00",
        )
        record_ingest_attempt(
            conn,
            attempt_id="v1-attempt",
            status="running",
            source_path="/tmp/v1-source.jsonl",
            origin="codex-session",
            phase="full_parse",
            started_at_ms=1_700_000_000_000,
            heartbeat_at_ms=1_700_000_002_500,
            parsed_raw_count=7,
            materialized_count=3,
        )
        record_daemon_stage_event(
            conn,
            attempt_id="v1-attempt",
            stage="full_parse",
            status="running",
            observed_at_ms=1_700_000_002_600,
            payload={
                "queued_file_count": 7,
                "needed_file_count": 7,
                "succeeded_file_count": 3,
                "failed_file_count": 1,
                "input_bytes": 1024,
                "source_payload_read_bytes": 512,
                "cursor_fingerprint_read_bytes": 256,
                "archive_write_bytes_delta": 4096,
                "total_time_s": 2.5,
                "stage_timings_json": '{"full_parse": 1.25, "convergence": 0.75}',
                "current_source": "codex-session",
                "current_path": "/tmp/v1-source.jsonl",
                "storage_route": "archive_full",
                "storage_tiers": "source,index",
                "payload_available_file_count": 6,
                "payload_unavailable_file_count": 1,
                "written_raw_count": 6,
                "rss_current_mb": 42.0,
                "cgroup_path": "/user.slice/v1.scope",
                "cgroup_memory_current_mb": 2048.0,
            },
            event_id="stage-1",
        )

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    live_cursor = payload["live_cursor"]
    assert isinstance(live_cursor, dict)
    assert live_cursor["tracked_file_count"] == 1
    assert live_cursor["failed_file_count"] == 1
    assert payload["failing_files"] == ["/tmp/v1-source.jsonl"]

    attempts = payload["live_ingest_attempts"]
    assert isinstance(attempts, dict)
    assert attempts["running_count"] == 1
    recent = attempts["recent"]
    assert isinstance(recent, list)
    latest = recent[0]
    assert isinstance(latest, dict)
    assert latest["attempt_id"] == "v1-attempt"
    assert latest["phase"] == "full_parse"
    assert latest["current_source"] == "codex-session"
    assert latest["current_path"] == "/tmp/v1-source.jsonl"
    assert latest["queued_file_count"] == 7
    assert latest["succeeded_file_count"] == 3
    assert latest["failed_file_count"] == 1
    assert latest["total_read_bytes"] == 768
    assert latest["read_amplification"] == 0.75
    assert latest["archive_write_bytes_delta"] == 4096
    assert latest["total_time_s"] == 2.5
    assert latest["stage_timings_s"] == {"full_parse": 1.25, "convergence": 0.75}
    assert latest["storage_route"] == "archive_full"
    assert latest["storage_tiers"] == "source,index"
    assert latest["payload_available_file_count"] == 6
    assert latest["payload_unavailable_file_count"] == 1
    assert latest["written_raw_count"] == 6
    assert latest["rss_current_mb"] == 42.0
    assert latest["cgroup_path"] == "/user.slice/v1.scope"
    assert latest["cgroup_memory_current_mb"] == 2048.0
    lines = format_daemon_status_lines(payload)
    assert any(line.startswith("Live ingest attempts: 1 running") for line in lines)
    assert any(line.startswith("  latest: running ") and line.endswith("full_parse 3/7 files") for line in lines)
    assert "  storage route: archive_full (source,index), 1 payload-unavailable" in lines
    assert "  memory: cgroup 2048.0 MiB" in lines


def test_daemon_status_reads_ops_tier_when_archive_db_exists(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    attempt_id = cursor.begin_ingest_attempt(paths=[source], input_bytes=source.stat().st_size, queued_file_count=1)
    cursor.update_ingest_attempt(
        attempt_id,
        phase="legacy_phase",
        current_source="legacy-source",
        current_path=source,
    )

    ops_db = tmp_path / "ops.db"
    with sqlite3.connect(ops_db) as conn:
        record_ingest_attempt(
            conn,
            attempt_id=attempt_id,
            status="running",
            source_path="/tmp/v1-preferred.jsonl",
            origin="codex-session",
            phase="archive_phase",
            started_at_ms=1_700_000_000_000,
            heartbeat_at_ms=1_700_000_001_000,
            parsed_raw_count=9,
            materialized_count=4,
        )

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    attempts = payload["live_ingest_attempts"]
    assert isinstance(attempts, dict)
    recent = attempts["recent"]
    assert isinstance(recent, list)
    latest = recent[0]
    assert isinstance(latest, dict)
    assert latest["phase"] == "archive_phase"
    assert latest["current_source"] == "codex-session"
    assert latest["current_path"] == "/tmp/v1-preferred.jsonl"
    assert latest["queued_file_count"] == 9
    assert latest["succeeded_file_count"] == 4


def test_daemon_status_reports_convergence_debt_separately(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="source_path",
        subject_id=str(source),
        error="legacy payload missing provenance",
    )

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    convergence = payload["convergence"]
    assert isinstance(convergence, dict)
    assert convergence["failed_count"] == 1
    stages = convergence["stage_summaries"]
    assert isinstance(stages, list)
    first_stage = stages[0]
    assert isinstance(first_stage, dict)
    assert first_stage["stage"] == "insights"
    recent = convergence["recent"]
    assert isinstance(recent, list)
    first_recent = recent[0]
    assert isinstance(first_recent, dict)
    assert first_recent["subject_id"] == str(source)
    lines = format_daemon_status_lines(payload)
    assert "Convergence debt: 1 failed, 0 retry due" in lines
    assert "  insights: 1 failed, 0 retry due" in lines


def test_archive_storage_info_uses_configured_archive_root(tmp_path: Path) -> None:
    default_root = tmp_path / "default"
    active_root = tmp_path / "active"
    default_root.mkdir()
    active_root.mkdir()
    initialize_archive_database(default_root / "ops.db", ArchiveTier.OPS)
    db_anchor = active_root / "custom.sqlite"
    db_anchor.touch()

    with (
        patch("polylogue.daemon.status.archive_root", return_value=default_root),
        patch("polylogue.daemon.status.db_path", return_value=db_anchor),
        patch("polylogue.daemon.status.index_db_path", return_value=default_root / "index.db"),
    ):
        storage = status_module._archive_storage_info()

    assert storage.archive_root == str(default_root)
    assert storage.configured_archive_root == str(default_root)
    assert storage.archive_root_matches_configured is True
    assert storage.active_db_path == str(default_root / "index.db")
    assert storage.active_store == "empty"
    assert storage.present_tiers == ["ops"]
    assert storage.missing_tiers == ["source", "index", "embeddings", "user"]


def test_build_daemon_status_downgrades_archive_ready_for_raw_materialization_debt(tmp_path: Path) -> None:
    storage = status_module.ArchiveStorageStatus(
        active_store="archive_file_set",
        active_db_path=str(tmp_path / "index.db"),
        archive_root=str(tmp_path),
        configured_archive_root=str(tmp_path),
        archive_ready=True,
        archive_materialization_ready=True,
        final_shape_ready=True,
        archive_schema_ready=True,
        present_tiers=["source", "index", "embeddings", "user", "ops"],
    )
    raw_readiness = status_module.RawMaterializationReadiness(
        available=True,
        total=1,
        warning=1,
        actionable=1,
        affected_total=4,
        affected_actionable=4,
    )

    with (
        patch("polylogue.daemon.status._active_status_db_path", return_value=tmp_path / "index.db"),
        patch("polylogue.daemon.status._archive_storage_info", return_value=storage),
        patch("polylogue.daemon.status._raw_materialization_readiness_info", return_value=raw_readiness),
        patch("polylogue.daemon.status._db_size_info", return_value={}),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
        patch("polylogue.daemon.status._live_cursor_summary_info", return_value=status_module.LiveCursorSummary()),
        patch(
            "polylogue.daemon.status._live_ingest_attempt_summary_info",
            return_value=status_module.LiveIngestAttemptSummary(),
        ),
        patch(
            "polylogue.daemon.status.convergence_debt_summary_info",
            return_value=status_module.ConvergenceDebtSummary(),
        ),
        patch("polylogue.daemon.status.cursor_lag_summary_info", return_value=status_module.CursorLagSummary()),
        patch("polylogue.daemon.status.catchup_status_info", return_value=status_module.CatchupStatus()),
        patch("polylogue.daemon.status._raw_failure_info", return_value={}),
        patch("polylogue.daemon.status.embedding_readiness_info", return_value={}),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
    ):
        status = build_daemon_status(sources=(), browser_capture_enabled=False)

    assert status.archive_storage.archive_schema_ready is True
    assert status.archive_storage.archive_materialization_ready is False
    assert status.archive_storage.archive_ready is False
    storage_component = cast(dict[str, object], status.component_readiness["archive_storage"])
    assert storage_component["state"] == "stale"
    assert storage_component["repair_hint"] == "polylogued run"
    assert storage_component["caveats"] == ["materialization_pending"]
    raw_component = cast(dict[str, object], status.component_readiness["raw_materialization"])
    assert raw_component["state"] == "stale"


def test_daemon_status_payload_reuses_bounded_probe_results(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    db.touch()
    db_info = Mock(
        return_value={
            "db_path": str(db),
            "db_size_bytes": 11,
            "wal_size_bytes": 7,
            "disk_free_bytes": 99,
        }
    )
    blob_info = Mock(return_value=0)
    fts_info = Mock(
        return_value={
            "indexed_surface": "messages_fts",
            "messages_ready": True,
            "session_work_events_ready": True,
            "threads_ready": True,
            "invariant_ready": True,
            "message_indexed_count": 4,
            "message_indexable_count": 4,
            "coverage_pct": 100.0,
            "surfaces": {
                "messages_fts": {
                    "ready": True,
                    "source_rows": 4,
                    "indexed_rows": 4,
                    "missing_rows": 0,
                }
            },
        }
    )
    freshness_info = Mock(return_value={"sessions_with_profiles": 3, "total_sessions": 4})

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=True),
        patch("polylogue.daemon.status._db_size_info", db_info),
        patch("polylogue.daemon.status._blob_size_info", blob_info),
        patch("polylogue.daemon.status._fts_readiness_info", fts_info),
        patch("polylogue.daemon.status._insight_freshness_info", freshness_info),
    ):
        payload = daemon_status_payload(sources=())

    assert payload["db_path"] == str(db)
    assert payload["db_size_bytes"] == 11
    assert payload["wal_size_bytes"] == 7
    assert payload["blob_dir_size_bytes"] == 0
    assert payload["disk_free_bytes"] == 99
    fts_readiness = payload["fts_readiness"]
    assert isinstance(fts_readiness, dict)
    assert fts_readiness["messages_ready"] is True
    assert fts_readiness["session_work_events_ready"] is True
    assert fts_readiness["threads_ready"] is True
    assert fts_readiness["invariant_ready"] is True
    assert fts_readiness["message_indexed_count"] == 4
    assert fts_readiness["message_indexable_count"] == 4
    assert fts_readiness["coverage_pct"] == 100.0
    surfaces = fts_readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages_surface = surfaces["messages_fts"]
    assert isinstance(messages_surface, dict)
    assert messages_surface["ready"] is True
    assert db_info.call_count == 1
    assert blob_info.call_count == 1
    assert fts_info.call_count == 1
    # Bounded probe results are reused across the status payload assembly,
    # but freshness probing is invoked a second time for the deep insight
    # readiness band. Both calls reuse the cached probe results.
    assert freshness_info.call_count >= 1


def test_insight_freshness_reads_archive_file_set_from_archive_tiers(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    archive_db = tmp_path / "index.db"
    initialize_archive_database(archive_db, ArchiveTier.INDEX)
    with sqlite3.connect(archive_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("native-1", "codex-session", bytes(32)),
        )
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("native-2", "codex-session", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO session_profiles (
                session_id, workflow_shape, search_text
            ) VALUES (?, ?, ?)
            """,
            ("codex-session:native-1", "debugging", "profile"),
        )
        conn.commit()

    with (
        patch("polylogue.daemon.status.db_path", return_value=db_anchor),
        patch("polylogue.daemon.status.index_db_path", return_value=archive_db),
    ):
        assert _insight_freshness_info() == {
            "sessions_with_profiles": 1,
            "total_sessions": 2,
        }


def test_daemon_status_fts_readiness_uses_lightweight_table_probe(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE messages_fts (text TEXT);
            """
        )

    with patch("polylogue.daemon.status.db_path", return_value=db):
        readiness = status_module._fts_readiness_info()

    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False


def test_daemon_status_fts_readiness_reads_archive_file_set_from_archive_tiers(tmp_path: Path) -> None:
    db_anchor = tmp_path / "index.db"
    archive_db = tmp_path / "index.db"
    initialize_archive_database(archive_db, ArchiveTier.INDEX)
    with sqlite3.connect(archive_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("native-1", "codex-session", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, message_type, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("codex-session:native-1", "message-1", 0, "user", "message", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("codex-session:native-1:message-1", "codex-session:native-1", 0, "text", "needle"),
        )
        conn.commit()

    with patch("polylogue.daemon.status.db_path", return_value=db_anchor):
        readiness = status_module._fts_readiness_info()

    assert readiness["indexed_surface"] == "messages_fts"
    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    assert readiness["coverage_exact"] is False
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    blocks = surfaces["messages_fts"]
    assert isinstance(blocks, dict)
    assert blocks["source_exists"] is True
    assert blocks["exists"] is True
    assert blocks["triggers_present"] is True
    assert blocks["ready"] is False


def test_daemon_status_fts_readiness_prefers_archive_when_present(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    archive_db = tmp_path / "index.db"
    with sqlite3.connect(db_anchor) as conn:
        conn.executescript(
            """
            CREATE TABLE messages (text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            """
        )
    initialize_archive_database(archive_db, ArchiveTier.INDEX)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db_anchor),
        patch("polylogue.daemon.status.index_db_path", return_value=archive_db),
    ):
        readiness = status_module._fts_readiness_info()

    assert readiness["indexed_surface"] == "messages_fts"
    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    assert set(surfaces) == {"messages_fts"}


def test_daemon_status_fts_readiness_uses_bounded_structural_probes(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    queries: list[str] = []
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE blocks (text TEXT);
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, message_count INTEGER NOT NULL);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TABLE messages_fts_docsize (id INTEGER PRIMARY KEY, sz BLOB);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks BEGIN SELECT 1; END;
            INSERT INTO sessions VALUES ('c1', 2), ('c2', 3);
            INSERT INTO blocks(rowid, text) VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e');
            INSERT INTO messages_fts_docsize VALUES (1, x''), (2, x''), (3, x''), (4, x''), (5, x'');
            """
        )

    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(queries.append)
        return cast(sqlite3.Connection, conn)

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.storage.sqlite.connection_profile.sqlite3.connect", side_effect=traced_connect),
    ):
        readiness = status_module._fts_readiness_info()

    assert readiness["messages_ready"] is True
    assert readiness["invariant_ready"] is True
    assert readiness["coverage_exact"] is False
    assert all("COUNT(*) FROM blocks" not in query for query in queries)
    assert all("COUNT(*) FROM messages_fts_docsize" not in query for query in queries)
    assert all("LEFT JOIN messages_fts_docsize" not in query for query in queries)


def test_fts_readiness_exact_detects_missing_docsize_row(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "index.db"
    initialize_archive_database(db_path, ArchiveTier.INDEX)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("native-1", "codex-session", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, message_type, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("codex-session:native-1", "message-1", 0, "user", "message", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("codex-session:native-1:message-1", "codex-session:native-1", 0, "text", "needle stale index"),
        )
        rowid = conn.execute(
            "SELECT rowid FROM blocks WHERE block_id = ?",
            ("codex-session:native-1:message-1:0",),
        ).fetchone()[0]
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
        conn.commit()

    structural = fts_readiness_info(db_path)
    exact = fts_readiness_info(db_path, exact=True)

    assert structural["messages_ready"] is False
    assert exact["messages_ready"] is False
    surfaces = exact["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["missing_rows"] == 1
    assert messages["ready"] is False


def test_fts_readiness_exact_uses_snapshot_transaction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from polylogue.daemon import fts_status

    db_path = tmp_path / "index.db"
    initialize_archive_database(db_path, ArchiveTier.INDEX)
    traced: list[str] = []

    def _open_traced_connection(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.set_trace_callback(traced.append)
        return conn

    monkeypatch.setattr(fts_status, "open_readonly_connection", _open_traced_connection)

    readiness = fts_status.fts_readiness_info(db_path, exact=True)

    assert readiness["coverage_exact"] is True
    assert any(statement == "BEGIN" for statement in traced)
    assert any(statement == "ROLLBACK" for statement in traced)


def test_fts_readiness_exact_detects_archive_missing_messages_fts_row(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    archive_db = tmp_path / "index.db"
    initialize_archive_database(archive_db, ArchiveTier.INDEX)
    with sqlite3.connect(archive_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("native-1", "codex-session", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, message_type, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("codex-session:native-1", "message-1", 0, "user", "message", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("codex-session:native-1:message-1", "codex-session:native-1", 0, "text", "needle"),
        )
        rowid = conn.execute(
            "SELECT rowid FROM blocks WHERE block_id = ?",
            ("codex-session:native-1:message-1:0",),
        ).fetchone()[0]
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
        conn.commit()

    readiness = fts_readiness_info(archive_db, exact=True)

    assert readiness["indexed_surface"] == "messages_fts"
    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    assert readiness["message_indexable_count"] == 1
    assert readiness["message_indexed_count"] == 0
    assert readiness["coverage_pct"] == 0.0
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    blocks = surfaces["messages_fts"]
    assert isinstance(blocks, dict)
    assert blocks["missing_rows"] == 1
    assert blocks["ready"] is False


def test_fts_readiness_requires_recorded_freshness_when_available(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE blocks (text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks BEGIN SELECT 1; END;
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL
            );
            INSERT INTO fts_freshness_state VALUES ('messages_fts', 'stale', '2026-05-24T00:00:00+00:00');
            """
        )

    readiness = fts_readiness_info(db_path)

    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["freshness_known"] is True
    assert messages["freshness_state"] == "stale"
    assert messages["ready"] is False


def test_fts_readiness_reports_recorded_freshness_counts_without_exact_scan(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "index.db"
    queries: list[str] = []
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE blocks (text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TABLE messages_fts_docsize (id INTEGER PRIMARY KEY, sz BLOB);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks BEGIN SELECT 1; END;
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0,
                indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0,
                excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                detail TEXT
            );
            INSERT INTO fts_freshness_state
                (surface, state, checked_at, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows, detail)
            VALUES
                ('messages_fts', 'ready', '2026-05-24T00:00:00+00:00', 200, 199, 1, 0, 0, 'repair pending');
            """
        )

    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(queries.append)
        return cast(sqlite3.Connection, conn)

    with patch("polylogue.storage.sqlite.connection_profile.sqlite3.connect", side_effect=traced_connect):
        readiness = fts_readiness_info(db_path)

    assert readiness["message_indexable_count"] == 200
    assert readiness["message_indexed_count"] == 199
    assert readiness["coverage_pct"] == 99.5
    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["missing_rows"] == 1
    assert messages["freshness_state"] == "stale"
    assert messages["freshness_recorded_state"] == "ready"
    assert messages["freshness_trusted"] is False
    assert messages["freshness_detail"] == "repair pending"
    assert all("COUNT(*) FROM blocks" not in query for query in queries)
    assert all("messages_fts_docsize" not in query for query in queries)


def test_fts_readiness_rejects_zero_count_ready_freshness_when_source_has_rows(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "index.db"
    queries: list[str] = []
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE blocks (text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks BEGIN SELECT 1; END;
            INSERT INTO blocks VALUES ('needs indexing');
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                source_rows INTEGER NOT NULL DEFAULT 0,
                indexed_rows INTEGER NOT NULL DEFAULT 0,
                missing_rows INTEGER NOT NULL DEFAULT 0,
                excess_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                detail TEXT
            );
            INSERT INTO fts_freshness_state
                (surface, state, checked_at, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows)
            VALUES ('messages_fts', 'ready', '2026-05-24T00:00:00+00:00', 0, 0, 0, 0, 0);
            """
        )

    original_connect = sqlite3.connect

    def traced_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(queries.append)
        return cast(sqlite3.Connection, conn)

    with patch("polylogue.storage.sqlite.connection_profile.sqlite3.connect", side_effect=traced_connect):
        readiness = fts_readiness_info(db_path)

    assert readiness["messages_ready"] is False
    assert readiness["invariant_ready"] is False
    assert readiness["coverage_pct"] == 0.0
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["freshness_state"] == "unknown"
    assert messages["freshness_recorded_state"] == "ready"
    assert messages["freshness_trusted"] is False
    assert all("count(*) from messages_fts" not in query.lower() for query in queries)


def test_fts_readiness_tolerates_malformed_recorded_counts(tmp_path: Path) -> None:
    from polylogue.daemon.fts_status import fts_readiness_info

    db_path = tmp_path / "index.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE blocks (text TEXT);
            CREATE TABLE messages_fts (text TEXT);
            CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks BEGIN SELECT 1; END;
            CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks BEGIN SELECT 1; END;
            CREATE TABLE fts_freshness_state (
                surface TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                source_rows TEXT,
                indexed_rows TEXT,
                missing_rows TEXT,
                excess_rows TEXT,
                duplicate_rows TEXT
            );
            INSERT INTO fts_freshness_state
                (surface, state, checked_at, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows)
            VALUES
                ('messages_fts', 'ready', '2026-05-24T00:00:00+00:00', 'not-int', NULL, 'bad', '0', 'bad');
            """
        )

    readiness = fts_readiness_info(db_path)
    surfaces = readiness["surfaces"]
    assert isinstance(surfaces, dict)
    messages = surfaces["messages_fts"]
    assert isinstance(messages, dict)
    assert messages["source_rows"] == 0
    assert messages["missing_rows"] == 0
    assert readiness["message_indexable_count"] == 0


def test_daemon_status_insight_freshness_uses_lightweight_counts(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY);
            CREATE TABLE session_profiles (session_id TEXT PRIMARY KEY);
            INSERT INTO sessions (session_id) VALUES ('a'), ('b');
            INSERT INTO session_profiles (session_id) VALUES ('a');
            """
        )

    with patch("polylogue.daemon.status.db_path", return_value=db):
        freshness = status_module._insight_freshness_info()

    assert freshness == {"sessions_with_profiles": 1, "total_sessions": 2}


@pytest.mark.frozen_clock_modules("polylogue.daemon.status")
def test_daemon_status_flags_stale_live_ingest_attempts(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    db = tmp_path / "index.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    attempt_id = cursor.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    old_updated_at_ms = int((frozen_clock.now() - timedelta(minutes=10)).timestamp() * 1000)
    with sqlite3.connect(db.with_name("ops.db")) as conn:
        conn.execute(
            "UPDATE ingest_attempts SET heartbeat_at_ms = ? WHERE attempt_id = ?",
            (old_updated_at_ms, attempt_id),
        )
        conn.commit()

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    attempts = payload["live_ingest_attempts"]
    assert isinstance(attempts, dict)
    assert attempts["running_count"] == 1
    assert attempts["stale_running_count"] == 1
    recent = attempts["recent"]
    assert isinstance(recent, list)
    latest = recent[0]
    assert isinstance(latest, dict)
    assert latest["stale"] is True
    # #1246: ``stale`` (existing) and the typed ``progress_classification``
    # together encode the same condition for an attempt that has not made
    # progress for at least ``STUCK_AFTER_S``.
    assert latest["progress_classification"] == "stuck"
    updated_age_s = latest["updated_age_s"]
    assert isinstance(updated_age_s, int | float)
    assert updated_age_s >= 600
    assert attempts["stuck_running_count"] == 1
    assert attempts["slow_running_count"] == 0
    lines = format_daemon_status_lines(payload)
    assert "Live ingest attempts: 1 running, 1 stuck" in lines
    assert "  latest: running stuck planning 0/1 files" in lines


@pytest.mark.frozen_clock_modules("polylogue.daemon.status")
def test_daemon_status_flags_slow_but_progressing_live_ingest_attempt(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    """Running attempts that exceed p95 historical duration but still
    report fresh progress are reported as ``slow``, not ``stuck`` (#1246)."""

    db = tmp_path / "index.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    cursor = CursorStore(db)
    # Seed completed attempts with short, uniform durations so the p95
    # baseline is well below the running attempt's elapsed time.
    base = frozen_clock.now() - timedelta(hours=1)
    with sqlite3.connect(db.with_name("ops.db")) as conn:
        for i in range(8):
            start = int((base + timedelta(minutes=i)).timestamp() * 1000)
            end = int((base + timedelta(minutes=i, seconds=2)).timestamp() * 1000)
            conn.execute(
                """
                INSERT INTO ingest_attempts (
                    attempt_id, status, phase, started_at_ms, heartbeat_at_ms, finished_at_ms
                ) VALUES (?, 'completed', 'convergence', ?, ?, ?)
                """,
                (f"hist-{i}", start, end, end),
            )
        conn.commit()

    attempt_id = cursor.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    # Make the running attempt look like it has been ticking for 90s —
    # well above the 2-second historical p95, but well under the 180s
    # stuck threshold. Set updated_at to "now" so it is not stale.
    started_at_ms = int((frozen_clock.now() - timedelta(seconds=90)).timestamp() * 1000)
    updated_at_ms = int(frozen_clock.now().timestamp() * 1000)
    with sqlite3.connect(db.with_name("ops.db")) as conn:
        conn.execute(
            "UPDATE ingest_attempts SET started_at_ms = ?, heartbeat_at_ms = ? WHERE attempt_id = ?",
            (started_at_ms, updated_at_ms, attempt_id),
        )
        conn.commit()

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    attempts = payload["live_ingest_attempts"]
    assert isinstance(attempts, dict)
    assert attempts["running_count"] == 1
    assert attempts["slow_running_count"] == 1
    assert attempts["stuck_running_count"] == 0
    assert attempts["stale_running_count"] == 0
    threshold = attempts["slow_threshold_s"]
    assert isinstance(threshold, int | float)
    assert threshold < 90.0
    recent = attempts["recent"]
    assert isinstance(recent, list)
    latest = recent[0]
    assert isinstance(latest, dict)
    assert latest["progress_classification"] == "slow"
    assert latest["stale"] is False
    lines = format_daemon_status_lines(payload)
    assert "Live ingest attempts: 1 running, 1 slow" in lines
    assert any(line.startswith("  latest: running slow ") for line in lines)


def test_daemon_status_summarizes_retry_due_and_excluded_live_cursor_files(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    failed = tmp_path / "failed.jsonl"
    failed.write_text('{"bad":true}\n')
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text('{"skip":true}\n')
    cursor = CursorStore(db)
    cursor.mark_failed(failed)
    cursor.set(excluded, excluded.stat().st_size)
    cursor.mark_excluded(excluded)

    with sqlite3.connect(db.with_name("ops.db")) as conn:
        conn.execute(
            "UPDATE ingest_cursor SET next_retry_at = ? WHERE source_path = ?",
            ("2000-01-01T00:00:00+00:00", str(failed)),
        )
        conn.commit()

    with (
        patch("polylogue.daemon.status.db_path", return_value=db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        status = build_daemon_status(sources=())

    assert status.live_cursor.tracked_file_count == 2
    assert status.live_cursor.failed_file_count == 1
    assert status.live_cursor.excluded_file_count == 1
    assert status.live_cursor.retry_due_file_count == 1
    assert status.live_cursor.in_backoff_file_count == 0
    assert [item.source_path for item in status.live_cursor.failing_files] == [str(excluded), str(failed)]
