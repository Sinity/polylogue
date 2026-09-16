"""Unmeasured observations are published as unmeasured (polylogue-xvwpi).

Three classes of defect are covered, one test each:

* a fabricated literal (``blob_dir_size_bytes = 0`` for a walk never run),
* a failure reported as zero (``statvfs`` raising, a probe raising),
* a missing measurement reported as healthy (``all({})`` is ``True``).

Each test names the literal whose restoration turns it red.
"""

from __future__ import annotations

import sqlite3
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import pytest


def test_blob_directory_size_is_unmeasured_not_zero() -> None:
    """The compact status path never walks the blob store, so it reports null.

    Anti-vacuity: restoring ``return 0`` in ``_blob_size_info`` (or the
    ``blob_dir_size_bytes: int = 0`` model default) turns this red -- the
    operator surface would again render a measured-looking "0 B".
    """
    from polylogue.daemon.status import DaemonStatus, _blob_size_info

    assert _blob_size_info() is None
    assert DaemonStatus.model_fields["blob_dir_size_bytes"].default is None


def test_failed_statvfs_reports_unknown_free_space_not_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed filesystem interrogation is unknown, never "0 bytes free".

    Anti-vacuity: restoring ``return 0`` in ``_disk_free_bytes`` turns this
    red, and would have the status surface announce a disk-full emergency
    that was never measured.
    """
    from polylogue.daemon import status_snapshot

    def _boom(_path: str) -> object:
        raise OSError("statvfs refused")

    monkeypatch.setattr("polylogue.daemon.status_snapshot.os.statvfs", _boom)
    assert status_snapshot._disk_free_bytes(tmp_path) is None


def test_uninspected_fts_triggers_do_not_report_all_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty trigger-presence map is unmeasured, not "all present".

    ``all({})`` is ``True``, so the gauge published
    ``polylogue_fts_triggers_all_present 1`` for a database whose triggers
    were never inspected.

    Anti-vacuity: restoring the unconditional
    ``samples=[(None, 1 if all(triggers.values()) else 0)]`` turns this red --
    the ``1`` reappears and the unmeasured marker does not.
    """
    from polylogue.daemon import metrics as metrics_module
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    monkeypatch.setattr(metrics_module, "_fts_trigger_presence", lambda _conn: {})

    body = metrics_module.format_metrics(index_db, now_monotonic=0.0)

    assert "polylogue_fts_triggers_all_present 1" not in body
    assert 'polylogue_probe_unmeasured{probe="fts_triggers_all_present"} 1' in body


def test_unreadable_live_cursor_is_typed_unavailable_not_all_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable live_cursor table is reported unreadable, not "0 failed".

    Input: the index tier exists but every connection to it raises
    ``sqlite3.Error`` -- a locked or replaced database.

    Anti-vacuity: restoring the bare ``return LiveCursorSummary()`` turns
    this red -- the model defaults would again render as "0 failed, 0
    excluded" with nothing marking them as unobserved.
    """
    from polylogue.daemon import status as status_module

    db = tmp_path / "index.db"
    db.write_bytes(b"")

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.Error("cursor table unreadable")

    monkeypatch.setattr(status_module, "_active_status_db_path", lambda: db)
    monkeypatch.setattr(status_module, "_archive_live_cursor_summary_info", lambda _path: None)
    monkeypatch.setattr("polylogue.daemon.status.sqlite3.connect", _boom)

    summary = status_module._live_cursor_summary_info()

    assert summary.available is False
    assert summary.unavailable_reason == "live_cursor_summary_unreadable"


def test_unpinned_embeddings_tier_reports_unknown_counts_not_zero() -> None:
    """An uninspected embeddings tier publishes null counts, not measured zeros.

    Anti-vacuity: restoring ``"embedded_messages": 0`` /
    ``"embedding_coverage_percent": 0.0`` turns this red -- the top-level
    status payload would again splice "0 embedded messages, 0.0 coverage" for
    a tier that was never opened.
    """
    from polylogue.operations.daemon_status import _unavailable_embedding_status
    from polylogue.storage.embeddings.status_payload import EmbeddingStatusSettings

    payload = _unavailable_embedding_status(
        EmbeddingStatusSettings(
            config_enabled=False,
            has_voyage_api_key=False,
            configured_model=None,
            configured_dimension=None,
            monthly_cost_cap_usd=None,
        )
    )

    assert payload["status"] == "unavailable"
    for key in (
        "total_sessions",
        "embedded_sessions",
        "embedded_messages",
        "pending_sessions",
        "embedding_coverage_percent",
        "failure_count",
    ):
        assert payload[key] is None, key


def test_status_route_etag_changes_when_the_liveness_probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A suppressed liveness probe must not ride under the previous ETag.

    Anti-vacuity: restoring ``contextlib.suppress(Exception)`` around
    ``_check_daemon_liveness`` drops the key, folds ``None`` into the ETag
    material, and a client holding the pre-failure ETag gets a 304 over an
    unmeasured value -- which makes the two ETags below equal and turns this
    red.
    """
    from io import BytesIO

    from polylogue.daemon.http import DaemonAPIHandler

    class _Headers:
        def get(self, _key: str, default: str | None = None) -> str | None:
            return default

    captured: list[tuple[HTTPStatus, object, dict[str, str]]] = []

    def _make() -> DaemonAPIHandler:
        handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
        handler.path = "/api/status"
        handler.command = "GET"
        handler.headers = cast("Any", _Headers())
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()
        handler.server = type("S", (), {})()

        def _send_json(status, payload, *, extra_headers=None):  # type: ignore[no-untyped-def]
            captured.append((status, payload, dict(extra_headers or {})))

        handler._send_json = _send_json  # type: ignore[method-assign]
        return handler

    monkeypatch.setattr("polylogue.daemon.http.get_latest_event_id", lambda: 7)
    monkeypatch.setattr("polylogue.daemon.http.get_status_snapshot_payload", lambda: {"ok": True})

    monkeypatch.setattr("polylogue.daemon.status._check_daemon_liveness", lambda: None)
    _make()._handle_status({})
    healthy_etag = captured[-1][2].get("ETag")

    def _boom() -> bool:
        raise RuntimeError("probe failed")

    monkeypatch.setattr("polylogue.daemon.status._check_daemon_liveness", _boom)
    _make()._handle_status({})
    failed_etag = captured[-1][2].get("ETag")

    assert healthy_etag is not None and failed_etag is not None
    assert healthy_etag != failed_etag
    final_payload = captured[-1][1]
    assert isinstance(final_payload, dict)
    assert final_payload["daemon_liveness_state"] == "unmeasured"
