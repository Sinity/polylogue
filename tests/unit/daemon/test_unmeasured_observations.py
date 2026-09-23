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
    from tests.infra.archive_templates import bootstrap_archive_root

    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
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


def test_build_identity_failure_does_not_attest_clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A build-info probe failure publishes ``dirty=unknown``, never false.

    Anti-vacuity: restoring the exception branch's ``build_dirty = False``
    makes the emitted label ``dirty=\"false\"`` and fails this assertion.
    """
    import polylogue.version as version_module
    from polylogue.daemon import metrics as metrics_module
    from tests.infra.archive_templates import bootstrap_archive_root

    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)

    class _BrokenVersion:
        version = "test"
        commit = "revision"

        @property
        def dirty(self) -> bool:
            raise RuntimeError("build identity unavailable")

    monkeypatch.setattr(version_module, "VERSION_INFO", _BrokenVersion())
    build_samples: list[object] = []
    real_emit_metric = metrics_module._emit_metric

    def _capture_emit(
        lines: list[str],
        *,
        name: str,
        help_text: str,
        metric_type: str,
        samples: list[tuple[dict[str, str] | None, float | int]],
        omit_when_empty: bool = False,
    ) -> None:
        if name == "polylogue_daemon_build_info":
            build_samples.extend(samples)
        real_emit_metric(
            lines,
            name=name,
            help_text=help_text,
            metric_type=metric_type,
            samples=samples,
            omit_when_empty=omit_when_empty,
        )

    monkeypatch.setattr(metrics_module, "_emit_metric", _capture_emit)
    body = metrics_module.format_metrics(index_db, now_monotonic=0.0)

    assert build_samples == [({"version": "unknown", "revision": "unknown", "dirty": "unknown"}, 1)]
    assert 'dirty="false"' not in body


def test_unreadable_status_fingerprint_invalidates_cached_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stat failure gets a changing discriminator, not a stable question mark.

    Anti-vacuity: restoring the constant ``name:?`` token makes the two calls
    equal and this cache-invalidation assertion fails.
    """
    from polylogue.daemon import status as status_module

    active = tmp_path / "index.db"
    ops = tmp_path / "ops.db"
    active.touch()
    ops.touch()
    monkeypatch.setattr(status_module, "_active_status_db_path", lambda: active)
    monkeypatch.setattr(status_module, "archive_root", lambda: tmp_path)
    original_stat = type(active).stat

    def _stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == active:
            raise OSError("stat refused")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(type(active), "stat", _stat)
    first = status_module._daemon_status_fingerprint(active)
    second = status_module._daemon_status_fingerprint(active)

    assert "index.db:unreadable-" in first
    assert "index.db:unreadable-" in second
    assert first != second


def test_missing_progress_classification_is_rendered_as_unknown() -> None:
    """An attempt without classification is visibly unmeasured."""
    from polylogue.daemon.status import format_daemon_status_lines

    lines = format_daemon_status_lines(
        {
            "live_ingest_attempts": {
                "running_count": 1,
                "recent": [{"status": "running", "phase": "parse"}],
            }
        }
    )

    assert "  latest: running progress-unknown parse 0/0 files" in lines
    assert all("healthy" not in line for line in lines)
