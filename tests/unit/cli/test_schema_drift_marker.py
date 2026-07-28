"""Tests for the last-resort format-drift marker (polylogue-da1).

AC: "Sentinel alerting does not depend on the daemon operating healthily --
a last-resort status-line marker is visible on any `polylogue` invocation."
``_emit_schema_drift_marker`` is called unconditionally from the root CLI
group callback; these tests exercise it directly (not through the full CLI
dispatch) since it is a pure, best-effort side-effecting function.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.cli.click_app import _emit_schema_drift_marker


def _patch_active_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    active_db = tmp_path / "archive" / "index.db"
    monkeypatch.setattr(
        "polylogue.storage.archive_identity.resolve_active_index_path",
        lambda _root: active_db,
    )


def test_marker_silent_when_sentinel_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_active_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "polylogue.cli.commands.status._schema_drift_status",
        lambda *a, **kw: {"available": False},
    )
    _emit_schema_drift_marker()
    captured = capsys.readouterr()
    assert captured.err == ""


def test_marker_silent_when_all_origins_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_active_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "polylogue.cli.commands.status._schema_drift_status",
        lambda *a, **kw: {
            "available": True,
            "since_ms": 1_700_000_000_000,
            "origins": [{"origin": "chatgpt-export", "total": 10, "risky_rate": 0.0, "severity": "ok"}],
        },
    )
    _emit_schema_drift_marker()
    captured = capsys.readouterr()
    assert captured.err == ""


def test_marker_prints_to_stderr_when_worst_origin_is_risky(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_active_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "polylogue.cli.commands.status._schema_drift_status",
        lambda *a, **kw: {
            "available": True,
            "since_ms": 1_700_000_000_000,
            "origins": [
                {"origin": "chatgpt-export", "total": 10, "risky_rate": 0.02, "severity": "ok"},
                {"origin": "codex-session", "total": 40, "risky_rate": 0.30, "severity": "error"},
            ],
        },
    )
    _emit_schema_drift_marker()
    captured = capsys.readouterr()
    assert "codex-session" in captured.err
    assert "30%" in captured.err
    assert "devtools lab schema generate/promote" in captured.err
    assert captured.out == ""


def test_marker_never_raises_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A broken active-db resolution must never turn an ordinary command into a failure."""

    def _boom(_root: Path) -> Path:
        raise RuntimeError("boom")

    monkeypatch.setattr("polylogue.storage.archive_identity.resolve_active_index_path", _boom)
    _emit_schema_drift_marker()  # must not raise
    captured = capsys.readouterr()
    assert captured.err == ""
