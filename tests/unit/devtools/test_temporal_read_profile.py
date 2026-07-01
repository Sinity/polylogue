from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from devtools import temporal_read_profile
from polylogue.cli.root_request import RootModeRequest
from polylogue.config import Config
from polylogue.surfaces.temporal_evidence import (
    TemporalEvidenceEvent,
    TemporalEvidenceWindow,
    build_temporal_evidence_window,
)


def test_temporal_read_profile_report_wraps_shared_builder(tmp_path: Path) -> None:
    config = Config(
        archive_root=tmp_path,
        db_path=tmp_path / "index.db",
        render_root=tmp_path / "render",
        sources=[],
    )
    window = build_temporal_evidence_window(
        [
            TemporalEvidenceEvent(
                event_id="session:abc:session",
                occurred_at=datetime(2026, 6, 30, 8, 0, tzinfo=UTC),
                family="archive-session",
                kind="session",
                label="Temporal profile",
                source_ref="session:abc",
                evidence_refs=("session:abc",),
            )
        ]
    )

    def fake_builder(
        _config: Config,
        _request: RootModeRequest,
        *,
        phase_recorder: Callable[[str, float, dict[str, object]], None],
    ) -> TemporalEvidenceWindow:
        phase_recorder("prepare", 1.25, {"archive_root": str(tmp_path), "limit": 1})
        phase_recorder("select_sessions", 2.5, {"session_count": 1})
        return window

    args = argparse.Namespace(
        query="repo:polylogue",
        limit=1,
        archive_root=None,
        out=None,
        include_window=False,
        json=True,
    )
    with (
        patch("devtools.temporal_read_profile.get_config", return_value=config),
        patch("devtools.temporal_read_profile.build_read_temporal_window", side_effect=fake_builder),
    ):
        report = temporal_read_profile.build_report(args)

    assert report["archive_root"] == str(tmp_path)
    assert report["query"] == "repo:polylogue"
    assert report["phase_summary"]["slowest_phase"] == "select_sessions"
    assert report["temporal_window_summary"]["family_counts"] == {"archive-session": 1}
    assert "temporal_window" not in report


def test_temporal_read_profile_main_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "profile.json"
    with patch(
        "devtools.temporal_read_profile.build_report",
        return_value={"report_version": 1, "total_elapsed_ms": 3.0},
    ):
        exit_code = temporal_read_profile.main(["--query", "repo:polylogue", "--out", str(out), "--json"])

    assert exit_code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["total_elapsed_ms"] == 3.0
    assert json.loads(capsys.readouterr().out)["report_version"] == 1
