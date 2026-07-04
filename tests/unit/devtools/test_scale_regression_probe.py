from __future__ import annotations

import io
import json
from pathlib import Path

from devtools.scale_regression_probe import main, run_scale_regression_probe


def test_scale_regression_probe_runs_seeded_bug_class_checks(tmp_path: Path) -> None:
    report = run_scale_regression_probe(tmp_path)

    assert report.ok is True
    assert {check.name for check in report.checks} == {
        "chunked_rebuild",
        "bounded_giant_session",
        "reset_source_preservation",
        "run_ref_no_drop",
        "raw_materialization_debt_detected",
        "insights_stage_resumable",
    }
    assert all(check.ok for check in report.checks)
    assert report.duration_ms >= 0


def test_scale_regression_probe_main_emits_json(tmp_path: Path) -> None:
    stdout = io.StringIO()

    exit_code = main(["--workdir", str(tmp_path), "--json"], stdout=stdout)

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["checks"]
    assert {check["name"] for check in payload["checks"]} >= {
        "chunked_rebuild",
        "bounded_giant_session",
        "run_ref_no_drop",
    }
