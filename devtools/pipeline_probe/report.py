"""Regression case capture and reporting for pipeline probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from devtools.pipeline_probe.request import ProbeSummary, RegressionCaseSummary
from devtools.regression_cases import DEFAULT_REGRESSION_CASE_DIR, RegressionCase, RegressionCaseStore
from polylogue.core.json import json_document


def _capture_regression_case(summary: ProbeSummary, args: argparse.Namespace) -> RegressionCaseSummary | None:
    name = getattr(args, "capture_regression", None)
    if not isinstance(name, str) or not name.strip():
        return None
    output_dir = getattr(args, "regression_output_dir", DEFAULT_REGRESSION_CASE_DIR)
    if not isinstance(output_dir, Path):
        output_dir = Path(str(output_dir))
    tags = tuple(str(tag) for tag in getattr(args, "regression_tag", []) or [])
    notes = tuple(str(note) for note in getattr(args, "regression_note", []) or [])
    case = RegressionCase.from_probe_summary(
        name=name,
        summary=json_document(summary),
        tags=tags,
        notes=notes,
    )
    path = RegressionCaseStore(output_dir).write(case)
    return {
        "case_id": case.case_id,
        "name": case.name,
        "path": str(path),
        "tags": list(case.tags),
    }


__all__ = [
    "_capture_regression_case",
]
