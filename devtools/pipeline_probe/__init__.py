"""Pipeline probe: exercise the real pipeline on bounded synthetic, archive-subset, or staged source corpora."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from devtools.pipeline_probe.engine import run_probe
from devtools.pipeline_probe.report import _capture_regression_case
from devtools.pipeline_probe.request import ProbeSummary, _parse_args, _request_from_args
from devtools.pipeline_probe.result import _build_budget_report
from devtools.pipeline_probe.staging import _write_probe_sources

__all__ = [
    "ProbeSummary",
    "_build_budget_report",
    "_write_probe_sources",
    "main",
    "run_probe",
]


def main(argv: list[str] | None = None) -> int:
    """Pipeline probe entry point: parse CLI args, run the probe, emit JSON summary."""
    args = _parse_args(argv)
    request = _request_from_args(args)
    active_request = request
    if request.workdir is None:
        with tempfile.TemporaryDirectory(prefix="polylogue-pipeline-probe-") as tempdir:
            active_request = replace(request, workdir=str(Path(tempdir)))
            with redirect_stdout(sys.stderr):
                summary = asyncio.run(run_probe(active_request))
    else:
        with redirect_stdout(sys.stderr):
            summary = asyncio.run(run_probe(active_request))
    budget_report = _build_budget_report(summary, active_request)
    if budget_report is not None:
        summary["budgets"] = budget_report
    regression_case = _capture_regression_case(summary, args)
    if regression_case is not None:
        summary["regression_case"] = regression_case
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if active_request.json_out is not None:
        json_out = Path(active_request.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if budget_report is not None and not budget_report["ok"] else 0
