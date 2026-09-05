"""Require test evidence in the receipt of one ``devtools verify`` run.

A hosted ``verify`` job that exits zero has not shown that tests ran: pytest
may have been skipped, refused, or never started. The receipt at
``.cache/verify/runs/<id>/run.json`` records what happened, and this check
accepts it only when the run succeeded and either a pytest step ran to success
or the selection was ``none`` for a recorded reason.

The run is named explicitly (``--run-id``) or read from
``.cache/verify/current-run.json``, which ``devtools verify`` writes for the
run it produced; the check never picks a receipt by age.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devtools.verify_runs import CURRENT_RUN_PATH, VERIFY_RUNS_DIR

__all__ = ["current_run_id", "main", "refusal", "run_receipt_path"]


def current_run_id(root: Path) -> str:
    """The run id ``devtools verify`` last wrote to ``current-run.json`` under ``root``."""
    path = root / CURRENT_RUN_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"no current verify run at {path}: {exc}") from exc
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{path} names no run_id")
    return run_id


def run_receipt_path(root: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"not a run id: {run_id!r}")
    return root / VERIFY_RUNS_DIR / run_id / "run.json"


def refusal(payload: Mapping[str, Any]) -> str | None:
    """Why the receipt is not test evidence, or None when it is."""
    status = payload.get("status")
    if status != "success":
        return f"run {payload.get('run_id')} ended {status!r} (exit {payload.get('exit_code')!r})"
    steps = [step for step in payload.get("steps") or [] if isinstance(step, Mapping)]
    pytest_steps = [step for step in steps if str(step.get("name") or step.get("label") or "").startswith("pytest")]
    if pytest_steps:
        failed = [step for step in pytest_steps if step.get("status") != "success" or step.get("exit") != 0]
        if failed:
            names = ", ".join(str(step.get("name") or step.get("step_id")) for step in failed)
            return f"pytest step(s) did not succeed: {names}"
        return None
    selection = payload.get("testmon_selection")
    selection = selection if isinstance(selection, Mapping) else {}
    if selection.get("selection_mode") == "none" and selection.get("selection_reason"):
        return None
    return (
        f"run {payload.get('run_id')} ran no pytest step and records no reason "
        f"(selection {selection.get('selection_mode')!r})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify_receipt_check")
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--run-id", help="the verify run to check (default: the run current-run.json names)")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    root = arguments.root or Path.cwd()
    try:
        run_id = arguments.run_id or current_run_id(root)
        receipt = run_receipt_path(root, run_id)
    except ValueError as exc:
        sys.stderr.write(f"verify receipt check: {exc}\n")
        return 1
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"verify receipt check: unreadable receipt {receipt}: {exc}\n")
        return 1
    if not isinstance(payload, dict):
        sys.stderr.write(f"verify receipt check: {receipt} is not a receipt document\n")
        return 1
    if payload.get("run_id") != run_id:
        sys.stderr.write(f"verify receipt check: {receipt} records run {payload.get('run_id')!r}, not {run_id!r}\n")
        return 1
    reason = refusal(payload)
    if reason is not None:
        sys.stderr.write(f"verify receipt check: {receipt}: {reason}\n")
        return 1
    selection = payload.get("testmon_selection") or {}
    sys.stderr.write(
        f"verify receipt check: {receipt}: success, selection {selection.get('selection_mode')!r}, "
        f"{sum(str(step.get('name', '')).startswith('pytest') for step in payload.get('steps') or [])} pytest step(s)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
