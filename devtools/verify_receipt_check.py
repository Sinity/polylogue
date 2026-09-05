"""Require test evidence in the newest ``devtools verify`` run receipt.

A hosted ``verify`` job that exits zero has not shown that tests ran: pytest
may have been skipped, refused, or never started. The receipt at
``.cache/verify/runs/<id>/run.json`` records what happened, and this check
accepts it only when the run succeeded and either a pytest step ran to success
or the selection was ``none`` for a recorded reason.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devtools.verify_runs import VERIFY_RUNS_DIR

__all__ = ["main", "newest_run_receipt", "refusal"]


def newest_run_receipt(root: Path) -> Path | None:
    """The most recently written ``run.json`` under the verify runs directory."""
    candidates = sorted(
        (root / VERIFY_RUNS_DIR).glob("*/run.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    return candidates[-1] if candidates else None


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
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(arguments[0]) if arguments else Path.cwd()
    receipt = newest_run_receipt(root)
    if receipt is None:
        sys.stderr.write(f"verify receipt check: no run receipt under {root / VERIFY_RUNS_DIR}\n")
        return 1
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"verify receipt check: unreadable receipt {receipt}: {exc}\n")
        return 1
    if not isinstance(payload, dict):
        sys.stderr.write(f"verify receipt check: {receipt} is not a receipt document\n")
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
