"""One intake route: the retired watcher chunk route stays retired.

polylogue-v4dcc. Acquisition has exactly one production route --
``FairIntakeDispatcher.run_once`` -> ``FileIntakeAdapter.admit_page`` ->
``LiveWatcher._ingest_files`` -> ``LiveBatchProcessor.ingest_files``. The
watcher's own catch-up scan, debounce queue, failed-retry scans, periodic
catch-up and hook-spool drain were a second route with its own defects and
its own performance profile; they are gone, and nothing may bring one of
their names back without this test saying so.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import polylogue

_PACKAGE_ROOT = Path(polylogue.__file__).parent

#: Every name the one-route deletion retired, with the route it belonged to.
RETIRED_NAMES = {
    "_catch_up": "watcher catch-up scan",
    "_catch_up_candidates": "watcher catch-up chunk loop",
    "_periodic_catch_up": "watcher periodic catch-up loop",
    "_cancel_periodic_catch_up": "watcher periodic catch-up loop",
    "_scan_catch_up_candidates": "watcher catch-up scan",
    "_plan_catch_up": "watcher catch-up planner",
    "_chunk_catch_up_paths": "watcher catch-up chunking",
    "_flush_catch_up_convergence": "watcher catch-up convergence flush",
    "_emit_catch_up_terminal": "watcher catch-up lifecycle receipts",
    "_redeem_whole_archive_pledges": "watcher catch-up whole-archive pledge",
    "_open_whole_archive_pledges": "watcher catch-up whole-archive pledge",
    "_release_whole_archive_pledges": "watcher catch-up whole-archive pledge",
    "CatchUpPlan": "watcher catch-up planner",
    "CandidateSourceFile": "watcher catch-up scan",
    "_interleave_by_source": "watcher catch-up fairness",
    "_debounced_batch": "watcher debounce queue",
    "_wait_for_pending_quiet": "watcher debounce queue",
    "_flush_pending": "watcher debounce queue",
    "_ensure_pending_scheduled": "watcher debounce queue",
    "_pending_paths": "watcher debounce queue",
    "_forced_reparse_paths": "watcher debounce queue",
    "cancel_pending": "watcher debounce queue",
    "_schedule_failed_retry_scan": "watcher failed-retry scan",
    "_schedule_failed_retry_wakeup": "watcher failed-retry scan",
    "_wake_failed_retries": "watcher failed-retry scan",
    "_cancel_failed_retry_task": "watcher failed-retry scan",
    "_defer_unaccounted_failed_retries": "watcher failed-retry scan",
    "intake_hints_only": "the watcher's two-mode switch",
    "catch_up_active": "the watcher's catch-up state",
    "run_live_watcher": "the standalone watcher entry point",
    "_drain_hook_spools": "watcher hook-spool drain",
    "drain_hook_event_spool": "watcher hook-spool drain",
}


#: Where an intake route can live: acquisition, the scheduler and its
#: adapters, plus the CLI that composes them. Deliberately not the whole
#: package -- ``_flush_pending`` is an ordinary local name elsewhere, and a
#: ledger that fails on an unrelated homonym stops being read.
_INTAKE_DIRECTORIES = ("sources/live", "daemon", "operations")


def _python_sources() -> list[Path]:
    return [
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if "daemon/static" not in str(path)
        and any(directory in str(path.relative_to(_PACKAGE_ROOT).parent) for directory in _INTAKE_DIRECTORIES)
    ]


@pytest.mark.parametrize("retired", sorted(RETIRED_NAMES))
def test_a_retired_intake_name_has_no_definition_or_caller(retired: str) -> None:
    """The name exists nowhere in the package -- not defined, not referenced.

    Anti-vacuity: reintroduce any retired member (a ``_catch_up`` method, a
    ``_pending_paths`` attribute, an ``intake_hints_only`` keyword) and its
    parametrization goes red naming the route it belonged to.
    """
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name: str | None = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.keyword):
                name = node.arg
            if name == retired:
                offenders.append(f"{path.relative_to(_PACKAGE_ROOT)}:{getattr(node, 'lineno', 0)}")
    assert not offenders, f"{retired} ({RETIRED_NAMES[retired]}) is back in: {', '.join(sorted(offenders))}"
