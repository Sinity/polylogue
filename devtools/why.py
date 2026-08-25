"""Answer "what happened and what do I do" from the verification receipts.

The receipts already record enough to answer this, but answering it meant
knowing which of ~500 run directories was the relevant one, which of two dozen
fields carried the cause, and what a diagnosis token implies. This session spent
several passes doing exactly that by hand -- reading step JSON to find a failing
lane, then opening the pytest-testmon SQLite to learn why a bootstrap happened.

So this renders rather than reconstructs: every diagnosis is mapped to a plain
cause and a concrete next command, and anything not in that table is reported
verbatim instead of guessed at. A wrong remedy is worse than none.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from devtools.verify_runs import VERIFY_HISTORY_PATH, VERIFY_RUNS_DIR

__all__ = ["main"]


@dataclass(frozen=True, slots=True)
class Explanation:
    """What a diagnosis means and what to do about it."""

    cause: str
    remedy: str


#: Diagnosis token -> (cause, remedy). Tokens absent here are rendered verbatim
#: with no invented advice.
_EXPLANATIONS: dict[str, Explanation] = {
    "pytest_passed": Explanation(
        "The selected tests all passed.",
        "Nothing to do.",
    ),
    "pytest_failed": Explanation(
        "Tests ran and some failed. The failures below are the run's own record, not a re-derivation.",
        "Re-run just the failing nodes: devtools test <nodeid>. If they pass alone, suspect ordering or shared state.",
    ),
    "pytest_interrupted": Explanation(
        "The run was interrupted before finishing, so its result is not a verdict on the suite.",
        "Re-run. If it was interrupted mid-bootstrap, the graph is retained and the next run resumes from it.",
    ),
    "pytest_report_incomplete": Explanation(
        "Pytest selected tests but did not record a terminal result for each one.",
        "Re-run the same selection; do not treat this receipt as a test verdict.",
    ),
    "pytest_collection_only": Explanation(
        "Pytest collected tests without running them.",
        "Re-run without --collect-only to execute the selected tests.",
    ),
    "pytest_no_tests_selected": Explanation(
        "Pytest's selection matched no tests.",
        "Use a selector that matches at least one test.",
    ),
    "gate_missing_executable": Explanation(
        "A required gate executable was unavailable.",
        "Install it or make it available on PATH, then re-run the gate.",
    ),
    "gate_missing_input": Explanation(
        "A required gate input was missing.",
        "Restore the input named in the gate details, then re-run the gate.",
    ),
    "gate_unreadable_input": Explanation(
        "A required gate input could not be read.",
        "Make the input named in the gate details readable, then re-run the gate.",
    ),
    "gate_semantic_violation": Explanation(
        "An import violates a package boundary declared by the layering policy.",
        "Read the detailed layering finding and fix the offending import. If the boundary is intentional, have the normal layering-policy owner deliberately update the tracked declaration or baseline through its review process; do not baseline an accidental violation.",
    ),
    "not_enforced": Explanation(
        "This gate was recorded but was not enforced.",
        "Enable the gate's enforcement option before relying on it as a check.",
    ),
    "checkout_import_mismatch": Explanation(
        "The resolved polylogue package was outside the checkout being verified.",
        "Run with an environment that imports polylogue from the invoked checkout.",
    ),
    "render_feeder_failed": Explanation(
        "A declared documentation feeder returned a nonzero result, so the site was not assembled.",
        "Fix the feeder renderer named in the render output, then run devtools render pages again.",
    ),
    "render_feeder_exception": Explanation(
        "A declared documentation feeder raised an exception before it completed.",
        "Fix the feeder renderer named in the render output, then run devtools render pages again.",
    ),
    "render_feeder_system_exit": Explanation(
        "A declared documentation feeder terminated through its command boundary.",
        "Read the feeder message, then run that feeder and devtools render pages again.",
    ),
    "render_feeder_invalid_result": Explanation(
        "A declared documentation feeder returned a non-integer result, so the site was not assembled.",
        "Fix the feeder renderer to return an integer status, then run devtools render pages again.",
    ),
    "render_pages_build_exception": Explanation(
        "The documentation site could not be assembled because its page build raised an exception.",
        "Read the recorded build error, fix the pages builder input or code, then run devtools render pages again.",
    ),
    "render_pagefind_failed": Explanation(
        "The declared Pagefind search-index output was not generated.",
        "Install or repair Pagefind, then run devtools render pages again.",
    ),
    "render_input_missing": Explanation(
        "A declared render input is missing, so freshness cannot be established.",
        "Restore the declared input named in the render details, then run devtools render all again.",
    ),
    "render_input_unreadable": Explanation(
        "A declared render input could not be read, so freshness cannot be established.",
        "Make the declared input readable, then run devtools render all again.",
    ),
    "render_input_invalid": Explanation(
        "A declared render input path or pattern is invalid.",
        "Fix the declared input path or pattern, then run devtools render all again.",
    ),
    "render_surface_exception": Explanation(
        "A generated surface raised an exception instead of producing its output.",
        "Read the recorded render details, fix that surface, then run devtools render all again.",
    ),
    "render_surface_failed": Explanation(
        "A generated surface returned a nonzero result, so its output was not accepted.",
        "Fix the generated surface named in the render output, then run devtools render all again.",
    ),
    "render_surface_invalid_result": Explanation(
        "A generated surface returned a non-integer result, so its output was not accepted.",
        "Fix the generated surface to return an integer status, then run devtools render all again.",
    ),
    "render_surface_system_exit": Explanation(
        "A generated surface terminated through its command boundary, so its output was not accepted.",
        "Read the recorded surface message, then run devtools render all again.",
    ),
    "render_stamp_write_failed": Explanation(
        "The generated surface completed but its freshness stamp could not be published.",
        "Make the cache path writable, then run devtools render all again.",
    ),
    "render_stamp_invalidate_failed": Explanation(
        "An old freshness stamp could not be invalidated before rerendering.",
        "Make the cache path writable, then run devtools render all again.",
    ),
}


def _latest_run(runs_dir: Path) -> Path | None:
    candidates = [path for path in runs_dir.glob("*/run.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"why: cannot read {path}: {exc}") from exc
    return loaded if isinstance(loaded, dict) else {}


def _render(payload: dict[str, Any], stream: Any) -> None:
    status = str(payload.get("status", "unknown"))
    tier = str(payload.get("tier", "unknown"))
    exit_code = payload.get("exit_code")
    duration = payload.get("duration_s")
    argv = payload.get("argv") or []

    headline = f"{tier} run {status}"
    if exit_code is not None:
        headline += f" (exit {exit_code})"
    if isinstance(duration, (int, float)):
        headline += f" in {duration:.1f}s"
    print(headline, file=stream)
    if argv:
        print(f"  invoked: devtools verify {' '.join(str(a) for a in argv)}", file=stream)

    diagnosis = payload.get("diagnosis") or payload.get("checkout_diagnosis")
    if diagnosis:
        print(f"\ndiagnosis: {diagnosis}", file=stream)
        explanation = _EXPLANATIONS.get(str(diagnosis))
        if explanation is None:
            print("  (no recorded explanation for this diagnosis -- reporting it verbatim)", file=stream)
        else:
            print(f"  cause : {explanation.cause}", file=stream)
            print(f"  do    : {explanation.remedy}", file=stream)

    selection = payload.get("testmon_selection")
    if isinstance(selection, dict):
        print(f"\nselection: {selection.get('selection_mode')} ({selection.get('state_status')})", file=stream)
        reason = selection.get("state_reason")
        if reason:
            print(f"  reason: {reason}", file=stream)
        missing = selection.get("missing_executable_paths") or []
        if missing:
            print(f"  {len(missing)} changed file(s) not covered by the graph:", file=stream)
            for path in list(missing)[:10]:
                print(f"    - {path}", file=stream)
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more", file=stream)
        runtime_data = selection.get("runtime_data_paths") or []
        if runtime_data:
            # Recorded exposure, not an escalation: untraceable changes no
            # longer force the complete corpus (operator decision 2026-08-18).
            print(
                f"  {len(runtime_data)} changed non-Python runtime file(s) are outside Python tracing"
                " (recorded exposure; only re-execution after an environment change covers them)",
                file=stream,
            )

    aggregate = payload.get("pytest_aggregate")
    if isinstance(aggregate, dict):
        selected = aggregate.get("selected_union_count")
        if selected is not None:
            attested = aggregate.get("attested_unchanged_count")
            attested_text = (
                f" ({attested} attested by unchanged recorded green)" if isinstance(attested, int) and attested else ""
            )
            print(
                f"\nselected {selected} test(s); complete corpus covered: "
                f"{aggregate.get('complete_corpus_covered')}{attested_text}",
                file=stream,
            )

    failing_steps = [
        step for step in (payload.get("steps") or []) if isinstance(step, dict) and step.get("exit") not in (0, None)
    ]
    if failing_steps:
        print("\nfailing steps:", file=stream)
        for step in failing_steps:
            print(
                f"  - {step.get('step_id')} exit={step.get('exit')} {step.get('diagnosis') or ''}".rstrip(), file=stream
            )
            output_path = step.get("output_path")
            if isinstance(output_path, str) and Path(output_path).is_file():
                tail = Path(output_path).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]
                for line in tail:
                    print(f"      {line}", file=stream)

    # Graph fate matters more than any single step: a deleted graph means the
    # NEXT run pays a ~9.5x complete-corpus bootstrap, and until this line
    # existed the only evidence was a buried receipt field.
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        cleanup_paths = step.get("testmon_cleanup_paths")
        if cleanup_paths:
            print("\nthis run DELETED the testmon graph (next run will bootstrap the complete corpus):", file=stream)
            for path in cleanup_paths:
                print(f"  - {path}", file=stream)
        if step.get("testmon_graph_retained"):
            print(
                f"\ntestmon graph retained despite receipt invalidation: {step['testmon_graph_retained']}",
                file=stream,
            )

    artifact_dir = payload.get("artifact_dir")
    if artifact_dir:
        print(f"\nartifacts: {artifact_dir}", file=stream)


def _aggregate(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    """The per-run pytest aggregate, which is where selection counts live."""
    aggregate = entry.get("pytest_aggregate")
    return aggregate if isinstance(aggregate, dict) else {}


def _render_history(hours: float, stream: Any) -> int:
    """Answer "where did the time go" from the durable verification history.

    This exists because the question kept being asked and kept requiring an ad
    hoc DuckDB query against the lynchpin substrate, which materialises on its
    own cadence and was 17 hours stale when it mattered. The history file is
    current by construction.
    """
    if not VERIFY_HISTORY_PATH.exists():
        print(f"why: no run history at {VERIFY_HISTORY_PATH}", file=sys.stderr)
        return 1
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    rows: list[dict[str, Any]] = []
    with VERIFY_HISTORY_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            started = str(entry.get("started_at") or "")
            try:
                when = datetime.fromisoformat(started)
            except ValueError:
                continue
            if when >= cutoff:
                rows.append(entry)

    if not rows:
        print(f"no verification runs in the last {hours:g}h", file=stream)
        return 0

    total = sum(float(entry.get("duration_s") or 0.0) for entry in rows)
    print(f"{len(rows)} run(s) in the last {hours:g}h, {total / 3600:.2f}h of wall time", file=stream)
    # Say this out loud, because the omission is biased rather than random. A
    # run only reaches the history append after it finishes, so anything killed
    # -- which in practice means the long bootstraps someone gave up waiting on
    # -- is absent. Measured 2026-08-18: a lane that burned 48 minutes on a
    # single terminated bootstrap contributed 0.04h to this view.
    print("(killed runs never reach this record, so long bootstraps are under-counted)\n", file=stream)

    def _summarise(key: str, label: str) -> None:
        buckets: dict[str, tuple[int, float]] = {}
        for entry in rows:
            name = str(entry.get(key) or "-")
            runs, seconds = buckets.get(name, (0, 0.0))
            buckets[name] = (runs + 1, seconds + float(entry.get("duration_s") or 0.0))
        print(f"by {label}:", file=stream)
        for name, (runs, seconds) in sorted(buckets.items(), key=lambda item: -item[1][1]):
            print(f"  {name:38} {runs:5} run(s)  {seconds / 3600:7.2f}h", file=stream)
        print(file=stream)

    _summarise("tier", "tier")
    _summarise("diagnosis", "diagnosis")

    # The expensive shape, called out by name: runs that selected nothing and
    # therefore executed everything.
    full = [
        entry
        for entry in rows
        if not _aggregate(entry).get("selected_union_count")
        and (_aggregate(entry).get("terminal_union_count") or 0) > 1000
    ]
    if full:
        wasted = sum(float(entry.get("duration_s") or 0.0) for entry in full)
        share = wasted / total * 100 if total else 0.0
        print(
            f"{len(full)} run(s) selected nothing and ran the full corpus: "
            f"{wasted / 3600:.2f}h ({share:.0f}% of the window)",
            file=stream,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain the most recent verification run.")
    parser.add_argument("--run", help="Explain a specific run id instead of the most recent.")
    parser.add_argument(
        "--history",
        nargs="?",
        type=float,
        const=24.0,
        metavar="HOURS",
        help="Summarise where verification time went over the last HOURS (default 24).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the receipt as JSON.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.history is not None:
        return _render_history(args.history, sys.stdout)

    if args.run:
        path = VERIFY_RUNS_DIR / args.run / "run.json"
        if not path.is_file():
            print(f"why: no such run: {args.run}", file=sys.stderr)
            return 1
    else:
        found = _latest_run(VERIFY_RUNS_DIR)
        if found is None:
            print("why: no verification runs recorded yet -- run 'devtools verify' first.", file=sys.stderr)
            return 1
        path = found

    payload = _load(path)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    _render(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
