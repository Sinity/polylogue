"""Measure ``--help`` wall-clock latency against the interactive-tier cold-CLI budget.

Use ``devtools bench help-latency`` to catch import-tax regressions on the
CLI ``--help`` path continuously (polylogue-20d.2). The interactive SLO tier
(polylogue-20d.14, ``docs/plans/slo-catalog.yaml``) states a <700ms budget for
a cold (no warm daemon) CLI invocation; ``--help`` is the cheapest possible
invocation of any command (no archive I/O, no query execution) so it is the
tightest floor on Python/import overhead. A regression here means every
daemonless invocation of that command pays the same tax.

Each target is run as a fresh subprocess (``python -m polylogue.cli <args>``)
several times; the *minimum* wall time is compared against budget rather than
the mean, because process-launch jitter (scheduler contention, page-cache
misses) only ever adds latency on a shared dev host, never subtracts it. This
mirrors the host-variable framing of the other ``devtools bench`` probes:
wall-clock is diagnostic and campaign-comparable, but the budget comparison
here IS a CI gate for "required" targets (unlike the wall-clock-only probes),
because import cost is deterministic given the source tree, not host load.

Targets marked ``gate="informational"`` are measured and reported but never
fail the check — they document a known-slow path with an open follow-up
(currently: ``ops maintenance migrate-tier``, whose own ``--help`` needs
``DURABLE_MIGRATION_TIERS`` at Click-decoration time to render its argument
choices, forcing the ``polylogue.storage.sqlite.archive_tiers`` package's
eager DDL-import chain even for `--help`; see
``polylogue/cli/commands/maintenance/_migrate_tier.py``'s module docstring).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

Gate = Literal["required", "informational"]


@dataclass(frozen=True, slots=True)
class HelpLatencyTarget:
    label: str
    args: tuple[str, ...]
    budget_ms: int
    gate: Gate


# Budget follows the 20d.14 interactive-tier "cold CLI (no daemon) <700ms"
# line in docs/plans/slo-catalog.yaml.
_DEFAULT_BUDGET_MS = 700

TARGETS: tuple[HelpLatencyTarget, ...] = (
    HelpLatencyTarget("root", ("--help",), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("find", ("find", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("read", ("read", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("mark", ("mark", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("select", ("select", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("analyze", ("analyze", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("import", ("import", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("config", ("config", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("dashboard", ("dashboard", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("ops", ("ops", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget("reset", ("reset", "--help"), _DEFAULT_BUDGET_MS, "required"),
    # `ops maintenance` is now a package of one lazily-dispatched submodule
    # per subcommand (polylogue-sod7): the group listing and every subcommand
    # except migrate-tier import only click/paths/config at `--help` time,
    # deferring ArchiveStore/blob_gc/blob_integrity/embeddings.reconcile/
    # migration_runner/the archive_tiers DDL stack into each command's own
    # function body.
    HelpLatencyTarget("ops-maintenance", ("ops", "maintenance", "--help"), _DEFAULT_BUDGET_MS, "required"),
    HelpLatencyTarget(
        "ops-maintenance-archive-read",
        ("ops", "maintenance", "archive-read", "--help"),
        _DEFAULT_BUDGET_MS,
        "required",
    ),
    # migrate-tier is the one documented exception: its --help needs
    # DURABLE_MIGRATION_TIERS at Click-decoration time (to render the `tier`
    # argument's valid choices), which forces the archive_tiers package's
    # eager DDL-import chain even for --help. See _migrate_tier.py.
    HelpLatencyTarget(
        "ops-maintenance-migrate-tier",
        ("ops", "maintenance", "migrate-tier", "--help"),
        _DEFAULT_BUDGET_MS,
        "informational",
    ),
)


def _time_invocation(args: tuple[str, ...], *, repeats: int) -> float:
    """Return the minimum wall-clock ms across ``repeats`` fresh subprocess runs."""

    best: float | None = None
    for _ in range(repeats):
        started = perf_counter()
        subprocess.run(
            [sys.executable, "-m", "polylogue.cli", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed_ms = (perf_counter() - started) * 1_000
        if best is None or elapsed_ms < best:
            best = elapsed_ms
    assert best is not None
    return best


def measure(*, repeats: int = 3, targets: tuple[HelpLatencyTarget, ...] = TARGETS) -> dict[str, object]:
    results = []
    for target in targets:
        elapsed_ms = _time_invocation(target.args, repeats=repeats)
        within_budget = elapsed_ms <= target.budget_ms
        results.append(
            {
                "label": target.label,
                "args": list(target.args),
                "gate": target.gate,
                "budget_ms": target.budget_ms,
                "elapsed_ms": round(elapsed_ms, 1),
                "within_budget": within_budget,
            }
        )
    violations = [r["label"] for r in results if r["gate"] == "required" and not r["within_budget"]]
    return {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "repeats": repeats,
        "results": results,
        "violations": violations,
        "ok": not violations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Subprocess runs per target; minimum wins.")
    parser.add_argument("--json", action="store_true", help="Emit the full JSON report instead of a table.")
    parser.add_argument("--out", type=Path, default=None, help="Also write the JSON report to this path.")
    args = parser.parse_args(argv)

    report = measure(repeats=max(1, args.repeats))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        results = cast("list[dict[str, object]]", report["results"])
        for result in results:
            marker = "OK" if result["within_budget"] else "OVER"
            flag = "" if result["gate"] == "required" else "  (informational)"
            print(f"{marker:>4}  {result['label']:<32} {result['elapsed_ms']:>7.1f}ms / {result['budget_ms']}ms{flag}")
        violations = cast("list[str]", report["violations"])
        if violations:
            print(f"\nBudget violations (required): {', '.join(violations)}")
        else:
            print("\nAll required targets within budget.")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
