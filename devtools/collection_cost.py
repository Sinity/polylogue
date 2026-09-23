"""Measure what a pytest selection pays before it runs a single test.

Every xdist worker collects its whole selection independently, so collection
is paid once per worker and is what bounds how many workers fit the pytest
slice. polylogue-016xl states its target as a number -- whole-corpus
``--collect-only`` peak RSS -- and a number nobody can reproduce on demand is
not a ratchet. This command is that reproduction, so "measure before and after
on the same head" is one invocation instead of a rewritten ad-hoc script.

What is measured, and why this way:

* The invocation is ``devtools.verify_test_collection``'s declared one, not a
  hand-rolled pytest line. That matters twice: the declared form collects the
  declared corpus without the spurious ``Plugin already registered`` error a
  bare ``-p no:cacheprovider`` run produces, and it carries the gate's
  ``HARNESS_RUN_ENV`` identity, so the measurement runs inside an agent lane
  rather than being refused by ``refuse_bare_pytest``. That refusal exists to
  keep unmetered *execution* off the host's pytest slot; ``--collect-only``
  runs no test body and holds no slot.
* The child's peak comes from ``resource.getrusage(RUSAGE_CHILDREN)`` around
  one subprocess, so nothing this process imports is charged to the selection.

``wall_clock_s`` is load-sensitive and is reported as an observation, not a
budget: the same whole-corpus collection measured 33.2 s in the bulk pool and
93.2 s in an agent lane while a corpus run and six sibling lanes were live.
The peak is the number to compare.

The budget is a ceiling the caller asks for, never a default: a run above it
exits 3, distinct from the selection's own failure exit, so a collection error
is never reported as a budget breach and a breach is never reported as an
error.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Final

from devtools import repo_root
from devtools.pytest_invocation import CLOSED_WORLD_COLLECTION_ARGS
from devtools.verify_test_collection import collection_command, collection_env

__all__ = [
    "COLLECTION_COST_TARGET_KIB_PER_ITEM",
    "CORPUS_COLLECTION_BUDGET_MIB",
    "OVER_BUDGET_EXIT",
    "collection_argv",
    "main",
    "measure_collection",
]

#: The declared corpus root inside the gate's invocation. A narrower selection
#: SUBSTITUTES for it; appending would silently add to it, which is how a
#: one-file measurement quietly collects 23,547 tests and reports the corpus
#: number as if it were the file's.
_CORPUS_ROOT_ARG: Final = CLOSED_WORLD_COLLECTION_ARGS[-1]

#: polylogue-016xl's target for the whole corpus: 100 MiB below the 533 MiB
#: measured 2026-09-06 over 20,681 tests. Recorded so the gap is a number
#: rather than a memory, and deliberately NOT enforced by default -- the
#: target has never been met, and a command that fails by default is a command
#: nobody runs.
#:
#: The 533 MiB baseline was taken with a bare ``pytest --collect-only -q
#: -p no:cacheprovider --continue-on-collection-errors``, which is not the
#: declared invocation this command uses. Compare numbers from this command
#: against each other.
CORPUS_COLLECTION_BUDGET_MIB: Final = 430

# The rewritten campaign criterion measures the collection footprint against
# the corpus it collected.  Keeping the target in KiB/item prevents a larger
# corpus from looking better merely because its absolute RSS happened to move.
COLLECTION_COST_TARGET_KIB_PER_ITEM: Final = 22.3

#: Over the budget the caller asked for. Distinct from the selection's own
#: non-zero exit so the two failures are never confused for each other.
OVER_BUDGET_EXIT: Final = 3

_COLLECTED_RE: Final = re.compile(r"(\d[\d,]*)\s+tests?\s+collected")


def _collected_count(output: str) -> int | None:
    """The count pytest reported, or None when it reported none.

    None rather than 0: "pytest did not say" and "the selection is empty" are
    different facts, and collapsing the first into the second is how an
    unavailable measurement becomes a zero.
    """
    last: int | None = None
    for match in _COLLECTED_RE.finditer(output):
        last = int(match.group(1).replace(",", ""))
    return last


def _cost_kib_per_item(peak_rss_mib: float, collected: int | None) -> float | None:
    """Return collection RSS per collected item, or ``None`` without a count."""
    if collected is None or collected <= 0:
        return None
    return round(peak_rss_mib * 1024 / collected, 2)


def collection_argv(selection: list[str], *, root: Path) -> list[str]:
    """The gate's declared collect-only invocation, narrowed to ``selection``.

    Built from ``verify_test_collection.collection_command`` rather than
    re-derived, so the measurement and the gate cannot drift into collecting
    different things.
    """

    command = list(collection_command(root=root))
    if not selection:
        return command
    try:
        index = len(command) - 1 - command[::-1].index(_CORPUS_ROOT_ARG)
    except ValueError as exc:  # pragma: no cover - the declared form changed
        raise RuntimeError(
            f"the declared collection invocation no longer names the corpus root {_CORPUS_ROOT_ARG!r}; "
            "narrowing it would add to the corpus instead of replacing it"
        ) from exc
    return [*command[:index], *selection, *command[index + 1 :]]


def measure_collection(selection: list[str], *, root: Path) -> dict[str, Any]:
    """Collect ``selection`` in a child process and report what it cost."""

    command = collection_argv(selection, root=root)
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.monotonic()
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False, env=collection_env())
    elapsed = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    peak_rss_mib = round(max(before, after) / 1024, 1)
    peak_rss_delta_mib = round(max(0, after - before) / 1024, 1)
    collected = _collected_count(output)
    return {
        "kind": "polylogue.collection-cost",
        "selection": list(selection) or ["<whole corpus>"],
        "collected": collected,
        "wall_clock_s": round(elapsed, 2),
        # ru_maxrss is the high-water mark across every reaped child, so the
        # later reading is this child's peak unless an earlier child in the
        # same process was larger. The delta is reported beside it, not
        # instead of it, so that case stays visible.
        "peak_rss_mib": peak_rss_mib,
        "peak_rss_delta_mib": peak_rss_delta_mib,
        # The delta excludes this command's already-paid child high-water
        # mark. It is the comparable collection cost when this process has
        # measured more than one selection; retain the absolute peak too.
        "collection_cost_kib_per_item": _cost_kib_per_item(peak_rss_delta_mib, collected),
        "returncode": completed.returncode,
        "tail": [line for line in output.strip().splitlines() if line.strip()][-3:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure a pytest selection's collection cost.")
    parser.add_argument(
        "selection",
        nargs="*",
        help="pytest selection to collect (default: the whole declared corpus)",
    )
    parser.add_argument(
        "--budget-kib-per-item",
        type=float,
        default=None,
        help="exit 3 when collection RSS per collected item exceeds this value",
    )
    parser.add_argument(
        "--budget-mib",
        type=int,
        default=None,
        help=(
            "exit 3 when the collection peak exceeds this many MiB "
            f"(polylogue-016xl's corpus target is {CORPUS_COLLECTION_BUDGET_MIB})"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    arguments = parser.parse_args(argv)

    result = measure_collection(list(arguments.selection), root=repo_root())
    if arguments.budget_mib is not None:
        result["budget_mib"] = arguments.budget_mib
        result["within_budget"] = result["peak_rss_mib"] <= arguments.budget_mib
    if arguments.budget_kib_per_item is not None:
        measured = result["collection_cost_kib_per_item"]
        result["budget_kib_per_item"] = arguments.budget_kib_per_item
        result["within_budget_kib_per_item"] = measured is not None and measured <= arguments.budget_kib_per_item

    if arguments.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        collected = result["collected"]
        counted = "count not reported" if collected is None else f"{collected} test(s)"
        print(f"collection: {counted} in {result['wall_clock_s']}s, peak {result['peak_rss_mib']} MiB")
        print(f"  selection: {' '.join(str(item) for item in result['selection'])}")
        if arguments.budget_mib is not None:
            print(f"  budget: {'within' if result['within_budget'] else 'OVER'} {arguments.budget_mib} MiB")
        if arguments.budget_kib_per_item is not None:
            measured = result["collection_cost_kib_per_item"]
            verdict = measured is not None and measured <= arguments.budget_kib_per_item
            print(
                f"  collection cost: {measured if measured is not None else 'unmeasured'} KiB/item ({'within' if verdict else 'OVER'} {arguments.budget_kib_per_item})"
            )
        for line in result["tail"]:
            print(f"  {line}")

    if result["returncode"] != 0:
        return int(result["returncode"])
    if arguments.budget_mib is not None and not result["within_budget"]:
        return OVER_BUDGET_EXIT
    if arguments.budget_kib_per_item is not None and not result["within_budget_kib_per_item"]:
        return OVER_BUDGET_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
