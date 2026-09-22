"""Prove the declared test corpus still collects, without running a test.

Gate classification: **blocking collectability check**.

``devtools verify --quick`` is the hosted merge gate and deliberately runs no
pytest: the suite's pre-merge evidence is the author's explicitly selected
local or hosted run. That policy has one blind spot it cannot see past -- the
author's selection describes *the tree they ran it on*, and a rebase can leave
behind a test module that no longer imports. Such a module raises on
collection, so the selection that "passed" selected zero tests, and nothing
downstream notices.

This gate collects the whole declared corpus and executes none of it. It is
honest about its reach:

* it catches a module that cannot be imported, a syntax error, a missing
  symbol, a renamed fixture referenced at module scope, and a collection root
  that no longer exists;
* it does **not** catch a failing assertion, a wrong result, or anything that
  needs a test body to run.

Collection uses the declared closed-world arguments from
``devtools.pytest_invocation`` -- the same values that define what the corpus
*is* for every other managed lane -- and the checkout venv's interpreter. It
loads neither testmon nor xdist: it writes no fingerprints and therefore
leaves the checkout's corpus graph exactly as it found it.

Usage:
  devtools gate test-collection
  devtools gate test-collection --json
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from devtools.agent_env import HARNESS_RUN_ENV
from devtools.pytest_invocation import (
    CLOSED_WORLD_COLLECTION_ARGS,
    IGNORED_COLLECTION_ARGS,
    managed_plugin_args,
)
from devtools.required_gate import evidence_gate_result
from devtools.toolchain import venv_python
from polylogue.core.json import dumps

ROOT = Path(__file__).resolve().parents[1]

#: ``-q --collect-only`` ends with a line such as ``1234 tests collected in 3s``
#: (or ``... collected, 2 errors``). The count is evidence that collection
#: reached the corpus rather than bailing out early.
_COLLECTED_RE = re.compile(r"(\d+)\s+tests?\s+collected")

#: pytest's exit code for "collection succeeded and selected nothing". For the
#: whole declared corpus that is never a legitimate outcome.
_EXIT_NO_TESTS_COLLECTED = 5

#: This step is a declared devtools gate, not a bare pytest session, so it
#: names itself to ``devtools.agent_env.refuse_bare_pytest``. That refusal
#: protects the host's pytest slot from unmetered *execution*; ``--collect-only``
#: runs no test body and holds no slot, and refusing it would make the gate red
#: inside every agent job -- exactly where ``verify --quick`` is run.
COLLECTION_RUN_ID = "gate-test-collection"

__all__ = ["COLLECTION_RUN_ID", "collection_command", "collection_env", "count_collected", "main"]


def collection_env() -> dict[str, str]:
    return {**os.environ, HARNESS_RUN_ENV: COLLECTION_RUN_ID}


def collection_command(*, root: Path = ROOT, paths: Sequence[str] | None = None) -> list[str]:
    """Return the declared collect-only invocation.

    ``paths`` narrows it to named files while keeping every other declared
    argument, so a caller that needs to count part of the corpus counts it
    under the same rules that define what the corpus is. The declared root
    (the trailing ``tests`` argument) is what the named paths replace.
    """
    roots = list(CLOSED_WORLD_COLLECTION_ARGS[:-1]) + list(paths) if paths else list(CLOSED_WORLD_COLLECTION_ARGS)
    return [
        venv_python(root=root),
        "-m",
        "pytest",
        "-q",
        "--collect-only",
        *IGNORED_COLLECTION_ARGS,
        *managed_plugin_args(testmon=False, xdist=False),
        *roots,
        "-p",
        "no:randomly",
    ]


def count_collected(paths: Sequence[str], *, root: Path = ROOT) -> int | None:
    """How many tests the declared rules collect from ``paths``.

    ``None`` when collection did not succeed -- an unimportable module, a
    missing path, a refused interpreter. A caller that cannot get this number
    has no bound on what those paths will execute, and ``None`` is that
    answer rather than a zero that would read as "nothing there".
    """
    if not paths:
        return 0
    command = collection_command(root=root, paths=paths)
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False, env=collection_env())
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    match = _COLLECTED_RE.search(f"{completed.stdout}\n{completed.stderr}")
    return int(match.group(1)) if match else None


def _failure_details(output: str, *, limit: int = 12) -> tuple[str, ...]:
    """Return the lines that name what failed to collect."""
    interesting = [
        line.rstrip() for line in output.splitlines() if line.startswith(("E  ", "ERROR", "_____", "ImportError"))
    ]
    return tuple(interesting[-limit:]) if interesting else tuple(output.splitlines()[-limit:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools gate test-collection",
        description="Collect the declared test corpus without running it.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv or []))

    command = collection_command(root=ROOT)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, env=collection_env())
    output = f"{completed.stdout}\n{completed.stderr}"
    match = _COLLECTED_RE.search(output)
    collected = int(match.group(1)) if match else 0

    ok = completed.returncode == 0 and collected > 0
    details: tuple[str, ...] = ()
    if not ok:
        details = _failure_details(output)
    gate = evidence_gate_result(
        gate="test-collection",
        executable=command[0],
        executable_available=Path(command[0]).exists(),
        required_count=1,
        inspected_count=1 if completed.returncode != 127 else 0,
        error_count=0 if ok else 1,
        details=details,
    )

    if args.json:
        print(
            dumps(
                {
                    "ok": ok,
                    "collected": collected,
                    "returncode": completed.returncode,
                    "no_tests_collected": completed.returncode == _EXIT_NO_TESTS_COLLECTED,
                    "required_gate": gate.to_payload(),
                }
            )
        )
    elif ok:
        print(f"  {collected} tests collect cleanly (none executed).")
    else:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        print(
            f"  collection failed (pytest exit {completed.returncode}): a test module in the declared corpus "
            "cannot be imported or collected. This gate runs no test bodies, so a failure here means a module "
            "could not run at all.",
            file=sys.stderr,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
