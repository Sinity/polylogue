"""Prove the declared test corpus still collects, without running a test.

Gate classification: **blocking collectability check**.

This is an explicit diagnostic gate. ``devtools verify --quick`` does not
collect the corpus; the complete-corpus run collects it while executing tests.
Focused tests remain the author's pre-merge test evidence. Use this gate when
collectability itself needs checking without executing the full corpus.

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
loads neither testmon nor xdist. Affected admission uses the same collection
owner with testmon against a caller-owned temporary graph snapshot and records
final selected node evidence. Neither route writes the checkout graph.

Usage:
  devtools gate test-collection
  devtools gate test-collection --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devtools.agent_env import HARNESS_RUN_ENV
from devtools.pytest_invocation import (
    CLOSED_WORLD_COLLECTION_ARGS,
    IGNORED_COLLECTION_ARGS,
    devtools_plugin_args,
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
#: runs no test body and holds no slot, so an explicit gate invocation may run
#: inside an agent job without an execution slot.
COLLECTION_RUN_ID = "gate-test-collection"

__all__ = [
    "COLLECTION_RUN_ID",
    "collection_command",
    "collection_env",
    "CollectedSelection",
    "collect_selection",
    "main",
]


def collection_env() -> dict[str, str]:
    return {**os.environ, HARNESS_RUN_ENV: COLLECTION_RUN_ID}


def collection_command(*, root: Path = ROOT, paths: Sequence[str] | None = None, testmon: bool = False) -> list[str]:
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
        *managed_plugin_args(testmon=testmon, xdist=False),
        *roots,
        "-p",
        "no:randomly",
    ]


@dataclass(frozen=True, slots=True)
class CollectedSelection:
    """Final collection evidence, with omitted identities explicitly visible."""

    selected_count: int
    nodeids: tuple[str, ...]
    omitted: int


def collect_selection(
    *,
    root: Path = ROOT,
    paths: Sequence[str] | None = None,
    datafile: Path | None = None,
    nodeid_limit: int,
    environment: Mapping[str, str],
) -> CollectedSelection | None:
    """Collect one actual launch against its supplied graph without executing it.

    The original graph is never opened here: ``datafile`` is the caller-owned
    admission snapshot. An unsuccessful collection or malformed evidence is
    unavailable, never an empty selection.
    """
    command = collection_command(root=root, paths=paths, testmon=datafile is not None)
    command.extend(devtools_plugin_args(testmon=datafile is not None))
    if datafile is not None:
        from devtools.testmon_provision import testmon_environment

        command.extend(
            (
                "--testmon",
                "--testmon-env=" + testmon_environment(root, environment.get("HYPOTHESIS_PROFILE")),
                "--testmon-forceselect",
            )
        )
    with tempfile.TemporaryDirectory(prefix="polylogue-selection-") as temporary:
        evidence = Path(temporary) / "selection.json"
        env = dict(environment)
        env.update(
            {
                HARNESS_RUN_ENV: COLLECTION_RUN_ID,
                "POLYLOGUE_PYTEST_SELECTION_PATH": str(evidence),
                "POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT": str(nodeid_limit),
            }
        )
        # Collection owns only its evidence, not the outer execution ledgers.
        for name in (
            "POLYLOGUE_PYTEST_EVENTS_PATH",
            "POLYLOGUE_PYTEST_EVENTS_DIR",
            "POLYLOGUE_PYTEST_SUMMARY_PATH",
            "PYTEST_XDIST_WORKER",
        ):
            env.pop(name, None)
        if datafile is not None:
            env["TESTMON_DATAFILE"] = str(datafile)
        else:
            env.pop("TESTMON_DATAFILE", None)
        guard = None
        try:
            if datafile is not None:
                from devtools.execution_source import start_execution
                from devtools.pytest_slot import _focused_worktree_provenance, _group_reaped

                env["POLYLOGUE_FOCUSED_WORKTREE_PROVENANCE"] = "1"
                guard = start_execution(root, env)
                provenance = _focused_worktree_provenance(str(root), env)
                assert guard is not None
                execution_command = guard.command(command, env, provenance)
                process = subprocess.Popen(
                    execution_command,
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    process_group=0,
                    pass_fds=guard.pass_fds,
                )
                guard.launched = True
                completed: subprocess.CompletedProcess[Any] = subprocess.CompletedProcess(command, process.wait())
                if not _group_reaped(process.pid):
                    guard.failure = "collection descendants remain alive"
            else:
                completed = subprocess.run(
                    command, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, env=env
                )
            if guard is not None:
                from devtools.execution_source import finish_execution

                stability = finish_execution(guard, env)
                guard = None
                if stability is None or stability["status"] != "stable":
                    return None
            if completed.returncode not in (0, _EXIT_NO_TESTS_COLLECTED):
                return None
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            count, omitted, nodes = (
                payload[key] for key in ("selected_count", "selected_nodeids_omitted", "selected_nodeids")
            )
            if (
                type(count) is not int
                or type(omitted) is not int
                or count < 0
                or omitted < 0
                or not isinstance(nodes, list)
                or any(not isinstance(node, str) or not node for node in nodes)
                or count != len(nodes) + omitted
            ):
                return None
            return CollectedSelection(count, tuple(nodes), omitted)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return None
        finally:
            if guard is not None:
                from devtools.execution_source import finish_execution

                finish_execution(guard, env)


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
