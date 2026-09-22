"""Executable invariant for the affected testmon selection contract."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from devtools.pytest_invocation import (
    CLOSED_WORLD_COLLECTION_ARGS,
    DEVTOOLS_PLUGIN_ARGS,
    IGNORED_COLLECTION_ARGS,
    managed_plugin_args,
)
from devtools.testmon_provision import TESTMON_COVERAGE_CORE, TESTMON_ENVIRONMENT, inspect_testmon_graph
from devtools.toolchain import venv_python
from devtools.verify import _pytest_worker_args
from devtools.worker_memory import CORPUS_MAX_WORKERS

#: The narrowest managed corpus width this gate accepts as a default. It is a
#: literal, not a second reading of :data:`CORPUS_MAX_WORKERS`, because the
#: mutation this gate claims to catch is an edit to that constant: comparing
#: the produced argument against the same constant moves both sides together,
#: so ``CORPUS_MAX_WORKERS = 0`` produced ``-n 0`` on both and the gate stayed
#: green while the corpus would have run without xdist at all.
MINIMUM_DEFAULT_WORKERS = 1


def worker_default_refusal(worker_args: list[str]) -> str | None:
    """Why the managed default width is unusable, or ``None``.

    Two independent claims, because neither catches the other's mutation:
    the produced arguments must still be the declared corpus width, AND that
    width must be a real xdist width on its own terms. Only the second
    survives an edit to :data:`CORPUS_MAX_WORKERS` itself.
    """
    if worker_args[:2] != ["--dist=loadgroup", "-n"] or len(worker_args) != 3:
        return f"managed verification does not request grouped xdist workers: {worker_args!r}"
    try:
        width = int(worker_args[2])
    except ValueError:
        return f"managed verification requested a non-numeric worker width: {worker_args[2]!r}"
    if width < MINIMUM_DEFAULT_WORKERS:
        return (
            f"managed verification defaults to {width} workers, below the "
            f"{MINIMUM_DEFAULT_WORKERS}-worker floor; the corpus would run without xdist"
        )
    if width != CORPUS_MAX_WORKERS:
        return f"managed verification defaults to {width} workers, not the declared corpus width {CORPUS_MAX_WORKERS}"
    return None


def main(_argv: list[str] | None = None) -> int:
    """Trace a generated corpus, edit one leaf, and require a small rerun.

    Anti-vacuity: changing ``--testmon-forceselect`` to ``--testmon-noselect``
    makes the second run execute all tests; removing xdist from this command
    would stop this gate from covering the hosted route; setting
    ``CORPUS_MAX_WORKERS = 0`` makes :func:`worker_default_refusal` fail;
    refusing the seed makes the first run fail to establish the graph.
    """
    configured_workers = os.environ.pop("POLYLOGUE_PYTEST_WORKERS", None)
    try:
        default_worker_args = _pytest_worker_args()
    finally:
        if configured_workers is not None:
            os.environ["POLYLOGUE_PYTEST_WORKERS"] = configured_workers
    refusal = worker_default_refusal(default_worker_args)
    if refusal is not None:
        print(f"testmon-selection: {refusal}")
        return 1
    with tempfile.TemporaryDirectory(prefix="polylogue-testmon-gate-") as temporary:
        root = Path(temporary)
        tests = root / "tests"
        tests.mkdir()
        for index in range(25):
            (root / f"leaf{index}.py").write_text(f"def value():\n    return {index}\n", encoding="utf-8")
            lines = [f"from leaf{index} import value", ""]
            for test_index in range(4):
                lines += [f"def test_{test_index}():", f"    assert value() == {index}", ""]
            (tests / f"test_leaf{index}.py").write_text("\n".join(lines), encoding="utf-8")
        (root / ".cache" / "testmon").mkdir(parents=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
        env["COVERAGE_CORE"] = TESTMON_COVERAGE_CORE
        env["TESTMON_DATAFILE"] = str(root / ".cache" / "testmon" / "testmondata")
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env.pop("PYTEST_ADDOPTS", None)
        env.pop("PYTEST_PLUGINS", None)
        base = [
            str(venv_python(root=Path(__file__).resolve().parents[1])),
            "-m",
            "pytest",
            "-q",
            *IGNORED_COLLECTION_ARGS,
            "-p",
            "pytest_jsonreport",
            *DEVTOOLS_PLUGIN_ARGS,
            *managed_plugin_args(testmon=True),
            *CLOSED_WORLD_COLLECTION_ARGS[:-1],
            "-p",
            "no:randomly",
            *_pytest_worker_args(),
        ]
        first = subprocess.run(
            [
                *base,
                "--testmon",
                f"--testmon-env={TESTMON_ENVIRONMENT}",
                "--testmon-noselect",
                "--json-report",
                f"--json-report-file={root / 'first.json'}",
                "tests",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if first.returncode != 0:
            print(first.stdout + first.stderr)
            return first.returncode or 1
        first_report = json.loads((root / "first.json").read_text(encoding="utf-8"))
        seeded = len(first_report.get("tests", [])) if isinstance(first_report, dict) else 0
        total = 100
        if seeded != total:
            print(f"testmon-selection: seed collected {seeded} of {total} tests")
            return 1
        seed = inspect_testmon_graph(root)
        if not seed.usable:
            print(f"testmon-selection: seed graph is not usable: {seed.reason}")
            return 1
        (root / "leaf0.py").write_text("def value():\n    return 0 + 0\n", encoding="utf-8")
        report_path = root / "report.json"
        second = subprocess.run(
            [
                *base,
                "--testmon",
                f"--testmon-env={TESTMON_ENVIRONMENT}",
                "--json-report",
                f"--json-report-file={report_path}",
                "--testmon-forceselect",
                "tests",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if second.returncode != 0:
            print(second.stdout + second.stderr)
            return second.returncode or 1
        output = second.stdout + second.stderr
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        selected_nodeids = (
            [
                test.get("nodeid")
                for test in report.get("tests", [])
                if isinstance(test, dict) and isinstance(test.get("nodeid"), str)
            ]
            if isinstance(report, dict)
            else []
        )
        selected = len(selected_nodeids)
        expected = {f"tests/test_leaf0.py::test_{index}" for index in range(4)}
        if set(selected_nodeids) != expected:
            print(
                f"testmon-selection: selected {selected} of {total}, expected {len(expected)} leaf0 tests; "
                f"nodeids={selected_nodeids!r}\n{output}"
            )
            return 1
    print(f"testmon-selection: selected {selected} of {total}; workers={CORPUS_MAX_WORKERS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
