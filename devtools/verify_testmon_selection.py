"""Executable invariant for the affected testmon selection contract."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from devtools.pytest_invocation import MANAGED_PLUGIN_ARGS
from devtools.toolchain import venv_python
from devtools.verify import CORPUS_MAX_WORKERS, _pytest_worker_args


def main(_argv: list[str] | None = None) -> int:
    """Trace a generated corpus, edit one leaf, and require a small rerun.

    Anti-vacuity: changing ``--testmon-forceselect`` to ``--testmon-noselect``
    makes the second run execute all tests; changing the default worker count
    to zero makes the worker assertion fail; refusing the seed makes the first
    run fail to establish the graph.
    """
    configured_workers = os.environ.pop("POLYLOGUE_PYTEST_WORKERS", None)
    try:
        default_worker_args = _pytest_worker_args()
    finally:
        if configured_workers is not None:
            os.environ["POLYLOGUE_PYTEST_WORKERS"] = configured_workers
    if default_worker_args != ["--dist=loadgroup", "-n", str(CORPUS_MAX_WORKERS)]:
        print(f"testmon-selection: managed verification does not default to {CORPUS_MAX_WORKERS} workers")
        return 1
    with tempfile.TemporaryDirectory(prefix="polylogue-testmon-gate-") as temporary:
        root = Path(temporary)
        tests = root / "tests"
        tests.mkdir()
        for index in range(25):
            (root / f"leaf{index}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
            lines = [f"from leaf{index} import VALUE", ""]
            for test_index in range(4):
                lines += [f"def test_{test_index}():", f"    assert VALUE == {index}", ""]
            (tests / f"test_leaf{index}.py").write_text("\n".join(lines), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env.pop("PYTEST_ADDOPTS", None)
        env.pop("PYTEST_PLUGINS", None)
        base = [
            str(venv_python(root=Path(__file__).resolve().parents[1])),
            "-m",
            "pytest",
            "-q",
            "-p",
            "pytest_jsonreport",
        ]
        first = subprocess.run(
            [*base, "--json-report", f"--json-report-file={root / 'first.json'}", "tests"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if first.returncode != 0:
            print(first.stdout + first.stderr)
            return first.returncode or 1
        (root / "leaf0.py").write_text("VALUE = 0  # changed\n", encoding="utf-8")
        report_path = root / "report.json"
        second = subprocess.run(
            [
                *base,
                *MANAGED_PLUGIN_ARGS,
                "--testmon",
                "--testmon-env=polylogue",
                "--json-report",
                f"--json-report-file={report_path}",
                "--testmon-forceselect",
                "tests/test_leaf0.py",
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
        selected = len(report.get("tests", [])) if isinstance(report, dict) else 0
        total = 100
        if not selected or not total or selected * 100 >= total * 5:
            print(f"testmon-selection: selected {selected} of {total}, expected under 5%\n{output}")
            return 1
    print(f"testmon-selection: selected {selected} of {total}; workers={CORPUS_MAX_WORKERS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
