"""``devtools test`` — focused pytest runner with project-owned semantics.

Agents and humans should never invoke raw ``pytest`` for inner-loop checks.
This command forwards a selection (paths, ``-k``/``-m`` expressions, ``-x``,
…) to pytest with:

- the repository's managed environment (``POLYLOGUE_ROOT`` and friends, a
  repo-local pycache prefix);
- a single-process default; parallelism is an explicit ``-n``
  request and is narrowed at the admitted pytest pool when necessary;
- the same pytest progress ledger, JSON report, and typed outcome receipt used
  by ``devtools verify``.

pytest runs only while holding the host's single pytest slot
(``devtools.pytest_slot``). A caller already inside that slot streams its
output; a caller outside it queues, waits, and reads the captured log.

For the full pre-PR gate use ``devtools verify``; this command is the inner
loop, not a substitute for it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
)
from devtools.pytest_invocation import (
    CLEAR_CONFIGURED_ADDOPTS,
    IGNORED_COLLECTION_ARGS,
    SUITE_COST_PLUGIN_NAME,
    devtools_plugin_args,
    effective_hypothesis_profile,
    managed_plugin_args,
)
from devtools.pytest_slot import (
    PytestSlotUnavailableError,
    basetemp_root,
    remove_temp_tree,
    run_pytest,
    run_pytest_isolated,
)
from devtools.pytest_stream_report import report_file_argument, spool_paths
from devtools.pytest_suite_cost_plugin import SUITE_COST_DIR_ENV, write_run_receipt
from devtools.testmon_provision import TESTMON_COVERAGE_CORE, inspect_testmon_graph
from devtools.toolchain import venv_python
from devtools.verify_runs import (
    PytestStepArtifacts,
    VerifyRun,
    append_verification_evidence,
    append_verify_history,
    env_for_pytest_step,
    git_head,
    prune_successful_verify_runs,
    pytest_command_worker_request,
)

ROOT = Path(__file__).resolve().parent.parent
PYTEST_REPORT_DIR = Path(".cache/verify")
PYTEST_REPORT_PATH = PYTEST_REPORT_DIR / "last-pytest.json"
PYTEST_PARALLEL_REPORT_PATTERN = "last-pytest-parallel-*.json"
DEFAULT_OUTLIER_COUNT = 10
PYTEST_PROGRESS_PATH = PYTEST_REPORT_DIR / "current-pytest-progress.json"
PYTEST_EVENTS_PATH = PYTEST_REPORT_DIR / "current-pytest-events.jsonl"
PYTEST_EVENTS_DIR = PYTEST_REPORT_DIR / "current-pytest-events"
PYTEST_SELECTION_PATH = PYTEST_REPORT_DIR / "current-pytest-selection.json"
PYTEST_SUMMARY_PATH = PYTEST_REPORT_DIR / "current-pytest-summary.json"
_PATH_VALUE_OPTIONS = frozenset(
    {
        "-c",
        "--basetemp",
        "--config-file",
        "--confcutdir",
        "--debug",
        "--ignore",
        "--ignore-glob",
        "--junit-xml",
        "--junitxml",
        "--log-file",
        "--rootdir",
    }
)
_ENV_EXPANDING_PATH_OPTIONS = frozenset({"--rootdir"})
_NON_PATH_VALUE_OPTIONS = frozenset(
    {
        "-k",
        "--keyword",
        "-m",
        "--mark",
        "--deselect",
        "--maxfail",
        "--tb",
        "--capture",
        "--durations",
        "--durations-min",
        "--override-ini",
        "-o",
    }
)


def _prepare_nodatacow_parent(path: Path) -> None:
    """Best-effortly mark the parent of a pytest basetemp as nodatacow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.Popen(
            ["chattr", "+C", str(path.parent)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.wait()
    except OSError:
        # The optimization is filesystem-specific and must not make tests
        # unavailable on hosts without chattr or the btrfs attribute.
        return


def _phase_duration(test: dict[str, Any]) -> float:
    return sum(float((test.get(phase) or {}).get("duration", 0) or 0) for phase in ("setup", "call", "teardown"))


def _format_duration(seconds: float) -> str:
    return f"{seconds / 60:.1f}m" if seconds >= 60 else f"{seconds:.2f}s"


def print_outliers(limit: int = DEFAULT_OUTLIER_COUNT, *, root: Path = ROOT) -> int:
    """Print slow tests and files from the latest full-run pytest reports."""
    report_paths = sorted((root / PYTEST_REPORT_DIR).glob(PYTEST_PARALLEL_REPORT_PATTERN))
    tests: list[tuple[str, str, float]] = []
    for path in report_paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        for test in report.get("tests", []):
            if not isinstance(test, dict) or not isinstance(test.get("nodeid"), str):
                continue
            duration = _phase_duration(test)
            tests.append((test["nodeid"], test["nodeid"].split("::", 1)[0], duration))
    if not tests:
        print(f"devtools test --outliers: no readable {PYTEST_PARALLEL_REPORT_PATTERN} receipts", file=sys.stderr)
        return 2

    serial_time = sum(duration for _nodeid, _filename, duration in tests)
    slowest_tests = sorted(tests, key=lambda item: (-item[2], item[0]))[:limit]
    file_totals: dict[str, float] = {}
    for _nodeid, filename, duration in tests:
        file_totals[filename] = file_totals.get(filename, 0.0) + duration
    slowest_files = sorted(file_totals.items(), key=lambda item: (-item[1], item[0]))[:limit]

    print(f"Full-run receipts: {len(report_paths)}; tests: {len(tests)}; serial time: {_format_duration(serial_time)}")
    test_share = sum(duration for _nodeid, _filename, duration in slowest_tests) / serial_time * 100
    print(f"Top {len(slowest_tests)} slowest tests ({test_share:.1f}% of serial time):")
    for nodeid, _filename, duration in slowest_tests:
        print(f"  {duration:8.2f}s ({duration / serial_time * 100:5.1f}%) {nodeid}")
    file_share = sum(duration for _filename, duration in slowest_files) / serial_time * 100
    print(f"Top {len(slowest_files)} slowest files ({file_share:.1f}% of serial time):")
    for filename, duration in slowest_files:
        print(f"  {duration:8.2f}s ({duration / serial_time * 100:5.1f}%) {filename}")
    return 0


def _parse_outliers(selection: list[str]) -> tuple[int | None, list[str]]:
    if not selection or not selection[0].startswith("--outliers"):
        return None, selection
    option, _, inline_limit = selection[0].partition("=")
    if option != "--outliers":
        return None, selection
    remaining = selection[1:]
    value = inline_limit or (remaining.pop(0) if remaining and not remaining[0].startswith("-") else "")
    try:
        limit = int(value) if value else DEFAULT_OUTLIER_COUNT
    except ValueError as exc:
        raise ValueError("--outliers expects a positive integer") from exc
    if limit < 1:
        raise ValueError("--outliers expects a positive integer")
    return limit, remaining


def _parse_runner(selection: list[str]) -> tuple[str, list[str]]:
    """Consume the runner mode without forwarding it to pytest."""
    runner = "managed"
    remaining: list[str] = []
    index = 0
    while index < len(selection):
        argument = selection[index]
        if argument == "--runner":
            index += 1
            if index == len(selection):
                raise ValueError("--runner expects managed or isolated")
            runner = selection[index]
        elif argument.startswith("--runner="):
            runner = argument.split("=", 1)[1]
        else:
            remaining.append(argument)
        index += 1
    if runner not in {"managed", "isolated"}:
        raise ValueError("--runner expects managed or isolated")
    return runner, remaining


def _verbose_output() -> bool:
    """Whether the caller asked for the full preamble and artifact footer."""
    return "--verbose" in sys.argv[1:] or bool(os.environ.get("POLYLOGUE_DEVTOOLS_VERBOSE"))


def _absolute_option_path(
    value: str,
    *,
    invocation_directory: Path,
    expand_environment_variables: bool = False,
) -> str:
    if expand_environment_variables:
        value = os.path.expandvars(value)
    path = Path(value)
    # pytest deliberately uses ``os.path.abspath`` for command-line paths:
    # resolving here would make ``-c config-link.ini`` select the linked
    # target as its rootdir instead of preserving the caller's spelling.
    return os.path.abspath(path if path.is_absolute() else invocation_directory / path)


def _normalize_selection_paths(selection: list[str], *, invocation_directory: Path) -> list[str]:
    """Preserve path selections relative to the directory that invoked devtools."""
    normalized: list[str] = []
    pending_option: str | None = None
    for argument in selection:
        if pending_option is not None:
            # pytest's --debug accepts an optional file name.  A following
            # option belongs to pytest, not to --debug's optional value.
            if pending_option == "--debug" and argument.startswith("-"):
                pending_option = None
            elif pending_option in _PATH_VALUE_OPTIONS:
                normalized.append(
                    _absolute_option_path(
                        argument,
                        invocation_directory=invocation_directory,
                        expand_environment_variables=pending_option in _ENV_EXPANDING_PATH_OPTIONS,
                    )
                )
                pending_option = None
                continue
            else:
                normalized.append(argument)
                pending_option = None
                continue
        if argument.startswith("-c="):
            normalized.append(
                "-c"
                + _absolute_option_path(
                    argument[len("-c=") :],
                    invocation_directory=invocation_directory,
                )
            )
            continue
        option_name, equals, option_value = argument.partition("=")
        if option_name in _PATH_VALUE_OPTIONS:
            if equals:
                normalized_value = _absolute_option_path(
                    option_value,
                    invocation_directory=invocation_directory,
                    expand_environment_variables=option_name in _ENV_EXPANDING_PATH_OPTIONS,
                )
                normalized.append(f"{option_name}={normalized_value}")
            else:
                normalized.append(argument)
                pending_option = option_name
            continue
        if option_name in _NON_PATH_VALUE_OPTIONS:
            normalized.append(argument)
            if not equals:
                pending_option = option_name
            continue
        if argument.startswith("-c") and len(argument) > len("-c"):
            normalized.append(
                "-c"
                + _absolute_option_path(
                    argument[len("-c") :],
                    invocation_directory=invocation_directory,
                )
            )
            continue
        if argument.startswith("-"):
            normalized.append(argument)
            continue
        path_text, separator, node_suffix = argument.partition("::")
        candidate = Path(path_text)
        if candidate.is_absolute() or not (invocation_directory / candidate).exists():
            normalized.append(argument)
            continue
        resolved = (invocation_directory / candidate).resolve()
        try:
            anchored = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            anchored = str(resolved)
        normalized.append(f"{anchored}{separator}{node_suffix}")
    return normalized


def _anchor_test_paths() -> None:
    """Anchor focused-test execution and artifacts to this checkout."""
    os.chdir(ROOT)


def _has_worker_flag(selection: list[str]) -> bool:
    """True when the caller already chose an xdist worker count."""
    return any(arg.startswith(("-n", "--numprocesses")) for arg in selection)


def _worker_args(selection: list[str]) -> list[str]:
    """Default focused runs to a single process; honor an explicit override.

    An ambient worker setting belongs to broad verification, not an inner-loop
    selection. The runner that owns the requested pool narrows an explicit
    request using its live cgroup budget.
    """
    if _has_worker_flag(selection):
        return []
    return []


def _xdist_distribution_args(selection: list[str], worker_args: list[str]) -> list[str]:
    """Keep declared shared-state groups together whenever xdist is active."""
    if any(arg == "--dist" or arg.startswith("--dist=") for arg in selection):
        return []
    command = [*selection, *worker_args]
    request = pytest_command_worker_request(command)
    if request in {None, "0"}:
        return []
    return ["--dist=loadgroup"]


def build_pytest_cmd(selection: list[str], *, report_path: Path = PYTEST_REPORT_PATH) -> list[str]:
    """Compose the pytest command for a focused selection."""
    worker_args = _worker_args(selection)
    collection_args = () if _selection_targets_benchmarks(selection) else IGNORED_COLLECTION_ARGS
    return [
        venv_python(root=ROOT),
        "-m",
        "pytest",
        *devtools_plugin_args(testmon=False),
        "-p",
        SUITE_COST_PLUGIN_NAME,
        *managed_plugin_args(testmon=False, xdist=_has_worker_flag(selection)),
        CLEAR_CONFIGURED_ADDOPTS,
        report_file_argument(report_path),
        *collection_args,
        *selection,
        *worker_args,
        *_xdist_distribution_args(selection, worker_args),
    ]


def focused_pytest_env(*, run: VerifyRun, artifacts: PytestStepArtifacts) -> dict[str, str]:
    """The environment a focused run executes under.

    Focused runs deliberately do not load testmon. They must not create a
    scratch graph, mutate the corpus graph, or load testmon's retention hook.
    """
    return env_for_pytest_step(dict(os.environ), run=run, artifacts=artifacts, testmon=False)


def _selection_targets_benchmarks(selection: list[str]) -> bool:
    """Keep benchmark collection available only when the caller asks for it."""
    return any("tests/benchmarks" in argument for argument in selection)


def _clear_pytest_report(report_path: Path) -> None:
    """Remove this focused invocation's stale pytest-domain artifacts."""
    for path in (
        report_path,
        *spool_paths(report_path),
    ):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _run(
    label: str,
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    run: VerifyRun,
    runner: str = "managed",
) -> tuple[int, float, dict[str, Any]]:
    """Run focused pytest through the host's pytest slot, preserving its receipt."""
    del label, run
    started = time.monotonic()
    try:
        executor = run_pytest if runner == "managed" else run_pytest_isolated
        outcome = executor(command, cwd=cwd, env=env, root=ROOT)
    except PytestSlotUnavailableError as exc:
        sys.stderr.write(f"devtools test: {exc}\n")
        return (
            125,
            time.monotonic() - started,
            {
                "diagnosis": "pytest_slot_unavailable",
                "error": str(exc),
                "termination_reason": "pytest_slot_unavailable",
            },
        )
    suite_cost_receipt = write_run_receipt(env.get(SUITE_COST_DIR_ENV))
    return (
        outcome.returncode,
        time.monotonic() - started,
        {
            "diagnosis": "pytest_passed" if outcome.returncode == 0 else "pytest_failed",
            "pytest_slot": outcome.slot,
            **({"suite_cost_receipt": str(suite_cost_receipt)} if suite_cost_receipt is not None else {}),
            # Named per client pid: the checkout accumulates one log per run,
            # and a glob over them reaches an arbitrary one.
            **({"pytest_slot_log": str(outcome.log_path)} if outcome.log_path is not None else {}),
            **({"pytest_slot_receipt": outcome.receipt} if outcome.receipt is not None else {}),
        },
    )


def _normalize_managed_pytest_environment(env: dict[str, str]) -> None:
    """Make focused collection independent of ambient pytest plugins/options."""
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    # A managed run launched from inside another pytest process (a test that
    # subprocesses `devtools test`) inherits the enclosing xdist worker
    # identity. The inner run is a fresh controller; a leaked worker id makes
    # the progress plugin skip controller-side selection/summary receipts and
    # the run fails closed despite collecting tests.
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # The CLI profile wins over this environment; absent both, focused runs
    # use the bounded verify profile (registered in tests/conftest.py).
    env.setdefault("HYPOTHESIS_PROFILE", "verify")
    env["COVERAGE_CORE"] = TESTMON_COVERAGE_CORE


def _publish_last_focused_pytest_report(report_path: Path) -> None:
    """Refresh the convenience report without making it receipt authority.

    Concurrent focused runs each write their receipt-local report.  This
    legacy location remains useful to interactive callers, but it is only a
    last-writer-wins copy and is never read while deciding a run's result.
    """

    if report_path.is_file():
        destination = ROOT / PYTEST_REPORT_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_path, destination)


def absent_selection_paths(selection: list[str], *, root: Path) -> list[str]:
    """The path selections that name nothing in the checkout.

    Resolved against the checkout root rather than the process working
    directory: a relative selection means the same file whichever directory
    ``devtools test`` was invoked from, and reporting every one of them as
    missing would turn "no tests collected" into a false explanation.
    """
    absent: list[str] = []
    for argument in selection:
        if argument.startswith("-") or not (argument.endswith(".py") or "/" in argument):
            continue
        candidate = Path(argument.split("::", 1)[0])
        if not (candidate if candidate.is_absolute() else root / candidate).exists():
            absent.append(argument)
    return absent


def main(argv: list[str] | None = None) -> int:
    invocation_directory = Path.cwd()
    selection = list(sys.argv[1:] if argv is None else argv)
    try:
        outlier_count, selection = _parse_outliers(selection)
        runner, selection = _parse_runner(selection)
    except ValueError as exc:
        sys.stderr.write(f"devtools test: {exc}\n")
        return 2
    if outlier_count is not None:
        return print_outliers(outlier_count)
    selection = _normalize_selection_paths(selection, invocation_directory=invocation_directory)
    _anchor_test_paths()
    try:
        assert_polylogue_matches_checkout(ROOT, context="devtools test")
    except CheckoutImportMismatchError as exc:
        sys.stderr.write(f"{exc}\n")
        return 125
    use_json = "--json" in selection
    # The control-plane dispatch may append a bare ``--json`` machine-readable
    # flag; it is meaningless for a streamed test run, so drop it before pytest.
    selection = [arg for arg in selection if arg != "--json"]
    if not selection:
        sys.stderr.write(
            "devtools test: give a selection, e.g.\n"
            "  devtools test tests/unit/pipeline\n"
            "  devtools test -k hybrid\n"
            "  devtools test tests/unit/storage -x\n"
            "For the full pre-PR gate use `devtools verify`.\n"
        )
        return 2

    run = VerifyRun(
        tier="focused-test",
        argv=selection,
        git_head=git_head(ROOT),
        root=ROOT,
    )
    # The report and its PID-named spool must live with this receipt.  A
    # checkout-global spool lets a later focused run delete an earlier run's
    # completed tests between teardown and controller-side assembly.
    report_path = run.run_dir / "steps" / "01-pytest-focused" / "pytest-report.json"
    cmd = build_pytest_cmd(selection, report_path=report_path)
    # Owning the basetemp here means the run can dispose of it when it is no
    # longer needed instead of relying on pytest's "keep the last three runs"
    # pruning, which never fires because each shell starts a fresh root. A
    # failed run keeps its tree: that is when the fixtures are worth reading.
    # Unique per invocation: pytest creates the basetemp itself and fails if it
    # already exists, so a fixed path makes two runs in one checkout collide —
    # and lanes, batches and the coordinator do run concurrently here.
    run_temp = basetemp_root(os.environ, root=ROOT) / f"tmp-{os.getpid()}-{time.time_ns():x}"
    remove_temp_tree(run_temp)
    _prepare_nodatacow_parent(run_temp)
    cmd = [*cmd, "--basetemp", str(run_temp)]
    _clear_pytest_report(report_path)
    artifacts = run.start_step(label="pytest focused", cmd=cmd)
    started = time.monotonic()
    try:
        pytest_env = focused_pytest_env(run=run, artifacts=artifacts)
        pytest_env.pop("POLYLOGUE_PYTEST_CONTAINMENT_PATH", None)
        pytest_env.pop("POLYLOGUE_BROAD_PREWARM", None)
        _normalize_managed_pytest_environment(pytest_env)
        hypothesis_profile, hypothesis_profile_source = effective_hypothesis_profile(
            selection, pytest_env, default="verify"
        )
        graph = inspect_testmon_graph(ROOT)
        rc, elapsed, metadata = _run(
            "pytest focused",
            cmd,
            cwd=str(ROOT),
            env=pytest_env,
            run=run,
            runner=runner,
        )
        _publish_last_focused_pytest_report(report_path)
        metadata["testmon_preselection"] = {
            "status": graph.status.value,
            "reason": graph.reason,
            "cause": graph.full_rerun_cause,
        }
        metadata["hypothesis_profile"] = hypothesis_profile
        metadata["hypothesis_profile_source"] = hypothesis_profile_source
        metadata["runner"] = runner
    except KeyboardInterrupt:
        rc = 130
        elapsed = time.monotonic() - started
        metadata = {"diagnosis": "pytest_interrupted", "termination_reason": "operator_interrupt"}
    except Exception as exc:
        rc = 125
        metadata = {
            "diagnosis": "focused_test_runner_exception",
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "termination_reason": "runner_exception",
        }
        elapsed = time.monotonic() - started
        sys.stderr.write(f"devtools test: cannot start pytest: {exc}\n")
    step = run.finish_step(
        step_id=artifacts.step_id,
        result={"duration_s": elapsed, **metadata, "exit": rc},
    )
    if step is not None:
        rc = int(step["exit"])
        metadata = step
    statistics: dict[str, Any] = cast(
        dict[str, Any], metadata.get("statistics") if isinstance(metadata.get("statistics"), dict) else {}
    )
    if rc == 5:
        # pytest exits 5 both when a selection genuinely matches nothing and
        # when a path in it does not exist, and the two are indistinguishable
        # to a caller. A selection derived from `git diff --name-only` contains
        # files the branch deleted, so the whole run silently collects nothing
        # and reads as "no work to do".
        absent = absent_selection_paths(selection, root=ROOT)
        if absent:
            sys.stderr.write(
                "devtools test: collected nothing because these paths do not exist: " + ", ".join(absent) + "\n"
            )
        else:
            sys.stderr.write(
                "devtools test: the selection collected no tests; every path exists, "
                "so check the -k/-m expression or retry if runs are contending.\n"
            )
    if rc == 0:
        remove_temp_tree(run_temp)
    payload = run.finish(
        exit_code=rc,
        duration_s=elapsed,
        diagnosis=metadata.get("diagnosis"),
        verification_scope="affected",
        final_git_head=git_head(ROOT),
        pytest_aggregate={
            "selection_mode": "focused",
            "selected_union_count": statistics.get("selected_count"),
            "terminal_union_count": statistics.get("terminal_count"),
            "terminal_green": statistics.get("ordinary_eligible", False),
            "outcomes": statistics.get("outcomes", {}),
        },
    )
    append_verify_history(payload)
    append_verification_evidence(payload)
    prune_successful_verify_runs(root=ROOT)
    if use_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    # The verdict is the last thing written, on every run. A pipeline exits with
    # its last command's status, so `devtools test ... | tail` reports tail's 0
    # whatever the run found; carrying the outcome in the stream keeps it out of
    # reach of that mistake. The receipt is this run's own file, never a
    # `current-*` name a concurrent run in the same checkout would overwrite.
    receipt = run.relative_run_dir / "run.json"
    sys.stderr.write(
        f"\ndevtools test: {'PASSED' if rc == 0 else 'FAILED'} exit={rc} "
        f"diagnosis={metadata.get('diagnosis') or 'unknown'} receipt={receipt}\n"
    )
    # The rest of the artifacts are reference material, not a result. Printing
    # them after every green run trains the reader to skip the tail of the
    # output, which is exactly where a failure summary appears. `devtools why`
    # reaches them on demand.
    if _verbose_output() or rc != 0:
        sys.stderr.write(f"devtools test: artifacts={run.relative_run_dir}/steps/{artifacts.step_id}\n")
    return rc
