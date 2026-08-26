"""``devtools test`` — focused pytest runner with project-owned semantics.

Agents and humans should never invoke raw ``pytest`` for inner-loop checks.
This command forwards a selection (paths, ``-k``/``-m`` expressions, ``-x``,
…) to pytest with:

- the repository's managed environment (``POLYLOGUE_ROOT`` and friends, a
  repo-local pycache prefix);
- a single-process worker default (``-n 0``) for fast focused runs, overridable
  with ``-n`` in the selection or ``POLYLOGUE_PYTEST_WORKERS``;
- live, streamed output (unlike ``devtools verify``, which captures);
- the same pytest progress ledger, JSON report, and typed outcome receipt used
  by ``devtools verify``.

Process placement, cancellation, resource admission, and timeout authority
belong to AgentCTL when a job needs them. A direct focused invocation is an
ordinary foreground subprocess and does not create a second local lifecycle.

For the full pre-PR gate use ``devtools verify``; this command is the inner
loop, not a substitute for it.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, cast

from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
)
from devtools.pytest_collection_contract import (
    CLEAR_CONFIGURED_ADDOPTS,
    IGNORED_COLLECTION_ARGS,
    MANAGED_PLUGIN_ARGS,
)
from devtools.pytest_scratch import Outcome, PytestScratchLease, run_managed_pytest, scratch_root_from_environment
from devtools.verify_runs import (
    PytestStepArtifacts,
    VerifyRun,
    append_verify_history,
    configured_pytest_worker_request,
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
PYTEST_OUTPUT_PATH = PYTEST_REPORT_DIR / "current-pytest-output.log"
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


class ManagedTestInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum


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


def _raise_managed_interruption(_signum: int, _frame: object) -> None:
    raise ManagedTestInterrupted(_signum)


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

    The override is read through the shared resolver rather than straight from
    the environment, so the cloud worker pin is scrubbed here exactly as it is
    for `devtools verify`. Reading it raw gave focused runs two processes where
    the policy intends one.
    """
    if _has_worker_flag(selection):
        return []
    requested = configured_pytest_worker_request(os.environ)
    return ["-n", str(requested if requested is not None else 0)]


def _xdist_distribution_args(selection: list[str], worker_args: list[str]) -> list[str]:
    """Keep declared shared-state groups together whenever xdist is active."""
    if any(arg == "--dist" or arg.startswith("--dist=") for arg in selection):
        return []
    command = [*selection, *worker_args]
    request = pytest_command_worker_request(command)
    if request in {None, "0"}:
        return []
    return ["--dist=loadgroup"]


def build_pytest_cmd(selection: list[str]) -> list[str]:
    """Compose the pytest command for a focused selection."""
    worker_args = _worker_args(selection)
    collection_args = () if _selection_targets_benchmarks(selection) else IGNORED_COLLECTION_ARGS
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "devtools.pytest_progress_plugin",
        *MANAGED_PLUGIN_ARGS,
        CLEAR_CONFIGURED_ADDOPTS,
        "--json-report",
        "--json-report-omit=collectors,log,streams,warnings",
        f"--json-report-file={PYTEST_REPORT_PATH}",
        *collection_args,
        *selection,
        *worker_args,
        *_xdist_distribution_args(selection, worker_args),
    ]


def _selection_targets_benchmarks(selection: list[str]) -> bool:
    """Keep benchmark collection available only when the caller asks for it."""
    return any("tests/benchmarks" in argument for argument in selection)


def _clear_pytest_report(_cmd: list[str]) -> None:
    """Remove this focused invocation's stale pytest-domain artifacts."""
    for path in (
        PYTEST_REPORT_PATH,
        PYTEST_PROGRESS_PATH,
        PYTEST_EVENTS_PATH,
        PYTEST_EVENTS_DIR,
        PYTEST_SELECTION_PATH,
        PYTEST_SUMMARY_PATH,
        PYTEST_OUTPUT_PATH,
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
) -> tuple[int, float, dict[str, str]]:
    """Run focused pytest directly while preserving its project receipt."""
    del label, run
    started = time.monotonic()
    handlers = {
        signum: signal.signal(signum, _raise_managed_interruption) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        completed = run_managed_pytest(command, cwd=Path(cwd), env=env)
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
    return (
        completed.returncode,
        time.monotonic() - started,
        {"diagnosis": "pytest_passed" if completed.returncode == 0 else "pytest_failed"},
    )


def _normalize_managed_pytest_environment(env: dict[str, str]) -> None:
    """Make focused collection independent of ambient pytest plugins/options."""
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


def _copy_focused_pytest_report(artifacts: PytestStepArtifacts) -> None:
    report_path = ROOT / PYTEST_REPORT_PATH
    if report_path.is_file():
        shutil.copyfile(report_path, artifacts.step_dir / "pytest-report.json")


def main(argv: list[str] | None = None) -> int:
    invocation_directory = Path.cwd()
    selection = list(sys.argv[1:] if argv is None else argv)
    try:
        outlier_count, selection = _parse_outliers(selection)
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

    cmd = build_pytest_cmd(selection)
    _clear_pytest_report(cmd)
    run = VerifyRun(
        tier="focused-test",
        argv=selection,
        git_head=git_head(ROOT),
        root=ROOT,
    )
    artifacts = run.start_step(label="pytest focused", cmd=cmd)
    started = time.monotonic()
    lease: PytestScratchLease | None = None
    try:
        pytest_env = env_for_pytest_step(dict(os.environ), run=run, artifacts=artifacts)
        pytest_env.pop("POLYLOGUE_PYTEST_CONTAINMENT_PATH", None)
        _normalize_managed_pytest_environment(pytest_env)
        lease = PytestScratchLease.acquire(
            root=scratch_root_from_environment(pytest_env),
            run_id=run.run_id,
            lane="focused",
            evidence_dir=artifacts.step_dir,
        )
        cmd = lease.command(cmd)
        pytest_env = lease.environment(pytest_env)
        rc, elapsed, metadata = _run(
            "pytest focused",
            cmd,
            cwd=str(ROOT),
            env=pytest_env,
            run=run,
        )
        _copy_focused_pytest_report(artifacts)
    except ManagedTestInterrupted as exc:
        rc = 128 + exc.signum
        elapsed = time.monotonic() - started
        metadata = {
            "diagnosis": "pytest_interrupted",
            "termination_reason": signal.Signals(exc.signum).name.lower(),
        }
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
    finally:
        if lease is not None:
            outcome: Outcome = (
                "success"
                if "rc" in locals() and rc == 0
                else "cancelled"
                if "rc" in locals() and rc == 130
                else "worker_crash"
                if "rc" in locals() and rc == 3
                else "failure"
            )
            lease.finalize(outcome)
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
    prune_successful_verify_runs(root=ROOT)
    if use_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    # The artifact-path footer is reference material, not a result. Printing six
    # paths after every green run trains the reader to skip the tail of the
    # output, which is exactly where a failure summary appears. `devtools why`
    # reaches the same artifacts on demand.
    if _verbose_output() or rc != 0:
        sys.stderr.write(
            f"\ndevtools test: progress={PYTEST_PROGRESS_PATH} selection={PYTEST_SELECTION_PATH} "
            f"summary={PYTEST_SUMMARY_PATH} events={PYTEST_EVENTS_PATH} "
            f"output={PYTEST_OUTPUT_PATH}\n"
        )
    return rc
