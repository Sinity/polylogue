"""Project semantic verification: gates, test selection, and typed receipts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from devtools.agent_env import refuse_verify_tier
from devtools.checkout_guard import CheckoutImportMismatchError, assert_polylogue_matches_checkout
from devtools.pytest_invocation import (
    CLEAR_CONFIGURED_ADDOPTS,
    CLOSED_WORLD_COLLECTION_ARGS,
    IGNORED_COLLECTION_ARGS,
    MANAGED_PLUGIN_ARGS,
    PROGRESS_PLUGIN_NAME,
)
from devtools.required_gate import executable_gate_result
from devtools.testmon_provision import (
    TESTMON_COVERAGE_CORE,
    TESTMON_DATA_RELPATH,
    TESTMON_ENVIRONMENT,
    TestmonGraphStatus,
    inspect_testmon_graph,
)
from devtools.toolchain import venv_bin, venv_python
from devtools.verification_authority import validate_authority_matrix
from devtools.verification_contracts import VerificationScope
from devtools.verification_result import declared_verification_result
from devtools.verify_runs import (
    CURRENT_EVENTS_DIR,
    PYTEST_CANONICAL_REPORT_NAME,
    VerifyRun,
    append_verification_evidence,
    append_verify_history,
    canonical_verification_receipt,
    copy_current_pytest_artifacts,
    env_for_pytest_step,
    git_head,
    prune_successful_verify_runs,
)
from polylogue.scenarios import (
    MeasurementScope,
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
    workload_adapter_declarations,
)

ROOT = Path(__file__).resolve().parents[1]
PYTEST_REPORT_DIR = Path(".cache/verify")
PYTEST_REPORT_PATH = PYTEST_REPORT_DIR / "last-pytest.json"
PYTEST_PROGRESS_PATH = PYTEST_REPORT_DIR / "current-pytest-progress.json"
PYTEST_EVENTS_PATH = PYTEST_REPORT_DIR / "current-pytest-events.jsonl"
PYTEST_EVENTS_DIR = CURRENT_EVENTS_DIR
PYTEST_SELECTION_PATH = PYTEST_REPORT_DIR / "current-pytest-selection.json"
PYTEST_SUMMARY_PATH = PYTEST_REPORT_DIR / "current-pytest-summary.json"
PYTEST_OUTPUT_PATH = PYTEST_REPORT_DIR / "current-pytest-output.log"
PYTEST_JUNIT_REPORT_DIR = PYTEST_REPORT_DIR / "junit"
#: The corpus runs unpartitioned. Eight workers peak near 10 GB, inside the
#: pytest pool's 12 GiB ceiling; CPU sits near 22% at that width, so the
#: ceiling, not the cores, is what bounds this.
CORPUS_MAX_WORKERS = 8
_AGENTCTL_OPERATION_ARGV = {"verify_affected": (), "verify_quick": ("--quick",), "verify_all": ("--all",)}
_UNMEASURED_WORKLOAD_DIMENSIONS = (
    "cpu_ms",
    "current_rss_bytes",
    "peak_rss_bytes",
    "current_pss_bytes",
    "peak_pss_bytes",
    "anon_bytes",
    "file_cache_bytes",
    "swap_bytes",
    "temp_storage_bytes",
    "storage_bytes",
    "read_io_bytes",
    "write_io_bytes",
    "response_bytes",
    "cancellation_latency_ms",
    "progress_completed",
    "progress_total",
    "queue_depth",
    "backpressure_ms",
    "cleanup_reclaimed_bytes",
    "sqlite_vm_steps",
)
_RENDER_DIAGNOSIS_RE = re.compile(r"^render all:.*diagnosis: (?P<token>[a-z][a-z0-9_]*)\b")


class VerificationInterrupted(KeyboardInterrupt):
    """A terminal signal reached the verifier after its receipt was created."""

    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum


def _raise_verification_interruption(signum: int, _frame: Any) -> None:
    raise VerificationInterrupted(signum)


def _declared_agentctl_operation(raw_argv: Sequence[str]) -> str | None:
    operation = os.environ.get("SINNIXD_OPERATION")
    return operation if _AGENTCTL_OPERATION_ARGV.get(operation or "") == tuple(raw_argv) else None


def _anchor_verification_paths() -> None:
    try:
        Path.cwd().resolve().relative_to(ROOT.resolve())
    except ValueError:
        return
    os.chdir(ROOT)


#: The daemon holds a type cache worth well over a gigabyte and is reparented to
#: the user manager, so without this it outlives every gate that starts one and
#: one accumulates per checkout. The idle clock resets on each connection, so a
#: checkout under active gating keeps its warm daemon.
DMYPY_IDLE_TIMEOUT_SECONDS = 900


def _mypy_cmd() -> list[str]:
    dmypy = venv_bin("dmypy", root=ROOT)
    try:
        result = subprocess.run([dmypy, "status"], capture_output=True, text=True, timeout=5, cwd=ROOT)
        if result.returncode == 0:
            # run can itself spawn the daemon (races, direct invocations);
            # the timeout must ride every spawning form or one immortal
            # daemon per checkout accumulates.
            return [dmypy, "run", f"--timeout={DMYPY_IDLE_TIMEOUT_SECONDS}", "--", "--no-error-summary"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(
            [dmypy, "start", f"--timeout={DMYPY_IDLE_TIMEOUT_SECONDS}", "--", "--no-error-summary"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=ROOT,
        )
        if result.returncode == 0:
            return [dmypy, "run", f"--timeout={DMYPY_IDLE_TIMEOUT_SECONDS}", "--", "--no-error-summary"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return [venv_bin("mypy", root=ROOT)]


def _devtools_cmd(*args: str) -> list[str]:
    return [venv_python(root=ROOT), "-m", "devtools", *(part for arg in args for part in arg.split())]


def _pytest_worker_args(*, maximum: int | None = None) -> list[str]:
    try:
        workers = max(0, int(os.environ.get("POLYLOGUE_PYTEST_WORKERS", "0")))
    except ValueError:
        workers = 0
    if maximum is not None:
        workers = min(workers, maximum)
    return ["--dist=loadgroup", "-n", str(workers)]


def _pytest_steps(*, selection: str, worker_args: Sequence[str]) -> list[tuple[str, list[str]]]:
    """The corpus runs as ONE collection.

    testmon drops every recorded test a run did not collect, so a partitioned
    run keeps only its last partition's edges. One run, always tracing: the
    graph is advanced by every managed run rather than recomputed.
    """
    command = [
        venv_python(root=ROOT),
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        *IGNORED_COLLECTION_ARGS,
        "--durations=10",
        f"--junitxml={PYTEST_JUNIT_REPORT_DIR}/verify-latest.xml",
        "--json-report",
        "--json-report-omit=collectors,log,streams,warnings",
        f"--json-report-file={PYTEST_REPORT_PATH}",
        "-p",
        PROGRESS_PLUGIN_NAME,
        *MANAGED_PLUGIN_ARGS,
        *CLOSED_WORLD_COLLECTION_ARGS,
        "--testmon",
        f"--testmon-env={TESTMON_ENVIRONMENT}",
        # `all` runs every test and still updates fingerprints; the default
        # tier selects from what the datafile records, and an empty datafile
        # selects everything, which is how it seeds.
        "--testmon-noselect" if selection == "all" else "--testmon-forceselect",
        "-p",
        "no:randomly",
        *worker_args,
    ]
    return [(f"pytest ({selection})", command)]


def build_verify_steps(*, quick: bool, selection: str = "all") -> list[tuple[str, list[str]]]:
    steps = [
        ("ruff format", [venv_bin("ruff", root=ROOT), "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", [venv_bin("ruff", root=ROOT), "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", _mypy_cmd()),
        ("render all", _devtools_cmd("render all", "--check")),
        ("verify layering", _devtools_cmd("verify layering", "--json")),
        ("verify patterns", _devtools_cmd("verify patterns", "--json")),
        ("verify doc-commands", _devtools_cmd("verify doc-commands")),
        ("verify schema-versioning", _devtools_cmd("verify schema-versioning")),
        ("verify oracle-integrity", _devtools_cmd("verify oracle-integrity")),
        ("verify consumer-reachability", _devtools_cmd("verify consumer-reachability", "--json")),
        ("verify timestamp-doctrine", _devtools_cmd("verify timestamp-doctrine")),
        ("schema privacy registry", [venv_python(root=ROOT), "-m", "devtools.verify_schema_privacy"]),
    ]
    if not quick:
        PYTEST_JUNIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        steps += _pytest_steps(selection=selection, worker_args=_pytest_worker_args(maximum=CORPUS_MAX_WORKERS))
    return steps


def _normalize_managed_pytest_environment(env: dict[str, str]) -> None:
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Property tests run under one full profile so a reduced Hypothesis budget
    # cannot quietly narrow what a recorded green stands for.
    env["HYPOTHESIS_PROFILE"] = "default"
    env["COVERAGE_CORE"] = TESTMON_COVERAGE_CORE
    env.pop("POLYLOGUE_CI", None)


def _clear_pytest_report(command: Sequence[str]) -> None:
    paths = [
        PYTEST_PROGRESS_PATH,
        PYTEST_EVENTS_PATH,
        PYTEST_EVENTS_DIR,
        PYTEST_SELECTION_PATH,
        PYTEST_SUMMARY_PATH,
        PYTEST_OUTPUT_PATH,
    ]
    paths += [Path(argument.split("=", 1)[1]) for argument in command if argument.startswith("--json-report-file=")]
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


MAX_RERUN_NODEIDS = 300


def _rerun_failed_once(command: Sequence[str], *, env: Mapping[str, str], artifacts: Any) -> dict[str, Any] | None:
    """Rerun exactly the failed tests once, alone and unselected.

    A test that fails twice is red and decides the step. A test that passes
    on rerun is flaky: recorded with both outcomes in the step and in the
    canonical report, where it counts as passed. A run beside other work can
    fail hundreds of tests on load alone; that is not a verdict on the head.
    """
    report_path = _pytest_report_path(command)
    report = _read_json(report_path)
    if not isinstance(report, Mapping):
        return None
    failed = [
        str(test["nodeid"])
        for test in report.get("tests", [])
        if isinstance(test, Mapping) and test.get("outcome") in {"failed", "error"} and test.get("nodeid")
    ]
    if not failed or len(failed) > MAX_RERUN_NODEIDS:
        return None
    rerun_report = artifacts.step_dir / "pytest-rerun.json"
    rerun_command = [
        venv_python(root=ROOT),
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--json-report",
        "--json-report-omit=collectors,log,streams,warnings",
        f"--json-report-file={rerun_report}",
        *MANAGED_PLUGIN_ARGS,
        "-p",
        "no:testmon",
        "-p",
        "no:randomly",
        CLEAR_CONFIGURED_ADDOPTS,
        *failed,
    ]
    sys.stderr.write(f"\n  rerun {len(failed)} failed test(s) alone ... ")
    sys.stderr.flush()
    rerun_env = {key: value for key, value in env.items() if not key.startswith("PYTEST_XDIST")}
    rerun_completed = subprocess.run(rerun_command, cwd=ROOT, env=rerun_env, stdout=sys.stderr, stderr=sys.stderr)
    second = _read_json(rerun_report)
    if not isinstance(second, Mapping) or rerun_completed.returncode not in (0, 1):
        # No report, or pytest itself did not finish cleanly (exit 3 is an
        # internal error): nothing here clears a failure.
        return {
            "attempted": failed,
            "still_failed": failed,
            "flaky": [],
            "rerun_report": None,
            "rerun_exit": rerun_completed.returncode,
        }
    second_outcome = {
        str(test["nodeid"]): str(test.get("outcome"))
        for test in second.get("tests", [])
        if isinstance(test, Mapping) and test.get("nodeid")
    }
    still_failed = [nodeid for nodeid in failed if second_outcome.get(nodeid) != "passed"]
    flaky = [nodeid for nodeid in failed if second_outcome.get(nodeid) == "passed"]
    if not still_failed and rerun_completed.returncode != 0:
        # Every node passed but the process did not: the run is not green.
        still_failed, flaky = failed, []
    if flaky:
        patched = dict(report)
        tests = []
        for test in report.get("tests", []):
            if isinstance(test, Mapping) and test.get("nodeid") in flaky:
                test = {**test, "first_outcome": test.get("outcome"), "outcome": "passed", "flaky": True}
            tests.append(test)
        patched["tests"] = tests
        summary = dict(report.get("summary") or {})
        for key in ("failed", "error"):
            if key in summary:
                summary[key] = max(
                    0,
                    int(summary[key])
                    - sum(
                        1
                        for t in report.get("tests", [])
                        if isinstance(t, Mapping) and t.get("nodeid") in flaky and t.get("outcome") == key
                    ),
                )
        summary["passed"] = int(summary.get("passed", 0)) + len(flaky)
        summary["flaky"] = len(flaky)
        if not still_failed:
            summary["exitstatus"] = 0
        patched["summary"] = summary
        report_path.write_text(json.dumps(patched), encoding="utf-8")
        # The progress plugin's own summary carries the first exit status;
        # evidence evaluation compares the two, so both tell the same story.
        plugin_summary_path = artifacts.step_dir / "summary.json"
        plugin_summary = _read_json(plugin_summary_path)
        if isinstance(plugin_summary, Mapping):
            updated = dict(plugin_summary)
            updated["flaky"] = len(flaky)
            if not still_failed:
                updated["exitstatus"] = 0
            plugin_summary_path.write_text(json.dumps(updated), encoding="utf-8")
    return {
        "attempted": failed,
        "still_failed": still_failed,
        "flaky": flaky,
        "rerun_report": str(rerun_report.relative_to(ROOT)),
    }


def _pytest_report_path(command: Sequence[str]) -> Path:
    return next(
        (Path(argument.split("=", 1)[1]) for argument in command if argument.startswith("--json-report-file=")),
        PYTEST_REPORT_PATH,
    )


def _copy_pytest_report(command: Sequence[str], artifacts: Any) -> dict[str, Any]:
    report = _read_json(_pytest_report_path(command))
    metadata: dict[str, Any] = {}
    if report is not None:
        destination = artifacts.step_dir / PYTEST_CANONICAL_REPORT_NAME
        shutil.copyfile(_pytest_report_path(command), destination)
        metadata["report_path"] = str(destination.relative_to(ROOT))
    return metadata


def _subprocess_env() -> dict[str, str]:
    return {**os.environ, "POLYLOGUE_ROOT": str(ROOT), "PYTHONPYCACHEPREFIX": str(ROOT / ".cache" / "pycache")}


def _run(label: str, command: list[str], *, run: VerifyRun) -> tuple[int, float, dict[str, Any]]:
    started = time.monotonic()
    sys.stderr.write(f"  {label} ... ")
    sys.stderr.flush()
    pytest_step = label.startswith("pytest")
    artifacts = run.start_step(label=label, cmd=command)
    env = _subprocess_env()
    completed: subprocess.CompletedProcess[Any]
    rerun: dict[str, Any] | None = None
    executable_result = executable_gate_result(command, gate=label, env=env)
    if not executable_result.ok:
        early_metadata = {
            "diagnosis": executable_result.diagnosis,
            "required_gate": executable_result.to_payload(),
        }
        run.finish_step(
            step_id=artifacts.step_id,
            result=_early_gate_failure_result(started, early_metadata),
        )
        sys.stderr.write("FAILED (missing executable)\n")
        return 127, time.monotonic() - started, early_metadata
    if pytest_step:
        _clear_pytest_report(command)
        _normalize_managed_pytest_environment(env)
        env = env_for_pytest_step(env, run=run, artifacts=artifacts)
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=sys.stderr, stderr=sys.stderr)
        rerun = _rerun_failed_once(command, env=env, artifacts=artifacts) if completed.returncode else None
    else:
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        except OSError as exc:
            early_metadata = {"diagnosis": "gate_subprocess_launch_failed", "error": str(exc)}
            run.finish_step(
                step_id=artifacts.step_id,
                result=_early_gate_failure_result(started, early_metadata),
            )
            sys.stderr.write("FAILED (subprocess launch)\n")
            return 127, time.monotonic() - started, early_metadata
    elapsed = time.monotonic() - started
    metadata: dict[str, Any] = {
        "diagnosis": "pytest_failed" if pytest_step else "gate_passed" if completed.returncode == 0 else "gate_failed"
    }
    if pytest_step:
        if rerun is not None:
            metadata["rerun"] = rerun
            if not rerun["still_failed"]:
                # Every failure passed alone: the step is green with its
                # flakes named, never green silently.
                metadata["diagnosis"] = "gate_passed"
                completed = subprocess.CompletedProcess(command, 0)
        metadata.update(_copy_pytest_report(command, artifacts))
        copy_current_pytest_artifacts(
            ROOT,
            artifacts,
            legacy_paths={
                "progress_path": PYTEST_PROGRESS_PATH,
                "events_merged_path": PYTEST_EVENTS_PATH,
                "selection_path": PYTEST_SELECTION_PATH,
                "summary_path": PYTEST_SUMMARY_PATH,
            },
        )
    else:
        stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        output = stdout + ("\n" if stdout and stderr else "") + stderr
        if output:
            artifacts.output_path.write_text(output, encoding="utf-8")
            metadata["output_path"] = str(artifacts.output_path.relative_to(ROOT))
        if label == "render all" and completed.returncode != 0:
            for line in output.splitlines():
                match = _RENDER_DIAGNOSIS_RE.match(line)
                if match is not None:
                    metadata["diagnosis"] = match.group("token")
                    break
        if "--json" in command:
            decoded: object = None
            with contextlib.suppress(json.JSONDecodeError):
                decoded = json.loads(output)
            if not isinstance(decoded, dict):
                for candidate in reversed(output.splitlines()):
                    with contextlib.suppress(json.JSONDecodeError):
                        possible = json.loads(candidate)
                        if isinstance(possible, dict):
                            decoded = possible
                            break
            if isinstance(decoded, dict):
                required_gate = decoded.get("required_gate")
                if isinstance(required_gate, dict):
                    metadata["required_gate"] = required_gate
                    if required_gate.get("gate_passed") is False:
                        metadata["diagnosis"] = str(required_gate.get("diagnosis") or "gate_failed")
    effective_exit = completed.returncode
    if not pytest_step:
        required_gate = metadata.get("required_gate")
        if isinstance(required_gate, Mapping) and required_gate.get("gate_passed") is False and effective_exit == 0:
            effective_exit = 1
    step = run.finish_step(
        step_id=artifacts.step_id, result={"duration_s": round(elapsed, 2), **metadata, "exit": effective_exit}
    )
    if pytest_step and step is not None:
        effective_exit = int(step["exit"])
        metadata = step
    sys.stderr.write(f"{'ok' if effective_exit == 0 else 'FAILED'} ({elapsed:.1f}s)\n")
    if not pytest_step and effective_exit and isinstance(completed.stdout, str):
        sys.stderr.write(completed.stdout)
    if not pytest_step and completed.returncode and isinstance(completed.stderr, str):
        sys.stderr.write(completed.stderr)
    return effective_exit, elapsed, metadata


def _early_gate_failure_result(started: float, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Finalize an early required-gate failure with its authoritative exit."""
    return {**metadata, "duration_s": round(time.monotonic() - started, 2), "exit": 127}


def _scope(*, quick: bool, selection: str) -> VerificationScope:
    if quick:
        return VerificationScope.NON_TEST
    return VerificationScope.AFFECTED if selection == "affected" else VerificationScope.COMPLETE


def _emit(payload: Mapping[str, Any], *, use_json: bool, operation: str | None) -> None:
    result = declared_verification_result(payload, operation=operation) if operation else dict(payload)
    if operation:
        # The operation result carries the same bounded receipt as the
        # evidence lane.  AgentCTL lifecycle fields remain outside this
        # projection and cannot turn process completion into semantic success.
        result["semantic_receipt"] = canonical_verification_receipt(payload)
    if use_json or operation:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))


def _verification_workload_receipt(
    *,
    tier: str,
    git_head: str | None,
    results: Sequence[Mapping[str, Any]],
    exit_code: int,
) -> dict[str, Any]:
    """Adapt verifier step timing into the shared workload receipt contract."""
    phases = tuple(str(result["name"]) for result in results)
    spec = WorkloadEnvelopeSpec(
        workload_id=f"devtools:verify:{tier}",
        family_id="verification",
        version=1,
        inputs=(WorkloadInputRef(input_id=f"git:{git_head}" if git_head else "git:unavailable"),),
        phases=phases,
        measurement_scope=MeasurementScope.PROCESS_TREE,
    )
    observations = tuple(
        WorkloadPhaseObservation(
            name=str(result["name"]),
            wall_ms=float(result["duration_s"]) * 1_000,
            unavailable=_UNMEASURED_WORKLOAD_DIMENSIONS,
        )
        for result in results
    )
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED if exit_code == 0 else WorkloadRunStatus.FAILED,
        build_id=f"git:{git_head}" if git_head else None,
        runtime_id=f"python:{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        archive_id=None,
        generation_id=None,
        frame_id=None,
        phases=observations,
        notes=(
            "Verifier adapter records step wall time only; resource dimensions are explicitly unavailable.",
            f"Measurement-path inventory contains {len(workload_adapter_declarations())} declared dispositions.",
        ),
    )
    return receipt.to_payload()


def _finish_interrupted_verification(
    *,
    run: VerifyRun,
    started: float,
    scope: VerificationScope,
    args: argparse.Namespace,
    agentctl_operation: str | None,
    exit_code: int,
    termination_reason: str,
) -> int:
    """Persist the terminal state when an outer runtime ends verification."""
    run.finish_interrupted_steps(
        exit_code=exit_code,
        diagnosis="verification_interrupted",
        termination_reason=termination_reason,
    )
    payload = _finish_and_record_verification(
        run=run,
        exit_code=exit_code,
        duration_s=time.monotonic() - started,
        diagnosis="verification_interrupted",
        verification_scope=scope.value,
        final_git_head=git_head(ROOT),
        pytest_aggregate={
            "selection_mode": "quick" if args.quick else "all" if args.all_tests else "affected",
            "outcomes": {},
            "terminal_green": False,
            "complete_corpus_covered": False,
            "termination_reason": termination_reason,
        },
    )
    _emit(payload, use_json=args.json, operation=agentctl_operation)
    return exit_code


def _finish_and_record_verification(
    *,
    run: VerifyRun,
    exit_code: int,
    duration_s: float,
    diagnosis: str | None = None,
    verification_scope: str | None = None,
    final_git_head: str | None = None,
    pytest_aggregate: Mapping[str, Any] | None = None,
    workload_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Finish, durably append, and prune every terminal verification path."""
    payload = run.finish(
        exit_code=exit_code,
        duration_s=duration_s,
        diagnosis=diagnosis,
        verification_scope=verification_scope,
        final_git_head=final_git_head,
        pytest_aggregate=pytest_aggregate,
        workload_receipt=workload_receipt,
    )
    append_verify_history(payload)
    append_verification_evidence(payload)
    prune_successful_verify_runs(root=ROOT)
    if exit_code != 0:
        try:
            from polylogue.context.failure_seed import write_failure_seed

            write_failure_seed(root=ROOT)
        except (FileNotFoundError, ValueError, OSError):
            pass
    return payload


def _aggregate_pytest_results(
    results: Sequence[Mapping[str, Any]], *, expected_step_count: int, mode: str, exit_code: int
) -> dict[str, Any]:
    pytest_results = [result for result in results if str(result.get("name", "")).startswith("pytest")]
    outcomes: dict[str, int] = {}
    selected_counts: list[int] = []
    terminal_counts: list[int] = []
    for result in pytest_results:
        raw_statistics: object = result.get("statistics")
        statistics: Mapping[str, Any] = raw_statistics if isinstance(raw_statistics, Mapping) else {}
        selected = statistics.get("selected_count")
        terminal = statistics.get("terminal_count")
        if isinstance(selected, int) and not isinstance(selected, bool):
            selected_counts.append(selected)
        if isinstance(terminal, int) and not isinstance(terminal, bool):
            terminal_counts.append(terminal)
        for outcome, count in (statistics.get("outcomes") or {}).items():
            outcomes[str(outcome)] = outcomes.get(str(outcome), 0) + int(count)
    complete = mode == "all" and exit_code == 0 and len(pytest_results) == expected_step_count
    return {
        "selection_mode": mode,
        # Full-corpus verification partitions the collection across managed
        # pytest steps, so these are disjoint populations and must be summed.
        "selected_union_count": sum(selected_counts),
        "terminal_union_count": sum(terminal_counts),
        "outcomes": outcomes,
        "terminal_green": exit_code == 0,
        "complete_corpus_covered": complete,
    }


def _main(argv: list[str] | None = None, *, agentctl_operation: str | None = None) -> int:
    refusal = refuse_verify_tier(list(argv or []), os.environ)
    if refusal is not None:
        sys.stderr.write(refusal + "\n")
        return 2
    parser = argparse.ArgumentParser(description="Run project semantic verification.")
    parser.add_argument("--quick", action="store_true", help="run the static gates only")
    parser.add_argument(
        "--all",
        "--full",
        dest="all_tests",
        action="store_true",
        help="explicit form of the default: the static gates plus the complete corpus",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    _anchor_verification_paths()
    validate_authority_matrix()
    started = time.monotonic()
    graph = inspect_testmon_graph(ROOT)
    selection = "all" if args.all_tests else "affected"
    scope = _scope(quick=args.quick, selection=selection)
    try:
        assert_polylogue_matches_checkout(ROOT, context="devtools verify")
    except CheckoutImportMismatchError as exc:
        payload = {
            "exit_code": 125,
            "duration_s": time.monotonic() - started,
            "diagnosis": "checkout_import_mismatch",
            "verification_scope": scope.value,
            "final_git_head": git_head(ROOT),
        }
        _emit(payload, use_json=args.json, operation=agentctl_operation)
        sys.stderr.write(f"verify: {exc}\n")
        return 125
    head = git_head(ROOT)
    tier = "quick" if args.quick else selection
    run = VerifyRun(
        tier=tier,
        argv=list(argv or []),
        git_head=head,
        root=ROOT,
        mirror_current=agentctl_operation is None,
        agentctl_operation=agentctl_operation,
    )
    if not args.quick:
        run.record_selection(
            selection_mode=selection,
            graph_status=str(graph.status),
            graph_reason=graph.reason,
            full_rerun_cause=graph.full_rerun_cause,
        )
        if graph.status is TestmonGraphStatus.UNUSABLE:
            payload = _finish_and_record_verification(
                run=run,
                exit_code=2,
                duration_s=time.monotonic() - started,
                diagnosis="graph_unusable",
                verification_scope=scope.value,
                final_git_head=git_head(ROOT),
            )
            sys.stderr.write(
                f"verify: {graph.reason}.\n"
                f"  remedy: delete {TESTMON_DATA_RELPATH} and rerun; the next run reseeds it.\n"
            )
            _emit(payload, use_json=args.json, operation=agentctl_operation)
            return 2
        if graph.status is TestmonGraphStatus.ABSENT:
            sys.stderr.write("verify: no testmon datafile: this run seeds it and runs every test.\n")
        elif graph.full_rerun_cause:
            sys.stderr.write(
                f"verify: {graph.full_rerun_cause} since the graph was written: this run re-executes every test.\n"
            )
    steps = build_verify_steps(quick=args.quick, selection=selection)
    try:
        results: list[dict[str, Any]] = []
        exit_code = 0
        for label, command in steps:
            rc, elapsed, metadata = _run(label, command, run=run)
            results.append({"name": label, "duration_s": round(elapsed, 2), "exit": rc, **metadata})
            if rc:
                exit_code = exit_code or rc
                if args.quick:
                    break
    except VerificationInterrupted as exc:
        return _finish_interrupted_verification(
            run=run,
            started=started,
            scope=scope,
            args=args,
            agentctl_operation=agentctl_operation,
            exit_code=128 + exc.signum,
            termination_reason=signal.Signals(exc.signum).name.lower(),
        )
    except KeyboardInterrupt:
        return _finish_interrupted_verification(
            run=run,
            started=started,
            scope=scope,
            args=args,
            agentctl_operation=agentctl_operation,
            exit_code=130,
            termination_reason="operator_interrupt",
        )
    aggregate = _aggregate_pytest_results(
        results,
        expected_step_count=sum(label.startswith("pytest") for label, _command in steps),
        mode="quick" if args.quick else selection,
        exit_code=exit_code,
    )
    # The retained exit code is the first failure's; its diagnosis must be too.
    diagnosis = next((str(result["diagnosis"]) for result in results if result["exit"] != 0), None)
    payload = _finish_and_record_verification(
        run=run,
        exit_code=exit_code,
        duration_s=time.monotonic() - started,
        diagnosis=diagnosis,
        verification_scope=scope.value,
        final_git_head=git_head(ROOT),
        pytest_aggregate=aggregate,
        workload_receipt=_verification_workload_receipt(
            tier=tier,
            git_head=head,
            results=results,
            exit_code=exit_code,
        ),
    )
    _emit(payload, use_json=args.json, operation=agentctl_operation)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    handlers = {
        signum: signal.signal(signum, _raise_verification_interruption) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return _main(raw_argv, agentctl_operation=_declared_agentctl_operation(raw_argv))
    except VerificationInterrupted as exc:
        return 128 + exc.signum
    except KeyboardInterrupt:
        return 130
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)


if __name__ == "__main__":
    raise SystemExit(main())
