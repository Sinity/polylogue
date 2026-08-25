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

from devtools.checkout_guard import CheckoutImportMismatchError, assert_polylogue_matches_checkout
from devtools.pytest_collection_contract import (
    CLOSED_WORLD_COLLECTION_ARGS,
    IGNORED_COLLECTION_ARGS,
    MANAGED_PLUGIN_ARGS,
    PARALLEL_MARKER_EXPRESSION,
    PROGRESS_PLUGIN_NAME,
    SERIAL_MARKER_EXPRESSION,
    STORAGE_SCALE_MARKER_EXPRESSION,
)
from devtools.pytest_scratch import PytestScratchLease, run_managed_pytest, scratch_root_from_environment
from devtools.required_gate import executable_gate_result
from devtools.testmon_bootstrap import (
    TESTMON_DATA_RELPATH,
    NativeTestmonDeadlineError,
    NativeTestmonRepairError,
    classify_native_testmon_changes,
    prepare_native_testmon_environment,
)
from devtools.verification_contracts import VerificationScope
from devtools.verification_result import declared_verification_result
from devtools.verify_runs import (
    CURRENT_EVENTS_DIR,
    PYTEST_CANONICAL_REPORT_NAME,
    VerifyRun,
    append_verify_history,
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
)

ROOT = Path(__file__).resolve().parents[1]
TESTMON_DATA = TESTMON_DATA_RELPATH
PYTEST_REPORT_DIR = Path(".cache/verify")
PYTEST_REPORT_PATH = PYTEST_REPORT_DIR / "last-pytest.json"
PYTEST_PROGRESS_PATH = PYTEST_REPORT_DIR / "current-pytest-progress.json"
PYTEST_EVENTS_PATH = PYTEST_REPORT_DIR / "current-pytest-events.jsonl"
PYTEST_EVENTS_DIR = CURRENT_EVENTS_DIR
PYTEST_SELECTION_PATH = PYTEST_REPORT_DIR / "current-pytest-selection.json"
PYTEST_SUMMARY_PATH = PYTEST_REPORT_DIR / "current-pytest-summary.json"
PYTEST_OUTPUT_PATH = PYTEST_REPORT_DIR / "current-pytest-output.log"
PYTEST_JUNIT_REPORT_DIR = PYTEST_REPORT_DIR / "junit"
SERIAL_LANE_MAX_WORKERS = 4
STORAGE_SCALE_LANE_MAX_WORKERS = 1
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


def _mypy_cmd() -> list[str]:
    try:
        result = subprocess.run(["dmypy", "status"], capture_output=True, text=True, timeout=5, cwd=ROOT)
        if result.returncode == 0:
            return ["dmypy", "run", "--", "--no-error-summary"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ["mypy"]


def _devtools_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "devtools", *(part for arg in args for part in arg.split())]


def _native_pytest_environment() -> dict[str, str | None]:
    return {
        "HYPOTHESIS_PROFILE": os.environ.get("HYPOTHESIS_PROFILE") or "default",
        "POLYLOGUE_CI": os.environ.get("POLYLOGUE_CI"),
    }


def _pytest_worker_args(*, maximum: int | None = None) -> list[str]:
    try:
        workers = max(0, int(os.environ.get("POLYLOGUE_PYTEST_WORKERS", "0")))
    except ValueError:
        workers = 0
    if maximum is not None:
        workers = min(workers, maximum)
    return ["--dist=loadgroup", "-n", str(workers)]


def _native_pytest_steps(
    *,
    testmon_mode: str,
    testmon_environment: str,
    parallel_worker_args: Sequence[str],
    serial_worker_args: Sequence[str],
    storage_scale_worker_args: Sequence[str],
) -> list[tuple[str, list[str]]]:
    base = [
        sys.executable,
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
        f"--testmon-env={testmon_environment}",
        "--testmon-noselect" if testmon_mode == "all" else "--testmon-forceselect",
    ]

    def lane_command(lane: str, marker: str, workers: Sequence[str]) -> list[str]:
        command = [
            f"--junitxml={PYTEST_JUNIT_REPORT_DIR}/verify-latest-{lane}.xml"
            if item.startswith("--junitxml=")
            else f"--json-report-file={PYTEST_REPORT_DIR / f'last-pytest-{lane}.json'}"
            if item.startswith("--json-report-file=")
            else item
            for item in base
        ]
        return [*command, "-m", marker, "-p", "no:randomly", *workers]

    return [
        (
            f"pytest native parallel ({testmon_mode})",
            lane_command("parallel", PARALLEL_MARKER_EXPRESSION, parallel_worker_args),
        ),
        (
            f"pytest native serial ({testmon_mode})",
            lane_command("serial", SERIAL_MARKER_EXPRESSION, serial_worker_args),
        ),
        (
            f"pytest native storage-scale ({testmon_mode})",
            lane_command("storage-scale", STORAGE_SCALE_MARKER_EXPRESSION, storage_scale_worker_args),
        ),
    ]


def build_verify_steps(
    *, quick: bool, lab: bool, commit: bool = False, testmon_mode: str = "affected", testmon_environment: str = ""
) -> list[tuple[str, list[str]]]:
    steps = [
        ("ruff format", ["ruff", "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", ["ruff", "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", _mypy_cmd()),
    ]
    if not commit:
        steps += [
            ("render all", _devtools_cmd("render all", "--check")),
            ("verify layering", _devtools_cmd("verify layering", "--json")),
            ("verify ci-commands", _devtools_cmd("verify ci-commands", "--json")),
            ("verify doc-commands", _devtools_cmd("verify doc-commands")),
            ("verify schema-roundtrip", _devtools_cmd("verify schema-roundtrip", "--all")),
            ("verify schema-versioning", _devtools_cmd("verify schema-versioning")),
            ("verify oracle-integrity", _devtools_cmd("verify oracle-integrity")),
            (
                "schema promotion audit",
                [
                    sys.executable,
                    "-m",
                    "polylogue.schemas.promotion_audit",
                    "polylogue/schemas",
                    "--output",
                    str(PYTEST_REPORT_DIR / "schema-promotion-audit.json"),
                ],
            ),
            ("schema privacy registry", [sys.executable, "-m", "devtools.verify_schema_privacy"]),
        ]
    if not quick and not commit:
        if testmon_mode not in {"affected", "all"} or not testmon_environment:
            raise ValueError("a valid native testmon selection and environment are required")
        PYTEST_JUNIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        steps += _native_pytest_steps(
            testmon_mode=testmon_mode,
            testmon_environment=testmon_environment,
            parallel_worker_args=_pytest_worker_args(),
            serial_worker_args=_pytest_worker_args(maximum=SERIAL_LANE_MAX_WORKERS),
            storage_scale_worker_args=_pytest_worker_args(maximum=STORAGE_SCALE_LANE_MAX_WORKERS),
        )
    if lab:
        steps += [
            ("verify scenario", _devtools_cmd("verify scenario", "run", "archive-smoke", "--tier", "0")),
            ("bench slo", _devtools_cmd("bench slo", "--include-lab")),
            ("verify timestamp-doctrine", _devtools_cmd("verify timestamp-doctrine")),
            ("verify insight-honesty", _devtools_cmd("verify insight-honesty")),
        ]
    return steps


def _normalize_managed_pytest_environment(env: dict[str, str]) -> None:
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


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
    if pytest_step:
        _clear_pytest_report(command)
        _normalize_managed_pytest_environment(env)
        env = env_for_pytest_step(env, run=run, artifacts=artifacts)
        lane = label.removeprefix("pytest native ").split(" ", 1)[0]
        lease = PytestScratchLease.acquire(
            root=scratch_root_from_environment(env), run_id=run.run_id, lane=lane, evidence_dir=artifacts.step_dir
        )
        managed_command = lease.command(command)
        try:
            completed = run_managed_pytest(
                managed_command, cwd=ROOT, env=lease.environment(env), stdout=sys.stderr, stderr=sys.stderr
            )
        except KeyboardInterrupt:
            lease.finalize("cancelled")
            raise
        except BaseException:
            lease.finalize("failure")
            raise
        else:
            lease.finalize(
                "success" if completed.returncode == 0 else "worker_crash" if completed.returncode == 3 else "failure"
            )
    else:
        executable_result = executable_gate_result(command, gate=label, env=env)
        if not executable_result.ok:
            early_metadata: dict[str, Any] = {
                "diagnosis": executable_result.diagnosis,
                "required_gate": executable_result.to_payload(),
            }
            run.finish_step(
                step_id=artifacts.step_id,
                result=_early_gate_failure_result(started, early_metadata),
            )
            sys.stderr.write("FAILED (missing executable)\n")
            return 127, time.monotonic() - started, early_metadata
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


def _changed_paths(base_commit: str, head_commit: str) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["git", "diff", "--no-renames", "--name-only", "-z", f"{base_commit}...{head_commit}", "--"],
            cwd=ROOT,
            capture_output=True,
            timeout=5,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise NativeTestmonRepairError("cannot determine changed paths for testmon selection") from exc
    return tuple(sorted(os.fsdecode(path) for path in result.stdout.split(b"\0") if path))


def _git_commit(ref: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _scope(*, quick: bool, commit: bool, all_tests: bool) -> VerificationScope:
    del all_tests
    if quick or commit:
        return VerificationScope.NON_TEST
    return VerificationScope.AFFECTED


def _emit(payload: Mapping[str, Any], *, use_json: bool, operation: str | None) -> None:
    result = declared_verification_result(payload, operation=operation) if operation else dict(payload)
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
        notes=("Verifier adapter records step wall time only; resource dimensions are explicitly unavailable.",),
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
            "selection_mode": "all" if args.all_tests else "affected",
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
    prune_successful_verify_runs(root=ROOT)
    return payload


def _main(argv: list[str] | None = None, *, agentctl_operation: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run project semantic verification.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--all", "--full", dest="all_tests", action="store_true")
    parser.add_argument("--lab", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    _anchor_verification_paths()
    started = time.monotonic()
    scope = _scope(quick=args.quick, commit=args.commit, all_tests=args.all_tests)
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
    tier = "quick" if args.quick else "commit" if args.commit else "all" if args.all_tests else "affected"
    run = VerifyRun(
        tier=tier,
        argv=list(argv or []),
        git_head=head,
        root=ROOT,
        mirror_current=agentctl_operation is None,
    )
    preparation = None
    mode = "all" if args.all_tests else "affected"
    if not args.quick and not args.commit:
        if head is None:
            raise RuntimeError("cannot resolve current Git head")
        base = _git_commit("origin/master") or head
        try:
            impact = classify_native_testmon_changes(ROOT, _changed_paths(base, head))
            preparation = prepare_native_testmon_environment(
                ROOT,
                required_executable_paths=impact.executable_paths,
                pytest_profile=_native_pytest_environment()["HYPOTHESIS_PROFILE"] or "default",
                pytest_environment=_native_pytest_environment(),
            )
        except (NativeTestmonDeadlineError, NativeTestmonRepairError) as exc:
            payload = _finish_and_record_verification(
                run=run,
                exit_code=125,
                duration_s=time.monotonic() - started,
                diagnosis="native_testmon_preparation_failed",
                verification_scope=scope.value,
                final_git_head=git_head(ROOT),
            )
            _emit(payload, use_json=args.json, operation=agentctl_operation)
            sys.stderr.write(f"verify: {exc}\n")
            return 125
        if preparation.selection_mode == "bootstrap" and not args.all_tests:
            payload = _finish_and_record_verification(
                run=run,
                exit_code=2,
                duration_s=time.monotonic() - started,
                diagnosis="native_testmon_graph_unavailable",
                verification_scope=scope.value,
                final_git_head=git_head(ROOT),
            )
            _emit(payload, use_json=args.json, operation=agentctl_operation)
            return 2
        mode = "all" if args.all_tests else "affected"
        run.record_selection(
            selection_mode=mode,
            state_status=preparation.local_state.status,
            state_reason=preparation.local_state.reason,
            missing_executable_paths=preparation.local_state.missing_executable_paths,
            runtime_data_paths=impact.runtime_data_paths,
            environment_digest=preparation.environment_name,
        )
    steps = build_verify_steps(
        quick=args.quick,
        commit=args.commit,
        lab=args.lab,
        testmon_mode=mode,
        testmon_environment=preparation.environment_name if preparation else "",
    )
    results: list[dict[str, Any]] = []
    exit_code = 0
    try:
        for label, command in steps:
            rc, elapsed, metadata = _run(label, command, run=run)
            results.append({"name": label, "duration_s": round(elapsed, 2), "exit": rc, **metadata})
            if rc:
                exit_code = rc
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
    aggregate = {
        "selection_mode": mode,
        "outcomes": {
            key: value
            for result in results
            if result["name"].startswith("pytest")
            for key, value in (result.get("statistics", {}).get("outcomes") or {}).items()
        },
        "terminal_green": exit_code == 0,
        "complete_corpus_covered": False,
    }
    diagnosis = next((str(result["diagnosis"]) for result in reversed(results) if result["exit"] != 0), None)
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
