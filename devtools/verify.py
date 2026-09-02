"""Project semantic verification: gates, test selection, and typed receipts."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from devtools.required_gate import executable_gate_result
from devtools.testmon_bootstrap import (
    TESTMON_DATA_RELPATH,
    NativeTestmonDeadlineError,
    NativeTestmonRepairError,
    classify_native_testmon_changes,
    installed_packages_digest,
    prepare_native_testmon_environment,
)
from devtools.toolchain import venv_bin, venv_python
from devtools.verification_authority import validate_authority_matrix
from devtools.verification_contracts import VerificationScope
from devtools.verification_ledger import (
    append_failure_ledger,
    ledger_records,
    policy_diagnostics,
    read_failure_ledger,
    read_verify_history,
)
from devtools.verification_result import declared_verification_result
from devtools.verify_js_tests import _STAMP_NAME as JS_INSTALL_STAMP
from devtools.verify_js_tests import JS_PACKAGES, available_cpus, extension_test_workers
from devtools.verify_runs import (
    CURRENT_EVENTS_DIR,
    PYTEST_CANONICAL_REPORT_NAME,
    VerifyRun,
    append_verification_evidence,
    append_verify_history,
    canonical_verification_receipt,
    copy_current_pytest_artifacts,
    env_for_pytest_step,
    git_dirty,
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


def _native_pytest_steps(
    *,
    testmon_mode: str,
    testmon_environment: str,
    parallel_worker_args: Sequence[str],
    serial_worker_args: Sequence[str],
    storage_scale_worker_args: Sequence[str],
) -> list[tuple[str, list[str]]]:
    if testmon_mode == "affected":
        testmon_args = ["--testmon", f"--testmon-env={testmon_environment}", "--testmon-forceselect"]
    elif testmon_mode == "all":
        # The complete corpus is what publishes the dependency graph that
        # affected verification selects from, so it always traces. It runs as
        # one collection: testmon deletes every recorded test a run did not
        # collect, so a partitioned run keeps only its last partition.
        testmon_args = ["--testmon", f"--testmon-env={testmon_environment}", "--testmon-noselect"]
    else:
        raise ValueError(f"unsupported native testmon mode: {testmon_mode}")
    base = [
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
        *testmon_args,
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

    parallel_steps = [
        (
            f"pytest native parallel ({testmon_mode})",
            lane_command("parallel", PARALLEL_MARKER_EXPRESSION, parallel_worker_args),
        )
    ]
    return [
        *parallel_steps,
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
    *,
    quick: bool,
    commit: bool = False,
    testmon_mode: str = "affected",
    testmon_environment: str = "",
) -> list[tuple[str, list[str]]]:
    steps = [
        ("ruff format", [venv_bin("ruff", root=ROOT), "format", "--check", "polylogue/", "tests/", "devtools/"]),
        ("ruff check", [venv_bin("ruff", root=ROOT), "check", "polylogue/", "tests/", "devtools/"]),
        ("mypy", _mypy_cmd()),
    ]
    if not commit:
        steps += [
            ("render all", _devtools_cmd("render all", "--check")),
            ("verify layering", _devtools_cmd("verify layering", "--json")),
            ("verify patterns", _devtools_cmd("verify patterns", "--json")),
            ("verify ci-commands", _devtools_cmd("verify ci-commands", "--json")),
            ("verify js-tests", _devtools_cmd("verify js-tests", "--json")),
            ("verify doc-commands", _devtools_cmd("verify doc-commands")),
            ("verify schema-roundtrip", _devtools_cmd("verify schema-roundtrip", "--all")),
            ("verify schema-versioning", _devtools_cmd("verify schema-versioning")),
            ("verify oracle-integrity", _devtools_cmd("verify oracle-integrity")),
            ("verify consumer-reachability", _devtools_cmd("verify consumer-reachability", "--json")),
            ("verify definition-closure", _devtools_cmd("verify definition-closure", "--json")),
            ("verify timestamp-doctrine", _devtools_cmd("verify timestamp-doctrine")),
            ("verify insight-honesty", _devtools_cmd("verify insight-honesty")),
            (
                "schema promotion audit",
                [
                    venv_python(root=ROOT),
                    "-m",
                    "polylogue.schemas.promotion_audit",
                    "polylogue/schemas",
                    "--output",
                    str(PYTEST_REPORT_DIR / "schema-promotion-audit.json"),
                ],
            ),
            ("schema privacy registry", [venv_python(root=ROOT), "-m", "devtools.verify_schema_privacy"]),
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
    return steps


def _normalize_managed_pytest_environment(env: dict[str, str]) -> None:
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Every managed pytest run writes the shared dependency graph: testmon
    # replaces the edges of each test it executes. A run under a reduced
    # Hypothesis budget records fewer edges for its property tests, and every
    # later affected selection inherits the blind spot, so all graph writers
    # trace under one full profile.
    env["HYPOTHESIS_PROFILE"] = "default"
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


def _execution_plan_digest() -> str:
    """Inputs of the whole verification plan beyond Python collection: worker
    count, the JavaScript toolchain and installed package trees the js-tests
    gate runs against, and whether each required Python gate executable can
    be launched."""
    node = shutil.which("node")
    node_version = ""
    if node:
        try:
            node_version = subprocess.run(
                [node, "--version"], capture_output=True, text=True, timeout=10, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            node_version = "unavailable"
    npm = shutil.which("npm")
    npm_version = ""
    if npm:
        try:
            npm_version = subprocess.run(
                [npm, "--version"], capture_output=True, text=True, timeout=10, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            npm_version = "unavailable"
    payload = {
        "workers": os.environ.get("POLYLOGUE_PYTEST_WORKERS", ""),
        "js_workers": os.environ.get("POLYLOGUE_EXTENSION_TEST_WORKERS")
        or str(extension_test_workers(available_cpus())),
        # Gates that read an authority override from the environment run a
        # different plan under it.
        "consumer_reachability": {
            name: os.environ.get(name, "") for name in ("CONSUMER_REACHABILITY_BASE", "CONSUMER_REACHABILITY_HEAD")
        },
        # Gates that compare against master read the tracking ref; a fetch
        # that moves it is a different plan at the same head.
        "master": _git_commit("origin/master") or "",
        "merge_base": _merge_base_with_master(),
        # A stored Hypothesis counterexample is replayed by a real run; the
        # example database is part of what the corpus executes.
        "hypothesis_examples": _tree_listing(ROOT / ".cache" / "hypothesis" / "examples"),
        "node": node_version,
        "npm": npm_version,
        # The Python gate executables the plan requires: a missing or
        # unlaunchable one turns a real run red, so its availability is part
        # of the plan a recorded green stands for.
        "gates": {name: _executable_identity(venv_bin(name, root=ROOT)) for name in ("ruff", "mypy", "dmypy")},
        # Every step of the complete plan names an executable; tools a gate
        # resolves at run time are listed here.
        "steps": {
            label: _executable_identity(command[0])
            for label, command in build_verify_steps(quick=False, testmon_mode="all", testmon_environment="plan")
            if command
        },
        "tools": {name: _executable_identity(name) for name in ("ast-grep", "node", "npm")},
        # The js-tests gate runs against each package's installed tree; the
        # lockfile, the install stamp and the installed binaries identify
        # what a green run executed.
        "js": {
            package: {
                "lock": _file_digest(ROOT / package / "package-lock.json"),
                "installed": _file_digest(ROOT / package / "node_modules" / JS_INSTALL_STAMP),
                "binaries": _tree_listing(ROOT / package / "node_modules" / ".bin"),
            }
            for package in JS_PACKAGES
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _merge_base_with_master() -> str:
    try:
        result = subprocess.run(
            ["git", "merge-base", "HEAD", "origin/master"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _executable_identity(executable: str) -> str:
    """Where an executable resolves and what it is, or "absent"."""
    if not executable_gate_result([executable], gate="plan").executable_available:
        return "absent"
    resolved = executable if os.path.dirname(executable) else shutil.which(executable)
    if resolved is None:
        return "absent"
    try:
        real = os.path.realpath(resolved)
        state = os.stat(real)
    except OSError:
        return "absent"
    return f"{real}:{state.st_size}:{int(state.st_mtime)}"


def _tree_listing(root: Path) -> str:
    """A digest of the names and sizes of the files directly under a directory."""
    try:
        entries = sorted((entry.name, entry.stat().st_size) for entry in root.iterdir() if not entry.is_dir())
    except OSError:
        return "absent"
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "absent"


def _packages_drift(packages: str) -> bool:
    """Whether the installed set differs from the newest complete run's.

    A dependency version does not rename the environment (testmon cannot see
    inside site-packages), so this is recorded on the receipt rather than
    used to discard the graph; the next complete run re-traces under the new
    set because its skip rule keys on packages_digest.
    """
    for row in reversed(read_verify_history(ROOT / ".cache/verify/history.jsonl")):
        if row.get("tier") != "all":
            continue
        selection = row.get("testmon_selection")
        recorded = selection.get("packages_digest") if isinstance(selection, Mapping) else None
        if isinstance(recorded, str):
            return recorded != packages
    return False


def _corpus_already_verified(*, head: str, environment: str, packages: str, plan: str) -> str | None:
    """The run id of the newest complete-corpus attempt at this head, environment,
    package set, and execution plan, if that attempt was a clean-tree success."""
    for row in reversed(read_verify_history(ROOT / ".cache/verify/history.jsonl")):
        if row.get("tier") != "all":
            continue
        receipt = row.get("semantic_receipt")
        selection = row.get("testmon_selection")
        aggregate = row.get("pytest_aggregate")
        if not (isinstance(receipt, Mapping) and isinstance(selection, Mapping) and isinstance(aggregate, Mapping)):
            continue
        if aggregate.get("covered_by_run"):
            # A skip is not an attempt; the attempt it reused is older.
            continue
        if receipt.get("source_revision") != head:
            # The newest complete attempt anywhere rewrote the shared graph
            # for its own head; a green at this head is no longer the graph
            # on disk, so the corpus runs and restores it.
            return None
        if (
            selection.get("environment_digest") != environment
            or selection.get("packages_digest") != packages
            or selection.get("plan_digest") != plan
        ):
            continue
        # The newest attempt decides: a later failed recompute must not be
        # hidden behind an older green.
        if (
            row.get("status") == "success"
            and row.get("git_dirty") is False
            and row.get("git_head") == head
            and aggregate.get("complete_corpus_covered") is True
        ):
            run_id = row.get("run_id")
            artifact_dir = row.get("artifact_dir")
            # Coverage is only reusable while its evidence still exists; a
            # run whose detail retention pruned is history, not a receipt.
            if isinstance(run_id, str) and isinstance(artifact_dir, str) and (ROOT / artifact_dir).is_dir():
                return run_id
            return None
        return None
    return None


def _emit(payload: Mapping[str, Any], *, use_json: bool, operation: str | None) -> None:
    result = declared_verification_result(payload, operation=operation) if operation else dict(payload)
    if operation:
        # The operation result carries the same bounded receipt as the
        # evidence lane.  AgentCTL lifecycle fields remain outside this
        # projection and cannot turn process completion into semantic success.
        result["semantic_receipt"] = canonical_verification_receipt(payload)
    if use_json or operation:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))


def _emit_native_testmon_refusal(*, preparation: Any, reason: str, stream: Any | None = None) -> None:
    """Explain why affected verification refused to measure this checkout."""
    if stream is None:
        stream = sys.stderr
    stream.write(
        "verify: refusing to measure affected verification; no compatible native testmon graph is available.\n"
        "  selection: affected\n"
        f"  environment: '{preparation.environment_name}' ({preparation.local_state.status})\n"
        f"  reason: {reason}\n"
        "  remedy: run 'devtools verify --all' to produce a compatible graph, then rerun 'devtools verify'.\n"
    )


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
    existing = read_failure_ledger(ROOT / ".cache/verify/failure-ledger.jsonl")
    verify_history = read_verify_history(ROOT / ".cache/verify/history.jsonl")
    records = ledger_records(payload, history=verify_history)
    if records:
        append_failure_ledger(records, path=ROOT / ".cache/verify/failure-ledger.jsonl")
        payload["failure_ledger"] = policy_diagnostics((*existing, *records))
        run._payload["failure_ledger"] = payload["failure_ledger"]
        run.write()
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
    parser = argparse.ArgumentParser(description="Run project semantic verification.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--all", "--full", dest="all_tests", action="store_true")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="run the complete corpus even when this head and environment already have a successful complete run",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.recompute and not args.all_tests:
        parser.error("--recompute applies to the complete corpus; pass --all")
    _anchor_verification_paths()
    validate_authority_matrix()
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
        agentctl_operation=agentctl_operation,
    )
    try:
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
                )
                packages = installed_packages_digest()
                plan = _execution_plan_digest()
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
            run.record_selection(
                selection_mode=mode,
                state_status=preparation.local_state.status,
                state_reason=preparation.local_state.reason,
                missing_executable_paths=preparation.local_state.missing_executable_paths,
                runtime_data_paths=impact.runtime_data_paths,
                environment_digest=preparation.environment_name,
                packages_digest=packages,
                plan_digest=plan,
                packages_drift=_packages_drift(packages),
            )
            if args.all_tests and not args.recompute and preparation.local_state.valid:
                # A graph that needs bootstrapping is itself a reason to run:
                # the skip covers verification, not the graph.
                covered = _corpus_already_verified(
                    head=head, environment=preparation.environment_name, packages=packages, plan=plan
                )
                if covered is not None and not git_dirty(ROOT) and git_head(ROOT) == head:
                    # The corpus at this head under this environment is already
                    # a recorded fact; rerunning it changes nothing but the
                    # electricity bill. A different head, a moved dependency
                    # set, a dirty tree, or a head that moved while this run
                    # prepared runs in full.
                    payload = _finish_and_record_verification(
                        run=run,
                        exit_code=0,
                        duration_s=time.monotonic() - started,
                        diagnosis="corpus_already_verified",
                        verification_scope=scope.value,
                        final_git_head=head,
                        pytest_aggregate={
                            "selection_mode": "all",
                            "outcomes": {},
                            "terminal_green": True,
                            "complete_corpus_covered": False,
                            "covered_by_run": covered,
                        },
                    )
                    sys.stderr.write(f"verify: complete corpus already verified at {head[:12]} by run {covered}\n")
                    _emit(payload, use_json=args.json, operation=agentctl_operation)
                    return 0
            if preparation.selection_mode == "bootstrap" and not args.all_tests:
                payload = _finish_and_record_verification(
                    run=run,
                    exit_code=2,
                    duration_s=time.monotonic() - started,
                    diagnosis="native_testmon_graph_unavailable",
                    verification_scope=scope.value,
                    final_git_head=git_head(ROOT),
                )
                _emit_native_testmon_refusal(preparation=preparation, reason=preparation.local_state.reason)
                _emit(payload, use_json=args.json, operation=agentctl_operation)
                return 2
        steps = build_verify_steps(
            quick=args.quick,
            commit=args.commit,
            testmon_mode=mode,
            testmon_environment=preparation.environment_name if preparation else "",
        )
        results: list[dict[str, Any]] = []
        exit_code = 0
        for label, command in steps:
            rc, elapsed, metadata = _run(label, command, run=run)
            results.append({"name": label, "duration_s": round(elapsed, 2), "exit": rc, **metadata})
            if rc:
                exit_code = exit_code or rc
                if not args.all_tests:
                    break
                # A complete-corpus run reports the whole corpus and finishes
                # publishing the graph; stopping at the first red step would
                # leave both partial.
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
        mode=mode,
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
