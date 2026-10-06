"""Project semantic verification: gates, test selection, and typed receipts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from devtools.agent_env import refuse_verify_tier, runtime_env
from devtools.checkout_guard import CheckoutImportMismatchError, assert_polylogue_matches_checkout
from devtools.checkout_identity import (
    ALLOW_DEFAULT_BRANCH_ENV,
    ON_DEFAULT_BRANCH_FLAG,
    REFUSAL_DIAGNOSIS,
    REFUSAL_EXIT,
    checkout_identity,
    default_branch_refusal,
)
from devtools.cloud_sentinels import cloud_sentinel_declined
from devtools.gate import quick_gates
from devtools.pytest_invocation import (
    CLOSED_WORLD_COLLECTION_ARGS,
    DEVTOOLS_PLUGIN_ARGS,
    IGNORED_COLLECTION_ARGS,
    SUITE_COST_PLUGIN_NAME,
    effective_hypothesis_profile,
    managed_plugin_args,
)
from devtools.pytest_slot import (
    OOM_KILLED_DIAGNOSIS,
    WORKTREE_PROVENANCE_ENV,
    PytestSlotUnavailableError,
    run_pytest,
    run_pytest_isolated,
    termination_metadata,
)
from devtools.pytest_stream_report import (
    REPORT_FILE_OPTION,
    report_file_argument,
    report_nodeid_to_selector,
    spool_paths,
)
from devtools.pytest_suite_cost_plugin import SUITE_COST_DIR_ENV, write_run_receipt
from devtools.required_gate import executable_gate_result
from devtools.testmon_provision import (
    TESTMON_COVERAGE_CORE,
    TestmonGraphStatus,
    declared_test_files,
    inspect_testmon_graph,
    primary_worktree,
    snapshot_testmon_graph,
    sync_testmon_graph,
    testmon_datafile,
    testmon_environment,
)
from devtools.toolchain import venv_python
from devtools.verification_admission import (
    AFFECTED_MAX_SELECTED_TESTS,
    AFFECTED_MAX_UNRECORDED_FILES,
    AFFECTED_MAX_WORKERS,
    AffectedAdmission,
    admit_affected_selection,
)
from devtools.verification_authority import validate_authority_matrix
from devtools.verification_contracts import VerificationScope
from devtools.verification_result import declared_verification_result
from devtools.verify_runs import (
    CURRENT_EVENTS_DIR,
    PYTEST_CANONICAL_REPORT_NAME,
    VERIFY_RUNS_DIR,
    VerifyRun,
    append_verification_evidence,
    append_verify_history,
    canonical_verification_receipt,
    copy_current_pytest_artifacts,
    env_for_pytest_step,
    git_head,
    git_worktree_content_sha256,
    prune_successful_verify_runs,
    pytest_command_worker_request,
    reconcile_and_record_verify_runs,
    verify_history_path,
)
from devtools.verify_test_collection import collect_selection
from devtools.worker_memory import CHARGE_PROFILE_ENV, CORPUS_MAX_WORKERS
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
PYTEST_JUNIT_REPORT_DIR = PYTEST_REPORT_DIR / "junit"
_AGENTCTL_OPERATION_ARGV = {"verify_affected": (), "verify_quick": ("--quick",), "verify_all": ("--all",)}
_PROJECT_DESCRIPTOR = ".agentctl/project.toml"
#: Path classes no test exercises: orchestration metadata, documentation and
#: hosted workflow definitions. A change set inside them selects no pytest
#: step; the static gates still run.
_NO_TEST_PATH_PREFIXES = (".agentctl/", ".github/")
_NO_TEST_PATH_SUFFIXES = (".md",)
#: Where the ``.md`` suffix stops meaning "documentation". Markdown under
#: ``tests/`` is fixture content a test reads and asserts --
#: ``tests/data/golden/chatgpt-simple.md`` is compared byte-for-byte by
#: ``tests/unit/ui/test_ui_visual.py::TestGoldenMarkdownRendering::test_chatgpt_simple_session``
#: -- so exempting it by suffix let a change set consisting only of that
#: fixture report "no test exercises them" and run no pytest at all.
_TEST_TREE_PREFIX = "tests/"
#: Selections that do not consult the testmon graph.
_GRAPH_FREE_SELECTIONS = frozenset({"descriptor", "none"})
# These tests read the AgentCTL descriptor directly. They are the bounded
# contract for a descriptor-only change; Python changes still use Testmon.
DESCRIPTOR_CONTRACT_TESTS = (
    "tests/unit/devtools/test_deployment_browser_smoke_service.py::test_declared_browser_smoke_has_no_private_browser_service_lease",
    "tests/unit/devtools/test_deployment_browser_smoke_service.py::test_declared_live_provider_proof_declares_no_port_lease",
    "tests/unit/devtools/test_deployment_browser_smoke_service.py::test_descriptor_declares_the_unleased_shared_chrome_operation_and_workspace_contract",
    "tests/unit/devtools/test_dev_loop_service.py::test_declared_operation_has_a_json_contract_and_no_retired_keys",
    "tests/unit/devtools/test_seeded_archive_cache_gc.py::test_declared_agentctl_operation_is_bounded_and_previewable",
    "tests/unit/devtools/test_agent_env.py::test_every_declared_pytest_pool_operation_classifies_its_own_worker",
    "tests/unit/devtools/test_verify.py::test_verify_quick_descriptor_accepts_the_declared_json_projection",
)
#: Tests that read a tracked document through ``git grep`` rather than a traced
#: Python import, so testmon never selects them for a change to that document.
#: A retired name reintroduced in AGENTS.md would otherwise pass the verifier.
CONTRACT_DOCUMENT_TESTS = (
    "tests/unit/architecture/test_retired_analysis_modules.py::test_no_tracked_reference_to_a_retired_analysis_name",
)
#: Documentation that a contract test reads. A change touching one of these
#: runs ``CONTRACT_DOCUMENT_TESTS`` whatever else the change selects.
_CONTRACT_READ_DOCUMENTS = frozenset({"AGENTS.md"})
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
    operation = runtime_env("AGENTCTL_OPERATION")
    return operation if _AGENTCTL_OPERATION_ARGV.get(operation or "") == tuple(raw_argv) else None


def _anchor_verification_paths() -> None:
    try:
        Path.cwd().resolve().relative_to(ROOT.resolve())
    except ValueError:
        return
    os.chdir(ROOT)


def _pytest_worker_args(*, maximum: int | None = None) -> list[str]:
    """xdist arguments for the corpus run.

    ``POLYLOGUE_PYTEST_WORKERS`` is an explicit override, ``0`` included (one
    process, no xdist). Unset means the corpus width, so a bare ``devtools
    verify`` and the CI runner (which exports nothing) run at the width the
    corpus was sized for rather than on a single worker. ``maximum`` is the
    ceiling the run must fit whatever was asked for: a request wider than the
    pool it runs in is reduced here rather than throttled there.
    """
    configured = os.environ.get("POLYLOGUE_PYTEST_WORKERS")
    if configured is None or not configured.strip() or cloud_sentinel_declined("POLYLOGUE_PYTEST_WORKERS", configured):
        workers = CORPUS_MAX_WORKERS
    else:
        try:
            workers = max(0, int(configured))
        except ValueError:
            workers = CORPUS_MAX_WORKERS
    if maximum is not None:
        workers = min(workers, maximum)
    return ["--dist=loadgroup", "-n", str(workers)]


def _pytest_steps(
    *,
    selection: str,
    worker_args: Sequence[str],
    hypothesis_profile: str | None = None,
    contract_documents_changed: bool = False,
) -> list[tuple[str, list[str]]]:
    """Build one complete collection, or an affected collection, both tracing.

    Both tiers load testmon so every managed corpus or affected run advances the
    one datafile. ``all`` deselects nothing -- it executes the whole collection
    and records what it traced, which is what makes the next affected run
    selectable. Only ``descriptor`` opts out: it collects a contract slice, not
    a corpus, so its fingerprints would describe a collection no later run has.

    An affected run whose change touches a contract document adds a second,
    untraced step for ``CONTRACT_DOCUMENT_TESTS``: testmon cannot select them,
    and a positional test id under ``--testmon-forceselect`` would only
    intersect the affected selection.
    """
    steps = [
        (
            f"pytest ({selection})",
            _pytest_command(
                selection=selection,
                worker_args=worker_args,
                hypothesis_profile=hypothesis_profile,
                explicit_tests=(
                    (*DESCRIPTOR_CONTRACT_TESTS, *CONTRACT_DOCUMENT_TESTS) if selection == "descriptor" else ()
                ),
            ),
        )
    ]
    if selection == "affected" and contract_documents_changed:
        steps.append(
            (
                "pytest (contract documents)",
                _pytest_command(
                    selection="descriptor",
                    worker_args=worker_args,
                    hypothesis_profile=hypothesis_profile,
                    explicit_tests=CONTRACT_DOCUMENT_TESTS,
                ),
            )
        )
    return steps


def _pytest_command(
    *,
    selection: str,
    worker_args: Sequence[str],
    hypothesis_profile: str | None,
    explicit_tests: Sequence[str],
) -> list[str]:
    testmon = selection != "descriptor"
    select_flag = "--testmon-noselect" if selection == "all" else "--testmon-forceselect"
    collection_args = CLOSED_WORLD_COLLECTION_ARGS[:-1] if selection == "descriptor" else CLOSED_WORLD_COLLECTION_ARGS
    command = [
        venv_python(root=ROOT),
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        *IGNORED_COLLECTION_ARGS,
        "--durations=10",
        f"--junitxml={PYTEST_JUNIT_REPORT_DIR}/verify-latest.xml",
        report_file_argument(PYTEST_REPORT_PATH),
        *DEVTOOLS_PLUGIN_ARGS,
        "-p",
        SUITE_COST_PLUGIN_NAME,
        *managed_plugin_args(testmon=testmon),
        *collection_args,
        *(
            ["--testmon", f"--testmon-env={testmon_environment(ROOT, hypothesis_profile)}", select_flag]
            if testmon
            else []
        ),
        "-p",
        "no:randomly",
        *([f"--hypothesis-profile={hypothesis_profile}"] if hypothesis_profile else []),
        *worker_args,
        *explicit_tests,
        # Never under pytest-cov: testmon owns the tracer, and refuses to share
        # it with branch coverage.
    ]
    return command


#: Labels whose verdict is recorded but does not decide the verifier's exit.
#: Static gates are independent processes, so they run side by side; the
#: quick tier then costs its slowest gate rather than the sum of all of them.
GATE_PARALLELISM = max(1, min(8, os.cpu_count() or 1))


def build_verify_steps(
    *,
    quick: bool,
    selection: str = "all",
    hypothesis_profile: str | None = None,
    changed_paths: frozenset[str] | None = None,
) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [(gate.label, gate.command(root=ROOT)) for gate in quick_gates()]
    if not quick and selection != "none":
        PYTEST_JUNIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        steps += _pytest_steps(
            selection=selection,
            worker_args=_pytest_worker_args(
                maximum=AFFECTED_MAX_WORKERS if selection == "affected" else CORPUS_MAX_WORKERS
            ),
            hypothesis_profile=hypothesis_profile,
            contract_documents_changed=_contract_documents_changed(changed_paths),
        )
    return steps


def _git_changed_paths(root: Path) -> frozenset[str] | None:
    """Return committed and working-tree paths, or ``None`` if Git is unavailable."""
    try:
        base = None
        for candidate in ("origin/master", "master", "HEAD^"):
            resolved = subprocess.run(
                ["git", "rev-parse", "--verify", candidate],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if resolved.returncode == 0 and resolved.stdout.strip():
                base = resolved.stdout.strip()
                break
        if base is None:
            return None
        paths: set[str] = set()
        for command in (
            ["git", "diff", "--name-only", "--no-renames", "-z", "--no-ext-diff", f"{base}...HEAD", "--"],
            ["git", "diff", "--name-only", "--no-renames", "-z", "--no-ext-diff", "HEAD", "--"],
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        ):
            result = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode != 0:
                return None
            paths.update(line for line in result.stdout.split("\0") if line)
        return frozenset(paths)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _no_test_path(path: str) -> bool:
    """Whether no test exercises ``path``.

    The suffix exemption is about documentation, so it stops at the test tree:
    a file under ``tests/`` is fixture content by construction, whatever its
    extension.
    """
    if path.startswith(_TEST_TREE_PREFIX):
        return False
    return path.startswith(_NO_TEST_PATH_PREFIXES) or path.endswith(_NO_TEST_PATH_SUFFIXES)


def _selection_for_changes(changed_paths: frozenset[str] | None) -> str:
    """The pytest selection a change set earns.

    ``affected``: the testmon graph selects. ``descriptor``: the change stays
    inside orchestration metadata and includes the AgentCTL descriptor, so the
    explicit descriptor contract tests are the whole selection. ``none``: the
    change stays inside orchestration metadata, documentation and hosted
    workflow definitions, which no test exercises. An unknown or empty change
    set is ``affected``.
    """
    if not changed_paths or not all(_no_test_path(path) for path in changed_paths):
        return "affected"
    if _PROJECT_DESCRIPTOR in changed_paths or changed_paths & _CONTRACT_READ_DOCUMENTS:
        return "descriptor"
    return "none"


def _contract_documents_changed(changed_paths: frozenset[str] | None) -> bool:
    return bool(changed_paths and changed_paths & _CONTRACT_READ_DOCUMENTS)


def _forced_tests(selection: str, changed_paths: frozenset[str] | None) -> tuple[str, ...]:
    """Tests an affected run adds beyond the testmon selection."""
    return CONTRACT_DOCUMENT_TESTS if selection == "affected" and _contract_documents_changed(changed_paths) else ()


def _selection_reason(selection: str, changed_paths: frozenset[str] | None = None) -> str | None:
    if selection == "none":
        return (
            "every changed path is orchestration metadata, documentation or a hosted workflow "
            f"({', '.join(f'{prefix}**' for prefix in _NO_TEST_PATH_PREFIXES)}, "
            f"{', '.join(f'*{suffix}' for suffix in _NO_TEST_PATH_SUFFIXES)} "
            f"outside {_TEST_TREE_PREFIX}**); no test exercises them"
        )
    if selection == "descriptor":
        if changed_paths is not None and _PROJECT_DESCRIPTOR not in changed_paths:
            documents = ", ".join(sorted(changed_paths & _CONTRACT_READ_DOCUMENTS))
            return f"the change stays inside documentation and includes a contract document a test reads ({documents})"
        return "the change stays inside orchestration metadata and includes the AgentCTL descriptor"
    return None


def _estimate_affected_selection(
    root: Path, graph: Any, forced_tests: Sequence[str] = (), *, hypothesis_profile: str | None = None
) -> tuple[int | None, float | None, str | None, int | None]:
    """Price final selected nodes against the same compatible graph snapshot.

    Each physical launch is counted separately. Within a launch, normalized
    node IDs are distinct, including parametrizations. Unknown nodes carry no
    invented duration; recorded durations remain a measured floor.
    """
    if getattr(graph, "status", None) is not TestmonGraphStatus.USABLE:
        return None, None, None, None
    if getattr(graph, "full_rerun_cause", None):
        return None, None, None, None
    source = testmon_datafile(root)
    if not source.is_file():
        return None, None, "the testmon graph disappeared before admission", None
    try:
        from testmon import db as testmon_db
        from testmon.testmon_core import TestmonData

        with tempfile.TemporaryDirectory(prefix="polylogue-affected-admission-") as temporary:
            destination = Path(temporary) / "testmondata"
            if not snapshot_testmon_graph(source, destination):
                return None, None, "the testmon graph could not be snapshotted for admission", None
            snapshot_state = inspect_testmon_graph(root, datafile=destination, profile=hypothesis_profile)
            if not snapshot_state.usable or snapshot_state.full_rerun_cause:
                return None, None, "the admission snapshot is unreadable or incompatible with this environment", None
            database = testmon_db.DB(str(destination), readonly=False)
            try:
                data = TestmonData.for_local_run(
                    rootdir=str(root), database=database, environment=testmon_environment(root, hypothesis_profile)
                )
                if data.system_packages_change:
                    return None, None, "the testmon environment changed; affected selection is unbounded", None
                recorded = {report_nodeid_to_selector(name): value for name, value in data.all_tests.items()}
                recorded_files = {name.split("::", 1)[0] for name in recorded}
                missing_files = declared_test_files(root) - recorded_files
                if len(missing_files) > AFFECTED_MAX_UNRECORDED_FILES:
                    return (
                        None,
                        None,
                        (
                            f"the testmon graph records no execution for {len(missing_files)} test files, "
                            f"more than the {AFFECTED_MAX_UNRECORDED_FILES} this estimate will price"
                        ),
                        None,
                    )
            finally:
                database.con.close()
            environment = dict(os.environ)
            _normalize_managed_pytest_environment(environment, ())
            if hypothesis_profile is not None:
                environment["HYPOTHESIS_PROFILE"] = hypothesis_profile
            launches: list[tuple[Sequence[str] | None, Path | None]] = [(None, destination)]
            if forced_tests:
                launches.append((tuple(dict.fromkeys(forced_tests)), None))
            launched: list[str] = []
            for paths, snapshot in launches:
                selection = collect_selection(
                    root=root,
                    paths=paths,
                    datafile=snapshot,
                    nodeid_limit=AFFECTED_MAX_SELECTED_TESTS + 1,
                    environment=environment,
                )
                if selection is None:
                    return None, None, "the actual selected test nodes could not be collected", None
                if selection.omitted:
                    return None, None, "the selected-node evidence exceeds the affected selection boundary", None
                launched.extend(sorted({report_nodeid_to_selector(name) for name in selection.nodeids}))
            unknown = sum(name not in recorded for name in launched)
            durations = [recorded[name].get("duration") for name in launched if name in recorded]
            estimated = None if any(value is None for value in durations) else sum(float(value) for value in durations)
            return len(launched), estimated, None, unknown
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
        return None, None, "the affected selection could not be measured from the graph", None


def _affected_admission(
    *, root: Path, graph: Any, forced_tests: Sequence[str] = (), hypothesis_profile: str | None = None
) -> AffectedAdmission:
    """Build the bounded affected admission decision and its measurement note."""
    selected_count, estimated_seconds, measurement_error, unrecorded_tests = _estimate_affected_selection(
        root, graph, forced_tests, hypothesis_profile=hypothesis_profile
    )
    decision = admit_affected_selection(
        graph_status=str(getattr(graph, "status", "unknown")),
        graph_reason=str(getattr(graph, "reason", "graph state unavailable")),
        full_rerun_cause=getattr(graph, "full_rerun_cause", None),
        selected_count=selected_count,
        estimated_seconds=estimated_seconds,
        unrecorded_tests=unrecorded_tests,
    )
    if measurement_error and decision.admitted:
        # This is defensive: the current estimator returns an unknown count
        # for every measurement failure, but keeping the check here prevents a
        # future estimator from accidentally admitting an unmeasured plan.
        decision = admit_affected_selection(
            graph_status="unknown",
            graph_reason=measurement_error,
            full_rerun_cause=None,
            selected_count=None,
            estimated_seconds=None,
            unrecorded_tests=unrecorded_tests,
        )
    if measurement_error and decision.status == "unknown":
        decision = AffectedAdmission(
            status=decision.status,
            selected_count=decision.selected_count,
            estimated_seconds=decision.estimated_seconds,
            reason=f"{decision.reason} ({measurement_error})",
            next_boundary=decision.next_boundary,
            max_selected_tests=decision.max_selected_tests,
            max_estimated_seconds=decision.max_estimated_seconds,
            max_workers=decision.max_workers,
            unrecorded_tests=decision.unrecorded_tests,
        )
    return decision


def _normalize_managed_pytest_environment(env: dict[str, str], command: Sequence[str] = ()) -> None:
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    # Broad verification is sized by the corpus charge model: an ambient
    # focused marker (``devtools test``'s) would admit corpus workers at the
    # focused per-worker budget and ceiling.
    env.pop(CHARGE_PROFILE_ENV, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Broad verification defaults to the complete profile. An explicit
    # environment value remains an intentional local policy and the CLI option
    # takes precedence inside pytest itself.
    env.setdefault("HYPOTHESIS_PROFILE", "default")
    # Broad verification warms every shared archive on the controller once,
    # instead of charging each worker's first consumer for a cold build.
    # This opt-in is the BROAD verifier's alone: `devtools.run_tests` defines
    # a function of the same name that deliberately does not set it, because a
    # focused selection would pay the warm-up for archives it never opens.
    if all(nodeid in command for nodeid in DESCRIPTOR_CONTRACT_TESTS):
        env.pop("POLYLOGUE_BROAD_PREWARM", None)
    else:
        env["POLYLOGUE_BROAD_PREWARM"] = "1"
    env["COVERAGE_CORE"] = TESTMON_COVERAGE_CORE
    env.pop("POLYLOGUE_CI", None)


def _clear_pytest_report(command: Sequence[str]) -> None:
    paths = [
        PYTEST_PROGRESS_PATH,
        PYTEST_EVENTS_PATH,
        PYTEST_EVENTS_DIR,
        PYTEST_SELECTION_PATH,
        PYTEST_SUMMARY_PATH,
    ]
    for report in (_pytest_report_path(command),):
        paths += [report, *spool_paths(report)]
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
    prefix = f"{REPORT_FILE_OPTION}="
    return next(
        (Path(argument.split("=", 1)[1]) for argument in command if argument.startswith(prefix)),
        PYTEST_REPORT_PATH,
    )


def _copy_pytest_report(command: Sequence[str], artifacts: Any) -> dict[str, Any]:
    source = _pytest_report_path(command)
    report = _read_json(source)
    metadata: dict[str, Any] = {}
    if report is not None:
        destination = artifacts.step_dir / PYTEST_CANONICAL_REPORT_NAME
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        metadata["report_path"] = str(destination.relative_to(ROOT))
    return metadata


def _bind_pytest_reports_to_step(command: Sequence[str], artifacts: Any) -> list[str]:
    """Rebind this run's report and junit output into its own step directory.

    The checkout-global report path is shared by every concurrent caller in one
    checkout: preparing a second run clears the first run's open spool, and the
    first run then assembles an incomplete report for tests that passed. Bind
    them per run; the global path stays as a last-writer-wins projection
    written after completion.
    """
    prefix = f"{REPORT_FILE_OPTION}="
    step_report = artifacts.step_dir / PYTEST_CANONICAL_REPORT_NAME
    step_junit = artifacts.step_dir / "pytest-junit.xml"
    rebound: list[str] = []
    for argument in command:
        if argument.startswith(prefix):
            rebound.append(report_file_argument(step_report))
        elif argument.startswith("--junitxml="):
            rebound.append(f"--junitxml={step_junit}")
        else:
            rebound.append(argument)
    return rebound


def _project_latest_pytest_report(command: Sequence[str]) -> None:
    """Publish the completed run's report at the shared interactive path."""
    source = _pytest_report_path(command)
    destination = ROOT / PYTEST_REPORT_PATH
    if source.resolve() == destination.resolve() or not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _subprocess_env() -> dict[str, str]:
    return {**os.environ, "POLYLOGUE_ROOT": str(ROOT), "PYTHONPYCACHEPREFIX": str(ROOT / ".cache" / "pycache")}


#: Serializes the output lines of gates running side by side.
_STEP_LOCK = threading.RLock()
#: Gate processes still running, so an interrupted run can stop them.
_LIVE_GATE_PROCESSES: set[subprocess.Popen[str]] = set()
#: Set while an interruption stops the gates, so a gate whose process it
#: terminated is left for the interrupted-run bookkeeping instead of being
#: recorded as an ordinary failure.
_GATES_INTERRUPTED = threading.Event()


class _GateInterruptedError(Exception):
    """A gate process ended because the run was interrupted."""


def _write_step_line(text: str, *, end: str = "\n") -> None:
    with _STEP_LOCK:
        sys.stderr.write(text + end)
        sys.stderr.flush()


def _write_step_result(label: str, pytest_step: bool, verdict: str, detail: str = "") -> None:
    """Write one step's verdict, and any failure output, as one uninterrupted block.

    A gate's line is written only when it finishes, so parallel gates never
    interleave a verdict with another gate's name.
    """
    prefix = "" if pytest_step else f"  {label} ... "
    _write_step_line(prefix + verdict + "\n" + detail, end="")


def _run_gate_process(command: list[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one gate to completion, registered so an interruption can stop it.

    Each gate leads its own process group: a gate that runs its checker as a
    child (``devtools.mypy_gate`` runs ``mypy``) is stopped with that child,
    which would otherwise hold the output pipes open after its parent died.
    """
    # Checking the interruption and registering the process are one step, so a
    # gate either starts before the interruption's snapshot of live processes,
    # and is stopped with them, or sees the interruption and never starts.
    with _STEP_LOCK:
        if _GATES_INTERRUPTED.is_set():
            raise _GateInterruptedError(command[0])
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _LIVE_GATE_PROCESSES.add(process)
    try:
        stdout, stderr = process.communicate()
    finally:
        with _STEP_LOCK:
            _LIVE_GATE_PROCESSES.discard(process)
    if _GATES_INTERRUPTED.is_set():
        raise _GateInterruptedError(command[0])
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _stop_gate_processes() -> None:
    with _STEP_LOCK:
        live = tuple(_LIVE_GATE_PROCESSES)
    for process in live:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGTERM)
    # One grace period for all of them, not one per gate.
    deadline = time.monotonic() + 10
    for process in live:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
    # The leader exiting does not mean its group did: a child that delays or
    # ignores SIGTERM still holds the gate's output pipes. Whatever of each
    # group survived the grace period is killed.
    for process in live:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)


@contextlib.contextmanager
def _signals_deferred() -> Iterator[None]:
    """Ignore SIGINT and SIGTERM for the duration; restore the handlers after."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {signum: signal.signal(signum, signal.SIG_IGN) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _run_steps(
    steps: Sequence[tuple[str, list[str]]], *, run: VerifyRun, runner: str
) -> list[tuple[str, tuple[int, float, dict[str, Any]]]]:
    """Run the static gates side by side, then any pytest step alone.

    Outcomes come back in declared order whatever order the gates finish in.
    An interruption stops the running gate processes and joins their workers
    before it propagates, so nothing records or prints a gate after the run's
    interrupted verdict; the stopped gates stay ``running`` for that verdict.
    """
    _GATES_INTERRUPTED.clear()
    gates = [(label, command) for label, command in steps if not label.startswith("pytest")]
    tests = [(label, command) for label, command in steps if label.startswith("pytest")]
    outcomes: list[tuple[str, tuple[int, float, dict[str, Any]]]] = []
    if gates:
        pool = ThreadPoolExecutor(max_workers=min(GATE_PARALLELISM, len(gates)), thread_name_prefix="gate")
        try:
            futures = [pool.submit(_run, label, command, run=run, runner=runner) for label, command in gates]
            done, _pending = wait(futures, return_when=FIRST_EXCEPTION)
            for future in done:
                future.result()
            outcomes.extend((label, future.result()) for (label, _command), future in zip(gates, futures, strict=True))
        except BaseException:
            # A second SIGINT/SIGTERM during cleanup must not abandon it: the
            # verdict would be written while gate processes still run.
            with _signals_deferred():
                with _STEP_LOCK:
                    _GATES_INTERRUPTED.set()
                pool.shutdown(wait=False, cancel_futures=True)
                _stop_gate_processes()
                pool.shutdown(wait=True)
            raise
        pool.shutdown(wait=True)
    for label, command in tests:
        outcomes.append((label, _run(label, command, run=run, runner=runner)))
    return outcomes


def _run(
    label: str, command: list[str], *, run: VerifyRun, runner: str = "managed"
) -> tuple[int, float, dict[str, Any]]:
    started = time.monotonic()
    pytest_step = label.startswith("pytest")
    if pytest_step:
        # A pytest step streams its own output; name it before that begins.
        _write_step_line(f"  {label} ... ", end="")
    artifacts = run.start_step(label=label, cmd=command)
    env = _subprocess_env()
    hypothesis_profile: str | None = None
    hypothesis_profile_source: str | None = None
    completed: subprocess.CompletedProcess[Any]
    termination: dict[str, Any] = {}
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
        _write_step_result(
            label,
            pytest_step,
            f"FAILED ({executable_result.diagnosis})",
            "".join(f"    {detail}\n" for detail in executable_result.details),
        )
        return 127, time.monotonic() - started, early_metadata
    slot = None
    metadata_receipt = None
    if pytest_step:
        command = _bind_pytest_reports_to_step(command, artifacts)
        _clear_pytest_report(command)
        _normalize_managed_pytest_environment(env, command)
        if "--testmon-noselect" in command:
            env["POLYLOGUE_TESTMON_COMPLETE"] = "1"
        env = env_for_pytest_step(env, run=run, artifacts=artifacts)
        # The pytest slot re-checks the branch and records what it executed
        # when the run starts, as it does for focused runs: the checkout can
        # switch branch while the run waits for the slot.
        env[WORKTREE_PROVENANCE_ENV] = "1"
        hypothesis_profile, hypothesis_profile_source = effective_hypothesis_profile(command, env, default="default")
        try:
            executor = run_pytest if runner == "managed" else run_pytest_isolated
            outcome = executor(command, cwd=str(ROOT), env=env, root=ROOT, stdout=sys.stderr)
        except PytestSlotUnavailableError as exc:
            early_metadata = {"diagnosis": "pytest_slot_unavailable", "error": str(exc)}
            runtime_evidence = getattr(exc, "runtime_evidence", None)
            if runtime_evidence is not None:
                early_metadata["pytest_slot_terminal"] = runtime_evidence
            run.finish_step(
                step_id=artifacts.step_id,
                result={**_early_gate_failure_result(started, early_metadata), "exit": 125},
            )
            _write_step_result(label, pytest_step, f"FAILED ({exc})")
            return 125, time.monotonic() - started, early_metadata
        slot = outcome.slot
        termination = termination_metadata(outcome)
        completed = subprocess.CompletedProcess(command, outcome.returncode)
        metadata_receipt = outcome.receipt
    else:
        try:
            completed = _run_gate_process(command, env=env)
        except OSError as exc:
            early_metadata = {"diagnosis": "gate_subprocess_launch_failed", "error": str(exc)}
            run.finish_step(
                step_id=artifacts.step_id,
                result=_early_gate_failure_result(started, early_metadata),
            )
            _write_step_result(label, pytest_step, "FAILED (subprocess launch)")
            return 127, time.monotonic() - started, early_metadata
    elapsed = time.monotonic() - started
    metadata: dict[str, Any] = {
        "diagnosis": "pytest_failed" if pytest_step else "gate_passed" if completed.returncode == 0 else "gate_failed"
    }
    if pytest_step:
        metadata["pytest_slot"] = slot
        metadata.update(termination)
        metadata["hypothesis_profile"] = hypothesis_profile
        metadata["hypothesis_profile_source"] = hypothesis_profile_source
        metadata["runner"] = runner
        if metadata_receipt is not None:
            metadata["pytest_slot_receipt"] = metadata_receipt
        suite_cost_receipt = write_run_receipt(env.get(SUITE_COST_DIR_ENV))
        if suite_cost_receipt is not None:
            metadata["suite_cost_receipt"] = str(suite_cost_receipt)
        metadata.update(_copy_pytest_report(command, artifacts))
        _project_latest_pytest_report(command)
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
        if label == "gate generated-surfaces" and completed.returncode != 0:
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
    detail = ""
    if not pytest_step and effective_exit and isinstance(completed.stdout, str):
        detail += completed.stdout
    if not pytest_step and completed.returncode and isinstance(completed.stderr, str):
        detail += completed.stderr
    _write_step_result(label, pytest_step, f"{'ok' if effective_exit == 0 else 'FAILED'} ({elapsed:.1f}s)", detail)
    return effective_exit, elapsed, metadata


def _early_gate_failure_result(started: float, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Finalize an early required-gate failure with its authoritative exit."""
    return {**metadata, "duration_s": round(time.monotonic() - started, 2), "exit": 127}


def _scope(*, quick: bool, selection: str) -> VerificationScope:
    if quick or selection == "none":
        return VerificationScope.NON_TEST
    return VerificationScope.AFFECTED if selection in {"affected", "descriptor"} else VerificationScope.COMPLETE


def _emit(payload: Mapping[str, Any], *, use_json: bool, operation: str | None) -> None:
    result = declared_verification_result(payload, operation=operation) if operation else dict(payload)
    if operation and payload.get("run_id"):
        # The operation result carries the same bounded receipt as the
        # evidence lane.  AgentCTL lifecycle fields remain outside this
        # projection and cannot turn process completion into semantic success.
        result["semantic_receipt"] = canonical_verification_receipt(payload)
        # The workload receipt persisted in run.json, not a second derivation:
        # stdout and the run record cite the same receipt id.
        if isinstance(payload.get("workload_receipt"), Mapping):
            result["workload_receipt"] = dict(payload["workload_receipt"])
    if use_json or operation:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        sys.stdout.flush()
    _write_verdict_line(payload, stream=sys.stderr)


def _write_verdict_line(payload: Mapping[str, Any], *, stream: Any) -> None:
    """State this run's verdict last, in the stream, on every terminal path.

    A pipeline exits with its last command's status, so ``devtools verify |
    tail`` reports tail's 0 whatever the gates found, and the per-gate ``ok``
    lines above do not close the gap: a run whose third gate failed still ends
    its output with the last gate's ``ok``, which reads as green. The verdict
    has to be the last thing said for a truncating reader to see it.

    ``devtools test`` already states its verdict this way (``run_tests.main``);
    this is the same sentence from the other runner, so one reader habit covers
    both. The receipt is this run's own ``run.json``, never a ``current-*``
    name a concurrent run in the same checkout would overwrite.
    """
    exit_code = int(payload.get("exit_code") or 0)
    verdict = "PASSED" if exit_code == 0 else "FAILED"
    # A verification that found nothing wrong carries no diagnosis, so the
    # clause is omitted rather than filled with ``unknown`` -- which would
    # claim the run failed to determine something it never had to.
    diagnosis = payload.get("diagnosis")
    named = f" diagnosis={diagnosis}" if diagnosis else ""
    artifact_dir = payload.get("artifact_dir")
    receipt_path = (ROOT / str(artifact_dir) / "run.json").resolve() if artifact_dir else None
    receipt = f" receipt={receipt_path}" if receipt_path else ""
    # The checkout this run tested, so the line that is cited says what it proves.
    head = payload.get("git_head")
    branch = payload.get("git_branch")
    tested = (
        f" checkout={ROOT.resolve()} branch={branch or '(detached)'} head={str(head)[:12] if head else 'unknown'}"
        if head or branch
        else ""
    )
    stream.write(f"\nverify: {verdict} exit={exit_code}{named}{receipt}{tested}\n")


def _emit_affected_admission_refusal(*, graph: Any, decision: AffectedAdmission, stream: Any | None = None) -> None:
    """Explain a bounded affected refusal without implying that tests ran."""
    if stream is None:
        stream = sys.stderr
    selected = "unknown" if decision.selected_count is None else str(decision.selected_count)
    stream.write(
        "verify: affected verification refused before pytest launch.\n"
        f"  selected: {selected} test(s)\n"
        f"  graph: {getattr(graph, 'status', 'unknown')} ({getattr(graph, 'reason', 'unavailable')})\n"
        f"  reason: {decision.reason}\n"
        f"  next boundary: {decision.next_boundary}\n"
    )


def _verification_build_id(*, head: str | None, dirty: bool, content_sha256: str | None) -> str | None:
    """Name the tree a verification runs against, never a HEAD it did not test.

    A clean checkout is its commit. A dirty one is only its Git-visible
    content, so the identity is the content digest the run already records;
    without one the tree is unidentified rather than borrowed from HEAD.
    """
    if dirty:
        return f"worktree-sha256:{content_sha256}" if content_sha256 else None
    return f"git:{head}" if head else None


def _planned_concurrency(steps: Sequence[tuple[str, Sequence[str]]]) -> int:
    """Widest process fan-out the plan admits: the gate pool, then each pytest step's xdist width."""
    gates = sum(not label.startswith("pytest") for label, _command in steps)
    widths = [min(GATE_PARALLELISM, gates)] if gates else []
    for label, command in steps:
        if label.startswith("pytest"):
            requested = pytest_command_worker_request(command)
            # ``-n 0`` (or no ``-n``) is one in-process pytest.
            widths.append(max(1, int(requested)) if requested and requested.isdigit() else 1)
    return max(widths, default=1)


def _verification_workload_spec(
    *, tier: str, steps: Sequence[tuple[str, Sequence[str]]], build_id: str | None
) -> WorkloadEnvelopeSpec:
    """Declare the complete intended plan, before admission can withhold any of it."""
    return WorkloadEnvelopeSpec(
        workload_id=f"devtools:verify:{tier}",
        family_id="verification",
        version=1,
        inputs=(WorkloadInputRef(input_id=build_id or "checkout:unidentified"),),
        phases=tuple(label for label, _command in steps),
        measurement_scope=MeasurementScope.PROCESS_TREE,
        concurrency=_planned_concurrency(steps),
    )


def _verification_workload_receipt(
    *,
    spec: WorkloadEnvelopeSpec,
    build_id: str | None,
    results: Sequence[Mapping[str, Any]],
    status: WorkloadRunStatus,
) -> dict[str, Any]:
    """Bind the phases actually observed to the plan declared before they ran."""
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
        status=status,
        build_id=build_id,
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


def _recorded_step_results(run: VerifyRun) -> list[dict[str, Any]]:
    """Steps the run finished with a measured duration, once each, in start order."""
    seen: set[str] = set()
    finished: list[dict[str, Any]] = []
    for step in run._payload["steps"]:
        name, duration = step.get("name"), step.get("duration_s")
        if isinstance(name, str) and name not in seen and isinstance(duration, int | float):
            seen.add(name)
            finished.append({"name": name, "duration_s": duration})
    return finished


def _finish_interrupted_verification(
    *,
    run: VerifyRun,
    started: float,
    scope: VerificationScope,
    args: argparse.Namespace,
    selection: str,
    agentctl_operation: str | None,
    exit_code: int,
    termination_reason: str,
    results: Sequence[Mapping[str, Any]],
    workload_spec: WorkloadEnvelopeSpec,
    build_id: str | None,
) -> int:
    """Persist the terminal state when an outer runtime ends verification.

    The receipt keeps the plan declared before execution; the phases it
    observes are the steps that finished before the interruption.
    """
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
            **_aggregate_pytest_results(
                results, expected_step_count=3, mode="quick" if args.quick else selection, exit_code=exit_code
            ),
            "terminal_green": False,
            "complete_corpus_covered": False,
            "termination_reason": termination_reason,
        },
        workload_receipt=_verification_workload_receipt(
            spec=workload_spec,
            build_id=build_id,
            results=_recorded_step_results(run),
            status=WorkloadRunStatus.INTERRUPTED,
        ),
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
    workload_receipt: Mapping[str, Any],
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
    history_path = verify_history_path(root=ROOT)
    append_verify_history(payload, path=history_path)
    append_verification_evidence(payload)
    prune_successful_verify_runs(root=ROOT, history_path=history_path)
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
        "selected_union_count": sum(selected_counts) if selected_counts else None,
        "terminal_union_count": sum(terminal_counts) if terminal_counts else None,
        "outcomes": outcomes,
        "terminal_green": exit_code == 0,
        "complete_corpus_covered": complete,
    }


#: Set by the devshell hook on every entry (flake.nix): ``complete`` when
#: ``.venv`` matches this checkout's pyproject.toml and uv.lock, ``incomplete``
#: when that sync failed and the previous environment was kept.
DEPENDENCY_SYNC_ENV = "POLYLOGUE_DEVSHELL_DEPENDENCY_SYNC"
DEPENDENCY_SYNC_INCOMPLETE = "incomplete"


def _main(argv: list[str] | None = None, *, agentctl_operation: str | None = None) -> int:
    arguments = list(argv or [])
    refusal = refuse_verify_tier(arguments, os.environ)
    if refusal is not None:
        # A caller that asked for JSON gets JSON, refusals included; otherwise
        # the one machine-readable contract has a prose-only hole in it.
        if "--json" in arguments:
            json.dump(
                {
                    "kind": "polylogue.verification-refusal",
                    "status": "refused",
                    "diagnosis": "agent_tier_refused",
                    "message": refusal,
                    "exit_code": 2,
                },
                sys.stdout,
            )
            sys.stdout.write("\n")
        else:
            sys.stderr.write(refusal + "\n")
        return 2
    parser = argparse.ArgumentParser(description="Run project semantic verification.")
    parser.add_argument("--quick", action="store_true", help="run the static gates only")
    parser.add_argument(
        "--all",
        dest="all_tests",
        action="store_true",
        help="the static gates plus the complete corpus; the default selects from the testmon graph instead",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--hypothesis-profile", help="profile passed to pytest; overrides HYPOTHESIS_PROFILE")
    parser.add_argument("--runner", choices=("managed", "isolated"), default="managed")
    parser.add_argument(
        ON_DEFAULT_BRANCH_FLAG,
        dest="on_default_branch",
        action="store_true",
        help="run on the default branch deliberately (a base comparison, or the hosted gate on a push)",
    )
    args = parser.parse_args(argv)
    if sys.flags.optimize > 0:
        message = (
            "devtools verify refuses optimized Python; run with the standard interpreter (sys.flags.optimize must be 0)"
        )
        if args.json:
            print(
                json.dumps({"status": "refused", "diagnosis": "optimized_python", "message": message, "exit_code": 125})
            )
        else:
            sys.stderr.write(message + "\n")
        return 125
    if os.environ.get(DEPENDENCY_SYNC_ENV) == DEPENDENCY_SYNC_INCOMPLETE:
        message = (
            "devtools verify refuses an unsynced environment: the devshell's `uv sync --frozen` failed, so .venv "
            "still holds the dependencies of an earlier pyproject.toml/uv.lock; fix the sync and re-enter the shell"
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "diagnosis": "dependency_sync_incomplete",
                        "message": message,
                        "exit_code": 125,
                    }
                )
            )
        else:
            sys.stderr.write(message + "\n")
        return 125
    _anchor_verification_paths()
    identity = checkout_identity(ROOT)
    branch_refusal = default_branch_refusal(identity, command="devtools verify", allowed=args.on_default_branch)
    if branch_refusal is not None:
        if args.json:
            json.dump(
                {
                    "kind": "polylogue.verification-refusal",
                    "status": "refused",
                    "diagnosis": REFUSAL_DIAGNOSIS,
                    "message": branch_refusal,
                    "exit_code": REFUSAL_EXIT,
                },
                sys.stdout,
            )
            sys.stdout.write("\n")
        else:
            sys.stderr.write(branch_refusal + "\n")
        return REFUSAL_EXIT
    # Carried to the pytest slot, which re-checks the branch at start; only
    # this invocation's flag may authorize it, never an inherited value.
    os.environ.pop(ALLOW_DEFAULT_BRANCH_ENV, None)
    if args.on_default_branch:
        os.environ[ALLOW_DEFAULT_BRANCH_ENV] = "1"
    sys.stderr.write(f"verify: {identity.describe()}\n")
    # Every step must see one tree: its Git-visible content is compared at the end.
    started_content = git_worktree_content_sha256(ROOT)
    validate_authority_matrix()
    started = time.monotonic()
    selection = "all" if args.all_tests else "affected"
    changed_paths: frozenset[str] | None = None
    if not args.quick and not args.all_tests:
        changed_paths = _git_changed_paths(ROOT)
        selection = _selection_for_changes(changed_paths)
    scope = _scope(quick=args.quick, selection=selection)
    # The import-root contract is checked before this run writes anything under
    # the verify cache: a mismatched checkout must leave no receipt or graph.
    try:
        assert_polylogue_matches_checkout(ROOT, context="devtools verify")
    except CheckoutImportMismatchError as exc:
        payload = {
            "exit_code": 125,
            "duration_s": time.monotonic() - started,
            "diagnosis": "checkout_import_mismatch",
            "verification_scope": scope.value,
            "final_git_head": git_head(ROOT),
            "git_head": identity.head,
            "git_branch": identity.branch,
        }
        # The detail first: the verdict, naming the checkout, is the last line.
        sys.stderr.write(f"verify: {exc}\n")
        _emit(payload, use_json=args.json, operation=agentctl_operation)
        return 125
    # Before this run writes its own ``running`` receipt, give a terminal state
    # to any earlier one whose process is gone. A verification killed outright
    # runs no handler of its own, so the next reader is the only thing that can
    # close it out.
    reconcile_and_record_verify_runs(runs_root=ROOT / VERIFY_RUNS_DIR)
    seeded_from_primary = sync_testmon_graph(
        ROOT, **({"profile": args.hypothesis_profile} if args.hypothesis_profile is not None else {})
    )
    graph = inspect_testmon_graph(
        ROOT, **({"profile": args.hypothesis_profile} if args.hypothesis_profile is not None else {})
    )
    head = git_head(ROOT)
    tier = "quick" if args.quick else selection
    # The complete plan, pytest included, is fixed before admission: a refusal
    # or an interruption withholds steps from execution, never from the plan.
    planned_steps = build_verify_steps(
        quick=args.quick,
        selection=selection,
        hypothesis_profile=args.hypothesis_profile,
        changed_paths=changed_paths,
    )
    run = VerifyRun(
        tier=tier,
        argv=list(argv or []),
        git_head=head,
        root=ROOT,
        mirror_current=agentctl_operation is None,
        agentctl_operation=agentctl_operation,
    )
    build_id = _verification_build_id(head=head, dirty=run.recorded_git_dirty, content_sha256=started_content)
    workload_spec = _verification_workload_spec(tier=tier, steps=planned_steps, build_id=build_id)
    run.declare_workload(workload_spec.to_payload())
    results: list[dict[str, Any]] = []
    # Everything from receipt creation to the terminal verdict runs under the
    # interruption handlers: a signal during selection, admission or the
    # post-step accounting still finishes this run instead of leaving it
    # ``running``. The publication itself stays outside, so a signal cannot
    # finish an already-finished run a second time.
    try:
        refused: AffectedAdmission | None = None
        if not args.quick:
            admission: AffectedAdmission | None = None
            if selection == "affected":
                admission = _affected_admission(
                    root=ROOT,
                    graph=graph,
                    forced_tests=_forced_tests(selection, changed_paths),
                    hypothesis_profile=args.hypothesis_profile,
                )
            run.record_selection(
                selection_mode=selection,
                graph_status=str(graph.status),
                graph_reason=graph.reason,
                full_rerun_cause=graph.full_rerun_cause if selection not in _GRAPH_FREE_SELECTIONS else None,
                graph_recorded_tests=getattr(graph, "recorded_tests", None),
                graph_source_dependencies=getattr(graph, "source_dependencies", None),
                seed_source=str(testmon_datafile(primary_worktree())) if seeded_from_primary else None,
                seed_source_mtime_ns=(
                    testmon_datafile(primary_worktree()).stat().st_mtime_ns if seeded_from_primary else None
                ),
                selection_reason=(
                    admission.reason if admission is not None else _selection_reason(selection, changed_paths)
                ),
                selected_count=admission.selected_count if admission is not None else None,
                estimated_seconds=admission.estimated_seconds if admission is not None else None,
                admission=admission.to_payload() if admission is not None else None,
            )
            if selection == "none":
                sys.stderr.write("verify: no pytest step: " + str(_selection_reason(selection, changed_paths)) + "\n")
            if admission is not None and not admission.admitted:
                refused = admission
        # A refused admission withholds pytest, not the static gates.
        steps = (
            [(label, command) for label, command in planned_steps if not label.startswith("pytest")]
            if refused is not None
            else planned_steps
        )
        exit_code = 0
        for label, (rc, elapsed, metadata) in _run_steps(steps, run=run, runner=args.runner):
            results.append({"name": label, "duration_s": round(elapsed, 2), "exit": rc, **metadata})
            if rc:
                exit_code = exit_code or rc
        executed: set[tuple[object, object, object]] = set()
        tree_unknown = False
        for result in results:
            slot_receipt = result.get("pytest_slot_receipt")
            provenance = slot_receipt.get("worktree_provenance") if isinstance(slot_receipt, Mapping) else None
            if isinstance(provenance, Mapping):
                # The receipt and verdict name what pytest executed, not what was admitted.
                run.record_execution_worktree(provenance)
                executed.add(
                    (
                        provenance.get("git_branch"),
                        provenance.get("git_head"),
                        provenance.get("git_worktree_content_sha256"),
                    )
                )
            elif result.get("diagnosis") == OOM_KILLED_DIAGNOSIS or (
                isinstance(slot_receipt, Mapping) and slot_receipt.get("diagnosis") == "execution_source_unavailable"
            ):
                # The kill took the slot receipt, so nothing identified the tree
                # pytest ran against; the admitted head is not that evidence.
                tree_unknown = True
        if tree_unknown:
            # Recorded after every step, so a later step's provenance cannot
            # stand in for the tree the killed step ran against.
            run.record_execution_worktree({"capture_source": "unavailable"})
        # The static gates read the checkout directly, with no slot to re-check it:
        # a run whose branch, HEAD or Git-visible content changed while it ran, or
        # whose pytest step executed other content, verified no single tree.
        finished_identity = checkout_identity(ROOT)
        finished_content = git_worktree_content_sha256(ROOT)
        checkout_moved = (
            (finished_identity.branch, finished_identity.head) != (identity.branch, identity.head)
            or finished_content != started_content
            or bool(executed - {(identity.branch, identity.head, started_content)})
        )
        if checkout_moved:
            sys.stderr.write(
                f"verify: the checkout moved during the run (started {identity.describe()}, "
                f"finished {finished_identity.describe()}); the result is void\n"
            )
            exit_code = exit_code or 1
        # The retained exit code is the first failure's; its diagnosis must be too.
        diagnosis = next(
            (str(result["diagnosis"]) for result in results if result["exit"] != 0),
            None,
        )
        if checkout_moved:
            diagnosis = "checkout_moved_during_run"
        if refused is not None:
            exit_code = exit_code or 2
            diagnosis = diagnosis or "affected_admission_refused"
            aggregate: dict[str, Any] = {
                "selection_mode": "affected",
                "selected_union_count": refused.selected_count,
                "terminal_union_count": 0,
                "outcomes": {},
                "terminal_green": False,
                "complete_corpus_covered": False,
                "admission": refused.to_payload(),
            }
        else:
            aggregate = _aggregate_pytest_results(
                results,
                expected_step_count=sum(label.startswith("pytest") for label, _command in steps),
                mode="quick" if args.quick else selection,
                exit_code=exit_code,
            )
        final_head = git_head(ROOT)
    except VerificationInterrupted as exc:
        return _finish_interrupted_verification(
            run=run,
            started=started,
            scope=scope,
            args=args,
            selection=selection,
            agentctl_operation=agentctl_operation,
            exit_code=128 + exc.signum,
            termination_reason=signal.Signals(exc.signum).name.lower(),
            results=results,
            workload_spec=workload_spec,
            build_id=build_id,
        )
    except KeyboardInterrupt:
        return _finish_interrupted_verification(
            run=run,
            started=started,
            scope=scope,
            args=args,
            selection=selection,
            agentctl_operation=agentctl_operation,
            exit_code=130,
            termination_reason="operator_interrupt",
            results=results,
            workload_spec=workload_spec,
            build_id=build_id,
        )
    payload = _finish_and_record_verification(
        run=run,
        exit_code=exit_code,
        duration_s=time.monotonic() - started,
        diagnosis=diagnosis,
        verification_scope=scope.value,
        final_git_head=final_head,
        pytest_aggregate=aggregate,
        workload_receipt=_verification_workload_receipt(
            spec=workload_spec,
            # A tree that moved under the run was not the declared build.
            build_id=None if checkout_moved else build_id,
            results=results,
            status=WorkloadRunStatus.SUCCEEDED if exit_code == 0 else WorkloadRunStatus.FAILED,
        ),
    )
    if refused is not None:
        _emit_affected_admission_refusal(graph=graph, decision=refused)
    _emit(payload, use_json=args.json, operation=agentctl_operation)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "schema-manifest":
        from devtools.verify_schema_manifest import main as verify_schema_manifest

        return verify_schema_manifest(raw_argv[1:])
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
