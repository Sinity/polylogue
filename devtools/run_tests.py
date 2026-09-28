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

import functools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import tomllib

from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
)
from devtools.checkout_identity import (
    ALLOW_DEFAULT_BRANCH_ENV,
    ON_DEFAULT_BRANCH_FLAG,
    REFUSAL_EXIT,
    checkout_identity,
    default_branch_refusal,
)
from devtools.pytest_invocation import (
    CLEAR_CONFIGURED_ADDOPTS,
    IGNORED_COLLECTION_ARGS,
    SUITE_COST_PLUGIN_NAME,
    devtools_plugin_args,
    effective_hypothesis_profile,
    managed_plugin_args,
)
from devtools.pytest_options import caller_plugins, operand_count, short_options_with_value, split_short_cluster
from devtools.pytest_rerun import RERUN_IN_SLOT_ENV, rerun_failed_once, semantic_rerun_options
from devtools.pytest_slot import (
    WORKTREE_PROVENANCE_ENV,
    PytestSlotUnavailableError,
    basetemp_root,
    guard_temp_trees,
    remove_temp_tree,
    run_pytest,
    run_pytest_isolated,
    sweep_stale_temp_trees,
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
    copy_current_pytest_artifacts,
    env_for_pytest_step,
    git_head,
    git_worktree_content_sha256,
    prune_successful_verify_runs,
    pytest_command_worker_request,
)
from devtools.worker_memory import CHARGE_PROFILE_ENV, FOCUSED_MAX_WORKERS

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
    report_paths = sorted(
        path for path in (root / PYTEST_REPORT_DIR).glob("last-pytest-*.json") if path.name != PYTEST_REPORT_PATH.name
    )
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
        print("devtools test --outliers: no readable full-run pytest receipts", file=sys.stderr)
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


#: How many recent run directories a reuse lookup reads. Successful detail is
#: already pruned to a small bound, so this only caps a pathological backlog.
REUSE_LOOKUP_LIMIT = 50
#: Set to ``0`` to always run, even when an identical green run exists.
REUSE_ENV = "POLYLOGUE_TEST_REUSE"


#: Held for the rest of the process once taken; the kernel releases it on exit.
#: Keyed by selection digest: a second ``flock`` from this same process on a
#: new descriptor would wait on its own lock forever.
_SELECTION_LOCKS: dict[str, int] = {}


def _hold_selection_lock(selection: list[str]) -> None:
    """Serialize identical selections within this checkout for this process's life."""
    import fcntl
    import hashlib

    lock_dir = ROOT / ".cache" / "verify" / "inflight"
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(json.dumps(selection).encode("utf-8")).hexdigest()[:24]
        if digest in _SELECTION_LOCKS:
            return
        handle = os.open(lock_dir / f"{digest}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("devtools test: the same selection is already running in this checkout; waiting for it.\n")
        sys.stderr.flush()
        fcntl.flock(handle, fcntl.LOCK_EX)
    _SELECTION_LOCKS[digest] = handle


def _parse_rerun(selection: list[str]) -> tuple[bool, list[str]]:
    """Consume ``--rerun`` (always run) without forwarding it to pytest."""
    return "--rerun" in selection, [argument for argument in selection if argument != "--rerun"]


#: Caller environment that can change what a selection executes or how
#: (Hypothesis profiles, pytest options, Polylogue test switches). Its values
#: are part of the reuse key, so a run under a different profile never answers
#: from a weaker one's receipt.
_EXECUTION_ENV_PREFIXES = ("HYPOTHESIS_", "PYTEST_", "POLYLOGUE_")
#: Individual switches the suite reads outside those prefixes: golden-file
#: regeneration, fuzz depth, colour, time zone and the XDG roots.
_EXECUTION_ENV_NAMES = frozenset(
    {
        # Tests reach tools (git, bash, compilers) through the search path,
        # and HOME is inherited into every job (.agentctl/project.toml).
        "PATH",
        "HOME",
        "UPDATE_GOLDEN",
        "FUZZ_ITERATIONS",
        "NO_COLOR",
        "TZ",
        "TZDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    }
)
#: The Hypothesis example database ``tests/conftest.py`` declares. A run can
#: save a new counterexample there, which the next run of the same selection
#: replays; its contents are therefore an input of every property test.
_HYPOTHESIS_DATABASE = Path(".cache/hypothesis/examples")


#: The only options a reusable selection may carry: ones that change neither
#: which tests run nor what the run leaves behind. Anything else -- report
#: files, cache-dependent selection (``--lf``), cache clearing, external
#: configuration -- has an effect a receipt cannot supply, so it always runs.
# Verbosity flags are excluded: they ask for output a receipt cannot replay.
_REUSABLE_FLAGS = frozenset({"-x", "--exitfirst", "-q", "--quiet", "--no-header"})
_REUSABLE_VALUE_OPTIONS = frozenset({"-k", "-m"})
_REUSABLE_PREFIXES = ("--tb=", "--maxfail=", "-k=", "-m=")


def _reuse_eligible(selection: list[str], *, root: Path) -> bool:
    """Whether every argument is a checkout-local selection or an inert option.

    The digest a receipt is keyed on covers the checkout's Git-visible tree,
    so a path outside it (``/tmp/test_x.py``) could change without changing
    the key; it is never reused.
    """
    resolved_root = root.resolve()
    index = 0
    while index < len(selection):
        argument = selection[index]
        if argument in _REUSABLE_VALUE_OPTIONS:
            index += 2
            continue
        if argument in _REUSABLE_FLAGS or argument.startswith(_REUSABLE_PREFIXES):
            index += 1
            continue
        if argument.startswith("-"):
            return False
        target = Path(argument.split("::", 1)[0])
        target = (target if target.is_absolute() else root / target).resolve()
        # A directory can hold ignored, collectable modules the tree digest
        # omits; only named files are reusable.
        if not target.is_relative_to(resolved_root) or not target.is_file() or _git_ignored(target, root=resolved_root):
            return False
        index += 1
    return True


def _ignored_python_sources(root: Path) -> bool:
    """Whether any test input under the source trees is ignored by Git.

    The tree digest omits ignored files, and a named test still loads its
    ancestors' ``conftest.py`` and whatever it imports; an ignored one could
    change the run without changing the key. Such a checkout never reuses.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                "tests",
                "polylogue",
                "devtools",
                # Root files pytest reads as configuration or plugins.
                "conftest.py",
                "pytest.ini",
                ".pytest.ini",
                "pyproject.toml",
                "tox.ini",
                "setup.cfg",
            ],
            cwd=root,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    if result.returncode != 0:
        return True
    # Any ignored file a test may read (a module, a conftest, a JSON fixture a
    # parametrization globs) is outside the digest; bytecode caches are not inputs.
    return any(
        line and "__pycache__/" not in line and not line.endswith((".pyc", ".pyo"))
        for line in result.stdout.decode("utf-8", "replace").splitlines()
    )


def _git_ignored(path: Path, *, root: Path) -> bool:
    """Whether Git ignores ``path``, so the tree digest does not cover it."""
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)], cwd=root, capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    # 0: ignored; 1: not ignored; anything else: cannot tell, so do not reuse.
    return result.returncode != 1


def execution_environment_key(environ: Mapping[str, str]) -> str:
    """A digest of the caller's execution-affecting environment."""
    import hashlib

    relevant = sorted(
        (key, value)
        for key, value in environ.items()
        if key.startswith(_EXECUTION_ENV_PREFIXES) or key in _EXECUTION_ENV_NAMES
    )
    return hashlib.sha256(json.dumps(relevant).encode("utf-8")).hexdigest()


def _reuse_environment_key() -> str:
    """The caller environment plus the example database this run starts from."""
    return f"{execution_environment_key(os.environ)}:{hypothesis_database_revision(ROOT)}"


def hypothesis_database_revision(root: Path) -> str:
    """The example database's declared revision marker, read without a walk.

    ``tests/conftest.py`` writes through ``RevisionedExampleDatabase``, which
    replaces the marker on every save, delete or move.
    """
    from devtools.hypothesis_database import read_revision

    return read_revision(root / _HYPOTHESIS_DATABASE)


def reusable_green_receipt(
    selection: list[str], *, root: Path, content_sha256: str | None, environment_key: str | None = None
) -> Path | None:
    """A green focused run of exactly this selection over exactly this tree.

    Keyed on the declared inputs only: the normalized selection, the
    worktree content digest the run was bound to at slot start, and the
    interpreter identity. A match means rerunning would execute the same
    tests over the same bytes with the same interpreter, so its receipt
    answers the question and the pool admission is skipped.
    """
    if content_sha256 is None or not _reuse_eligible(selection, root=root) or _ignored_python_sources(root):
        return None
    runs_root = root / ".cache" / "verify" / "runs"
    try:
        # Run ids carry time to the second only, so the name bounds the
        # lookup and the recorded start orders the runs within it.
        entries = sorted(
            (entry for entry in runs_root.iterdir() if "-focused-test-" in entry.name),
            reverse=True,
        )[:REUSE_LOOKUP_LIMIT]
    except OSError:
        return None
    loaded: list[tuple[str, Path, dict[str, Any]]] = []
    for entry in entries:
        try:
            payload = json.loads((entry / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            loaded.append((_run_order(entry.name, payload), entry, payload))
    loaded.sort(key=lambda item: item[0], reverse=True)
    interpreter = (sys.executable, platform.python_version())
    for _order, entry, payload in loaded:
        receipt = entry / "run.json"
        fingerprint = payload.get("environment_fingerprint") or {}
        same_inputs = (
            payload.get("argv") == selection
            and payload.get("execution_environment_key") == environment_key
            and payload.get("git_worktree_content_sha256") == content_sha256
            and (str(Path(fingerprint.get("python_executable", "")).resolve()), fingerprint.get("python_version"))
            == (str(Path(interpreter[0]).resolve()), interpreter[1])
        )
        if not same_inputs:
            continue
        # The newest run of these exact inputs decides: an older green never
        # outranks a later red of the same selection on the same tree.
        green = (
            payload.get("status") == "success"
            and payload.get("exit_code") == 0
            and (payload.get("pytest_aggregate") or {}).get("terminal_green") is True
        )
        if not green or _later_failure_pruned(root, after=_order):
            return None
        return receipt
    return None


def _run_order(run_id: str, payload: Mapping[str, Any]) -> str:
    """Chronological sort key: run ids carry the second, ``started_at`` the rest."""
    return f"{run_id[:16]}|{payload.get('started_at') or ''}"


def _later_failure_pruned(root: Path, *, after: str) -> bool:
    """Whether a focused run later than ``after`` failed and lost its detail.

    ``after`` is the green run's :func:`_run_order` key.

    Retention keeps fewer failed details than green ones, so a later red of
    the same inputs can be pruned while the older green survives. The
    append-only history still names every run; a failed one without its
    detail directory has unknown inputs and may be that red. Unreadable
    history answers the same way: reuse is refused, never assumed. Without
    a history file nothing was pruned, since retention prunes only runs the
    history records.
    """
    runs_root = root / ".cache" / "verify" / "runs"
    try:
        with (root / ".cache" / "verify" / "history.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if "-focused-test-" not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                run_id = row.get("run_id") if isinstance(row, dict) else None
                if (
                    isinstance(run_id, str)
                    and "-focused-test-" in run_id
                    and _run_order(run_id, row) > after
                    and row.get("status") != "success"
                    and not (runs_root / run_id).exists()
                ):
                    return True
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return False


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
    """True when the caller already chose an xdist worker count.

    A short-option cluster is walked as argparse walks it: ``-vn2`` is ``-v``
    plus ``-n2``, while in ``-kn`` the ``n`` is ``-k``'s attached value.
    """
    value_short: frozenset[str] | None = None
    for argument in selection:
        if argument.startswith("--"):
            if argument.split("=", 1)[0] == "--numprocesses":
                return True
            continue
        if not argument.startswith("-") or len(argument) < 2:
            continue
        if argument.startswith("-n"):
            return True
        if value_short is None:
            value_short = short_options_with_value(caller_plugins(selection))
        for letter in argument[1:]:
            if letter == "n":
                return True
            if f"-{letter}" in value_short:
                break
    return False


#: A selection naming at least this many test modules runs under xdist.
#: Measured 2026-09-27 over 212 focused receipts: runs above 300 tests were
#: 13% of runs and 60% of pool time; their median selection named 16 modules.
#: A handful of modules runs faster in one process than xdist can start.
LARGE_SELECTION_MODULES = 8
#: From this many modules a selection is sized as corpus work: the focused
#: profile's per-worker bound was measured on selections far below it.
BROAD_SELECTION_MODULES = 100


@functools.cache
def _configured_test_module_globs() -> tuple[str, ...]:
    """The suite's ``python_files`` collection globs, from ``pyproject.toml``.

    A directory expansion must match every pattern pytest is configured to
    collect (``fuzz_*.py`` alongside ``test_*.py``), or modules that pattern
    alone would collect are missing from the module count and the large- or
    broad-selection thresholds under-fire.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("python_files")
    if not patterns:
        return ("test_*.py",)
    return tuple(patterns) if isinstance(patterns, list) else (str(patterns),)


def _selected_test_modules(selection: list[str]) -> int:
    """How many test modules the selection names, directories expanded."""
    modules: set[Path] = set()
    # An option's value is skipped exactly as pytest's parser consumes it:
    # standalone flags (``-x``, ``--strict-markers``, ...) take none, and the
    # path after them is still a selection.
    certain: list[str] = []
    index = 0
    while index < len(selection):
        certain.append(selection[index])
        index += 1 + operand_count(selection, index)
    if not any(not argument.startswith("-") for argument in certain):
        # No path operand: pytest collects its configured ``testpaths``, the
        # whole test tree, so the selection is counted as that tree.
        certain = [*certain, "tests"]
    for argument in certain:
        if argument.startswith("-"):
            continue
        target = Path(argument.split("::", 1)[0])
        target = target if target.is_absolute() else ROOT / target
        if target.is_dir():
            for glob in _configured_test_module_globs():
                modules.update(path.resolve() for path in target.rglob(glob) if path.is_file())
        elif target.is_file():
            # Node ids of one file are one module, not several.
            modules.add(target.resolve())
    return len(modules)


def _xdist_disabled(selection: list[str]) -> bool:
    """Whether the caller disabled xdist or asked for output it cannot carry.

    ``-p no:xdist`` disables it outright; ``-s``/``--capture=no`` asks for
    live output, and ``--pdb``/``--trace`` for an interactive debugger, neither
    of which xdist workers can provide.
    """
    if any(
        argument in {"-pno:xdist", "-p=no:xdist", "-s", "--capture=no", "--pdb", "--trace"}
        or (argument == "no:xdist" and index and selection[index - 1] == "-p")
        or (argument == "no" and index and selection[index - 1] == "--capture")
        for index, argument in enumerate(selection)
    ):
        return True
    # ``-sv`` is ``-s -v``: a clustered ``-s`` asks for live output too.
    value_short: frozenset[str] | None = None
    for argument in selection:
        if argument.startswith("--") or not argument.startswith("-") or len(argument) <= 2:
            continue
        if value_short is None:
            value_short = short_options_with_value(caller_plugins(selection))
        flags, _option, _value = split_short_cluster(argument, value_short)
        if "-s" in flags:
            return True
    return False


def _worker_args(selection: list[str]) -> list[str]:
    """Run a large selection under xdist; keep a small one in one process.

    An explicit ``-n`` is the caller's and is honored. An ambient worker
    setting belongs to broad verification, not an inner-loop selection. The
    slot narrows the width to its live cgroup budget under the focused charge
    profile, so a busy pool runs a large selection narrower, never over its
    ceiling.
    """
    if _has_worker_flag(selection) or _selection_targets_benchmarks(selection) or _xdist_disabled(selection):
        # Benchmarks run in one process by contract (``-p no:xdist``).
        return []
    if _selected_test_modules(selection) >= LARGE_SELECTION_MODULES:
        return ["-n", str(FOCUSED_MAX_WORKERS)]
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
        *managed_plugin_args(testmon=False, xdist=_has_worker_flag(selection) or bool(worker_args)),
        CLEAR_CONFIGURED_ADDOPTS,
        report_file_argument(report_path),
        *collection_args,
        *_before_separator(selection, [*worker_args, *_xdist_distribution_args(selection, worker_args)]),
    ]


def _before_separator(selection: list[str], options: list[str]) -> list[str]:
    """``selection`` with ``options`` placed before any ``--``.

    After ``--`` pytest reads every argument as a path, so generated options
    appended there would be looked up as files.
    """
    if "--" not in selection:
        return [*selection, *options]
    index = selection.index("--")
    return [*selection[:index], *options, *selection[index:]]


def focused_pytest_env(*, run: VerifyRun, artifacts: PytestStepArtifacts) -> dict[str, str]:
    """The environment a focused run executes under.

    Focused runs deliberately do not load testmon. They must not create a
    scratch graph, mutate the corpus graph, or load testmon's retention hook.
    """
    env = env_for_pytest_step(dict(os.environ), run=run, artifacts=artifacts, testmon=False)
    # Sized as a focused selection, not as a share of the whole corpus.
    env[CHARGE_PROFILE_ENV] = "focused"
    return env


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
    artifacts: PytestStepArtifacts,
    report_path: Path,
    runner: str = "managed",
    stdout: Any = None,
) -> tuple[int, float, dict[str, Any]]:
    """Run focused pytest through the host's pytest slot, preserving its receipt."""
    del label, run
    started = time.monotonic()
    try:
        executor = run_pytest if runner == "managed" else run_pytest_isolated
        env[WORKTREE_PROVENANCE_ENV] = "1"
        # A queued job reruns its own failures before releasing the slot, so a
        # red run is adjudicated without a second queue wait.
        env[RERUN_IN_SLOT_ENV] = json.dumps(
            {
                "report_path": str(report_path),
                "step_dir": str(artifacts.step_dir),
                "root": str(ROOT),
                "options": semantic_rerun_options(command),
            }
        )
        output_option = {"stdout": stdout} if stdout is not None else {}
        outcome = executor(command, cwd=cwd, env=env, root=ROOT, **output_option)
    except PytestSlotUnavailableError as exc:
        sys.stderr.write(f"devtools test: {exc}\n")
        runtime_evidence = getattr(exc, "runtime_evidence", None)
        return (
            125,
            time.monotonic() - started,
            {
                "diagnosis": "pytest_slot_unavailable",
                "error": str(exc),
                "termination_reason": "pytest_slot_unavailable",
                **({"pytest_slot_terminal": runtime_evidence} if runtime_evidence is not None else {}),
            },
        )
    returncode = outcome.returncode
    if outcome.slot.startswith("agentctl job") and (
        not isinstance(outcome.receipt, dict) or not isinstance(outcome.receipt.get("worktree_provenance"), dict)
    ):
        return (
            125,
            time.monotonic() - started,
            {"diagnosis": "worktree_provenance_unavailable", "pytest_slot": outcome.slot},
        )
    # Exit 1 is "tests failed", the only outcome a rerun can speak to. Exit 2
    # (interrupted), 3 (internal error), 4 (usage) and the signal codes
    # describe the run itself. This is the same adjudication `devtools verify`
    # performs, from the same module: a focused run is the one MORE likely to
    # sit beside six sibling jobs in the pool, so it needs it at least as much.
    rerun = (
        rerun_failed_once(
            report_path=report_path,
            step_dir=artifacts.step_dir,
            env=env,
            root=ROOT,
            runner=runner,
            first_provenance=(
                outcome.receipt.get("worktree_provenance") if isinstance(outcome.receipt, dict) else None
            ),
            options=semantic_rerun_options(command),
        )
        if returncode == 1
        else None
    )
    if rerun is not None and not rerun["still_failed"]:
        # Every failure passed alone: the run is green with its flakes named,
        # never green silently.
        returncode = 0
    suite_cost_receipt = write_run_receipt(env.get(SUITE_COST_DIR_ENV))
    return (
        returncode,
        time.monotonic() - started,
        {
            "diagnosis": "pytest_passed" if returncode == 0 else "pytest_failed",
            "pytest_slot": outcome.slot,
            **({"rerun": rerun} if rerun is not None else {}),
            **({"suite_cost_receipt": str(suite_cost_receipt)} if suite_cost_receipt is not None else {}),
            # Named per client pid: the checkout accumulates one log per run,
            # and a glob over them reaches an arbitrary one.
            **({"pytest_slot_log": str(outcome.log_path)} if outcome.log_path is not None else {}),
            **({"pytest_slot_receipt": outcome.receipt} if outcome.receipt is not None else {}),
            **(
                {"worktree_provenance": outcome.receipt["worktree_provenance"]}
                if isinstance(outcome.receipt, dict) and isinstance(outcome.receipt.get("worktree_provenance"), dict)
                else {}
            ),
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


def _certain_selections(selection: list[str]) -> list[str]:
    """The arguments that cannot be an option's value.

    An argument directly after a space-separated option (``--ignore
    tests/test_x.py``, ``-p plugin``) may be that option's value, and pytest's
    option table is open-ended, so it is left to pytest. Missing one here only
    defers the refusal to pytest; refusing a value would block a valid run.
    """
    certain: list[str] = []
    for index, argument in enumerate(selection):
        previous = selection[index - 1] if index else ""
        if previous.startswith("-") and "=" not in previous:
            continue
        certain.append(argument)
    return certain


def _is_test_module_name(name: str) -> bool:
    """Whether ``name`` follows pytest's default test-module naming."""
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


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
        force_rerun, selection = _parse_rerun(selection)
    except ValueError as exc:
        sys.stderr.write(f"devtools test: {exc}\n")
        return 2
    if outlier_count is not None:
        return print_outliers(outlier_count)
    on_default_branch = ON_DEFAULT_BRANCH_FLAG in selection
    selection = [arg for arg in selection if arg != ON_DEFAULT_BRANCH_FLAG]
    selection = _normalize_selection_paths(selection, invocation_directory=invocation_directory)
    _anchor_test_paths()
    identity = checkout_identity(ROOT)
    refusal = default_branch_refusal(identity, command="devtools test", allowed=on_default_branch)
    if refusal is not None:
        sys.stderr.write(refusal + "\n")
        return REFUSAL_EXIT
    sys.stderr.write(f"devtools test: {identity.describe()}\n")
    try:
        assert_polylogue_matches_checkout(ROOT, context="devtools test")
    except CheckoutImportMismatchError as exc:
        # The detail first: the verdict, naming the checkout, is the last line.
        sys.stderr.write(f"{exc}\n")
        sys.stderr.write(f"devtools test: FAILED exit=125 diagnosis=checkout_import_mismatch {identity.describe()}\n")
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

    # Refuse a missing path before queueing for the host pytest slot: pytest
    # fails such a run at collection anyway, but only after it has waited out
    # the pool admission, which on a contended pool is many minutes.
    absent_before_admission = [
        argument
        for argument in absent_selection_paths(_certain_selections(selection), root=ROOT)
        # Only a node id or a test module is certain to be a selection: an
        # option value (``--junit-xml reports/out.xml``, ``--log-file
        # reports/out.py``) also contains a slash and legitimately does not
        # exist before the run.
        if "::" in argument or _is_test_module_name(Path(argument).name)
    ]
    if absent_before_admission:
        sys.stderr.write(
            "devtools test: these selected paths do not exist, so nothing was queued: "
            + ", ".join(absent_before_admission)
            + "\n"
        )
        return 4

    if runner == "managed":
        # Two callers in one checkout asking for the same selection share one
        # run: the second waits here, then finds the first's receipt below. A
        # forced or non-reusing run takes the lock too, so its new receipt can
        # never land while another caller is answering from an older one.
        _hold_selection_lock(selection)
    # An isolated run exists to execute outside the managed slot; a managed
    # receipt cannot stand in for it.
    if not force_rerun and runner == "managed" and os.environ.get(REUSE_ENV, "1") != "0":
        # The wait for the lock can be long: the checkout may have changed
        # branch meanwhile, so admission is decided again before any reuse.
        identity = checkout_identity(ROOT)
        refusal = default_branch_refusal(identity, command="devtools test", allowed=on_default_branch)
        if refusal is not None:
            sys.stderr.write(refusal + "\n")
            return REFUSAL_EXIT
        digest = git_worktree_content_sha256(ROOT)
        environment_key = _reuse_environment_key()
        reused = reusable_green_receipt(
            selection,
            root=ROOT,
            content_sha256=digest,
            environment_key=environment_key,
        )
        if reused is not None and (
            git_worktree_content_sha256(ROOT) != digest or _reuse_environment_key() != environment_key
        ):
            # A save landed during the lookup: the receipt no longer describes
            # this tree, so the selection runs.
            reused = None
        if reused is not None:
            # And the branch may have moved under an unchanged tree: admission
            # is decided on the checkout as it is at the moment of reuse.
            identity = checkout_identity(ROOT)
            refusal = default_branch_refusal(identity, command="devtools test", allowed=on_default_branch)
            if refusal is not None:
                sys.stderr.write(refusal + "\n")
                return REFUSAL_EXIT
        reused_payload: object = None
        if reused is not None:
            # Read now: retention pruning may remove the run directory at any
            # moment, and a receipt that cannot be read answers nothing.
            try:
                reused_payload = json.loads(reused.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                reused = None
        if reused is not None:
            if use_json:
                print(json.dumps(reused_payload, indent=2, ensure_ascii=False))
            sys.stderr.write(
                "devtools test: this selection already passed on this exact tree; not queueing again "
                "(--rerun to force).\n"
                f"\ndevtools test: PASSED exit=0 diagnosis=pytest_passed_reused receipt={reused} {identity.describe()}\n"
            )
            return 0

    run = VerifyRun(
        tier="focused-test",
        argv=selection,
        git_head=git_head(ROOT),
        root=ROOT,
    )
    run.record_execution_environment_key(_reuse_environment_key())
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
    temp_root = basetemp_root(os.environ, root=ROOT)
    # A killed run leaves its basetemp behind; the next run is the first moment
    # anything can notice, so it reclaims every tree whose owner is gone.
    sweep_stale_temp_trees(temp_root)
    run_temp = temp_root / f"tmp-{os.getpid()}-{time.time_ns():x}"
    remove_temp_tree(run_temp)
    # Guards the abnormal exits: a signal or an interpreter teardown that never
    # reaches the disposition below. Cancelled once that disposition is made.
    temp_guard = guard_temp_trees(run_temp)
    _prepare_nodatacow_parent(run_temp)
    if not any(arg == "--basetemp" or arg.startswith("--basetemp=") for arg in selection):
        cmd = [*cmd, "--basetemp", str(run_temp)]
    _clear_pytest_report(report_path)
    artifacts = run.start_step(label="pytest focused", cmd=cmd)
    started = time.monotonic()
    try:
        pytest_env = focused_pytest_env(run=run, artifacts=artifacts)
        if _selected_test_modules(selection) >= BROAD_SELECTION_MODULES:
            # A selection this broad accumulates like the corpus does, so it
            # is sized by the corpus model, not the focused profile.
            pytest_env.pop(CHARGE_PROFILE_ENV, None)
        pytest_env.pop("POLYLOGUE_PYTEST_CONTAINMENT_PATH", None)
        # A named selection builds only what it asked for: the shared-archive
        # warm-up in tests/conftest.py's pytest_sessionstart is the broad
        # verifier's, and costs a focused run ~13 s and 12 archive-tier
        # initializations it has no use for (polylogue-62j1f).
        #
        # HAZARD: `devtools.verify` defines a function of the SAME NAME whose
        # body SETS this variable. The call below must stay this module's.
        pytest_env.pop("POLYLOGUE_BROAD_PREWARM", None)
        _normalize_managed_pytest_environment(pytest_env)
        # Only this invocation's flag authorizes the default branch; an
        # inherited value must not reach the slot's start-time re-check.
        pytest_env.pop(ALLOW_DEFAULT_BRANCH_ENV, None)
        if on_default_branch:
            pytest_env[ALLOW_DEFAULT_BRANCH_ENV] = "1"
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
            artifacts=artifacts,
            report_path=report_path,
            runner=runner,
            stdout=sys.stderr if use_json else None,
        )
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
        slot_log = metadata.get("pytest_slot_log")
        if isinstance(slot_log, str) and Path(slot_log).is_file():
            shutil.copyfile(slot_log, ROOT / PYTEST_REPORT_DIR / "current-pytest-output.log")
        _publish_last_focused_pytest_report(report_path)
        metadata["testmon_preselection"] = {
            "status": graph.status.value,
            "reason": graph.reason,
            "cause": graph.full_rerun_cause,
            "recorded_tests": getattr(graph, "recorded_tests", None),
            "source_dependencies": getattr(graph, "source_dependencies", None),
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
    provenance = metadata.get("worktree_provenance")
    if isinstance(provenance, dict):
        run.record_execution_worktree(provenance)
        # Report what actually ran, not what was admitted at submission.
        identity = replace(identity, branch=provenance.get("git_branch"), head=provenance.get("git_head"))
        # pytest may import files at any point of its run: content that moved
        # after the slot identified it means no single tree was tested.
        finished = checkout_identity(ROOT)
        if (finished.branch, finished.head) != (identity.branch, identity.head) or git_worktree_content_sha256(
            ROOT
        ) != provenance.get("git_worktree_content_sha256"):
            sys.stderr.write(
                f"devtools test: the checkout moved during the run (tested {identity.describe()}, "
                f"finished {finished.describe()}); the result is void\n"
            )
            rc = rc or 1
            metadata = {**metadata, "diagnosis": "checkout_moved_during_run"}
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
    temp_guard.cancel()
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
    # Absolute, so the line names the checkout that ran.
    receipt = ROOT / run.relative_run_dir / "run.json"
    # The rest of the artifacts are reference material, not a result. Printing
    # them after every green run trains the reader to skip the tail of the
    # output, which is exactly where a failure summary appears. `devtools why`
    # reaches them on demand. When they are printed, it is before the verdict,
    # so the verdict and the checkout it tested stay the last line.
    if _verbose_output() or rc != 0:
        sys.stderr.write(f"\ndevtools test: artifacts={run.relative_run_dir}/steps/{artifacts.step_id}")
    sys.stderr.write(
        f"\ndevtools test: {'PASSED' if rc == 0 else 'FAILED'} exit={rc} "
        f"diagnosis={metadata.get('diagnosis') or 'unknown'} receipt={receipt} {identity.describe()}\n"
    )
    return rc
