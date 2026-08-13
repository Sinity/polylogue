"""``devtools test`` — focused pytest runner through the managed harness.

Agents and humans should never invoke raw ``pytest`` for inner-loop checks.
This command forwards a selection (paths, ``-k``/``-m`` expressions, ``-x``,
…) to pytest with:

- the repository's managed environment (``POLYLOGUE_ROOT`` and friends, a
  repo-local pycache prefix);
- a single-process worker default (``-n 0``) for fast focused runs, overridable
  with ``-n`` in the selection or ``POLYLOGUE_PYTEST_WORKERS``;
- live, streamed output (unlike ``devtools verify``, which captures);
- the same pytest progress ledger, external deadline supervisor, owned process
  group/cgroup containment, heartbeat, and stall timeout used by
  ``devtools verify``;
- a checkout-scoped lock that serializes overlapping runs so two suites from
  the same checkout do not race and burn CPU. Concurrency is already
  *correctness*-safe at the conftest level (#1785, per-run tmpfs basetemp); the
  lock is the throughput guard. Set ``POLYLOGUE_TEST_NO_LOCK=1`` to bypass it.

For the full pre-PR gate use ``devtools verify``; this command is the inner
loop, not a substitute for it.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
)
from devtools.verify import (
    PYTEST_CONTAINMENT_PATH,
    PYTEST_EVENTS_PATH,
    PYTEST_OUTPUT_PATH,
    PYTEST_PROGRESS_PATH,
    PYTEST_REPORT_PATH,
    PYTEST_SELECTION_PATH,
    PYTEST_SUMMARY_PATH,
    _clear_pytest_report,
    _run,
)
from devtools.verify_runs import (
    VerifyRun,
    append_verify_history,
    git_head,
    pytest_command_worker_request,
    worktree_fingerprint,
)

ROOT = Path(__file__).resolve().parent.parent
_LOCK_PATH = ROOT / ".cache" / "test-run.lock"
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


def _absolute_option_path(
    value: str,
    *,
    invocation_directory: Path,
    expand_environment_variables: bool = False,
) -> str:
    if expand_environment_variables:
        value = os.path.expandvars(value)
    path = Path(value)
    return str(path if path.is_absolute() else (invocation_directory / path).resolve())


def _normalize_selection_paths(selection: list[str], *, invocation_directory: Path) -> list[str]:
    """Preserve path selections relative to the directory that invoked devtools."""
    normalized: list[str] = []
    pending_option: str | None = None
    for argument in selection:
        if pending_option is not None:
            normalized.append(
                _absolute_option_path(
                    argument,
                    invocation_directory=invocation_directory,
                    expand_environment_variables=pending_option in _ENV_EXPANDING_PATH_OPTIONS,
                )
            )
            pending_option = None
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
    """Default focused runs to a single process; honor an explicit override."""
    if _has_worker_flag(selection):
        return []
    workers = os.environ.get("POLYLOGUE_PYTEST_WORKERS", "0").strip() or "0"
    return ["-n", workers]


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
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "devtools.pytest_progress_plugin",
        "--json-report",
        "--json-report-omit=collectors,log,streams,warnings",
        f"--json-report-file={PYTEST_REPORT_PATH}",
        *selection,
        *worker_args,
        *_xdist_distribution_args(selection, worker_args),
    ]


@contextlib.contextmanager
def _run_lock(*, enabled: bool) -> Iterator[None]:
    """Serialize concurrent ``devtools test`` runs from the same checkout."""
    if not enabled:
        yield
        return
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            holder = handle.read().strip() or "another run"
            sys.stderr.write(
                f"devtools test: waiting for in-flight run ({holder}) — set POLYLOGUE_TEST_NO_LOCK=1 to skip\n"
            )
            sys.stderr.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}")
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()


def main(argv: list[str] | None = None) -> int:
    invocation_directory = Path.cwd()
    selection = list(sys.argv[1:] if argv is None else argv)
    selection = _normalize_selection_paths(selection, invocation_directory=invocation_directory)
    _anchor_test_paths()
    try:
        fingerprint = assert_polylogue_matches_checkout(ROOT, context="devtools test")
    except CheckoutImportMismatchError as exc:
        sys.stderr.write(f"{exc}\n")
        return 125
    polylogue_import_path = fingerprint.polylogue_import_path
    environment_fingerprint = fingerprint.as_dict()
    sys.stderr.write(f"devtools test: polylogue package → {polylogue_import_path}\n")

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
    no_lock = os.environ.get("POLYLOGUE_TEST_NO_LOCK") == "1"
    with _run_lock(enabled=not no_lock):
        _clear_pytest_report(cmd)
        initial_worktree_fingerprint = worktree_fingerprint(ROOT)
        run = VerifyRun(
            tier="focused-test",
            argv=selection,
            git_head=git_head(ROOT),
            root=ROOT,
            polylogue_import_path=str(polylogue_import_path),
            environment_fingerprint=environment_fingerprint,
            worktree_fingerprint=initial_worktree_fingerprint,
        )
        started = time.monotonic()
        try:
            rc, _elapsed, metadata = _run("pytest focused", cmd, cwd=str(ROOT), run=run)
        except KeyboardInterrupt:
            rc = 130
            metadata = {"diagnosis": "pytest_interrupted", "termination_reason": "operator_interrupt"}
            run.finish_interrupted_steps(exit_code=rc, diagnosis=str(metadata["diagnosis"]))
        final_worktree_fingerprint = worktree_fingerprint(ROOT)
        if "unavailable" in {initial_worktree_fingerprint, final_worktree_fingerprint}:
            metadata["diagnosis"] = "checkout_fingerprint_unavailable"
            if rc == 0:
                rc = 125
            sys.stderr.write("devtools test: checkout fingerprint unavailable; evidence is not exact-head.\n")
        elif final_worktree_fingerprint != initial_worktree_fingerprint:
            metadata["diagnosis"] = "checkout_changed_during_focused_test"
            if rc == 0:
                rc = 125
            sys.stderr.write("devtools test: checkout contents changed during pytest; evidence is not exact-head.\n")
        payload = run.finish(
            exit_code=rc,
            duration_s=time.monotonic() - started,
            diagnosis=metadata.get("diagnosis"),
            verification_scope="affected",
            release_baseline_allowed=False,
            final_worktree_fingerprint=final_worktree_fingerprint,
        )
        append_verify_history(payload)
    if use_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    sys.stderr.write(
        f"\ndevtools test: progress={PYTEST_PROGRESS_PATH} selection={PYTEST_SELECTION_PATH} "
        f"summary={PYTEST_SUMMARY_PATH} events={PYTEST_EVENTS_PATH} containment={PYTEST_CONTAINMENT_PATH} "
        f"output={PYTEST_OUTPUT_PATH}\n"
    )
    return rc
