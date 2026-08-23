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
import subprocess
import sys
import time
from pathlib import Path

from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
)
from devtools.verify_runs import (
    VerifyRun,
    append_verify_history,
    configured_pytest_worker_request,
    env_for_pytest_step,
    git_head,
    pytest_command_worker_request,
)

ROOT = Path(__file__).resolve().parent.parent
PYTEST_REPORT_DIR = Path(".cache/verify")
PYTEST_REPORT_PATH = PYTEST_REPORT_DIR / "last-pytest.json"
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


def _verbose_output() -> bool:
    """Whether the caller asked for the full preamble and artifact footer."""
    return "--verbose" in sys.argv[1:] or bool(os.environ.get("POLYLOGUE_DEVTOOLS_VERBOSE"))


def _import_path_is_unsurprising(polylogue_import_path: Path | str) -> bool:
    """Whether the resolved package is the one this checkout owns.

    Fails toward announcing. An empty or unresolvable path is NOT evidence that
    the environment is fine -- and it would otherwise read as unsurprising,
    because ``Path("").resolve()`` is the current directory, which normally sits
    inside the checkout.
    """
    if not str(polylogue_import_path).strip():
        return False
    try:
        return Path(polylogue_import_path).resolve().is_relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return False


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
    completed = subprocess.run(command, cwd=cwd, env=env)
    return (
        completed.returncode,
        time.monotonic() - started,
        {"diagnosis": "pytest_passed" if completed.returncode == 0 else "pytest_failed"},
    )


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
    # Announce the resolved package ONLY when it is surprising. This line exists
    # for the 2026-07-31 wrong-checkout incident, where a worktree silently ran
    # the main checkout's code; it earns its place when it contradicts the
    # checkout, and is pure noise on the overwhelming majority of runs where it
    # does not. The fact itself is never lost -- it is recorded as
    # polylogue_import_path in the run receipt either way.
    if _verbose_output() or not _import_path_is_unsurprising(polylogue_import_path):
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
    _clear_pytest_report(cmd)
    run = VerifyRun(
        tier="focused-test",
        argv=selection,
        git_head=git_head(ROOT),
        root=ROOT,
        polylogue_import_path=str(polylogue_import_path),
        environment_fingerprint=environment_fingerprint,
    )
    artifacts = run.start_step(label="pytest focused", cmd=cmd)
    started = time.monotonic()
    try:
        pytest_env = env_for_pytest_step(dict(os.environ), run=run, artifacts=artifacts)
        pytest_env.pop("POLYLOGUE_PYTEST_CONTAINMENT_PATH", None)
        rc, elapsed, metadata = _run(
            "pytest focused",
            cmd,
            cwd=str(ROOT),
            env=pytest_env,
            run=run,
        )
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
    run.finish_step(
        step_id=artifacts.step_id,
        result={"duration_s": elapsed, "exit": rc, **metadata},
    )
    payload = run.finish(
        exit_code=rc,
        duration_s=elapsed,
        diagnosis=metadata.get("diagnosis"),
        verification_scope="affected",
        final_git_head=git_head(ROOT),
    )
    append_verify_history(payload)
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
