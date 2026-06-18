"""``devtools test`` — focused pytest runner through the managed harness.

Agents and humans should never invoke raw ``pytest`` for inner-loop checks.
This command forwards a selection (paths, ``-k``/``-m`` expressions, ``-x``,
…) to pytest with:

- the repository's managed environment (``POLYLOGUE_ROOT`` and friends, a
  repo-local pycache prefix);
- a single-process worker default (``-n 0``) for fast focused runs, overridable
  with ``-n`` in the selection or ``POLYLOGUE_PYTEST_WORKERS``;
- live, streamed output (unlike ``devtools verify``, which captures);
- the same pytest progress ledger, heartbeat, and stall timeout used by
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
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from devtools.verify import (
    PYTEST_EVENTS_PATH,
    PYTEST_OUTPUT_PATH,
    PYTEST_PROGRESS_PATH,
    _clear_pytest_report,
    _run_pytest_with_heartbeat,
)

ROOT = Path(__file__).resolve().parent.parent
_LOCK_PATH = ROOT / ".cache" / "test-run.lock"


def _managed_env() -> dict[str, str]:
    """Mirror devtools.verify's subprocess environment for parity."""
    env = os.environ.copy()
    env["POLYLOGUE_ROOT"] = str(ROOT)
    env["POLYLOGUE_REPO_ROOT"] = str(ROOT)
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / ".cache" / "pycache")
    env["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(ROOT / PYTEST_EVENTS_PATH)
    return env


def _has_worker_flag(selection: list[str]) -> bool:
    """True when the caller already chose an xdist worker count."""
    return any(arg.startswith(("-n", "--numprocesses")) for arg in selection)


def _worker_args(selection: list[str]) -> list[str]:
    """Default focused runs to a single process; honor an explicit override."""
    if _has_worker_flag(selection):
        return []
    workers = os.environ.get("POLYLOGUE_PYTEST_WORKERS", "0").strip() or "0"
    return ["-n", workers]


def build_pytest_cmd(selection: list[str]) -> list[str]:
    """Compose the pytest command for a focused selection."""
    return ["pytest", "-p", "devtools.pytest_progress_plugin", *selection, *_worker_args(selection)]


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
    selection = list(sys.argv[1:] if argv is None else argv)
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
        result = _run_pytest_with_heartbeat(cmd, cwd=str(ROOT), env=_managed_env(), t0=time.monotonic())
    if result.stderr:
        sys.stderr.write(result.stderr)
    sys.stderr.write(
        f"\ndevtools test: progress={PYTEST_PROGRESS_PATH} events={PYTEST_EVENTS_PATH} output={PYTEST_OUTPUT_PATH}\n"
    )
    return result.returncode
