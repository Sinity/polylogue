from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from pathlib import Path
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Any, TextIO

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

# Basetemp placement is selected once by ``resolve_pytest_basetemp_root``.
# This conftest owns only pytest-side claims, stale-tree reclamation, and the
# optional btrfs no-CoW mark for a disk-backed selection.
from devtools import verify_runs
from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
    resolved_polylogue_path,
)
from devtools.pytest_supervisor import _process_start_ticks
from devtools.verify_runs import (
    PytestResourceError,
    clear_managed_pytest_basetemp_claim,
    normalize_pytest_basetemp_env,
    resolve_pytest_basetemp_root,
)
from devtools.verify_runs import pytest_basetemp_claim_path as _basetemp_claim_path

# Resolve (but don't yet raise on) the polylogue-vs-checkout mismatch check
# before the first `from polylogue...` import below: a shared/editable venv's
# `.pth` entry can point at a different checkout than the one this pytest
# process is actually running from (e.g. a linked git worktree reusing the
# main checkout's `.venv`), and whichever tree `import polylogue` resolves to
# here is the one every test in this run uses (sys.modules caches by name).
# The actual `pytest.UsageError` is raised from `pytest_configure` below —
# matching the existing basetemp-preflight precedent in this file — so pytest
# reports a clean usage error instead of a raw conftest-import traceback.
_TESTS_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    assert_polylogue_matches_checkout(_TESTS_REPO_ROOT, context="pytest (tests/conftest.py)")
    _CHECKOUT_GUARD_ERROR: CheckoutImportMismatchError | None = None
except CheckoutImportMismatchError as _checkout_exc:
    _CHECKOUT_GUARD_ERROR = _checkout_exc

from polylogue.archive.models import Session
from polylogue.scenarios import CorpusSpec, build_default_corpus_specs
from polylogue.storage.runtime import RawSessionRecord
from tests.infra.builders import make_conv, make_msg
from tests.infra.timeout_policy import timeout_marker_error

pytest_plugins = (
    "tests.infra.corpus_fixtures",
    "tests.infra.scale_fixtures",
    "tests.infra.frozen_clock",
    "tests.infra.clock_guard",
)

# Pytest supports nested in-process ``pytest.main()`` calls. Every controller
# invocation occupies this process-local stack, including caller-owned scopes.
# A completed invocation therefore cannot authorize stale environment values,
# and an unmanaged middle scope cannot resurrect a managed outer identity.
_ACTIVE_PYTEST_SCOPES: list[tuple[str, str] | None] = []
_ACTIVE_PYTEST_BASETEMPS: set[Path] = set()
_PYTEST_SCOPE_ATTR = "_polylogue_pytest_scope"
_PYTEST_SCOPE_BASETEMP_ATTR = "_polylogue_pytest_scope_basetemp"
_PYTEST_SCOPE_ENV_ATTR = "_polylogue_pytest_scope_environment"
_NESTED_BASETEMP_POLICY_ENV = ("POLYLOGUE_PYTEST_BASETEMP_ROOT", "POLYLOGUE_PYTEST_TMPFS")

if TYPE_CHECKING:
    from click.testing import CliRunner

    from polylogue.config import Source
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite import SQLiteBackend
    from tests.infra.storage_records import SessionBuilder

# ---------------------------------------------------------------------------
# Scale markers for data-gravity and long-haul validation (Workstream H)
# ---------------------------------------------------------------------------


def _set_managed_pytest_identity(identity: tuple[str, str] | None) -> None:
    """Expose only the managed identity owned by the active invocation."""
    if identity is None:
        os.environ.pop("POLYLOGUE_PYTEST_RUN_ID", None)
        os.environ.pop("POLYLOGUE_PYTEST_MANAGED_BASETEMP", None)
        return
    run_id, basetemp = identity
    os.environ["POLYLOGUE_PYTEST_RUN_ID"] = run_id
    os.environ["POLYLOGUE_PYTEST_MANAGED_BASETEMP"] = basetemp


def _push_pytest_scope(
    config: pytest.Config, identity: tuple[str, str] | None, *, basetemp: Path | None = None
) -> None:
    """Register one controller invocation, managed or caller-owned."""
    _ACTIVE_PYTEST_SCOPES.append(identity)
    setattr(config, _PYTEST_SCOPE_ATTR, identity)
    if basetemp is not None:
        claim_path = _basetemp_claim_path(basetemp, kind="lock")
        _ACTIVE_PYTEST_BASETEMPS.add(claim_path)
        setattr(config, _PYTEST_SCOPE_BASETEMP_ATTR, claim_path)


def _force_nested_pytest_scratch(config: pytest.Config) -> None:
    """Keep an unsupervised nested controller out of an outer tmpfs budget."""
    previous = {name: os.environ.get(name) for name in _NESTED_BASETEMP_POLICY_ENV}
    setattr(config, _PYTEST_SCOPE_ENV_ATTR, previous)
    forced = verify_runs.force_managed_pytest_scratch(os.environ)
    for name in _NESTED_BASETEMP_POLICY_ENV:
        value = forced.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _restore_nested_pytest_scratch(config: pytest.Config) -> None:
    previous = getattr(config, _PYTEST_SCOPE_ENV_ATTR, None)
    if previous is None:
        return
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and choose the managed test temp root."""
    if _CHECKOUT_GUARD_ERROR is not None:
        # Refuse before collection: every test in this run would otherwise
        # exercise a `polylogue` package from a different checkout than the
        # one this pytest process was invoked against (see
        # devtools/checkout_guard.py for the full hazard writeup).
        raise pytest.UsageError(f"pytest: {_CHECKOUT_GUARD_ERROR}") from _CHECKOUT_GUARD_ERROR
    sys.stderr.write(f"pytest: polylogue package → {resolved_polylogue_path()} (checkout: {_TESTS_REPO_ROOT})\n")
    config.addinivalue_line("markers", "scale(level): parametric scale marker (small/medium/large/stretch)")
    # Tiered scale markers (issue #1183); definitions also live in
    # pyproject.toml `markers` so xfail_strict + filterwarnings agree.
    config.addinivalue_line(
        "markers", "scale_small: small-tier scale fixture (~100 convs / ~1k msgs); default verify gate (#1183)"
    )
    config.addinivalue_line(
        "markers", "scale_medium: medium-tier scale fixture (~1k convs / ~10k msgs); verify --lab gate (#1183)"
    )
    config.addinivalue_line(
        "markers",
        "scale_large: large-tier scale fixture (~10k convs / ~100k msgs); nightly CI / campaigns only (#1183)",
    )

    if config.option.basetemp is not None:
        configured_basetemp = str(config.option.basetemp)
        run_id = os.environ.get("POLYLOGUE_PYTEST_RUN_ID")
        managed_basetemp = os.environ.get("POLYLOGUE_PYTEST_MANAGED_BASETEMP")
        supervised_managed = (
            run_id is not None
            and os.environ.get("POLYLOGUE_VERIFY_RUN_ID") == run_id
            and managed_basetemp == configured_basetemp
        )
        if supervised_managed and not hasattr(config, "workerinput"):
            assert run_id is not None
            if _basetemp_claim_path(Path(configured_basetemp), kind="lock") in _ACTIVE_PYTEST_BASETEMPS:
                raise pytest.UsageError(
                    f"pytest: explicit basetemp is already active in this pytest process: {configured_basetemp}"
                )
            identity = (run_id, configured_basetemp)
            _push_pytest_scope(config, identity, basetemp=Path(configured_basetemp))
            return
        # A second in-process pytest.main() inherits os.environ from the first
        # run. Explicit basetemp ownership is per invocation, so stale managed
        # markers must not turn the caller-owned diagnostic tree into cleanup
        # fodder at session finish.
        if not hasattr(config, "workerinput"):
            _mark_caller_owned_basetemp(Path(configured_basetemp))
            _push_pytest_scope(config, None, basetemp=Path(configured_basetemp))
            _set_managed_pytest_identity(None)
        return

    if config.option.basetemp is None:
        prior_scope = (
            _ACTIVE_PYTEST_SCOPES[-1] if _ACTIVE_PYTEST_SCOPES and not hasattr(config, "workerinput") else None
        )
        if _ACTIVE_PYTEST_SCOPES and not hasattr(config, "workerinput"):
            _set_managed_pytest_identity(None)
            _force_nested_pytest_scratch(config)
        normalized_basetemp_env = normalize_pytest_basetemp_env(os.environ)
        configured_root = normalized_basetemp_env.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
        unmanaged_tmpfs_root = configured_root is not None and verify_runs._is_beneath(
            Path(configured_root), verify_runs.PYTEST_TMPFS_ROOT
        )
        if "POLYLOGUE_VERIFY_RUN_ID" not in os.environ and (
            "POLYLOGUE_PYTEST_BASETEMP_ROOT" not in normalized_basetemp_env or unmanaged_tmpfs_root
        ):
            # Bare pytest has no devtools supervisor to enforce a tmpfs cap.
            # Keep its basetemp on scratch; managed devtools runs carry the
            # verify-run id and may opt into bounded tmpfs safely.
            os.environ.pop("POLYLOGUE_PYTEST_BASETEMP_ROOT", None)
            os.environ["POLYLOGUE_PYTEST_TMPFS"] = "0"
        checkout = hashlib.sha1(str(config.rootpath).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        os.environ["POLYLOGUE_PYTEST_CHECKOUT"] = checkout
        run_id = os.environ.get("POLYLOGUE_PYTEST_RUN_ID")
        if not hasattr(config, "workerinput"):
            _sweep_stale_polylogue_basetemps()
        if run_id is None:
            run_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
            os.environ["POLYLOGUE_PYTEST_RUN_ID"] = run_id
        try:
            root, label = _managed_pytest_temp_root()
            basetemp = root / f"pytest-polylogue-{checkout}-{run_id}"
            if not hasattr(config, "workerinput"):
                _mark_basetemp_owner(basetemp)
        except PytestResourceError as exc:
            _restore_nested_pytest_scratch(config)
            _set_managed_pytest_identity(prior_scope)
            # Fail loudly and early: refuse before pytest starts collecting,
            # rather than crashing an unrelated command later with a bare
            # OSError once the chosen basetemp fills up.
            raise pytest.UsageError(f"pytest: {exc}") from exc
        config.option.basetemp = str(basetemp)
        os.environ["POLYLOGUE_PYTEST_MANAGED_BASETEMP"] = str(basetemp)
        if not hasattr(config, "workerinput"):
            identity = (run_id, str(basetemp))
            _push_pytest_scope(config, identity, basetemp=basetemp)
        sys.stderr.write(f"pytest: basetemp → {config.option.basetemp} ({label})\n")


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore or retire process-local managed ownership after one invocation."""
    if not hasattr(config, _PYTEST_SCOPE_ATTR):
        return
    scope = getattr(config, _PYTEST_SCOPE_ATTR)
    if not _ACTIVE_PYTEST_SCOPES or _ACTIVE_PYTEST_SCOPES[-1] != scope:
        return
    _ACTIVE_PYTEST_SCOPES.pop()
    active_basetemp = getattr(config, _PYTEST_SCOPE_BASETEMP_ATTR, None)
    if active_basetemp is not None:
        _ACTIVE_PYTEST_BASETEMPS.discard(active_basetemp)
    _restore_nested_pytest_scratch(config)
    _set_managed_pytest_identity(_ACTIVE_PYTEST_SCOPES[-1] if _ACTIVE_PYTEST_SCOPES else None)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Reject unbounded or effectively disabled per-test timeout markers."""
    for item in items:
        marker = item.get_closest_marker("timeout")
        if marker is None:
            continue
        issue = timeout_marker_error(marker)
        if issue is not None:
            raise pytest.UsageError(f"{item.nodeid}: {issue}")


# Per-run basetemps are freed on sessionfinish. A run killed before
# sessionfinish (SIGKILL, OOM) leaks its basetemp, so the controller reclaims
# clearly-dead orphans on startup. Seeded corpora (``pytest-polylogue-seeded-*``)
# are excluded because they are shared, reusable across runs, and bounded to
# one per checkout. Placement policy (which root, and whether it has enough
# free space) is a single shared source of truth in
# ``devtools.verify_runs.resolve_pytest_basetemp_root`` — this module only
# adds the mkdir/no-CoW-marking side effects and the stale-directory sweep.
_STALE_BASETEMP_MAX_AGE_S = 30 * 60
_BASE_TEMP_CLAIM_LOCKS: dict[Path, TextIO] = {}
_BASE_TEMP_CLAIM_THREAD_LOCKS: dict[Path, threading.Lock] = {}


def _managed_pytest_temp_root() -> tuple[Path, str]:
    """Return the temp root for managed pytest basetemps."""
    root, label = resolve_pytest_basetemp_root(os.environ)
    root.mkdir(parents=True, exist_ok=True)
    if label in ("scratch", "disk fallback"):
        # No-CoW marking is only meaningful (and only ever was applied) for
        # the disk-backed NVMe/cloud-fallback candidates. It shells out to
        # findmnt/lsattr/chattr with real subprocess-spawn latency, so it
        # must stay off the tmpfs and explicit-override paths — both are
        # startup-latency-sensitive (every managed pytest invocation pays
        # this once) and tmpfs never benefits from a btrfs-specific flag.
        _mark_btrfs_nocow(root)
    return root, label


def _mark_basetemp_owner(basetemp: Path) -> None:
    """Claim a managed tree outside pytest's replaceable basetemp directory."""
    handle = _acquire_basetemp_claim_lock(basetemp, blocking=False)
    if handle is None:
        raise PytestResourceError(f"managed pytest basetemp is already claimed: {basetemp}")
    pid = os.getpid()
    start_ticks = _process_start_ticks(pid)
    identity = f"{pid}:{start_ticks}" if start_ticks is not None else str(pid)
    try:
        _basetemp_claim_path(basetemp, kind="managed").write_text(identity, encoding="utf-8")
    except OSError as exc:
        _release_basetemp_claim_lock(basetemp)
        raise PytestResourceError(f"cannot record managed pytest basetemp claim: {basetemp}") from exc


def _mark_caller_owned_basetemp(basetemp: Path) -> None:
    """Claim an explicit ``--basetemp`` before pytest may replace its tree."""
    lock_path = _basetemp_claim_path(basetemp, kind="lock")
    if lock_path in _ACTIVE_PYTEST_BASETEMPS:
        raise pytest.UsageError(f"pytest: explicit basetemp is already active in this pytest process: {basetemp}")
    handle = _acquire_basetemp_claim_lock(basetemp, blocking=True)
    if handle is None:
        raise pytest.UsageError(f"pytest: cannot claim the explicit basetemp: {basetemp}")
    with contextlib.suppress(OSError):
        basetemp.mkdir(parents=True, exist_ok=True)
        clear_managed_pytest_basetemp_claim(basetemp)
        _basetemp_claim_path(basetemp, kind="caller-owned").write_text("explicit\n", encoding="utf-8")


def _acquire_basetemp_claim_lock(basetemp: Path, *, blocking: bool) -> TextIO | None:
    """Serialize a claim against a stale sweep for the same basetemp path."""
    lock_path = _basetemp_claim_path(basetemp, kind="lock")
    thread_lock = _BASE_TEMP_CLAIM_THREAD_LOCKS.setdefault(lock_path, threading.Lock())
    if not thread_lock.acquire(blocking=blocking):
        return None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError:
        thread_lock.release()
        return None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except OSError:
        handle.close()
        thread_lock.release()
        return None
    _BASE_TEMP_CLAIM_LOCKS[lock_path] = handle
    return handle


def _release_basetemp_claim_lock(basetemp: Path) -> None:
    """Release this pytest process's claim lock after its session ends."""
    lock_path = _basetemp_claim_path(basetemp, kind="lock")
    handle = _BASE_TEMP_CLAIM_LOCKS.pop(lock_path, None)
    if handle is not None:
        with contextlib.suppress(OSError):
            handle.close()
    thread_lock = _BASE_TEMP_CLAIM_THREAD_LOCKS.get(lock_path)
    if thread_lock is not None and thread_lock.locked():
        thread_lock.release()


def _remove_stale_basetemp(entry: Path) -> None:
    """Remove an already-adjudicated stale tree, including read-only fixtures."""
    if entry.is_symlink():
        return
    owner_directory_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    owner_file_mode = stat.S_IRUSR | stat.S_IWUSR
    for current_root, directories, files in os.walk(entry):
        root = Path(current_root)
        with contextlib.suppress(OSError):
            root.chmod(root.stat().st_mode | owner_directory_mode)
        for name in directories:
            child = root / name
            if child.is_symlink():
                continue
            with contextlib.suppress(OSError):
                child.chmod(child.stat().st_mode | owner_directory_mode)
        for name in files:
            child = root / name
            if child.is_symlink():
                continue
            with contextlib.suppress(OSError):
                child.chmod(child.stat().st_mode | owner_file_mode)
    shutil.rmtree(entry)


def _mark_btrfs_nocow(path: Path) -> None:
    """Best-effort no-CoW marking for SQLite-heavy pytest scratch roots."""
    if shutil.which("chattr") is None or _filesystem_type(path) != "btrfs":
        return
    try:
        current = subprocess.run(["lsattr", "-d", str(path)], capture_output=True, text=True, timeout=2)
        if current.returncode == 0 and current.stdout.split(maxsplit=1)[0].find("C") >= 0:
            return
        subprocess.run(["chattr", "+C", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return


def _filesystem_type(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["findmnt", "-T", str(path), "-no", "FSTYPE"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _polylogue_basetemp_roots() -> tuple[Path, ...]:
    """Every root a current or past placement policy could have used.

    Deliberately broader than "the root this run picked" so a placement
    policy change (e.g. this fix) still reclaims leftovers on a root the
    current run no longer chooses.
    """
    return verify_runs.pytest_basetemp_known_roots(os.environ)


def _sweep_stale_polylogue_basetemps(
    *, max_age_s: int = _STALE_BASETEMP_MAX_AGE_S, roots: tuple[Path, ...] | None = None
) -> None:
    """Best-effort reclaim of per-run basetemps left by crashed runs.

    Safety invariant: reclamation requires a durable, positive managed claim
    plus a confirmed-dead owner. Unknown paths are never deleted: they may be
    an explicit caller path racing a startup sweep. The claim lock makes the
    decision and a caller's claim mutually exclusive, while the claim itself
    survives pytest's lazy replacement of the basetemp directory. Explicit
    caller-owned paths carry their own durable claim and are excluded.
    Seeded corpora
    (``pytest-polylogue-*-seeded-*``) are never touched here — they are
    shared, reusable, and built once behind their own ``.build.done`` guard.
    """

    cutoff = time.time() - max_age_s
    for root in roots or _polylogue_basetemp_roots():
        for entry in root.glob("pytest-polylogue-*"):
            if "-seeded-" in entry.name:
                continue
            try:
                if not entry.is_dir():
                    continue
                if _basetemp_claim_path(entry, kind="caller-owned").is_file():
                    continue
                if not _basetemp_claim_path(entry, kind="managed").is_file():
                    continue
                handle = _acquire_basetemp_claim_lock(entry, blocking=False)
                if handle is None:
                    continue
                try:
                    if _basetemp_claim_path(entry, kind="caller-owned").is_file():
                        continue
                    if not _basetemp_claim_path(entry, kind="managed").is_file():
                        continue
                    owner_alive = verify_runs.managed_pytest_basetemp_owner_alive(entry)
                    if owner_alive is not False:
                        continue
                    if entry.stat().st_mtime < cutoff:
                        _remove_stale_basetemp(entry)
                        clear_managed_pytest_basetemp_claim(entry)
                finally:
                    _release_basetemp_claim_lock(entry)
            except OSError:
                pass


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Free this run's per-run tmpfs basetemp (controller only).

    xdist workers share the controller's basetemp, so only the controller
    (no ``PYTEST_XDIST_WORKER`` in env) removes it, and only when we minted
    a per-run name — never a caller-supplied ``--basetemp`` or the shared
    seeded corpus root. Parallel xdist runs leave their per-run basetemp for
    the next startup sweep instead of deleting it during session-finish
    teardown, where xdist/json-report may still be flushing controller/worker
    artifacts.
    """
    basetemp = session.config.option.basetemp
    if not basetemp:
        return
    basetemp_path = Path(str(basetemp))
    try:
        if os.environ.get("PYTEST_XDIST_WORKER"):
            return
        if not os.environ.get("POLYLOGUE_PYTEST_RUN_ID"):
            return
        numprocesses = getattr(session.config.option, "numprocesses", None)
        if numprocesses not in (None, 0, "0"):
            return
        if str(basetemp_path) == os.environ.get("POLYLOGUE_PYTEST_MANAGED_BASETEMP"):
            shutil.rmtree(basetemp_path, ignore_errors=True)
            if not basetemp_path.exists():
                clear_managed_pytest_basetemp_claim(basetemp_path)
    finally:
        _release_basetemp_claim_lock(basetemp_path)


@pytest.fixture(autouse=True)
def _reclaim_test_tmp_path(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Release each test's private tree as soon as its teardown finishes.

    Failure evidence belongs in the managed event/longrepr/resource receipts,
    not in an unbounded filesystem witness. Retaining every failed tree made
    full-suite tmpfs usage proportional to the number of failures and caused
    a calm 8-worker run to exceed 2 GiB before completing. A failing node can
    still be rerun with an explicit basetemp when its files matter. That
    explicit diagnostic path retains its per-test trees for inspection.
    """
    configured = getattr(request.config.option, "basetemp", None)
    managed = os.environ.get("POLYLOGUE_PYTEST_MANAGED_BASETEMP")
    if configured and str(configured) != managed:
        yield
        return
    try:
        yield
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Hypothesis profiles: `--hypothesis-profile ci` uses fewer examples for speed
# ---------------------------------------------------------------------------
_HYPOTHESIS_HOME = Path(".cache/hypothesis")
set_hypothesis_home_dir(_HYPOTHESIS_HOME)
_HYPOTHESIS_DB = DirectoryBasedExampleDatabase(_HYPOTHESIS_HOME / "examples")

settings.register_profile(
    "ci",
    max_examples=5 if os.environ.get("POLYLOGUE_CI") else 30,
    # deadline=None: per-example wall time is dominated by xdist scheduling
    # jitter under `-n 16`, not algorithmic cost, so a deadline only produces
    # load-dependent flakes (#1775). Real perf regressions are caught by the
    # dedicated TestPerformanceBudget / benchmark tests, not Hypothesis timing.
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    database=_HYPOTHESIS_DB,
)
# differing_executors fires when mutmut runs tests in threads — suppress globally since
# it never indicates a real bug (only fires in threaded test runners, not normal pytest).
settings.register_profile(
    "default",
    max_examples=100,
    # deadline=None + suppress too_slow: see the ci profile note (#1775). The
    # property tests in test_schema_privacy / test_null_guard_properties /
    # test_machine_contract have no per-test deadline override and previously
    # inherited Hypothesis's 200ms default, which DB- and subprocess-touching
    # examples blow past under full-suite worker contention.
    deadline=None,
    suppress_health_check=[HealthCheck.differing_executors, HealthCheck.too_slow],
    database=_HYPOTHESIS_DB,
)
# "verify" profile: fast bounded-example pass for routine devtools verify cycles.
# Use HYPOTHESIS_PROFILE=verify to cut hypothesis runtime ~8× while retaining
# regression detection on the most recent counterexamples in the database.
settings.register_profile(
    "verify",
    max_examples=10,
    deadline=5000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.differing_executors,
    ],
    database=_HYPOTHESIS_DB,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


_CONNECTION_MACHINERY = (
    "storage/sqlite/connection.py",
    "storage/sqlite/connection_profile.py",
    "contextlib.py",
)
_TESTS_ROOT = str(Path(__file__).resolve().parent)

# These variables are emitted by the managed verification supervisor and must
# survive the host-configuration scrub below.  They are test-run evidence
# plumbing, not operator configuration; removing them after collection makes
# setup/call reports disappear from the event ledger while teardown still gets
# recorded, which makes interrupted seed shards look falsely successful.
_MANAGED_VERIFY_ENV = frozenset(
    {
        "POLYLOGUE_VERIFY_RUN_ID",
        "POLYLOGUE_PYTEST_EVENTS_DIR",
        "POLYLOGUE_PYTEST_EVENTS_PATH",
        "POLYLOGUE_PYTEST_RUN_ID",
        "POLYLOGUE_PYTEST_SELECTION_PATH",
        "POLYLOGUE_PYTEST_SUMMARY_PATH",
        "POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT",
        "POLYLOGUE_PYTEST_MANAGED_BASETEMP",
    }
)


@pytest.fixture(autouse=True)
def _close_test_opened_sqlite_connections(
    monkeypatch: pytest.MonkeyPatch,
    _reclaim_test_tmp_path: None,
) -> Iterator[None]:
    """Close sync ``sqlite3`` connections that *test code* opened but never closed.

    Many tests use ``with sqlite3.connect(...) as conn:`` which commits on exit
    but does NOT close the connection — a per-call leak that surfaces as
    ``ResourceWarning: unclosed database`` now that ``ignore::ResourceWarning``
    is removed from ``filterwarnings``. Rather than hand-close hundreds of call
    sites (and re-fight the same battle on every new test), track connections
    whose real opening call site lives under ``tests/`` and close them at
    teardown — per-thread, so each is closed from the thread that created it
    (``sqlite3`` connections are ``check_same_thread`` by default).

    Connections opened from *production* code (``polylogue/...``) are
    deliberately NOT tracked. If production leaks a connection the warning still
    fires, so this fixture cannot silently mask a real production leak — the
    kind this change just fixed in the live cursor store, the cursor-lag
    baseline, and the embedding progress ledger.

    The async (``aiosqlite``) leg is symmetric in spirit but simpler: every
    production async path opens its connection with ``async with
    aiosqlite.connect(...)`` (auto-closed), so a connection still open at
    teardown is necessarily a test that built a facade/backend and never
    ``await``-ed ``close()``. Those are registered at construction and closed
    via a fresh event loop (``aiosqlite`` bridges ``close()`` onto its own
    worker thread, so a new loop closes it cleanly).
    """
    import aiosqlite

    main_ident = threading.get_ident()
    sync_local = threading.local()
    async_tracked: list[aiosqlite.Connection] = []
    real_async_init = aiosqlite.Connection.__init__

    def _bucket() -> list[sqlite3.Connection]:
        conns: list[sqlite3.Connection] | None = getattr(sync_local, "conns", None)
        if conns is None:
            conns = []
            sync_local.conns = conns
        return conns

    def _close_current_thread() -> None:
        for conn in _bucket():
            try:
                conn.close()
            except Exception:
                pass
        sync_local.conns = []

    real_connect = sqlite3.connect
    real_thread_run = threading.Thread.run

    def _opened_by_test() -> bool:
        # Walk past the connection-opener machinery to the real call site.
        frame: FrameType | None = sys._getframe(2)
        while frame is not None:
            filename = frame.f_code.co_filename
            if not any(part in filename for part in _CONNECTION_MACHINERY):
                return filename.startswith(_TESTS_ROOT)
            frame = frame.f_back
        return False

    def _tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn: sqlite3.Connection = real_connect(*args, **kwargs)
        if _opened_by_test():
            _bucket().append(conn)
        return conn

    def _thread_run(self: threading.Thread) -> None:
        try:
            real_thread_run(self)
        finally:
            _close_current_thread()
            # A worker thread that used the production thread-local connection
            # cache (``connection_context``/``open_connection`` from
            # ``connection.py``) leaves its cached connection open — the cache is
            # per-thread reuse with no thread-exit hook. Clear it here, from the
            # worker thread itself, so the cached connection is closed rather
            # than leaked when the thread ends. This only closes the production
            # cache (designed to be cleared), so it does not mask a real leak.
            from polylogue.storage.sqlite.connection import _clear_connection_cache

            try:
                _clear_connection_cache()
            except Exception:
                pass

    def _tracking_async_init(self: aiosqlite.Connection, *args: object, **kwargs: object) -> None:
        real_async_init(self, *args, **kwargs)  # type: ignore[arg-type]
        async_tracked.append(self)

    monkeypatch.setattr(sqlite3, "connect", _tracking_connect)
    monkeypatch.setattr(threading.Thread, "run", _thread_run)
    monkeypatch.setattr(aiosqlite.Connection, "__init__", _tracking_async_init)
    try:
        yield
    finally:
        monkeypatch.undo()
        if threading.get_ident() == main_ident:
            _close_current_thread()
        still_open = [c for c in async_tracked if getattr(c, "_connection", None) is not None]
        async_tracked.clear()
        if still_open:
            import asyncio

            async def _close_async() -> None:
                for conn in still_open:
                    try:
                        await conn.close()
                    except Exception:
                        pass

            try:
                asyncio.run(_close_async())
            except Exception:
                pass


@pytest.fixture(autouse=True)
def _clear_polylogue_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _reclaim_test_tmp_path: None,
    request: pytest.FixtureRequest,
) -> None:
    # Close any cached SQLite connections to prevent WAL sidecar corruption
    # when tests create/move/delete temp database files.
    from polylogue.storage.sqlite.connection import _clear_connection_cache

    _clear_connection_cache()

    # The daemon's structural-fault gate is intentionally process-local in
    # production, but tests simulate that fault routinely.  Reset it on both
    # sides of every test so a simulated mismatch cannot silently turn a later
    # live-ingest or demo test into an all-skipped batch.
    from polylogue.core.degraded import clear_degraded

    clear_degraded()
    request.addfinalizer(clear_degraded)

    # Clear search runtime state to prevent monkeypatched package-level search
    # adapters and cached results from leaking across tests.
    import sys

    from polylogue.storage.sqlite.connection import open_read_connection

    for module_name in ("polylogue.storage.search", "polylogue.storage.search.runtime"):
        search_module = sys.modules.get(module_name)
        search_module_any: Any = search_module
        cache = getattr(search_module_any, "search_messages_cached", None) if search_module is not None else None
        if cache is not None:
            cache.cache_clear()
        if module_name.endswith(".runtime") and search_module is not None:
            search_module_any.open_read_connection = open_read_connection

    # Clear schema-validator cache so monkeypatched registry/schema tests
    # cannot leak compiled validators into later integration runs.
    from polylogue.schemas.validator import SchemaValidator
    from polylogue.schemas.validator_resolution import reset_registry_cache

    SchemaValidator._cache.clear()
    reset_registry_cache()

    # polylogue-kmqwm: PROVIDERS' ``session_dir`` entries are computed once at
    # import time from ``Path.home()`` (e.g. ``~/.codex/sessions``,
    # ``~/.gemini/tmp``) -- long before any per-test XDG/env patching above
    # takes effect. Point every provider's session_dir at an inert per-test
    # tmp path so any residual schema-inference session-directory fallback
    # (opt-in via ``allow_session_dir_fallback``, see
    # ``polylogue/schemas/sampling.py``) can never read this operator's real
    # local session directories from inside the test suite, even from a test
    # that deliberately exercises the fallback path.
    from polylogue.schemas.observation_models import PROVIDERS

    harness_session_dir_root = tmp_path / "harness-session-dir-guard"
    for provider_config in PROVIDERS.values():
        if provider_config.session_dir is not None:
            monkeypatch.setattr(
                provider_config,
                "session_dir",
                harness_session_dir_root / provider_config.name.value,
                raising=False,
            )

    # Reset blob store singleton to prevent cross-test pollution when
    # tests write blobs to the temp XDG_DATA_HOME. Only reset if we're
    # about to set new XDG paths (i.e., after we've set up the env above).
    # Tests that create their own temp dirs will have their blobs written
    # to the correct location automatically since blob_store uses XDG_DATA_HOME.
    from polylogue.storage.blob_store import reset_blob_store

    reset_blob_store()

    # Reset the MCP runtime-services singleton. It is lazily built and cached
    # in ``polylogue.mcp.server_support._runtime_services`` the first time a
    # tool resolves config/archive paths; the cached services capture the
    # archive root that was active at build time. Tests that point
    # ``POLYLOGUE_ARCHIVE_ROOT`` at a seeded temp archive (e.g.
    # tests/integration/test_mcp.py) leave that cached services object behind,
    # so a later MCP tool call reads the polluter's archive instead of the
    # caller's isolated one. Resetting here forces a fresh build per test.
    from polylogue.mcp.server_support import _set_runtime_services

    _set_runtime_services(None)

    # Drop the cached daemon status snapshot. ``polylogue.daemon.status_snapshot``
    # holds a process-wide ``_SNAPSHOT`` singleton; tests that prime it via
    # ``refresh_status_snapshot(payload=...)`` (e.g. with a deliberately minimal
    # 3-key payload) otherwise leak it into later status-surface tests, whose
    # ``GET /api/status`` then reads the stale payload and fails the
    # required-keys / component_state contracts.
    from polylogue.daemon.status_snapshot import reset_status_snapshot

    reset_status_snapshot()

    # Strip every POLYLOGUE_* host env var so tests never inherit operator
    # configuration (archive root, daemon api host/port, validation mode,
    # notification webhook, etc.) from the developer host (#1325). A live
    # ``polylogued`` on the dev machine sets several of these and previously
    # caused the CLI under test to connect to the host daemon instead of the
    # fixture archive. Iterate ``os.environ`` so future POLYLOGUE_* additions
    # are stripped automatically.
    from tests.infra.schema_access import ALLOW_MISSING_SCHEMAS_ENV

    for key in list(os.environ):
        # ALLOW_MISSING_SCHEMAS_ENV is a test-only escape hatch (not operator
        # config) for lanes that intentionally run without packaged provider
        # schema data; it must survive this sweep or it could never take
        # effect inside the test suite that is its only consumer.
        if key.startswith("POLYLOGUE_") and key not in {ALLOW_MISSING_SCHEMAS_ENV, *_MANAGED_VERIFY_ENV}:
            monkeypatch.delenv(key, raising=False)

    for key in (
        # Prevent tests from hitting external Voyage API
        "VOYAGE_API_KEY",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    # Disable the site-wide config (``/etc/polylogue/polylogue.toml``) so that
    # operator-installed daemon settings (api host/port, archive root) cannot
    # bleed into tests. Empty string is the documented "disable" sentinel
    # honoured by :func:`polylogue.config._load_site_config` (#1325).
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", "")
    # Route the CLI status surface to an unreachable URL so that an operator
    # ``polylogued`` listening on the built-in ``127.0.0.1:8766`` cannot
    # respond to in-process or subprocess CLI invocations. Port 1 is reserved
    # (TCPMUX) and reliably refuses on a developer host (#1325).
    monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "http://127.0.0.1:1")


@pytest.fixture
def workspace_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_archive_template: Path,
) -> dict[str, Path]:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    archive_root = tmp_path / "archive"

    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    # Most tests using this fixture assert pipeline/query behavior, not schema
    # contract strictness. Keep validation deterministic and opt-in per test.
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")

    _clone_archive_template(empty_archive_template, archive_root)

    return {
        "archive_root": archive_root,
        "data_root": data_dir,
        "state_dir": state_dir,
    }


@pytest.fixture
def db_without_fts(tmp_path: Path) -> Path:
    """Database with schema but WITHOUT the FTS table (simulates fresh install)."""
    from polylogue.storage.sqlite.connection import open_connection

    db_path = tmp_path / "no_fts.db"
    with open_connection(db_path) as conn:
        # Drop the FTS table and all triggers to simulate fresh install state
        conn.execute("DROP TRIGGER IF EXISTS messages_fts_ai")
        conn.execute("DROP TRIGGER IF EXISTS messages_fts_au")
        conn.execute("DROP TRIGGER IF EXISTS messages_fts_ad")
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.commit()
    return db_path


@pytest.fixture
def storage_repository(workspace_env: dict[str, Path]) -> SessionRepository:
    """Storage repository with its own write lock.

    Use this fixture in tests that need thread-safe storage operations.
    The repository encapsulates the write lock and provides methods for
    saving sessions and recording runs.

    Depends on workspace_env to ensure XDG_DATA_HOME is set before
    creating the default backend.
    """
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.connection import create_default_backend

    backend = create_default_backend()
    return SessionRepository(backend=backend)


@pytest.fixture
def cli_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_archive_template: Path,
) -> dict[str, Path]:
    """
    Isolated CLI workspace with archive roots and database.

    Creates a complete test environment for CLI command testing:
    - Archive root directory
    - Data directory for database (XDG_DATA_HOME)
    - Inbox directory for test data
    - Pre-configured environment variables

    Returns:
        dict with paths: archive_root, data_root, inbox_dir, db_path
    """
    # Create directory structure
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    archive_root = tmp_path / "archive"
    inbox_dir = tmp_path / "inbox"
    render_root = archive_root / "render"

    for path in [data_dir, state_dir, archive_root, inbox_dir, render_root]:
        path.mkdir(parents=True, exist_ok=True)

    # The archive is the index database under the archive root. Seeding helpers
    # and the CLI read and write the same store, so the workspace db_path is the
    # archive root's index.db rather than a separate file.
    db_path = archive_root / "index.db"

    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")  # Plain output for tests
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")

    _clone_archive_template(empty_archive_template, archive_root)

    return {
        "archive_root": archive_root,
        "data_root": data_dir,
        "state_dir": state_dir,
        "inbox_dir": inbox_dir,
        "render_root": render_root,
        "db_path": db_path,
    }


def _clone_archive_template(source: Path, destination: Path) -> None:
    """Clone one immutable empty archive into a test-private workspace."""
    destination.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["cp", "-a", "--reflink=auto", f"{source}/.", str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        shutil.copytree(source, destination, dirs_exist_ok=True)

    bootstrap_marker = destination / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    if bootstrap_marker.is_file():
        from polylogue.storage.sqlite.durable_change_train import _record_fresh_durable_bootstrap

        bootstrap_marker.unlink()
        _record_fresh_durable_bootstrap(destination)


@pytest.fixture(scope="session")
def empty_archive_template(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
) -> Path:
    """Build a census-complete empty archive once per pytest run.

    A complete five-tier layout alone is no longer sufficient to claim raw
    materialization readiness: the raw-authority frontier must have a
    completed census too.  The shared fixture represents a usable empty
    archive, so establish that real durable state through the production
    census route before sharing clones with CLI and insight tests.
    """
    import fcntl

    from polylogue.config import Config
    from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    worker_base = tmp_path_factory.getbasetemp()
    run_root = worker_base.parent if worker_id != "master" else worker_base
    template = run_root / ".empty-archive-template"
    ready = run_root / ".empty-archive-template.ready"
    lock_path = run_root / ".empty-archive-template.lock"

    with lock_path.open("a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if ready.exists() and template.is_dir():
            return template

        building = run_root / f".empty-archive-template.building-{os.getpid()}"
        shutil.rmtree(building, ignore_errors=True)
        try:
            with ArchiveStore(building):
                pass
            inspect_raw_authority_frontier(
                Config(
                    archive_root=building,
                    render_root=building / "render",
                    sources=[],
                    db_path=building / "index.db",
                )
            )
            building.replace(template)
            ready.touch()
        finally:
            shutil.rmtree(building, ignore_errors=True)

    return template


@pytest.fixture
def mock_drive_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """
    Mock Google Drive OAuth credentials for testing.

    Creates mock credentials.json and token.json files and sets up
    environment variables to point to them.

    Returns:
        dict with paths: credentials_path, token_path, and MockCredentials instance
    """
    from tests.infra.drive_mocks import MockCredentials

    creds_dir = tmp_path / "drive_creds"
    creds_dir.mkdir(parents=True, exist_ok=True)

    # Create mock credentials.json (OAuth client config)
    creds_path = creds_dir / "credentials.json"
    creds_path.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "mock_client_id.apps.googleusercontent.com",
                    "client_secret": "mock_client_secret",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
        )
    )

    # Create mock token.json (OAuth access/refresh tokens)
    token_path = creds_dir / "token.json"
    mock_creds = MockCredentials()
    token_path.write_text(mock_creds.to_json())

    # Set environment variables
    monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(creds_path))
    monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))

    return {
        "credentials_path": creds_path,
        "token_path": token_path,
        "mock_credentials": mock_creds,
    }


@pytest.fixture
def mock_drive_service(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """
    Mock Google Drive service for testing.

    Patches google.auth and googleapiclient.discovery to return mock objects.

    Returns:
        dict with: service (MockDriveService), files (dict), file_content (dict)
    """
    from tests.infra.drive_mocks import MockCredentials, MockDriveService, mock_drive_file

    # Sample file data
    files_data = {
        "folder1": mock_drive_file(
            file_id="folder1",
            name="Google AI Studio",
            mime_type="application/vnd.google-apps.folder",
        ),
        "prompt1": mock_drive_file(
            file_id="prompt1",
            name="Test Prompt",
            mime_type="application/vnd.google-makersuite.prompt",
            parents=["folder1"],
        ),
    }

    file_content: dict[str, bytes | str] = {
        "prompt1": b'{"title": "Test Prompt", "content": "Test content"}',
    }

    mock_service = MockDriveService(files_data=files_data, file_content=file_content)
    mock_creds = MockCredentials()

    # Patch google.auth.default
    def mock_google_auth_default(*args: object, **kwargs: object) -> tuple[MockCredentials, None]:
        del args, kwargs
        return mock_creds, None

    # Patch googleapiclient.discovery.build
    def mock_discovery_build(
        service_name: str,
        version: str,
        credentials: object | None = None,
        *args: object,
        **kwargs: object,
    ) -> MockDriveService:
        del service_name, version, credentials, args, kwargs
        return mock_service

    # Patch google.auth.transport.requests.Request
    class MockRequest:
        pass

    monkeypatch.setattr("google.auth.default", mock_google_auth_default, raising=False)
    monkeypatch.setattr("googleapiclient.discovery.build", mock_discovery_build, raising=False)
    monkeypatch.setattr("google.auth.transport.requests.Request", MockRequest, raising=False)

    return {
        "service": mock_service,
        "files": files_data,
        "file_content": file_content,
        "credentials": mock_creds,
    }


@pytest.fixture
def mock_media_downloader(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """
    Patch GoogleAPI's MediaIoBaseDownload with our mock.

    This fixture patches the lazy import mechanism in drive_gateway.py
    to return our MockMediaIoBaseDownload when MediaIoBaseDownload is requested.
    """
    from tests.infra.drive_mocks import MockMediaIoBaseDownload

    # Patch the _import_module function to return mock for MediaIoBaseDownload
    class MockHttpModule(ModuleType):
        MediaIoBaseDownload: type[MockMediaIoBaseDownload]

    def mock_import_module(name: str) -> ModuleType:
        if name == "googleapiclient.http":
            mock_http = MockHttpModule("googleapiclient.http")
            mock_http.MediaIoBaseDownload = MockMediaIoBaseDownload
            return mock_http
        # Fall through to original for other modules
        import importlib

        return importlib.import_module(name)

    import polylogue.sources.drive.gateway as drive_gateway

    monkeypatch.setattr(drive_gateway, "_import_module", mock_import_module)

    return {"MockMediaIoBaseDownload": MockMediaIoBaseDownload}


# =============================================================================
# NEW FIXTURES FOR TEST CONSOLIDATION (added during aggressive parametrization)
# =============================================================================


@pytest.fixture
def db_path(workspace_env: Mapping[str, Path]) -> Path:
    """Shortcut fixture for database path setup.

    Usage in tests:
        def test_something(db_path):
            builder = SessionBuilder(db_path, "test-conv")
    """
    from tests.infra.storage_records import db_setup

    return db_setup(workspace_env)


@pytest.fixture
def session_builder(db_path: Path) -> Callable[[str], SessionBuilder]:
    """Fixture that provides SessionBuilder factory.

    Usage in tests:
        def test_something(session_builder):
            conv = (session_builder("test-conv")
                   .add_message("m1", text="Hello")
                   .save())
    """
    from tests.infra.storage_records import SessionBuilder

    def _builder(session_id: str = "test-conv") -> SessionBuilder:
        return SessionBuilder(db_path, session_id)

    return _builder


# =============================================================================
# CONSOLIDATED FIXTURES (moved from individual test files)
# =============================================================================


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Create an isolated test database with schema initialized.

    Replaces duplicate fixtures in: test_store.py, test_pipeline.py, test_lib.py,
    test_repository_render.py, test_search_index.py
    """
    from polylogue.storage.sqlite.connection import open_connection

    db_path = tmp_path / "test.db"
    with open_connection(db_path):
        pass  # Schema auto-initializes
    return db_path


@pytest.fixture
def test_conn(test_db: Path) -> Iterator[sqlite3.Connection]:
    """Provide a connection to the test database.

    Replaces duplicate fixtures in: test_store.py, test_pipeline.py, test_search_index.py
    """
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(test_db) as conn:
        yield conn


@pytest.fixture
async def sqlite_backend(tmp_path: Path) -> AsyncIterator[SQLiteBackend]:
    """Create a SQLite backend for testing."""

    from polylogue.storage.sqlite import SQLiteBackend
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    db_path = tmp_path / "index.db"
    backend = SQLiteBackend(db_path=db_path)
    yield backend
    await backend.close()


@pytest.fixture
def sample_session() -> Session:
    """Create a diverse session for filter/projection testing.

    Includes:
    - User messages (m1, m5)
    - Assistant messages (m2, m6 short, m7 substantive)
    - System message (m3)
    - Tool message (m4)
    - Mix of short/noise and substantive messages

    Replaces duplicate fixtures in: test_projections.py
    """
    messages = [
        make_msg(id="m1", role="user", text="User question", timestamp="2024-01-01T10:00:00"),
        make_msg(id="m2", role="assistant", text="Assistant response", timestamp="2024-01-01T10:01:00"),
        make_msg(id="m3", role="system", text="System prompt", timestamp="2024-01-01T10:02:00"),
        make_msg(id="m4", role="tool", text="Tool output", timestamp="2024-01-01T10:03:00"),
        make_msg(
            id="m5",
            role="user",
            text="Another question with searchterm here",
            timestamp="2024-01-01T10:04:00",
        ),
        make_msg(id="m6", role="assistant", text="ok", timestamp="2024-01-01T10:05:00"),
        make_msg(
            id="m7",
            role="assistant",
            text="Substantial answer with details",
            timestamp="2024-01-01T10:06:00",
        ),
    ]
    return make_conv(id="conv1", provider="test", messages=messages)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CliRunner for testing CLI commands.

    Replaces duplicate definitions across CLI test files.
    """
    from click.testing import CliRunner

    return CliRunner()


# =============================================================================
# SYNTHETIC RAW SAMPLES (replaces the old raw_db_samples fixture)
# =============================================================================


@pytest.fixture(scope="session")
def raw_synthetic_samples() -> list[RawSessionRecord]:
    """Generate synthetic raw session records for all providers.

    Uses SyntheticCorpus to generate wire-format bytes, wrapped in
    RawSessionRecord objects. This replaces the old raw_db_samples
    fixture that required a real database with imported data.

    Returns:
        List of RawSessionRecord objects (synthetic data, always available)
    """
    import hashlib
    from datetime import datetime, timezone

    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.storage.runtime import RawSessionRecord

    specs = build_default_corpus_specs(
        providers=SyntheticCorpus.available_providers(),
        count=5,
        seed=42,
        origin="generated.test-raw-samples",
        tags=("synthetic", "test", "raw-samples"),
    )

    samples: list[RawSessionRecord] = []
    for spec in specs:
        batch = SyntheticCorpus.generate_batch_for_spec(spec)
        for idx, raw_bytes in enumerate(batch.raw_items):
            raw_id = hashlib.sha256(raw_bytes).hexdigest()
            samples.append(
                RawSessionRecord(
                    raw_id=raw_id,
                    source_name=spec.provider,
                    source_path=f"<synthetic:{spec.provider}:{idx}>",
                    blob_size=len(raw_bytes),
                    acquired_at=datetime.now(timezone.utc).isoformat(),
                )
            )
    return samples


# =============================================================================
# SYNTHETIC SOURCE FIXTURE (replaces FIXTURES_DIR-based sources)
# =============================================================================


@pytest.fixture
def synthetic_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, int, range, int], Source]:
    """Factory fixture that generates synthetic Source objects for any provider.

    Writes SyntheticCorpus output to temp files, returning Source objects that
    can be fed through the full pipeline — just like the old fixture-based
    sources, but always available and schema-driven.

    Usage::

        def test_something(synthetic_source):
            source = synthetic_source("chatgpt")
            for convo in iter_source_sessions(source):
                ...

            # Multiple files:
            source = synthetic_source("claude-code", count=3)
    """
    from polylogue.config import Source
    from polylogue.schemas.synthetic import SyntheticCorpus
    from tests.infra.source_builders import SyntheticAntigravityLanguageServerClient

    def _factory(
        provider: str,
        count: int = 1,
        messages_per_session: range = range(4, 12),
        seed: int = 42,
    ) -> Source:
        spec = CorpusSpec.for_provider(
            provider,
            count=count,
            messages_min=messages_per_session.start,
            messages_max=messages_per_session.stop - 1,
            seed=seed,
            origin="generated.test-source",
            tags=("synthetic", "test", "source"),
        )
        provider_dir = tmp_path / "synthetic" / provider
        written = SyntheticCorpus.write_spec_artifacts(spec, provider_dir, prefix="synth")

        if provider == "antigravity":
            monkeypatch.setattr(
                "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
                SyntheticAntigravityLanguageServerClient,
            )
            return Source(name=provider, path=provider_dir)

        if count == 1:
            return Source(name=f"{provider}-test", path=written.files[0])
        # For multiple files, return Source pointing to first file
        # (pipeline processes individual files, not directories)
        return Source(name=f"{provider}-test", path=written.files[0])

    return _factory
