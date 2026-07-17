from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat as _stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Mapping
from pathlib import Path
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

# ---------------------------------------------------------------------------
# Place pytest temp directories on the NVMe scratch area by default.
# Test SQLite databases are write-heavy; /tmp lives on the root SSD on the
# operator workstation, while /realm/tmp is the high-write-budget scratch
# volume. /dev/shm remains available for explicit performance lanes, but it
# must not be the default: interrupted full/xdist runs can otherwise leave
# multi-GiB RAM-backed basetemps resident until reboot.
# ---------------------------------------------------------------------------
from polylogue.archive.models import Session
from polylogue.scenarios import CorpusSpec, build_default_corpus_specs
from polylogue.storage.runtime import RawSessionRecord
from tests.infra.builders import make_conv, make_msg

pytest_plugins = (
    "tests.infra.corpus_fixtures",
    "tests.infra.scale_fixtures",
    "tests.infra.frozen_clock",
)

if TYPE_CHECKING:
    from click.testing import CliRunner

    from polylogue.config import Source
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite import SQLiteBackend
    from tests.infra.storage_records import SessionBuilder

# ---------------------------------------------------------------------------
# Scale markers for data-gravity and long-haul validation (Workstream H)
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and choose the managed test temp root."""
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

    if config.option.basetemp is None:
        checkout = hashlib.sha1(str(config.rootpath).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        os.environ["POLYLOGUE_PYTEST_CHECKOUT"] = checkout
        run_id = os.environ.get("POLYLOGUE_PYTEST_RUN_ID")
        if not hasattr(config, "workerinput"):
            _sweep_stale_polylogue_basetemps()
        if run_id is None:
            run_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
            os.environ["POLYLOGUE_PYTEST_RUN_ID"] = run_id
        root, label = _managed_pytest_temp_root()
        config.option.basetemp = str(root / f"pytest-polylogue-{checkout}-{run_id}")
        sys.stderr.write(f"pytest: basetemp → {config.option.basetemp} ({label})\n")


# Per-run basetemps are freed on sessionfinish. A run killed before
# sessionfinish (SIGKILL, OOM) leaks its basetemp, so the controller reclaims
# clearly-dead orphans on startup. Seeded corpora (``pytest-polylogue-seeded-*``)
# are excluded because they are shared, reusable across runs, and bounded to
# one per checkout.
_STALE_BASETEMP_MAX_AGE_S = 30 * 60
_DEFAULT_SCRATCH_ROOT = Path("/realm/tmp/polylogue-pytest")


def _is_tmpfs(path: Path) -> bool:
    try:
        return path.is_dir() and bool(_stat.S_ISVTX & path.stat().st_mode)
    except OSError:
        return False


def _managed_pytest_temp_root() -> tuple[Path, str]:
    """Return the temp root for managed pytest basetemps."""

    configured = os.environ.get("POLYLOGUE_PYTEST_BASETEMP_ROOT")
    if configured:
        root = Path(configured)
        root.mkdir(parents=True, exist_ok=True)
        return root, "configured"

    shm = Path("/dev/shm")
    if os.environ.get("POLYLOGUE_PYTEST_TMPFS") == "1" and _is_tmpfs(shm):
        return shm, "tmpfs opt-in"

    try:
        _DEFAULT_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        _mark_btrfs_nocow(_DEFAULT_SCRATCH_ROOT)
    except OSError:
        fallback = Path("/tmp/polylogue-pytest")
        fallback.mkdir(parents=True, exist_ok=True)
        _mark_btrfs_nocow(fallback)
        return fallback, "disk fallback"
    return _DEFAULT_SCRATCH_ROOT, "scratch"


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
    roots = [_managed_pytest_temp_root()[0]]
    shm = Path("/dev/shm")
    if _is_tmpfs(shm):
        roots.append(shm)
    return tuple(dict.fromkeys(roots))


def _sweep_stale_polylogue_basetemps(
    *, max_age_s: int = _STALE_BASETEMP_MAX_AGE_S, roots: tuple[Path, ...] | None = None
) -> None:
    """Best-effort reclaim of per-run basetemps left by crashed runs."""

    cutoff = time.time() - max_age_s
    for root in roots or _polylogue_basetemp_roots():
        for entry in root.glob("pytest-polylogue-*"):
            if "-seeded-" in entry.name:
                continue
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
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
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    if not os.environ.get("POLYLOGUE_PYTEST_RUN_ID"):
        return
    numprocesses = getattr(session.config.option, "numprocesses", None)
    if numprocesses not in (None, 0, "0"):
        return
    basetemp = session.config.option.basetemp
    if not basetemp:
        return
    basetemp_path = Path(str(basetemp))
    if basetemp_path.name.startswith("pytest-polylogue-") and "-seeded-" not in basetemp_path.name:
        shutil.rmtree(basetemp_path, ignore_errors=True)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Retain the call outcome so passing test temp trees can be reclaimed."""
    report = yield
    setattr(item, f"rep_{report.when}", report)
    return report


@pytest.fixture(autouse=True)
def _reclaim_passing_test_tmp_path(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[None]:
    """Bound broad-run temp growth while preserving failed-test evidence."""
    yield
    report: pytest.TestReport | None = getattr(request.node, "rep_call", None)
    if report is not None and report.passed:
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


@pytest.fixture(autouse=True)
def _close_test_opened_sqlite_connections(
    monkeypatch: pytest.MonkeyPatch,
    _reclaim_passing_test_tmp_path: None,
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
    baseline, the embedding progress ledger, and the OTLP correlation reader.

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
    _reclaim_passing_test_tmp_path: None,
) -> None:
    # Close any cached SQLite connections to prevent WAL sidecar corruption
    # when tests create/move/delete temp database files.
    from polylogue.storage.sqlite.connection import _clear_connection_cache

    _clear_connection_cache()

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
    for key in list(os.environ):
        if key.startswith("POLYLOGUE_"):
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


@pytest.fixture(scope="session")
def empty_archive_template(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
) -> Path:
    """Build the empty five-tier archive once per pytest run, shared read-only."""
    import fcntl

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
def synthetic_source(tmp_path: Path) -> Callable[[str, int, range, int], Source]:
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

        if count == 1:
            return Source(name=f"{provider}-test", path=written.files[0])
        # For multiple files, return Source pointing to first file
        # (pipeline processes individual files, not directories)
        return Source(name=f"{provider}-test", path=written.files[0])

    return _factory
