"""Canonical SQLite connection profiles and factory functions shared by sync and async backends.

Factories
---------
``open_connection(path)`` returns a read-write connection with write pragmas applied.
``open_daemon_connection(path)`` returns a read-write connection with a smaller
daemon/ops cache profile.
``open_readonly_connection(path)`` returns a uri=ro connection with read pragmas applied.
``connection_context(path)`` is a context manager for a single-use read-write connection.

These are lightweight one-shot wrappers around ``sqlite3.connect()``.  For the
thread-local cached connection used by the async runtime, use the factories in
``connection.py`` instead.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from polylogue.logging import BoundLoggerLike


@dataclass(frozen=True, slots=True)
class SQLiteConnectionProfile:
    """SQLite timeout and PRAGMA profile for one connection role."""

    role: Literal["read", "write"]
    timeout_seconds: int
    busy_timeout_ms: int
    cache_size_kib: int
    mmap_size_bytes: int
    foreign_keys: bool = False
    journal_mode: str | None = None
    synchronous: str | None = None
    temp_store: str = "MEMORY"
    wal_autocheckpoint_pages: int | None = None
    journal_size_limit_bytes: int | None = None
    query_only: bool = False
    locking_mode: str | None = None

    @property
    def pragma_statements(self) -> tuple[str, ...]:
        statements: list[str] = []
        if self.foreign_keys:
            statements.append("PRAGMA foreign_keys = ON")
        if self.journal_mode is not None:
            statements.append(f"PRAGMA journal_mode={self.journal_mode}")
        statements.extend(
            (
                f"PRAGMA busy_timeout = {self.busy_timeout_ms}",
                f"PRAGMA cache_size = -{self.cache_size_kib}",
            )
        )
        if self.synchronous is not None:
            statements.append(f"PRAGMA synchronous = {self.synchronous}")
        statements.extend(
            (
                # Qualify the schema explicitly.  An unqualified mmap_size
                # pragma becomes the default for databases attached later,
                # charging every sibling tier against a budget that counts
                # this profile once.
                f"PRAGMA main.mmap_size = {self.mmap_size_bytes}",
                f"PRAGMA temp_store = {self.temp_store}",
            )
        )
        if self.wal_autocheckpoint_pages is not None:
            statements.append(f"PRAGMA wal_autocheckpoint = {self.wal_autocheckpoint_pages}")
        if self.journal_size_limit_bytes is not None:
            statements.append(f"PRAGMA journal_size_limit = {self.journal_size_limit_bytes}")
        if self.query_only:
            statements.append("PRAGMA query_only = ON")
        if self.locking_mode is not None:
            # Deliberately qualified to ``main``: an unqualified locking_mode
            # pragma also applies to every attached database (and becomes the
            # default for later ATTACHes), which would exclusively lock shared
            # durable tiers (user.db/ops.db) out from under concurrent readers.
            statements.append(f"PRAGMA main.locking_mode = {self.locking_mode}")
        return tuple(statements)


DB_TIMEOUT = 30
# Read busy_timeout. WAL readers normally don't block on a writer, but the
# brief window where a writer holds an exclusive lock (commit + TRUNCATE
# checkpoint) can exceed a second on a multi-GiB archive. A 1 s timeout turned
# that transient window into a hard "database is locked" error on interactive
# read surfaces (e.g. `polylogue find` during daemon ingest); 5 s lets the read
# wait out the checkpoint and succeed while staying far below the 30 s writer
# timeout, so reads remain responsive.
READ_DB_TIMEOUT = 5
MEMORY_BUDGET_ENV_VAR = "POLYLOGUE_MEMORY_BUDGET_BYTES"
DEFAULT_MEMORY_BUDGET_BYTES = 18 * 1024**3


def _read_declared_memory_budget_bytes() -> int:
    """Resolve the optional typed config/env budget, preserving current defaults."""
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().memory_budget_bytes
    return configured if configured is not None else DEFAULT_MEMORY_BUDGET_BYTES


MEMORY_BUDGET_BYTES = _read_declared_memory_budget_bytes()


def _scale_profile_size(default_size: int) -> int:
    """Scale one mmap/cache limit proportionally to the effective budget."""
    return max(1, round(default_size * MEMORY_BUDGET_BYTES / DEFAULT_MEMORY_BUDGET_BYTES))


# The measured defaults remain unchanged when no budget is configured. The
# service unit can export MEMORY_BUDGET_ENV_VAR from the same declared budget
# used for its cgroup limits, moving every SQLite mmap/cache allowance together.
WRITE_CACHE_SIZE_KIB = _scale_profile_size(131072)  # 128 MiB
DAEMON_WRITE_CACHE_SIZE_KIB = _scale_profile_size(16384)  # 16 MiB
READ_CACHE_SIZE_KIB = _scale_profile_size(32768)  # 32 MiB
WRITE_MMAP_SIZE_BYTES = _scale_profile_size(1073741824)  # 1 GiB
DAEMON_WRITE_MMAP_SIZE_BYTES = _scale_profile_size(67108864)  # 64 MiB
READ_MMAP_SIZE_BYTES = _scale_profile_size(134217728)  # 128 MiB
# The bounded FTS repair connection is opened separately from the daemon's
# ordinary writer and must remain inside the same process budget.
BOUNDED_REPAIR_CACHE_SIZE_KIB = _scale_profile_size(32768)  # 32 MiB
BOUNDED_REPAIR_MMAP_SIZE_BYTES = _scale_profile_size(134217728)  # 128 MiB
# Schema inference keeps its own WAL journal connection alive while it scans
# provider artifacts. It has no mmap allowance, only this page-cache limit.
OBSERVATION_JOURNAL_CACHE_SIZE_KIB = _scale_profile_size(65536)  # 64 MiB
WAL_AUTOCHECKPOINT_PAGES = 10000
# #1614: soft cap on the WAL file. After any checkpoint that frees
# pages, SQLite truncates the WAL down to this size. Without this cap
# the WAL grows unbounded when a TRUNCATE checkpoint is blocked by a
# long-running reader — the dogfood probe reproducibly grew it from
# ~750 MB to ~1 GB in 60 s during catch-up. 160 MiB = 4x the
# autocheckpoint threshold (40 MiB), so a healthy autocheckpoint
# cycle does not trip the limit but a reader-blocked WAL eventually
# hits it and shrinks on the next successful checkpoint.
WAL_JOURNAL_SIZE_LIMIT_BYTES = 160 * 1024 * 1024

WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="NORMAL",
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
    journal_size_limit_bytes=WAL_JOURNAL_SIZE_LIMIT_BYTES,
)

DAEMON_WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=DAEMON_WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=DAEMON_WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="NORMAL",
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
    journal_size_limit_bytes=WAL_JOURNAL_SIZE_LIMIT_BYTES,
)

# polylogue-623q: an owned INACTIVE index generation (bulk offline
# rebuild/backfill) is never read by anything until
# ``IndexGenerationStore.promote()`` swaps the ``index.db`` symlink, and is
# unconditionally discarded (``discard_if_inactive``) if the pass raises --
# see ``maintenance/rebuild_index.py``'s ``_rebuild_index_from_source_owned``.
# That licenses a much more aggressive durability/speed tradeoff than the
# live writer profile above, which must survive a crash mid-write against the
# ONE active index a concurrent reader may be using right now:
#   - ``journal_mode=MEMORY`` (not WAL, not OFF): keeps the rollback journal
#     resident in RAM instead of round-tripping through the filesystem/WAL
#     checkpoint machinery, but still gives ``sqlite3.Connection.rollback()``
#     something to roll back to. ``revision_backfill.py``'s batched
#     census/replay loops call ``archive.rollback()`` on a recoverable batch
#     failure and re-processes that batch -- ``journal_mode=OFF`` disables
#     the rollback journal entirely, so that call would silently no-op and
#     the retry could double-apply against already-partially-written rows.
#     MEMORY is the fastest mode that keeps this real, already-exercised
#     recovery path correct.
#   - ``synchronous=OFF``: no fsync at all. A host crash mid-build can leave
#     ``index.db`` corrupt, but a corrupt INACTIVE generation is simply
#     discarded and rebuilt -- never promoted, never read.
#   - A much larger ``cache_size``/``mmap_size`` than even the live writer
#     profile: a bulk rebuild's working set (the whole generation being
#     built) is far larger than one incremental daemon write, and there is no
#     competing live-writer cgroup budget to share (this is a throwaway,
#     single-purpose process).
BULK_BUILD_CACHE_SIZE_KIB = _scale_profile_size(524288)  # 512 MiB
BULK_BUILD_MMAP_SIZE_BYTES = _scale_profile_size(4294967296)  # 4 GiB

BULK_BUILD_WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=BULK_BUILD_CACHE_SIZE_KIB,
    mmap_size_bytes=BULK_BUILD_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="MEMORY",
    synchronous="OFF",
    # An owned inactive generation has exactly one writer and zero readers
    # until promoted, so per-transaction lock acquisition/release syscall
    # churn is pure waste. EXCLUSIVE holds the file lock for the connection
    # lifetime. The promote path closes this connection before the pointer
    # swap, so the exclusive hold never outlives the build.
    locking_mode="EXCLUSIVE",
)

READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=READ_DB_TIMEOUT,
    busy_timeout_ms=READ_DB_TIMEOUT * 1000,
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    # #1614: explicit read-only signal. ``open_readonly_connection``
    # opens with the ``mode=ro`` URI flag which is already enforced
    # by SQLite at the file level, but the pragma additionally
    # rejects accidental writes via the same connection at SQL parse
    # time instead of waiting for the write lock.
    query_only=True,
)

DAEMON_WRITE_CONNECTION_PRAGMA_STATEMENTS = DAEMON_WRITE_CONNECTION_PROFILE.pragma_statements
WRITE_CONNECTION_PRAGMA_STATEMENTS = WRITE_CONNECTION_PROFILE.pragma_statements
READ_CONNECTION_PRAGMA_STATEMENTS = READ_CONNECTION_PROFILE.pragma_statements
BULK_BUILD_WRITE_CONNECTION_PRAGMA_STATEMENTS = BULK_BUILD_WRITE_CONNECTION_PROFILE.pragma_statements


# ---------------------------------------------------------------------------
# Mapped-bytes budget vs. the cgroup memory limit (polylogue-e98k)
# ---------------------------------------------------------------------------
#
# 2026-07-31 incident: the mmap/cache profile sizes above (this file) and the
# systemd cgroup limits (sinnix repo, `modules/services/polylogue.nix`) were
# picked independently, in two different repos, with nothing tying them
# together. A 4 GiB `BULK_BUILD_MMAP_SIZE_BYTES` window over a 38 GB
# `index.db` fills completely under any scan-heavy work; one bulk-build
# connection alone therefore accounted for ~4.5 GiB against a 6 GiB
# `MemoryHigh` ceiling, leaving no headroom for the daemon's own writer and
# concurrent readers. `MemoryHigh` was the wrong instrument for reclaimable,
# file-backed mmap'd pages (it throttles anon growth; mapped pages just
# evict-and-refault under pressure -- the observed `slow_write` signature),
# not a leak, so pinning at the ceiling was structurally guaranteed rather
# than a bug in either repo. A runtime bump to `MemoryHigh=14G` stopped the
# throttling dead; that finding is now the committed default
# (`MemoryHigh=14G` / `MemoryMax=18G`) -- see the comment beside that
# override for the measurement. `mapped_bytes_budget` below is the
# mechanical anchor so nobody has to re-derive that arithmetic from scratch
# next time either side's constants move: the sinnix `MemoryMax`/`MemoryHigh`
# override for `polylogued.service` must stay comfortably above the value
# this returns, and `check_mapped_bytes_budget_against_cgroup_limit` makes
# that comparison observable at runtime instead of only discoverable hours
# into an incident.
#
# `mmap_size` is an upper bound SQLite MAY map into, never a guaranteed
# allocation, so `mapped_bytes_budget()` is a conservative ceiling estimate,
# not a live-RSS prediction -- it will typically overstate actual usage.


def mapped_bytes_budget(*, concurrent_read_connections: int = 4) -> int:
    """Plausible peak concurrent SQLite mmap+cache footprint for one polylogued process.

    Models the worst case that actually bit us: one bulk-build connection
    (an offline `polylogue ops maintenance rebuild-index`, or a
    daemon-triggered bulk rebuild via `daemon/bulk_rebuild.py`) running
    concurrently with the daemon's own long-lived write connection
    (`DAEMON_WRITE_CONNECTION_PROFILE`) and a handful of concurrent
    short-lived read connections (CLI/MCP/API reads against the live
    archive while a rebuild is in flight), plus one ordinary writer, one
    bounded FTS repair connection, and one schema-observation journal
    connection. This is a conservative upper bound across the production
    profiles, including one-shot maintenance/CLI writers, so cgroup allowance
    does not depend on an assumed lifecycle ordering.

    `concurrent_read_connections` defaults to 4: a conservative but not
    extreme estimate of simultaneous interactive reads (CLI/MCP/API) during
    a bulk rebuild. Callers with better knowledge of their own concurrency
    (e.g. a fixed MCP worker pool size) may override it.
    """
    return (
        BULK_BUILD_MMAP_SIZE_BYTES
        + BULK_BUILD_CACHE_SIZE_KIB * 1024
        + WRITE_MMAP_SIZE_BYTES
        + WRITE_CACHE_SIZE_KIB * 1024
        + DAEMON_WRITE_MMAP_SIZE_BYTES
        + DAEMON_WRITE_CACHE_SIZE_KIB * 1024
        + concurrent_read_connections * (READ_MMAP_SIZE_BYTES + READ_CACHE_SIZE_KIB * 1024)
        + BOUNDED_REPAIR_MMAP_SIZE_BYTES
        + BOUNDED_REPAIR_CACHE_SIZE_KIB * 1024
        + OBSERVATION_JOURNAL_CACHE_SIZE_KIB * 1024
    )


@dataclass(frozen=True, slots=True)
class MappedBytesBudgetCheck:
    """Result of comparing :func:`mapped_bytes_budget` to the detected cgroup limits."""

    budget_bytes: int
    memory_max_bytes: int | None
    memory_high_bytes: int | None
    concurrent_read_connections: int
    memory_budget_bytes: int | None = None

    @property
    def budget_mb(self) -> float:
        return round(self.budget_bytes / (1024 * 1024), 1)

    @property
    def effective_memory_budget_bytes(self) -> int:
        return self.memory_budget_bytes if self.memory_budget_bytes is not None else MEMORY_BUDGET_BYTES

    @property
    def memory_budget_mb(self) -> float:
        return round(self.effective_memory_budget_bytes / (1024 * 1024), 1)

    @property
    def concurrent_read_budget_bytes(self) -> int:
        return self.concurrent_read_connections * (READ_MMAP_SIZE_BYTES + READ_CACHE_SIZE_KIB * 1024)

    @property
    def concurrent_profile_budget_bytes(self) -> int:
        return (
            BULK_BUILD_MMAP_SIZE_BYTES
            + BULK_BUILD_CACHE_SIZE_KIB * 1024
            + WRITE_MMAP_SIZE_BYTES
            + WRITE_CACHE_SIZE_KIB * 1024
            + DAEMON_WRITE_MMAP_SIZE_BYTES
            + DAEMON_WRITE_CACHE_SIZE_KIB * 1024
            + BOUNDED_REPAIR_MMAP_SIZE_BYTES
            + BOUNDED_REPAIR_CACHE_SIZE_KIB * 1024
            + OBSERVATION_JOURNAL_CACHE_SIZE_KIB * 1024
        )

    @property
    def concurrent_allowance_bytes(self) -> int:
        return self.concurrent_read_budget_bytes + self.concurrent_profile_budget_bytes

    @property
    def memory_max_mb(self) -> float | None:
        return round(self.memory_max_bytes / (1024 * 1024), 1) if self.memory_max_bytes is not None else None

    @property
    def memory_high_mb(self) -> float | None:
        return round(self.memory_high_bytes / (1024 * 1024), 1) if self.memory_high_bytes is not None else None

    @property
    def at_risk_limits(self) -> tuple[str, ...]:
        """Which cgroup limit file(s), if any, sit at or below the computed budget.

        Either limit landing at or below the budget reproduces the 2026-07-31
        incident shape: `memory.high` throttles mapped/reclaimable pages before
        `memory.max` would ever OOM-kill, so `memory.high` is actually the
        more precise reproduction of what happened -- but a `memory.max` this
        low is also worth flagging, since it means the hard ceiling itself
        cannot even hold one worst-case concurrent footprint.
        """
        at_risk: list[str] = []
        if self.memory_max_bytes is not None and self.memory_max_bytes <= self.budget_bytes:
            at_risk.append("memory.max")
        if self.memory_high_bytes is not None and self.memory_high_bytes <= self.budget_bytes:
            at_risk.append("memory.high")
        return tuple(at_risk)


def check_mapped_bytes_budget_against_cgroup_limit(*, concurrent_read_connections: int = 4) -> MappedBytesBudgetCheck:
    """Compare the computed mmap/cache budget to this process' cgroup v2 memory limits.

    Reads `memory.max`/`memory.high` under `/sys/fs/cgroup/<this process' unified
    cgroup path>` via `polylogue.core.metrics`. Both are `None` when cgroup v2
    is not mounted, the controller isn't delegated (e.g. outside a cgroup, or a
    container without the `memory` controller), or the limit is literally
    `max` (unlimited) -- callers must treat `None` as "no limit detected", not
    as an error.
    """
    from polylogue.core.metrics import read_cgroup_memory_high_bytes, read_cgroup_memory_max_bytes

    return MappedBytesBudgetCheck(
        budget_bytes=mapped_bytes_budget(concurrent_read_connections=concurrent_read_connections),
        memory_max_bytes=read_cgroup_memory_max_bytes(),
        memory_high_bytes=read_cgroup_memory_high_bytes(),
        concurrent_read_connections=concurrent_read_connections,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
    )


def log_mapped_bytes_budget_check(logger: BoundLoggerLike, check: MappedBytesBudgetCheck | None = None) -> None:
    """Log the mapped-bytes budget vs. detected cgroup memory limit at startup.

    Call once at daemon startup and once at the start of an offline bulk
    rebuild -- the two paths that can hold a `BULK_BUILD_WRITE_CONNECTION_PROFILE`
    connection. Degrades gracefully (a debug-level line, never a raised
    exception) when no cgroup limit is detected at all, since that is the
    ordinary case for a dev-machine or non-cgroup-confined run, not an error.
    """
    if check is None:
        check = check_mapped_bytes_budget_against_cgroup_limit()
    if check.memory_max_bytes is None and check.memory_high_bytes is None:
        logger.debug(
            "mmap_budget_no_cgroup_limit_detected",
            memory_budget_bytes=check.effective_memory_budget_bytes,
            memory_budget_mb=check.memory_budget_mb,
            budget_bytes=check.budget_bytes,
            budget_mb=check.budget_mb,
            concurrent_allowance_bytes=check.concurrent_allowance_bytes,
            concurrent_read_budget_bytes=check.concurrent_read_budget_bytes,
            concurrent_profile_budget_bytes=check.concurrent_profile_budget_bytes,
            concurrent_read_connections=check.concurrent_read_connections,
        )
        return
    at_risk = check.at_risk_limits
    if at_risk:
        logger.warning(
            "mmap_budget_at_or_above_cgroup_limit",
            memory_budget_bytes=check.effective_memory_budget_bytes,
            memory_budget_mb=check.memory_budget_mb,
            budget_bytes=check.budget_bytes,
            budget_mb=check.budget_mb,
            concurrent_allowance_bytes=check.concurrent_allowance_bytes,
            concurrent_read_budget_bytes=check.concurrent_read_budget_bytes,
            concurrent_profile_budget_bytes=check.concurrent_profile_budget_bytes,
            memory_max_mb=check.memory_max_mb,
            memory_high_mb=check.memory_high_mb,
            at_risk_limits=list(at_risk),
            concurrent_read_connections=check.concurrent_read_connections,
        )
    else:
        logger.info(
            "mmap_budget_within_cgroup_limit",
            memory_budget_bytes=check.effective_memory_budget_bytes,
            memory_budget_mb=check.memory_budget_mb,
            budget_bytes=check.budget_bytes,
            budget_mb=check.budget_mb,
            concurrent_allowance_bytes=check.concurrent_allowance_bytes,
            concurrent_read_budget_bytes=check.concurrent_read_budget_bytes,
            concurrent_profile_budget_bytes=check.concurrent_profile_budget_bytes,
            memory_max_mb=check.memory_max_mb,
            memory_high_mb=check.memory_high_mb,
            concurrent_read_connections=check.concurrent_read_connections,
        )


# ---------------------------------------------------------------------------
# Lightweight factory functions — open + apply pragmas, no caching / schema / vec
# ---------------------------------------------------------------------------


_SIBLING_TIER_ATTACHMENTS: tuple[tuple[str, str], ...] = (
    ("source_tier", "source.db"),
    ("user_tier", "user.db"),
    ("embeddings", "embeddings.db"),
    ("ops_tier", "ops.db"),
)


def _attach_sibling_tiers(conn: sqlite3.Connection) -> None:
    """Attach sibling archive tiers to an ``index.db`` connection (idempotent).

    Lets one-shot sync connections resolve cross-tier tables (e.g. source.db's
    ``raw_sessions``/``blob_refs``) by unqualified name. SQLite resolves
    unqualified names to ``main`` first, so index-tier tables are unaffected;
    only sibling-only tables resolve to their attached tier.
    """
    main_path: str | None = None
    attached: set[str] = set()
    for row in conn.execute("PRAGMA database_list").fetchall():
        schema_name = str(row[1])
        if schema_name == "main":
            main_path = str(row[2]) if row[2] else None
        else:
            attached.add(schema_name)
    if not main_path:
        return
    main = Path(main_path)
    if main.name != "index.db":
        return
    root = main.parent
    for schema_name, filename in _SIBLING_TIER_ATTACHMENTS:
        if schema_name in attached:
            continue
        sibling = root / filename
        if sibling.exists():
            conn.execute(f"ATTACH DATABASE ? AS {schema_name}", (str(sibling),))


def open_connection(path: str | Path, *, timeout: float = DB_TIMEOUT) -> sqlite3.Connection:
    """Open a read-write SQLite connection with canonical write pragmas applied.

    This is a lightweight one-shot factory: it opens the file, applies the
    write-time PRAGMA profile, attaches sibling archive tiers (so cross-tier
    reads resolve), and returns the connection.  The caller owns the connection
    lifecycle (must close it).

    For the thread-local cached archive connection used by the async runtime,
    use ``connection_context`` from ``connection.py`` instead.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        for stmt in WRITE_CONNECTION_PRAGMA_STATEMENTS:
            conn.execute(stmt)
        _attach_sibling_tiers(conn)
    except BaseException:
        # A pragma can fail (e.g. a WAL-mode write pragma against a
        # lock-held database). Close the just-opened connection before
        # propagating so it is not orphaned by the caller's ``with``/``closing``.
        conn.close()
        raise
    return conn


def open_daemon_connection(
    path: str | Path,
    *,
    timeout: float = DB_TIMEOUT,
    busy_timeout_ms: int | None = None,
) -> sqlite3.Connection:
    """Open a read-write SQLite connection for daemon maintenance/ops writes.

    Long-running daemon loops write small status, cursor, telemetry, and
    maintenance rows. They should not inherit the full batch-ingest cache and
    mmap profile, because systemd charges their SQLite page cache to the
    service cgroup for the lifetime of the process.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        for stmt in DAEMON_WRITE_CONNECTION_PRAGMA_STATEMENTS:
            if busy_timeout_ms is not None and stmt.startswith("PRAGMA busy_timeout"):
                stmt = f"PRAGMA busy_timeout = {busy_timeout_ms}"
            conn.execute(stmt)
        _attach_sibling_tiers(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def descriptor_alias_path(opened_fd: int) -> Path | None:
    """Return a validated portable pathname alias for an opened descriptor."""

    descriptor_metadata = os.fstat(opened_fd)
    for directory in ("/dev/fd", "/proc/self/fd"):
        candidate = Path(directory) / str(opened_fd)
        try:
            alias_metadata = os.stat(candidate)
        except OSError:
            continue
        if (alias_metadata.st_dev, alias_metadata.st_ino) == (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
        ):
            return candidate
    return None


def _descriptor_database_uri(opened_main_fd: int, suffix: str) -> str | None:
    """Return a validated descriptor URI on platforms that expose one."""
    alias = descriptor_alias_path(opened_main_fd)
    return None if alias is None else f"file:{alias}{suffix}"


def open_readonly_connection(
    path: str | Path,
    *,
    timeout: float = READ_DB_TIMEOUT,
    immutable: bool = False,
    opened_main_fd: int | None = None,
) -> sqlite3.Connection:
    """Open a read-only SQLite connection with canonical read pragmas applied.

    Uses ``file:...?mode=ro`` URI mode to guarantee no write locks are taken.
    Returns ``None`` / raises ``sqlite3.OperationalError`` if the database file
    does not exist.

    ``immutable`` additionally sets SQLite's ``immutable=1`` URI parameter,
    which tells SQLite the file is guaranteed not to change for the lifetime
    of the connection: it skips locking and WAL/journal presence checks, and
    will not create a ``-shm``/``-wal`` sidecar itself. This is only correct
    against a verified-stable snapshot (e.g. a stopped-daemon clone the caller
    has already confirmed has no WAL/SHM/journal sidecars) -- never against a
    database a live process (such as ``polylogued``) might still be writing.
    Callers passing ``immutable=True`` own that precondition check; this
    helper does not perform it, since the check is specific to how the caller
    obtained the snapshot.

    When ``opened_main_fd`` is supplied, the reader is bound to that opened
    inode through a validated ``/dev/fd`` or ``/proc/self/fd`` alias. A caller
    that needs descriptor binding fails closed when neither alias is available.
    """
    suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    if opened_main_fd is not None and immutable:
        raise ValueError("an opened SQLite file descriptor cannot use immutable mode")
    opened_fd = opened_main_fd
    if opened_fd is None:
        database_uri = f"file:{path}{suffix}"
    else:
        descriptor_uri = _descriptor_database_uri(opened_fd, suffix)
        if descriptor_uri is None:
            raise RuntimeError(f"cannot open selected SQLite database through a descriptor-bound path: {path}")
        database_uri = descriptor_uri
    conn = sqlite3.connect(database_uri, uri=True, timeout=timeout)
    try:
        for stmt in READ_CONNECTION_PRAGMA_STATEMENTS:
            conn.execute(stmt)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def connection_context(path: str | Path, *, timeout: float = DB_TIMEOUT) -> Iterator[sqlite3.Connection]:
    """Context manager for a single-use read-write connection.

    Opens a connection with write pragmas, yields it, and closes on exit.
    """
    conn = open_connection(path, timeout=timeout)
    try:
        yield conn
    finally:
        conn.close()


__all__ = [
    "DB_TIMEOUT",
    "DEFAULT_MEMORY_BUDGET_BYTES",
    "BOUNDED_REPAIR_CACHE_SIZE_KIB",
    "BOUNDED_REPAIR_MMAP_SIZE_BYTES",
    "BULK_BUILD_CACHE_SIZE_KIB",
    "BULK_BUILD_MMAP_SIZE_BYTES",
    "BULK_BUILD_WRITE_CONNECTION_PRAGMA_STATEMENTS",
    "BULK_BUILD_WRITE_CONNECTION_PROFILE",
    "DAEMON_WRITE_CACHE_SIZE_KIB",
    "DAEMON_WRITE_CONNECTION_PRAGMA_STATEMENTS",
    "DAEMON_WRITE_CONNECTION_PROFILE",
    "DAEMON_WRITE_MMAP_SIZE_BYTES",
    "MEMORY_BUDGET_BYTES",
    "MEMORY_BUDGET_ENV_VAR",
    "MappedBytesBudgetCheck",
    "OBSERVATION_JOURNAL_CACHE_SIZE_KIB",
    "READ_CACHE_SIZE_KIB",
    "READ_CONNECTION_PRAGMA_STATEMENTS",
    "READ_CONNECTION_PROFILE",
    "READ_DB_TIMEOUT",
    "READ_MMAP_SIZE_BYTES",
    "SQLiteConnectionProfile",
    "WAL_AUTOCHECKPOINT_PAGES",
    "WRITE_CACHE_SIZE_KIB",
    "WRITE_CONNECTION_PRAGMA_STATEMENTS",
    "WRITE_CONNECTION_PROFILE",
    "WRITE_MMAP_SIZE_BYTES",
    "check_mapped_bytes_budget_against_cgroup_limit",
    "connection_context",
    "descriptor_alias_path",
    "log_mapped_bytes_budget_check",
    "mapped_bytes_budget",
    "open_daemon_connection",
    "open_connection",
    "open_readonly_connection",
]
