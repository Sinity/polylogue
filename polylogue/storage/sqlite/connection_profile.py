"""Canonical SQLite connection profiles and factory functions shared by sync and async backends.

Factories
---------
``open_connection(path)`` returns a read-write connection with write pragmas applied.
``open_daemon_connection(path)`` returns a read-write connection with a smaller
daemon/ops cache profile.
``open_readonly_connection(path)`` returns a uri=ro connection with read pragmas applied.
Pass ``validate_schema=False`` only for diagnostic readers that must inspect a
stale tier and report its version.
``connection_context(path)`` is a context manager for a single-use read-write connection.

These are lightweight one-shot wrappers around ``sqlite3.connect()``.  For the
thread-local cached connection used by the async runtime, use the factories in
``connection.py`` instead.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import quote

from polylogue.storage.sqlite.write_lease import require_write_lease

if TYPE_CHECKING:
    from polylogue.logging import BoundLoggerLike
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


SCRATCH_SYNCHRONOUS_ENV = "POLYLOGUE_SQLITE_SYNCHRONOUS"


def scratch_synchronous_override() -> str | None:
    """``POLYLOGUE_SQLITE_SYNCHRONOUS=OFF`` drops fsync for throwaway archives.

    The test harness sets it: every archive under a pytest scratch tree is
    deleted seconds after it is written, and on a copy-on-write filesystem
    its fsyncs are the dominant disk load of a run. Only ``OFF`` is honoured
    and it applies to every profile that syncs at all; the value is read
    when statements are built, so it must be set before this module loads.
    """
    value = os.environ.get(SCRATCH_SYNCHRONOUS_ENV, "")
    return "OFF" if value.strip().upper() == "OFF" else None


@dataclass(frozen=True, slots=True)
class SQLiteConnectionProfile:
    """SQLite timeout and PRAGMA profile for one connection role."""

    role: Literal["read", "write"]
    timeout_seconds: float
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
    generation_identity: Literal["live", "sealed"] = "live"
    immutable: bool = False
    max_snapshot_age_s: float | None = None
    cancellation_supported: bool = False

    @property
    def pragma_statements(self) -> tuple[str, ...]:
        statements: list[str] = []
        synchronous = scratch_synchronous_override() or self.synchronous
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
        if synchronous is not None:
            statements.append(f"PRAGMA synchronous = {synchronous}")
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

# The four named lock-wait classes. ``interactive-read`` is READ_DB_TIMEOUT
# above: short enough that a stuck read surfaces rather than hangs. The other
# three wait out a full writer hold rather than fail a job a retry would only
# repeat, so they sit at the writer's own busy timeout.
TIMEOUT_CLASS_BACKGROUND_READ_S = 30.0
TIMEOUT_CLASS_PUBLICATION_S = 30.0
TIMEOUT_CLASS_OFFLINE_BULK_S = 30.0

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
OWNED_WAL_AUTOCHECKPOINT_PAGES = 0
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

# An owned INACTIVE index generation is never read by anything until
# ``IndexGenerationStore.promote()`` swaps the ``index.db`` symlink, and is
# unconditionally discarded (``discard_if_inactive``) if the pass raises.
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


# polylogue-6xcqj: the cold-build shape for the ACTIVE index generation, held
# under the single-writer lease and proven empty before this profile is used.
#
# index.db is rebuildable, and an empty active generation has nothing a crash
# could lose that a restart would not simply re-derive from source.db, so the
# durability levers of the bulk-build profile apply by the same argument:
#   - ``synchronous=OFF``: no fsync per commit (measured ~15% of a cold build).
#   - a raised autocheckpoint threshold: a cold build commits constantly, and
#     an autocheckpoint inside a 256 MiB catch-up page charges its whole WAL
#     copy-back to whichever commit crossed the threshold.
#
# Foreign-key enforcement stays ON. Turning it off would need a verification
# pass at a boundary, and this shape has no boundary that may mutate the
# connection (see ``ArchiveStore.finish_active_cold_build``). Keeping it on
# means the cold shape relaxes durability only, and cannot change what a pass
# writes, defers or refuses -- which is the property that makes it safe to
# select automatically on the live route.
#
# What is deliberately NOT taken from ``BULK_BUILD_WRITE_CONNECTION_PROFILE``:
# ``journal_mode=MEMORY`` and ``locking_mode=EXCLUSIVE``. The active generation
# is read concurrently by the CLI, MCP and the daemon's own readers, and both
# of those would either lock them out or remove the WAL they read through.
# Those two -- and dropping reader indexes, which a read-only open reports as a
# schema manifest mismatch -- belong to an owned inactive generation.
COLD_BUILD_ACTIVE_WAL_AUTOCHECKPOINT_PAGES = 200_000

COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    # The live writer's cache and mmap budget, NOT the bulk build's. The bulk
    # profile's 512 MiB / 4 GiB window is sized for a throwaway single-purpose
    # process that owns the machine; this connection is the daemon's own live
    # writer, sharing a cgroup budget with its readers (see the mapped-bytes
    # note below). Measured 2026-09-16 on a 500-file synthetic cold build
    # through the dispatcher: the bulk sizes cost 433 MiB peak RSS against the
    # live profile's 261 MiB, for a shape whose window on this route is a
    # single intake page.
    cache_size_kib=WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="OFF",
    wal_autocheckpoint_pages=COLD_BUILD_ACTIVE_WAL_AUTOCHECKPOINT_PAGES,
    journal_size_limit_bytes=WAL_JOURNAL_SIZE_LIMIT_BYTES,
)

# What a live-generation reader pins is its *open read transaction*, not its
# connection. Two measurements on a synthetic WAL archive with
# ``wal_autocheckpoint=0`` (what the armed recurring owner leaves every daemon
# writer) and a 512-byte row payload:
#
#   40k rows written, then one PASSIVE. No reader and an idle ``mode=ro``
#   connection both checkpointed 6079 of 6079 frames; a reader holding one
#   lazily stepped cursor checkpointed 0 of 6079.
#
#   Eight bursts of 5k rows with one PASSIVE after each. Behind the idle
#   connection the WAL plateaued (2,962,312 -> 2,978,792 bytes); behind the
#   stepped cursor it grew monotonically every burst, 2,962,312 -> 23,776,552
#   bytes -- 8.0x, the full concurrent write volume, for the transaction's
#   whole lifetime.
#
# So the bound that matters is on how long a read transaction may stay open, and
# ``ReadFrame.stream`` below is the route that applies it per row. Every live
# read profile declares that maximum age; a sealed generation cannot change
# under a reader and declares none.
INTERACTIVE_READ_SNAPSHOT_AGE_S = 30.0
BACKGROUND_READ_SNAPSHOT_AGE_S = 300.0

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
    generation_identity="live",
    max_snapshot_age_s=INTERACTIVE_READ_SNAPSHOT_AGE_S,
    cancellation_supported=True,
)

BACKGROUND_READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=TIMEOUT_CLASS_BACKGROUND_READ_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_BACKGROUND_READ_S * 1000),
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    query_only=True,
    generation_identity="live",
    max_snapshot_age_s=BACKGROUND_READ_SNAPSHOT_AGE_S,
    cancellation_supported=True,
)

OFFLINE_BULK_READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=TIMEOUT_CLASS_OFFLINE_BULK_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_OFFLINE_BULK_S * 1000),
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    query_only=True,
    generation_identity="live",
    max_snapshot_age_s=BACKGROUND_READ_SNAPSHOT_AGE_S,
    cancellation_supported=True,
)

# SQLite's ``immutable=1`` skips locking and WAL/journal detection outright, so
# it is a claim about the generation rather than about the caller's intent: it
# is correct only where nothing can still write the file. This is the one
# profile that carries it, and ``open_readonly_connection`` selects this profile
# whenever a caller asks for immutability, so the two cannot drift apart.
SEALED_READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=TIMEOUT_CLASS_OFFLINE_BULK_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_OFFLINE_BULK_S * 1000),
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    query_only=True,
    generation_identity="sealed",
    immutable=True,
    max_snapshot_age_s=None,
    cancellation_supported=True,
)

# Historical continuity classification needs SQLite's connection-local TEMP
# relations for its bounded candidate stream.  ``query_only`` rejects TEMP
# writes as well as durable writes, so this deliberately private profile is
# not part of ``READ_PROFILES``: the factory below is the only route that can
# use it.  The URI's ``mode=ro&immutable=1`` still makes the authenticated
# main database immutable; only SQLite's private TEMP schema is writable.
SEALED_STAGING_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=TIMEOUT_CLASS_OFFLINE_BULK_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_OFFLINE_BULK_S * 1000),
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    temp_store="MEMORY",
    query_only=False,
    generation_identity="sealed",
    immutable=True,
    max_snapshot_age_s=None,
    cancellation_supported=True,
)


# This is intentionally a small, positive allowlist.  The liveness and legacy
# hook matcher need these aggregate/scalar functions plus the registered UDF;
# all other function calls, including ``load_extension``, are refused.
_SEALED_STAGING_FUNCTIONS = frozenset(
    {
        "coalesce",
        "count",
        "min",
        "sum",
        "polylogue_deterministic_raw_session_id",
    }
)
_SEALED_STAGING_READ_PRAGMAS = frozenset(
    {"data_version", "query_only", "schema_version", "table_info", "temp_store", "user_version"}
)


def _authorize_sealed_staging_operation(
    action: int,
    argument1: str | None,
    argument2: str | None,
    database: str | None,
    _trigger: str | None,
) -> int:
    """Allow only liveness reads and connection-local TEMP staging.

    SQLite invokes this callback while compiling each statement.  Returning
    ``SQLITE_DENY`` by default is important: adding a new operation to the
    classifier must explicitly earn an entry here rather than silently
    widening an authenticated immutable reader.
    """

    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        # SQLite reports COUNT(*)'s synthetic empty-column read without a
        # database name; it still belongs to the statement's main/temp table.
        return sqlite3.SQLITE_OK if database in {None, "main", "temp"} else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (argument2 or argument1 or "").lower()
        return sqlite3.SQLITE_OK if function_name in _SEALED_STAGING_FUNCTIONS else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA:
        pragma_name = (argument1 or "").lower()
        # A non-NULL second argument is a PRAGMA assignment.  Setup pragmas
        # run before this authorizer is installed; callers get reads only.
        if pragma_name == "table_info":
            return sqlite3.SQLITE_OK
        return (
            sqlite3.SQLITE_OK
            if argument2 is None and pragma_name in _SEALED_STAGING_READ_PRAGMAS
            else sqlite3.SQLITE_DENY
        )

    # TEMP DML and TEMP table/index creation, deletion, and reindexing are the
    # complete staging vocabulary.  SQLite reports its internal
    # sqlite_temp_master updates with database="temp", so those are included
    # by the same database check rather than by table-name exceptions.
    temp_dml = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}
    temp_schema = {
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
    }
    if action in temp_dml:
        return sqlite3.SQLITE_OK if database == "temp" else sqlite3.SQLITE_DENY
    if action in temp_schema:
        return sqlite3.SQLITE_OK if database == "temp" else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_REINDEX:
        return sqlite3.SQLITE_OK if database == "temp" else sqlite3.SQLITE_DENY

    # This explicitly denies ATTACH/DETACH, all main-schema writes, virtual
    # tables, triggers/views, unsafe PRAGMAs and every future/unlisted action.
    return sqlite3.SQLITE_DENY


# Named timeout classes are the only supported policy vocabulary.  Callers
# select a role, not an arbitrary lock-wait duration.
TIMEOUT_CLASSES: Mapping[str, float] = {
    "interactive-read": float(READ_DB_TIMEOUT),
    "background-read": TIMEOUT_CLASS_BACKGROUND_READ_S,
    "publication": TIMEOUT_CLASS_PUBLICATION_S,
    "offline-bulk": TIMEOUT_CLASS_OFFLINE_BULK_S,
    # The active cold-build writer runs under the daemon's normal publication
    # hold budget and therefore uses the same declared lock-wait class.  Keep
    # it in the shared vocabulary so every named writer profile is validated
    # against an explicit timeout class.
    "active-cold-build": TIMEOUT_CLASS_PUBLICATION_S,
}
READ_PROFILES: Mapping[str, SQLiteConnectionProfile] = {
    "interactive-read": READ_CONNECTION_PROFILE,
    "background-read": BACKGROUND_READ_CONNECTION_PROFILE,
    "offline-bulk": OFFLINE_BULK_READ_CONNECTION_PROFILE,
}
# ``publication`` and ``offline-bulk`` name different profiles on the write side
# than on the read side, so the two vocabularies stay separate maps: merging
# them silently shadowed the offline-bulk *read* profile with the bulk-build
# writer.
WRITE_PROFILES: Mapping[str, SQLiteConnectionProfile] = {
    "publication": DAEMON_WRITE_CONNECTION_PROFILE,
    "offline-bulk": BULK_BUILD_WRITE_CONNECTION_PROFILE,
    "active-cold-build": COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE,
}

# One tier, no sibling attach. An excision apply and a backup snapshot both
# commit a single tier at a time so a mid-operation failure leaves at most one
# tier mutated; attaching siblings would draw them into the same transaction
# scope, which is the thing those routes exist to avoid. Journal mode and
# foreign-key enforcement are deliberately left as the file already has them:
# these routes adopt a tier, they do not reconfigure it. Autocheckpoint remains
# explicit because it is connection-local: an unowned process keeps the bounded
# fallback while the daemon's recurring owner disables it for this writer too.
ISOLATED_TIER_WRITE_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=TIMEOUT_CLASS_PUBLICATION_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_PUBLICATION_S * 1000),
    cache_size_kib=DAEMON_WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=DAEMON_WRITE_MMAP_SIZE_BYTES,
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
)

READ_CONNECTION_PRAGMA_STATEMENTS = READ_CONNECTION_PROFILE.pragma_statements


# ---------------------------------------------------------------------------
# Recurring checkpoint ownership and WAL escalation policy
# ---------------------------------------------------------------------------

#: What each escalation may attempt, in attempt order.
#:
#: PASSIVE never waits and never blocks a reader, so it is the only mode a
#: recurring owner may run against a live archive. RESTART additionally waits
#: for existing readers to drain before resetting the WAL, which is bounded
#: only at a declared quiescent boundary. TRUNCATE also takes the writer lock to
#: shrink the file, and belongs to seal, shutdown and offline generation
#: lifecycle after readers have drained -- never to fight a busy live reader.
CheckpointEscalation = Literal["recurring", "quiescent", "exclusive"]

CHECKPOINT_ESCALATION_MODES: Mapping[CheckpointEscalation, tuple[str, ...]] = {
    "recurring": ("PASSIVE",),
    "quiescent": ("PASSIVE", "RESTART"),
    "exclusive": ("PASSIVE", "RESTART", "TRUNCATE"),
}

#: WAL size at which a checkpoint is worth running at all, and the size past
#: which an escalation may reach its next mode.
WAL_WARN_BYTES = 256 * 1024 * 1024
WAL_ESCALATION_BYTES = 512 * 1024 * 1024

#: Checkpoint hold budget. Declared here rather than beside the publication
#: budgets in ``daemon/write_coordinator.py`` so checkpoint time is accounted
#: against its own ceiling instead of disappearing into whichever publication
#: hold happened to contain it.
CHECKPOINT_HOLD_BUDGET_S = 20.0

_RECURRING_CHECKPOINT_OWNER = threading.Event()


def recurring_checkpoint_owner_armed() -> bool:
    """Whether this process runs the recurring checkpoint coordinator."""
    return _RECURRING_CHECKPOINT_OWNER.is_set()


def _set_recurring_checkpoint_owner(armed: bool) -> None:
    if armed:
        _RECURRING_CHECKPOINT_OWNER.set()
    else:
        _RECURRING_CHECKPOINT_OWNER.clear()


@contextmanager
def arm_recurring_checkpoint_owner(*, armed: bool = True) -> Iterator[None]:
    """Claim recurring checkpoint ownership for this process.

    Process-global rather than thread-local like the write lease: the daemon
    opens writable connections from several threads and one coordinator owns
    checkpointing for all of them.
    """
    previous = _RECURRING_CHECKPOINT_OWNER.is_set()
    _set_recurring_checkpoint_owner(armed)
    try:
        yield
    finally:
        _set_recurring_checkpoint_owner(previous)


def write_connection_pragma_statements(profile: SQLiteConnectionProfile) -> tuple[str, ...]:
    """Write pragmas for ``profile`` under this process' checkpoint ownership.

    Under an armed recurring owner an implicit autocheckpoint runs inside
    whichever writer's commit crossed the page threshold, charging its wait and
    hold to that writer's publication budget where no checkpoint accounting can
    see it. Resolved per open rather than frozen at import so a process that
    arms ownership after loading this module still gets the owned profile.
    """
    if profile.role != "write" or profile.wal_autocheckpoint_pages is None:
        return profile.pragma_statements
    if not recurring_checkpoint_owner_armed():
        return profile.pragma_statements
    return replace(profile, wal_autocheckpoint_pages=OWNED_WAL_AUTOCHECKPOINT_PAGES).pragma_statements


def write_connection_local_pragma_statements(profile: SQLiteConnectionProfile) -> tuple[str, ...]:
    """Return a writer profile without database-mode initialization.

    ``journal_mode`` changes shared database state and needs an exclusive
    transition lock. A later connection to an already-initialized durable tier
    must therefore not replay it while another writer owns that tier's
    transaction; the remaining statements are connection-local policy.
    """
    return tuple(
        statement
        for statement in write_connection_pragma_statements(profile)
        if not statement.startswith("PRAGMA journal_mode")
    )


def initialize_source_tier_database_mode(conn: sqlite3.Connection) -> None:
    """Set source.db's shared WAL mode while its fresh bootstrap owns the file.

    This is deliberately separate from :func:`open_source_tier_write_connection`.
    ``source.db`` is authoritative material: its normal writer profile uses
    WAL/NORMAL with replay from retained input after a process crash, while a
    power-loss guarantee remains the durable publication/cursor boundary, not
    a claim made by this connection-local setting.
    """
    journal_mode = WRITE_CONNECTION_PROFILE.journal_mode
    if journal_mode is None:
        raise RuntimeError("the source-tier writer profile must declare a journal mode")
    conn.execute(f"PRAGMA journal_mode={journal_mode}")


def open_source_tier_write_connection(
    path: str | Path,
    *,
    archive_root: str | Path | None = None,
) -> sqlite3.Connection:
    """Open a source-tier writer with the normal local policy only.

    Bootstrap owns the one-time database-mode transition. This factory is
    shared by the persistent archive handle and publication reservations so a
    fresh source tier and a reservation transaction cannot drift on
    synchronous, busy-timeout, or foreign-key policy.
    """
    require_write_lease(f"open_source_tier_write_connection({path})", archive_root=archive_root)
    conn = sqlite3.connect(str(path), timeout=WRITE_CONNECTION_PROFILE.timeout_seconds)
    try:
        for statement in write_connection_local_pragma_statements(WRITE_CONNECTION_PROFILE):
            conn.execute(statement)
    except BaseException:
        conn.close()
        raise
    return conn


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
    running concurrently with the daemon's own long-lived write connection
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
            tier = _archive_tier_for_path(sibling)
            if tier is not None:
                sibling_conn = open_readonly_connection(
                    sibling, tier=tier, validate_schema=False, timeout_class="background-read"
                )
                try:
                    _assert_schema_supported(sibling_conn, sibling, tier)
                finally:
                    sibling_conn.close()
            conn.execute(f"ATTACH DATABASE ? AS {schema_name}", (str(sibling),))


def _archive_tier_for_path(path: str | Path) -> ArchiveTier | None:
    """Resolve a conventional archive filename without importing at module load."""
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    return next((tier for tier in ArchiveTier if Path(path).name == f"{tier.value}.db"), None)


def _schema_skew_remedy(tier: ArchiveTier) -> str:
    """Describe the safe recovery route for a mismatched archive tier."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec

    spec = archive_tier_spec(tier)
    if spec.durability in {"rebuildable", "expensive_rebuild", "disposable"}:
        return (
            f"{tier.value}.db is {spec.durability} derived state; rebuild or recreate this tier "
            "from durable evidence with the current runtime before retrying"
        )
    return (
        f"{tier.value}.db is durable state; do not rebuild it. Run `polylogue ops maintenance migrate-tier "
        f"{tier.value}` with a verified backup manifest before retrying"
    )


def _tier_holds_no_schema(conn: sqlite3.Connection) -> bool:
    """Report whether a tier file carries any non-internal schema object."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1").fetchone()
    return row is None


def _assert_schema_supported(conn: sqlite3.Connection, path: str | Path, tier: ArchiveTier | None) -> None:
    """Reject a known archive tier before any caller can issue SQL against it."""
    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    resolved_tier = tier if tier is not None else _archive_tier_for_path(path)
    if resolved_tier is None:
        return
    try:
        expected = ARCHIVE_VERSION_BY_TIER[resolved_tier]
    except KeyError as exc:
        raise ValueError(f"unknown archive tier: {resolved_tier!r}") from exc
    found = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if found == 0 and _tier_holds_no_schema(conn):
        # A tier file with neither a version stamp nor any schema object has
        # never been provisioned. Reading it is reading an absent tier: the
        # caller fails on the missing table it asked for, which is a truthful
        # not-provisioned answer, where skew would misreport corruption.
        return
    if resolved_tier is ArchiveTier.INDEX and found == 0:
        return
    if found != expected:
        raise SchemaSkew(
            tier=resolved_tier.value,
            expected=expected,
            found=found,
            remedy=_schema_skew_remedy(resolved_tier),
        )


def _assert_derived_identity_supported(conn: sqlite3.Connection, tier: ArchiveTier | None) -> None:
    """Check a derived tier's identity, not merely its version cursor.

    ``user_version`` tracks the numbered schema; it says nothing about the
    lowering, materializer and routing fingerprints the derived identity also
    covers, and a bootstrap route restamps it before this check ever reads it.
    A tier stamped by a different runtime therefore passes the version gate
    while carrying read models this runtime cannot interpret.
    """
    from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier

    if tier is None:
        return
    try:
        derived_tier = DerivedTier(tier.value)
    except ValueError:
        return
    from polylogue.storage.sqlite.schema_bootstrap import assert_derived_schema_identity

    assert_derived_schema_identity(conn, derived_tier.value)


def assert_tier_schema_supported(
    conn: sqlite3.Connection,
    path: str | Path,
    tier: ArchiveTier | None = None,
) -> None:
    """Reject a tier this runtime cannot serve, by version and by identity.

    Public so a bootstrap route that opens a not-yet-materialised tier with
    ``validate_schema=False`` can apply the check once it has stamped the
    schema. That route rewrites ``user_version`` while materialising, so the
    version alone proves nothing about the tier it just wrote over; the
    derived identity is what the stamp is for and is checked here rather than
    on every ordinary open.
    """
    _assert_schema_supported(conn, path, tier)
    _assert_derived_identity_supported(conn, tier if tier is not None else _archive_tier_for_path(path))


def open_connection(
    path: str | Path,
    *,
    timeout: float = DB_TIMEOUT,
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
    profile: SQLiteConnectionProfile = WRITE_CONNECTION_PROFILE,
    archive_root: str | Path | None = None,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a read-write SQLite connection with canonical write pragmas applied.

    This is a lightweight one-shot factory: it opens the file, applies the
    write-time PRAGMA profile, attaches sibling archive tiers (so cross-tier
    reads resolve), and returns the connection.  The caller owns the connection
    lifecycle (must close it).

    For the thread-local cached archive connection used by the async runtime,
    use ``connection_context`` from ``connection.py`` instead.

    ``check_same_thread=False`` is for a handle that outlives the thread that
    opened it -- the cold build's ``ops.db`` checkpoint holder is opened on one
    write-coordinator thread, re-asserted on the next pass' thread and closed
    on whichever thread settles the generation. It is not a concurrency
    licence: ``sqlite3`` here is serialized (``threadsafety == 3``), so the
    handle is thread-safe, but the *writes* it serves are still ordered by the
    single-writer lease. A caller that has no such ordering must leave this
    ``True`` and get the thread check.
    """
    if profile.role != "write":
        raise ValueError("open_connection requires a write profile")
    require_write_lease(f"open_connection({path})", archive_root=archive_root)
    conn = sqlite3.connect(str(path), timeout=timeout, check_same_thread=check_same_thread)
    try:
        if validate_schema:
            _assert_schema_supported(conn, path, tier)
        for stmt in write_connection_pragma_statements(profile):
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
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
    archive_root: str | Path | None = None,
) -> sqlite3.Connection:
    """Open a read-write SQLite connection for daemon maintenance/ops writes.

    Long-running daemon loops write small status, cursor, telemetry, and
    maintenance rows. They should not inherit the full batch-ingest cache and
    mmap profile, because systemd charges their SQLite page cache to the
    service cgroup for the lifetime of the process.
    """
    require_write_lease(f"open_daemon_connection({path})", archive_root=archive_root)
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        if validate_schema:
            _assert_schema_supported(conn, path, tier)
        for stmt in write_connection_pragma_statements(DAEMON_WRITE_CONNECTION_PROFILE):
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
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
    profile: SQLiteConnectionProfile = READ_CONNECTION_PROFILE,
    timeout_class: str = "interactive-read",
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

    ``validate_schema=False`` is reserved for diagnostic readers that need to
    inspect a tier before reporting its schema mismatch. It does not change the
    read-only connection profile or grant write access.
    """
    if profile.role != "read" or not profile.query_only:
        raise ValueError("open_readonly_connection requires a query-only read profile")
    if timeout_class not in READ_PROFILES:
        raise ValueError(f"unknown SQLite timeout class: {timeout_class}")
    if profile is READ_CONNECTION_PROFILE and timeout_class != "interactive-read":
        profile = READ_PROFILES[timeout_class]
    if immutable and profile.generation_identity != "sealed":
        if not any(profile is named for named in (READ_CONNECTION_PROFILE, *READ_PROFILES.values())):
            raise ValueError(
                "immutable SQLite mode requires a sealed-generation read profile; a live-generation "
                "profile cannot promise the file will not change under the connection"
            )
        profile = SEALED_READ_CONNECTION_PROFILE
    immutable = immutable or profile.immutable
    timeout = profile.timeout_seconds if timeout == READ_DB_TIMEOUT else timeout
    suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    if opened_main_fd is not None and immutable:
        raise ValueError("an opened SQLite file descriptor cannot use immutable mode")
    opened_fd = opened_main_fd
    if opened_fd is None:
        # Percent-encode the path: an unescaped '?' or '#' in a filename would
        # otherwise be parsed as the URI's own query or fragment delimiter and
        # silently open a different file, or none.
        database_uri = f"file:{quote(str(path))}{suffix}"
    else:
        descriptor_uri = _descriptor_database_uri(opened_fd, suffix)
        if descriptor_uri is None:
            raise RuntimeError(f"cannot open selected SQLite database through a descriptor-bound path: {path}")
        database_uri = descriptor_uri
    conn = sqlite3.connect(database_uri, uri=True, timeout=timeout)
    try:
        if validate_schema:
            _assert_schema_supported(conn, path, tier)
        for stmt in profile.pragma_statements:
            conn.execute(stmt)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def one_shot_diagnostic_read(
    path: str | Path,
    *,
    tier: ArchiveTier | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open one non-paginated diagnostic probe under the interactive read profile.

    This is deliberately narrower than :func:`open_readonly_connection`:
    it is for a small probe whose result is consumed before request or
    presentation work begins (for example, checking a schema version).  A
    caller that retains SQLite state while paging or rendering must use a
    :func:`read_frame` instead, so its live snapshot has a declared bounded
    lifetime and can rebind safely.

    Diagnostics may need to read an unsupported schema in order to explain
    it, hence schema validation is intentionally disabled here.  That does
    not relax SQLite's ``mode=ro`` or ``query_only`` enforcement.
    """
    conn = open_readonly_connection(
        path,
        tier=tier,
        validate_schema=False,
        timeout_class="interactive-read",
    )
    try:
        yield conn
    finally:
        conn.close()


def open_sealed_staging_connection(
    path: str | Path,
    *,
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
) -> sqlite3.Connection:
    """Open an immutable source image with sealed, TEMP-only staging.

    This is a deliberately dedicated exception for historical liveness
    classification.  The main database is opened with SQLite's
    ``mode=ro&immutable=1`` URI and remains protected by a fail-closed
    authorizer; the only writes admitted after setup target the connection's
    in-memory TEMP schema.  It does not accept caller-selected profiles,
    descriptors, attachments, or write options.
    """

    profile = SEALED_STAGING_CONNECTION_PROFILE
    database_uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
    conn = sqlite3.connect(database_uri, uri=True, timeout=profile.timeout_seconds)
    try:
        if validate_schema:
            _assert_schema_supported(conn, path, tier)
        # Apply only this bounded profile's setup statements.  In particular,
        # do not copy READ_CONNECTION_PROFILE here: its query_only=ON is
        # exactly what prevents TEMP staging.
        for statement in profile.pragma_statements:
            conn.execute(statement)
        conn.set_authorizer(_authorize_sealed_staging_operation)
    except BaseException:
        conn.close()
        raise
    return conn


def open_profiled_connection(
    path: str | Path,
    *,
    profile: SQLiteConnectionProfile,
    timeout: float | None = None,
    immutable: bool = False,
    opened_main_fd: int | None = None,
    tier: ArchiveTier | None = None,
) -> sqlite3.Connection:
    """Open a connection from an explicit named profile.

    Read profiles are always enforced at SQLite's boundary.  Write profiles
    use the existing writer factory so attachment and schema checks remain
    identical to ordinary archive writes.
    """
    if profile.role == "read":
        if not profile.query_only:
            raise ValueError("read profiles must enable query_only")
        return open_readonly_connection(
            path,
            timeout=profile.timeout_seconds if timeout is None else timeout,
            immutable=immutable,
            opened_main_fd=opened_main_fd,
            tier=tier,
            profile=profile,
        )
    if immutable or opened_main_fd is not None:
        raise ValueError("writer profiles cannot use immutable or descriptor-bound reads")
    return open_connection(
        path,
        timeout=profile.timeout_seconds if timeout is None else timeout,
        tier=tier,
        profile=profile,
    )


def open_isolated_write_connection(
    path: str | Path,
    *,
    purpose: str,
    profile: SQLiteConnectionProfile = ISOLATED_TIER_WRITE_PROFILE,
    timeout: float | None = None,
    archive_root: str | Path | None = None,
) -> sqlite3.Connection:
    """Open one writable tier without attaching sibling databases.

    Snapshot/checkpoint and other one-tier operations must still pass through
    the same lease boundary as ordinary archive writes.  Keeping this factory
    separate prevents those operations from accidentally widening their
    transaction to attached tiers.
    """
    if profile.role != "write":
        raise ValueError("open_isolated_write_connection requires a write profile")
    require_write_lease(purpose, archive_root=archive_root)
    conn = sqlite3.connect(str(path), timeout=profile.timeout_seconds if timeout is None else timeout)
    try:
        for statement in write_connection_pragma_statements(profile):
            conn.execute(statement)
    except BaseException:
        conn.close()
        raise
    return conn


# ---------------------------------------------------------------------------
# Read frames: a bounded, rebindable read connection over one generation
# ---------------------------------------------------------------------------
#
# A ``mode=ro`` connection is not by itself a bounded read, and it is also not
# by itself a cost: an idle one holds no read mark and a recurring PASSIVE
# checkpoint recycles the log straight past it. What turns PASSIVE into a no-op
# and lets the WAL grow without limit is an *open read transaction* -- in
# practice a cursor that is still being stepped. ``sqlite3.Connection`` cannot
# report that (``in_transaction`` stays False for a SELECT in autocommit while
# the cursor pins frames), so the frame has to own the stepping to bound it.
#
# A read frame therefore gives a live-generation reader three things: an age
# past which it must rebind, a generation identity that says whether what it
# rebound to is still the thing it was reading, and ``stream`` -- the one route
# that steps a cursor while re-checking that age between rows and releases the
# cursor before raising, so an expiry ends the WAL pin instead of reporting it.


class ReadFrameExpiredError(RuntimeError):
    """A live read frame outlived the maximum snapshot age its profile declares."""

    code = "read_frame_expired"


class StaleContinuationError(RuntimeError):
    """A continuation cannot be resumed against an equivalent frame.

    Raised instead of resuming where the anchor no longer holds the position:
    continuing there would skip or duplicate rows, and only the caller can
    decide which of those it can tolerate.
    """

    code = "stale_continuation"


class ReadFrameCancelledError(RuntimeError):
    """The frame's in-flight statement was cancelled by its owner."""

    code = "read_frame_cancelled"


@dataclass(frozen=True, slots=True)
class GenerationToken:
    """Identity of the generation a frame is bound to.

    File identity, because that is what a generation swap moves and what stays
    comparable between two connections. Content freshness *within* one
    generation is a separate question that only the open connection can answer
    (``PRAGMA data_version`` is explicitly not meaningful across connections),
    so the frame tracks that separately.
    """

    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class ReadContinuation:
    """A resumable position in a paged read, and the anchor that proves it.

    ``anchor_sql`` must select the row the page stopped at and return its
    position as the first column, so a rebound frame can prove the position
    still means the same thing rather than assuming it.
    """

    position: object
    anchor_sql: str
    anchor_params: tuple[object, ...] = ()
    generation: GenerationToken | None = None
    #: Which frame incarnation produced this continuation. A rebind starts a
    #: new one, so a continuation can never be waved through on generation
    #: identity alone after the frame it was produced on was replaced.
    epoch: int = 0


@dataclass(frozen=True, slots=True)
class ReadFrameStatus:
    """One live read frame, as the recurring checkpoint owner sees it."""

    path: Path
    timeout_class: str
    age_s: float
    max_snapshot_age_s: float | None
    #: Whether a cursor opened through :meth:`ReadFrame.stream` is in flight.
    #: This is the only state that provably pins WAL frames, and the only
    #: reason a frame belongs in a blocked checkpoint's evidence.
    streaming: bool
    reason: str | None = None

    @property
    def overdue(self) -> bool:
        return self.max_snapshot_age_s is not None and self.age_s > self.max_snapshot_age_s

    def describe(self) -> str:
        declared = "none" if self.max_snapshot_age_s is None else f"{self.max_snapshot_age_s:.0f}s"
        suffix = f" ({self.reason})" if self.reason else ""
        return f"{self.path.name}:{self.timeout_class} age={self.age_s:.1f}s max={declared}{suffix}"


# The process-local read-snapshot registry. ``connection_profile`` owns both
# halves of the coupled decision, so the recurring checkpoint owner can name the
# frames that pinned it instead of only naming a PID from a ``/proc`` walk that
# cannot distinguish an idle handle from an open read transaction.
_LIVE_READ_FRAMES: weakref.WeakSet[ReadFrame] = weakref.WeakSet()
_LIVE_READ_FRAMES_LOCK = threading.Lock()


def live_read_frames() -> tuple[ReadFrameStatus, ...]:
    """Every read frame still open in this process, oldest first."""
    with _LIVE_READ_FRAMES_LOCK:
        frames = tuple(_LIVE_READ_FRAMES)
    statuses = [frame.status() for frame in frames]
    return tuple(sorted(statuses, key=lambda status: status.age_s, reverse=True))


def pinning_read_frames(path: Path | str | None = None) -> tuple[ReadFrameStatus, ...]:
    """Live frames holding an open read transaction, optionally over one file.

    ``path`` is compared by resolved filesystem identity so a generation reached
    through a symlinked ``index.db`` matches the tier a checkpoint names.
    """
    target = Path(path).resolve(strict=False) if path is not None else None
    return tuple(
        status
        for status in live_read_frames()
        if status.streaming and (target is None or status.path.resolve(strict=False) == target)
    )


def _generation_token(path: Path) -> GenerationToken:
    stat = path.stat()
    return GenerationToken(device=stat.st_dev, inode=stat.st_ino)


def _data_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA main.data_version").fetchone()[0])


class ReadFrame:
    """A read connection bound to one generation for a declared maximum age.

    Not thread-safe: a frame belongs to the request or page loop that opened
    it. ``cancel`` is the one exception, and only for a profile that declares
    cancellation support -- it interrupts an in-flight statement from another
    thread, which is what SQLite's ``interrupt`` is for.
    """

    __slots__ = (
        "__weakref__",
        "_cancelled",
        "_conn",
        "_data_version",
        "_epoch",
        "_generation",
        "_opened_at",
        "_path",
        "_profile",
        "_reason",
        "_streaming",
        "_tier",
        "_timeout_class",
    )

    def __init__(
        self,
        path: Path | str,
        *,
        profile: SQLiteConnectionProfile,
        tier: ArchiveTier | None = None,
        timeout_class: str = "unnamed",
        reason: str | None = None,
    ) -> None:
        if profile.role != "read" or not profile.query_only:
            raise ValueError("a read frame requires a query-only read profile")
        if profile.generation_identity == "live" and profile.max_snapshot_age_s is None:
            # A live generation changes under the reader, so a frame over one
            # with no declared maximum is precisely the unbounded WAL pin this
            # class exists to prevent. Refusing here is what stops a caller
            # from obtaining one by handing in a profile with the bound removed.
            raise ValueError(
                f"a live-generation read frame over {path} must declare max_snapshot_age_s; "
                "use read_frame(..., max_snapshot_age_s=..., reason=...) to extend the bound, "
                "or a sealed-generation profile if the file genuinely cannot change"
            )
        self._path = Path(path)
        self._profile = profile
        self._tier = tier
        self._timeout_class = timeout_class
        self._reason = reason
        self._cancelled = False
        self._epoch = 0
        self._streaming = 0
        self._conn = self._open()
        self._opened_at = time.monotonic()
        self._generation = _generation_token(self._path)
        self._data_version = _data_version(self._conn)
        with _LIVE_READ_FRAMES_LOCK:
            _LIVE_READ_FRAMES.add(self)

    def _open(self) -> sqlite3.Connection:
        conn = open_readonly_connection(
            self._path,
            profile=self._profile,
            immutable=self._profile.immutable,
            tier=self._tier,
        )
        conn.row_factory = sqlite3.Row
        return conn

    # -- identity and lifetime ------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        """The bound connection, refused once the frame has expired.

        Refusing here is what makes the declared maximum age load-bearing: a
        caller that holds a frame past it gets a typed error instead of a
        silently unbounded WAL pin.
        """
        self.check()
        return self._conn

    @property
    def generation(self) -> GenerationToken:
        return self._generation

    @property
    def epoch(self) -> int:
        """How many times this frame has been rebound."""
        return self._epoch

    @property
    def profile(self) -> SQLiteConnectionProfile:
        return self._profile

    @property
    def age_s(self) -> float:
        return time.monotonic() - self._opened_at

    @property
    def expired(self) -> bool:
        max_age = self._profile.max_snapshot_age_s
        return max_age is not None and self.age_s > max_age

    @property
    def streaming(self) -> bool:
        """Whether a :meth:`stream` cursor is in flight, i.e. pinning WAL frames."""
        return self._streaming > 0

    def status(self) -> ReadFrameStatus:
        return ReadFrameStatus(
            path=self._path,
            timeout_class=self._timeout_class,
            age_s=self.age_s,
            max_snapshot_age_s=self._profile.max_snapshot_age_s,
            streaming=self.streaming,
            reason=self._reason,
        )

    def check(self) -> None:
        """Raise if this frame may no longer be read from."""
        if self._cancelled:
            raise ReadFrameCancelledError(f"read frame over {self._path} was cancelled")
        if self.expired:
            declared = f"{self._profile.max_snapshot_age_s:.1f}s"
            extension = f" (extended for {self._reason})" if self._reason else ""
            raise ReadFrameExpiredError(
                f"read frame over {self._path} reached {self.age_s:.1f}s against a declared "
                f"{declared} maximum for timeout class {self._timeout_class}{extension}; "
                "rebind it or finish the read"
            )

    # -- bounded streaming ----------------------------------------------------

    def stream(
        self,
        sql: str,
        parameters: Sequence[object] = (),
    ) -> Generator[sqlite3.Row, None, None]:
        """Step one cursor under this frame's declared maximum snapshot age.

        This is the only supported way to hold a SQLite read transaction open
        across other work. A cursor that is stepped lazily pins every WAL frame
        written since it started, which is what makes the recurring PASSIVE
        checkpoint reclaim nothing; a caller that pulls rows through here gets
        the declared bound applied between rows rather than only at the moment
        it first asked for the connection.

        On expiry the cursor is closed before :class:`ReadFrameExpiredError`
        reaches the caller, so the refusal *ends* the pin rather than merely
        reporting it. That ordering is the point: a typed error that left the
        cursor open would name the problem and keep causing it.

        Deliberately a generator rather than a plain iterator: a caller that
        abandons the read part-way calls ``close()`` to end the pin at a point
        it chooses, instead of leaving it to garbage collection.
        """
        self.check()
        cursor = self._conn.execute(sql, tuple(parameters))
        self._streaming += 1
        try:
            for row in cursor:
                self.check()
                yield row
        finally:
            self._streaming -= 1
            cursor.close()

    def revalidate(self) -> bool:
        """Whether this frame still sees exactly what it was opened on.

        Both halves matter: the generation may have been swapped under the
        frame, or another connection may have committed into the same one.
        """
        try:
            if _generation_token(self._path) != self._generation:
                return False
            return _data_version(self._conn) == self._data_version
        except (OSError, sqlite3.Error):
            return False

    def rebind(self) -> GenerationToken:
        """Reopen against the current generation, releasing the pinned frames.

        A sealed generation has nothing to rebind to: it cannot change, so a
        rebind request against one is a caller error rather than a no-op that
        hides a wrong profile choice.
        """
        if self._profile.generation_identity == "sealed":
            raise ValueError(f"a sealed-generation read frame over {self._path} has nothing to rebind to")
        if self.streaming:
            # Rebinding closes the connection the in-flight cursor is stepping.
            # Refusing is the loud half of the bound: a stream that outlived its
            # age must end with a typed expiry, not be silently re-opened
            # underneath and resumed against different rows.
            raise ReadFrameExpiredError(
                f"read frame over {self._path} cannot rebind while a stream is in flight; "
                "finish or abandon the stream first"
            )
        self._conn.close()
        self._cancelled = False
        self._epoch += 1
        self._conn = self._open()
        self._opened_at = time.monotonic()
        self._generation = _generation_token(self._path)
        self._data_version = _data_version(self._conn)
        return self._generation

    def cancel(self) -> None:
        """Interrupt an in-flight statement on this frame."""
        if not self._profile.cancellation_supported:
            raise ValueError(f"read profile for {self._path} does not declare cancellation support")
        self._cancelled = True
        self._conn.interrupt()

    # -- continuations --------------------------------------------------------

    def bind(self, continuation: ReadContinuation) -> ReadContinuation:
        """Stamp a continuation with the frame incarnation that produced it."""
        return replace(continuation, generation=self._generation, epoch=self._epoch)

    def resume(self, continuation: ReadContinuation) -> ReadContinuation:
        """Return a continuation valid against a current frame, or refuse.

        Rebinds an expired frame first, then either confirms the continuation
        is still equivalent -- same generation, or an anchor row that still
        holds the same position -- or raises :class:`StaleContinuationError`. It
        never advances or rewinds the position to make one fit.
        """
        if self.expired:
            self.rebind()
        unchanged = (
            continuation.generation == self._generation and continuation.epoch == self._epoch and self.revalidate()
        )
        if unchanged:
            return continuation
        row = self._conn.execute(continuation.anchor_sql, continuation.anchor_params).fetchone()
        if row is None or row[0] != continuation.position:
            raise StaleContinuationError(
                f"continuation at {continuation.position!r} cannot be resumed against the current "
                f"generation of {self._path}: its anchor row no longer holds that position"
            )
        return replace(continuation, generation=self._generation, epoch=self._epoch)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        with _LIVE_READ_FRAMES_LOCK:
            _LIVE_READ_FRAMES.discard(self)
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_frame(
    path: Path | str,
    *,
    timeout_class: str = "interactive-read",
    tier: ArchiveTier | None = None,
    max_snapshot_age_s: float | None = None,
    reason: str | None = None,
) -> ReadFrame:
    """Open a read frame under one of the declared read timeout classes.

    A caller whose work legitimately outlives its class default extends the
    bound explicitly with ``max_snapshot_age_s`` and says why in ``reason``;
    the reason travels into the expiry error and into
    :func:`live_read_frames`, so a long snapshot is a declared decision rather
    than an anonymous one. There is deliberately no way to spell "no bound":
    that is what a sealed generation is for.
    """
    if timeout_class not in READ_PROFILES:
        raise ValueError(f"unknown SQLite timeout class: {timeout_class}")
    profile = READ_PROFILES[timeout_class]
    if max_snapshot_age_s is None:
        if reason is not None:
            raise ValueError("a read-frame reason describes an extended bound; pass max_snapshot_age_s with it")
    else:
        if not reason:
            raise ValueError(f"extending the {timeout_class} snapshot bound to {max_snapshot_age_s}s requires a reason")
        if max_snapshot_age_s <= 0:
            raise ValueError("an extended read-frame snapshot bound must be a positive number of seconds")
        profile = replace(profile, max_snapshot_age_s=float(max_snapshot_age_s))
    return ReadFrame(path, profile=profile, tier=tier, timeout_class=timeout_class, reason=reason)


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
    "BULK_BUILD_WRITE_CONNECTION_PROFILE",
    "COLD_BUILD_ACTIVE_WAL_AUTOCHECKPOINT_PAGES",
    "COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE",
    "DAEMON_WRITE_CACHE_SIZE_KIB",
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
    "READ_PROFILES",
    "SEALED_READ_CONNECTION_PROFILE",
    "SEALED_STAGING_CONNECTION_PROFILE",
    "BACKGROUND_READ_CONNECTION_PROFILE",
    "OFFLINE_BULK_READ_CONNECTION_PROFILE",
    "BACKGROUND_READ_SNAPSHOT_AGE_S",
    "INTERACTIVE_READ_SNAPSHOT_AGE_S",
    "CHECKPOINT_ESCALATION_MODES",
    "CHECKPOINT_HOLD_BUDGET_S",
    "CheckpointEscalation",
    "OWNED_WAL_AUTOCHECKPOINT_PAGES",
    "WAL_ESCALATION_BYTES",
    "WAL_WARN_BYTES",
    "WRITE_PROFILES",
    "arm_recurring_checkpoint_owner",
    "recurring_checkpoint_owner_armed",
    "write_connection_pragma_statements",
    "write_connection_local_pragma_statements",
    "initialize_source_tier_database_mode",
    "open_source_tier_write_connection",
    "SQLiteConnectionProfile",
    "TIMEOUT_CLASSES",
    "WAL_AUTOCHECKPOINT_PAGES",
    "WRITE_CACHE_SIZE_KIB",
    "WRITE_CONNECTION_PROFILE",
    "WRITE_MMAP_SIZE_BYTES",
    "check_mapped_bytes_budget_against_cgroup_limit",
    "GenerationToken",
    "ReadContinuation",
    "ReadFrame",
    "ReadFrameCancelledError",
    "ReadFrameExpiredError",
    "ReadFrameStatus",
    "StaleContinuationError",
    "live_read_frames",
    "pinning_read_frames",
    "read_frame",
    "connection_context",
    "descriptor_alias_path",
    "open_sealed_staging_connection",
    "one_shot_diagnostic_read",
    "log_mapped_bytes_budget_check",
    "mapped_bytes_budget",
    "assert_tier_schema_supported",
    "open_isolated_write_connection",
    "open_daemon_connection",
    "open_connection",
    "open_readonly_connection",
]
