"""Fresh bootstrap helpers for archive databases.

Writer module: ops.
Fresh Ops destinations receive their event lifetime inside bootstrap custody.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from polylogue.storage.archive_tuple_location import InactiveTierDestination
    from polylogue.storage.sqlite.population_admission import _PopulationAdmission

from polylogue.core.sql_settlement import current_native_sql_lifetimes
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers import (
    ARCHIVE_BASELINE_DDL_BY_TIER,
    ARCHIVE_BASELINE_VERSION_BY_TIER,
    ARCHIVE_DDL_BY_TIER,
    ARCHIVE_VERSION_BY_TIER,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.audit_leaf import AuditLeafError, assert_verified_audit_leaf
from polylogue.storage.sqlite.connection_profile import (
    WRITE_CONNECTION_PROFILE,
    NativeSQLCustodyOwner,
    _close_failed_native_construction,
    _connect_archive_writer,
    open_readonly_connection,
    retained_native_sql_owners_for_lifetime,
)
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

# Kept locally so schema metadata can import the bootstrap module while the
# migration runner is still importing the archive-tier package.
DURABLE_MIGRATION_TIERS = frozenset({ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT})


def _is_schema_inventory_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class RuntimeTierProbeAuthority:
    """The schema/version of an authenticated post-apply train candidate."""

    tier: ArchiveTier
    version: int
    schema_inventory_sha256: str

    def __post_init__(self) -> None:
        if (
            self.tier not in DURABLE_MIGRATION_TIERS
            or type(self.version) is not int
            or self.version < 1
            or not _is_schema_inventory_digest(self.schema_inventory_sha256)
        ):
            raise ValueError("invalid runtime tier probe schema authority")


_RUNTIME_PROBE_AUTHORITY: ContextVar[RuntimeTierProbeAuthority | None] = ContextVar(
    "polylogue_runtime_probe_authority", default=None
)


@contextlib.contextmanager
def runtime_tier_probe_authority(authority: RuntimeTierProbeAuthority) -> Iterator[None]:
    """Carry one train's accepted schema through its existing consumer probes."""
    token = _RUNTIME_PROBE_AUTHORITY.set(authority)
    try:
        yield
    finally:
        _RUNTIME_PROBE_AUTHORITY.reset(token)


DurabilityClass = Literal["irreplaceable", "rebuildable", "expensive_rebuild", "human", "disposable"]


@dataclass(frozen=True, slots=True)
class ArchiveTierSpec:
    """Runtime metadata for one archive database file."""

    tier: ArchiveTier
    filename: str
    durability: DurabilityClass
    backup_required: bool

    @property
    def version(self) -> int:
        return ARCHIVE_VERSION_BY_TIER[self.tier]

    @property
    def ddl(self) -> str:
        return ARCHIVE_DDL_BY_TIER[self.tier]

    @property
    def baseline_version(self) -> int:
        return ARCHIVE_BASELINE_VERSION_BY_TIER[self.tier]

    @property
    def baseline_ddl(self) -> str:
        return ARCHIVE_BASELINE_DDL_BY_TIER[self.tier]


ARCHIVE_TIER_SPECS: dict[ArchiveTier, ArchiveTierSpec] = {
    ArchiveTier.SOURCE: ArchiveTierSpec(
        ArchiveTier.SOURCE,
        filename="source.db",
        durability="irreplaceable",
        backup_required=True,
    ),
    ArchiveTier.INDEX: ArchiveTierSpec(
        ArchiveTier.INDEX,
        filename="index.db",
        durability="rebuildable",
        backup_required=False,
    ),
    ArchiveTier.EMBEDDINGS: ArchiveTierSpec(
        ArchiveTier.EMBEDDINGS,
        filename="embeddings.db",
        durability="expensive_rebuild",
        backup_required=True,
    ),
    ArchiveTier.USER: ArchiveTierSpec(
        ArchiveTier.USER,
        filename="user.db",
        durability="human",
        backup_required=True,
    ),
    ArchiveTier.OPS: ArchiveTierSpec(
        ArchiveTier.OPS,
        filename="ops.db",
        durability="disposable",
        backup_required=False,
    ),
    ArchiveTier.AUDIT: ArchiveTierSpec(
        ArchiveTier.AUDIT,
        filename="audit.db",
        durability="irreplaceable",
        backup_required=True,
    ),
}


# ``OwnedArchiveLocation`` protects against other processes. Its deliberate
# reentrancy permits nested production opens in one process, so bootstrap also
# needs this process-local serialization around the fresh durable receipt
# protocol.
_ACTIVE_ARCHIVE_BOOTSTRAP_LOCK = threading.RLock()

#: Per-root generation token of the archive this process last validated, keyed
#: by absolute configured root. ``initialize_active_archive_root`` runs on
#: every index-tier sync write open and once per ingest batch, and its body
#: opens and validates all six tiers -- a fixed ~0.14 s that a cold build
#: repeats per chunk for no new information (polylogue-q53j4 AC1). Guarded by
#: ``_ACTIVE_ARCHIVE_BOOTSTRAP_LOCK``; see ``_archive_generation_token`` for
#: what counts as a different generation and why this is not a once-flag.
_ACTIVE_ARCHIVE_BOOTSTRAP_GENERATIONS: dict[str, tuple[object, ...]] = {}

#: Count of validations actually executed. Diagnostic; read by tests.
_ACTIVE_ARCHIVE_BOOTSTRAP_VALIDATIONS = 0


def archive_tier_spec(tier: ArchiveTier) -> ArchiveTierSpec:
    """Return the database-file spec for one durability tier."""
    return ARCHIVE_TIER_SPECS[tier]


# Creating an empty tier costs one ``executescript`` of that tier's whole DDL.
# Profiling a single storage module (37 tests, 195s wall) attributed 171.8s of
# self time to 207 such calls at ~830ms each -- 90% of the run -- because every
# archive built from scratch re-parses hundreds of CREATE statements. The DDL is
# deterministic, so the first materialised tier of a given (tier, version) is a
# faithful prototype for every later one in the same process, and SQLite's own
# backup API restores it as a page copy instead of re-parsing.
#
# Embeddings requires sqlite-vec to be loaded on the target connection before
# a cached page copy is restored.  The restore path performs that readiness
# step, so its empty vec0 schema is cacheable like every other tier.
_TIER_PROTOTYPE_LOCK = threading.Lock()
#: Per-process tally of how each tier initialization resolved, keyed
#: ``(tier, outcome)``. Three outcomes, because they have three different
#: costs and three different fixes:
#:
#: * ``prototype_hit``  -- restored by page copy from this process's cache.
#: * ``ddl_fresh``      -- verifiably-empty database, real DDL executed. Paid
#:   once per tier/DDL identity per process for cacheable tiers.
#: * ``ddl_reapply``    -- NON-empty database, whole-tier DDL re-executed for
#:   its ``IF NOT EXISTS`` idempotence and the same-version convergence steps
#:   that follow it. Redundant by construction, but no longer uniformly
#:   expensive: once the unconditional ``user_version`` header write was
#:   removed (polylogue-c1jgh) the no-op ``CREATE ... IF NOT EXISTS`` pass
#:   costs 0.3-4ms for source/user/ops/audit/embeddings. ``index`` is the
#:   outlier at ~44ms, and the route that reached it was paying a further
#:   ~102ms re-stamping an unchanged derived identity -- which is why an
#:   already-current tier now takes :func:`converge_same_version_tier`.
#:
#: Kept because the split is not observable from the outside: all of them look
#: like "initialize a tier" to a caller, while their costs differ by two orders
#: of magnitude. Do not re-derive the ranking from this comment: the counters
#: are in every managed run receipt under ``suite_cost.tier_init``, and the
#: ranking has already changed once underneath a comment that claimed it.
#: An integer increment against work that already runs SQL; the read side is
#: :func:`archive_tier_init_counts`.
_TIER_INIT_COUNTS: dict[tuple[str, str], int] = {}
_TIER_INIT_COUNTS_LOCK = threading.Lock()

_TIER_PROTOTYPES: dict[tuple[str, int, str, int, str], Path] = {}
_TIER_PROTOTYPE_DIR: Path | None = None
_PROTOTYPE_CACHEABLE_TIERS = frozenset(ArchiveTier)

#: polylogue-kc8eq: the page size a fresh derived index database is created
#: with. SQLite fixes ``page_size`` permanently when the first page is
#: allocated, so this is a creation-time decision that no later connection
#: profile can revise -- before this constant existed the only choice was
#: SQLite's compiled-in 4096, taken silently and recorded nowhere.
#:
#: Measured 2026-09-16 on a 149 MB index built through the production parse
#: and write route (400 codex sessions x 40 turns: 64,400 blocks, 18.3 MB of
#: block text), re-laid out with VACUUM INTO at each candidate:
#:
#:   page_size  file bytes   pages   blocks-by-session p95  FTS p95  trigram p95
#:   4096       148,967,424  36,369  0.071 ms               1.927 ms  12.532 ms
#:   8192       149,118,976  18,203  0.072 ms               1.926 ms  13.499 ms
#:   16384      149,995,520   9,155  0.069 ms               1.954 ms  14.088 ms
#:
#: File bytes and point-read latency are flat across the three (0.1 and 0.7
#: percent on bytes). The trigram LIKE probe is not: it degrades monotonically,
#: 7.7 percent at 8192 and 12.4 percent at 16384, because a trigram scan reads
#: whole pages to reach a few postings and a larger page moves more bytes per
#: hit. So 4096 wins or ties on every axis measured and is the recorded
#: default. That is also SQLite's own default, which is the point: the value
#: did not change, the fact that it is now chosen, validated and recorded did.
#:
#: Re-measure before changing it. A change re-creates generations rather than
#: migrating them, and the sweep above is worth re-running at the real corpus
#: size, where the trigram index is far larger relative to cache.
DEFAULT_ARCHIVE_PAGE_SIZE = 4096

#: SQLite accepts only these; anything else is silently ignored by the pragma,
#: which would make a wrong value look like it took effect.
_VALID_PAGE_SIZES = frozenset({512, 1024, 2048, 4096, 8192, 16384, 32768, 65536})


def apply_creation_page_size(conn: sqlite3.Connection, page_size: int) -> None:
    """Fix ``page_size`` on a database that has not allocated its first page.

    Refuses rather than silently no-opping: ``PRAGMA page_size`` on a database
    that already has pages is accepted by SQLite and then ignored (outside a
    VACUUM), so a caller that believed it chose 8192 would be handed 4096 with
    no error anywhere.
    """
    if page_size not in _VALID_PAGE_SIZES:
        raise ValueError(f"page_size must be a SQLite power of two between 512 and 65536, not {page_size}")
    if int(conn.execute("PRAGMA page_count").fetchone()[0]) != 0:
        raise RuntimeError("page_size can only be chosen before the database allocates its first page")
    conn.execute(f"PRAGMA page_size = {page_size}")
    applied = int(conn.execute("PRAGMA page_size").fetchone()[0])
    if applied != page_size:
        raise RuntimeError(f"requested page_size {page_size} but SQLite reports {applied}")


def _record_tier_init(tier: ArchiveTier, outcome: str) -> None:
    with _TIER_INIT_COUNTS_LOCK:
        key = (tier.value, outcome)
        _TIER_INIT_COUNTS[key] = _TIER_INIT_COUNTS.get(key, 0) + 1


def archive_tier_init_counts() -> dict[str, int]:
    """This process's tier-initialization tally as ``"tier.outcome" -> count``."""
    with _TIER_INIT_COUNTS_LOCK:
        return {f"{tier}.{outcome}": count for (tier, outcome), count in sorted(_TIER_INIT_COUNTS.items())}


def _cleanup_tier_prototype_dir(directory: Path) -> None:
    if retained_native_sql_owners_for_lifetime(directory):
        raise RuntimeError("tier prototype directory retains unsettled native SQL")
    shutil.rmtree(directory, ignore_errors=True)


def _tier_prototype_dir() -> Path:
    global _TIER_PROTOTYPE_DIR
    if _TIER_PROTOTYPE_DIR is None:
        directory = Path(tempfile.mkdtemp(prefix="polylogue-tier-prototype-"))
        atexit.register(_cleanup_tier_prototype_dir, directory)
        _TIER_PROTOTYPE_DIR = directory
    return _TIER_PROTOTYPE_DIR


def connection_page_size(conn: sqlite3.Connection) -> int:
    """The page size this connection's database has, or will have on first page."""
    return int(conn.execute("PRAGMA page_size").fetchone()[0])


def _tier_prototype_key(
    conn: sqlite3.Connection, tier: ArchiveTier, required_version: int
) -> tuple[str, int, str, int, str]:
    """Identify a prototype by every input that shapes its SQLite pages.

    Page size and journal mode belong in the key because a prototype is
    restored with the SQLite backup API, which makes an empty destination
    adopt the source's page size and header (WAL or rollback) outright.
    Without them a caller that asked for 8192, or for an inactive generation
    in rollback mode, would get the other shape back from the cache.
    """
    ddl_digest = hashlib.sha256(archive_tier_spec(tier).baseline_ddl.encode()).hexdigest()
    journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    return tier.value, required_version, ddl_digest, connection_page_size(conn), journal_mode


def _restore_tier_prototype(conn: sqlite3.Connection, tier: ArchiveTier, required_version: int) -> bool:
    """Page-copy a cached empty tier onto ``conn``; False when unavailable or unfaithful."""
    key = _tier_prototype_key(conn, tier, required_version)
    with _TIER_PROTOTYPE_LOCK:
        prototype = _TIER_PROTOTYPES.get(key)
    if prototype is None or not prototype.is_file():
        return False
    try:
        if tier is ArchiveTier.EMBEDDINGS:
            loaded, _error = try_load_sqlite_vec(conn)
            if not loaded:
                return False
        from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, _close_failed_native_construction

        source = open_readonly_connection(prototype.resolve(strict=True), immutable=True, validate_schema=False)
        source_owner = NativeSQLCustodyOwner(
            source,
            lifetime_dependencies=(
                *current_native_sql_lifetimes(),
                *((_TIER_PROTOTYPE_DIR,) if _TIER_PROTOTYPE_DIR is not None else ()),
            ),
        )
        try:
            source.backup(conn)
        except BaseException as primary:
            _close_failed_native_construction(source_owner, primary)
            raise
        else:
            source_owner.close()
        stored = int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error:
        return False
    # Fail closed: a prototype that does not reproduce the expected version is
    # discarded rather than trusted, and the caller falls back to real DDL.
    if stored != required_version:
        with _TIER_PROTOTYPE_LOCK:
            _TIER_PROTOTYPES.pop(key, None)
        return False
    return True


def _record_tier_prototype(conn: sqlite3.Connection, tier: ArchiveTier, required_version: int) -> None:
    """Snapshot a freshly materialised empty tier for reuse in this process."""
    key = _tier_prototype_key(conn, tier, required_version)
    with _TIER_PROTOTYPE_LOCK:
        if key in _TIER_PROTOTYPES:
            return
    staging: Path | None = None
    directory = _tier_prototype_dir()
    try:
        destination = directory / f"{tier.value}-v{required_version}-{key[2]}-p{key[3]}-{key[4]}.db"
        staging_fd, staging_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(staging_fd)
        staging = Path(staging_name)
        from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, _close_failed_native_construction

        target = connect_measured(staging)
        target_owner = NativeSQLCustodyOwner(target, lifetime_dependencies=(*current_native_sql_lifetimes(), directory))
        try:
            conn.backup(target)
        except BaseException as primary:
            _close_failed_native_construction(target_owner, primary)
            raise
        else:
            target_owner.close()
        staging.chmod(staging.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        os.replace(staging, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, sqlite3.Error):
        return
    finally:
        if staging is not None and not retained_native_sql_owners_for_lifetime(directory):
            with contextlib.suppress(OSError):
                staging.unlink(missing_ok=True)
    with _TIER_PROTOTYPE_LOCK:
        _TIER_PROTOTYPES.setdefault(key, destination)


def converge_same_version_tier(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    """Bring an already-materialised tier at the current version up to date.

    The declared same-version policy, and the whole of it. A tier whose stored
    ``user_version`` is the spec's has already had its canonical DDL executed
    and committed, so re-running ``executescript`` over it buys nothing but the
    ``IF NOT EXISTS`` no-ops -- and re-stamping its derived schema identity buys
    nothing but the commit fsync that stamp needs. Measured warm on this
    checkout, re-opening the six tiers of an initialized archive root costs
    123.7ms through the full initialization route and 26.7ms through this one, with ddl_reapply going 50 -> 0 over ten passes. index.db
    is where the saving is (35.5ms -> 21.4ms per open); the other five tiers
    were only paying 0.3-4ms of no-op DDL each.

    Existing index and ops tiers verify their derived identity before any
    schema statement. A stale or absent stamp raises ``SchemaSkew``; only a
    fresh empty tier is materialized and stamped by this runtime. An admitted
    index tier gains only the canonical runtime performance indexes before
    manifest validation, as on ordinary writable sync and async opens. User-tier
    annotation rows and the embeddings connection's extension remain owned
    by their respective same-version policies.
    """
    if tier in (ArchiveTier.INDEX, ArchiveTier.OPS):
        from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported

        # Admission precedes every DDL path, including runtime indexes: an
        # foreign derived identity is never patched or restamped into currency.
        assert_tier_schema_supported(conn, f"{tier.value}.db", tier)
        if tier is ArchiveTier.INDEX:
            from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync
            from polylogue.storage.sqlite.schema_manifest import assert_schema_manifest

            ensure_runtime_indexes_sync(conn)
            assert_schema_manifest(conn, tier)
    elif tier is ArchiveTier.USER:
        _ensure_user_annotation_schemas(conn)
        conn.commit()
    elif tier is ArchiveTier.EMBEDDINGS:
        # Every embeddings connection needs the extension loaded, not just the
        # freshly initialized one: sqlite-vec state is connection-local, so a
        # same-version open that skipped this returned a connection on which
        # every vec0 query fails as "no such module".
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            raise RuntimeError("archive embeddings initialization requires sqlite-vec") from error
    conn.commit()


def initialize_archive_tier(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    """Materialise a tier on an open connection.

    Derived tiers (index, ops) are stamped with their schema identity by
    :func:`_apply_archive_tier_convergence`, in the transaction that commits
    ``user_version``.
    """
    _materialize_archive_tier(conn, tier)


def initialize_runtime_tier_probe(
    conn: sqlite3.Connection, tier: ArchiveTier, *, probe_path: Path | None = None
) -> None:
    """Build an isolated empty probe through baseline and installed migrations.

    Probe consumers need the runtime schema without recursively releasing a
    train whose riders they are proving. This is not an archive initializer:
    populated connections refuse before any DDL.
    """
    databases = conn.execute("PRAGMA database_list").fetchall()
    main_path = next((str(row[2]) for row in databases if row[1] == "main"), "")
    if any(row[1] not in {"main", "temp"} for row in databases):
        raise RuntimeError("runtime tier probe cannot have attached databases")
    if main_path and (probe_path is None or Path(main_path).resolve() != probe_path.resolve()):
        raise RuntimeError("file-backed runtime tier probe requires its declared isolated path")
    if conn.in_transaction or int(conn.execute("PRAGMA user_version").fetchone()[0]) != 0:
        raise RuntimeError("runtime tier probe requires an empty connection")
    if conn.execute("SELECT 1 FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' LIMIT 1").fetchone() is not None:
        raise RuntimeError("runtime tier probe requires an empty connection")
    if tier in DURABLE_MIGRATION_TIERS:
        from polylogue.storage.sqlite.durable_change_train import (
            _canonical_schema_inventory,
            validate_durable_migration_sidecars,
        )
        from polylogue.storage.sqlite.migration_runner import (
            _execute_proved_migration_sql,
            _load_migrations,
            capture_durable_schema_inventory,
        )

        authority = _RUNTIME_PROBE_AUTHORITY.get()
        target = (
            authority.version if authority is not None and authority.tier is tier else ARCHIVE_VERSION_BY_TIER[tier]
        )
        expected = _canonical_schema_inventory(tier, target)
        if authority is not None and authority.tier is tier and authority.schema_inventory_sha256 != expected.sha256:
            raise RuntimeError("runtime tier probe authority differs from the installed numbered schema")
        steps = _load_migrations(tier)
        validate_durable_migration_sidecars(tier, tuple((step.name, step.sql) for step in steps))
        initialize_archive_tier(conn, tier)
        floor = ARCHIVE_BASELINE_VERSION_BY_TIER[tier]
        pending = tuple(step for step in steps if floor < step.version <= target)
        if tuple(step.version for step in pending) != tuple(range(floor + 1, target + 1)):
            raise RuntimeError("runtime tier probe lacks its complete numbered schema chain")
        for step in pending:
            before = capture_durable_schema_inventory(conn)
            if before.sha256 != _canonical_schema_inventory(tier, step.version - 1).sha256:
                raise RuntimeError("runtime tier probe differs before its numbered schema step")
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                _execute_proved_migration_sql(conn, step)
                if not conn.in_transaction:
                    raise RuntimeError("runtime tier probe schema step escaped its owned transaction")
                conn.execute(f"PRAGMA user_version = {step.version}")
        if (
            int(conn.execute("PRAGMA user_version").fetchone()[0]) != target
            or capture_durable_schema_inventory(conn).sha256 != expected.sha256
        ):
            raise RuntimeError("runtime tier probe did not reproduce its accepted schema/version")
        return
    initialize_archive_tier(conn, tier)


def _materialize_archive_tier(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    """Initialize a fresh archive tier database on an already-open connection.

    When the connection's database is verifiably empty (no objects in
    ``sqlite_master``), the tier is restored from this process's cached
    prototype as a page copy instead of re-parsing hundreds of CREATE
    statements -- the same reuse ``initialize_active_archive_root`` gets,
    extended here because the open-connection route is what nearly a
    hundred test modules build every archive through (~0.8s of DDL per
    database, multiplied by four tiers and thousands of tests). A
    non-empty derived database is admitted as-is by its schema identity;
    other tiers retain their declared initialization policy.
    """
    spec = archive_tier_spec(tier)
    # Foreign-key enforcement is connection state, not schema: every branch
    # below must leave it enabled.
    conn.execute("PRAGMA foreign_keys = ON")
    if tier in DURABLE_MIGRATION_TIERS:
        stored_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if stored_version > spec.baseline_version:
            if stored_version != spec.version:
                from polylogue.core.errors import SchemaSkew

                raise SchemaSkew(tier=tier.value, expected=spec.version, found=stored_version)
            converge_same_version_tier(conn, tier)
            return
    if tier in (ArchiveTier.INDEX, ArchiveTier.OPS):
        object_count = int(conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0])
        if object_count or int(conn.execute("PRAGMA user_version").fetchone()[0]) != 0:
            converge_same_version_tier(conn, tier)
            return
    if tier in _PROTOTYPE_CACHEABLE_TIERS:
        object_count = int(conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0])
        if object_count == 0:
            if _restore_tier_prototype(conn, tier, spec.baseline_version):
                conn.execute("PRAGMA foreign_keys = ON")
                if tier is ArchiveTier.OPS:
                    # A prototype copies pages, including its seed row. This
                    # destination was proved empty above and owns a new event
                    # lifetime; existing admitted ledgers never take this path.
                    conn.execute(
                        "UPDATE daemon_event_retention SET lifetime=lower(hex(randomblob(16))) "
                        "WHERE ledger='daemon_events'"
                    )
                if tier is ArchiveTier.INDEX:
                    from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

                    ensure_runtime_indexes_sync(conn)
                _apply_archive_tier_convergence(conn, tier, spec, version=spec.baseline_version)
                _record_tier_init(tier, "prototype_hit")
                return
            _initialize_archive_tier_ddl(conn, tier)
            _record_tier_prototype(conn, tier, spec.baseline_version)
            _record_tier_init(tier, "ddl_fresh")
            return
        _initialize_archive_tier_ddl(conn, tier)
        _record_tier_init(tier, "ddl_reapply")
        return
    # Explicit escape hatch for a future tier that cannot safely be restored
    # from a page-copy prototype (for example, an extension with connection-
    # local state that cannot be prepared before backup restoration).
    empty = int(conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]) == 0
    _initialize_archive_tier_ddl(conn, tier)
    _record_tier_init(tier, "ddl_fresh" if empty else "ddl_reapply")


def _initialize_archive_tier_ddl(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    """The canonical DDL route: parse and execute the tier's whole schema."""
    spec = archive_tier_spec(tier)
    conn.execute("PRAGMA foreign_keys = ON")
    if tier is ArchiveTier.EMBEDDINGS:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            raise RuntimeError("archive embeddings initialization requires sqlite-vec") from error
    # One transaction for the whole schema. In autocommit every CREATE is
    # its own commit, and each commit is a sync of the journal: a fresh tier
    # paid one per statement (hundreds on the index tier), which under host
    # I/O load stretched a fresh root's bootstrap into minutes. The
    # transaction stays open through the convergence steps, whose final
    # commit publishes schema, stamp and version together, so an interrupted
    # initialization leaves an empty file rather than a partial schema.
    conn.executescript(f"BEGIN;\n{spec.baseline_ddl}\n;")
    if tier is ArchiveTier.INDEX:
        from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

        # The manifest assertion refuses an index tier without its runtime
        # indexes; they belong to the same schema transaction.
        ensure_runtime_indexes_sync(conn)
    _apply_archive_tier_convergence(conn, tier, spec, version=spec.baseline_version)


def _apply_archive_tier_convergence(
    conn: sqlite3.Connection,
    tier: ArchiveTier,
    spec: ArchiveTierSpec,
    *,
    version: int | None = None,
) -> None:
    """Replay same-version convergence after either DDL or page-copy restore.

    Prototypes accelerate canonical DDL; they are not a second schema
    authority.  The steps here (user annotation schemas, the ops schema-state
    record, the derived identity stamp, the version write) run on cache hits
    too.
    """
    if tier is ArchiveTier.USER:
        _ensure_user_annotation_schemas(conn)
    if tier in (ArchiveTier.INDEX, ArchiveTier.OPS):
        from polylogue.storage.sqlite.schema_bootstrap import stamp_derived_schema_identity

        # Stamp before the version write so both land in the commit below. A
        # tier that reaches its current ``user_version`` is therefore always
        # stamped; an interrupted initialization leaves version 0, which the
        # next open re-materialises instead of meeting an unstamped tier.
        stamp_derived_schema_identity(conn, tier.value)
    # Write the version ONLY when it actually changes. ``PRAGMA user_version = N``
    # rewrites the database header even when N is already the stored value, so an
    # unconditional write dirties a page on every same-version reapply and turns a
    # no-op schema pass into a full commit fsync. Measured on NVMe: the whole
    # reapply is 148.80ms with this write and 0.34ms without it, while the
    # ``executescript`` it accompanies -- hundreds of no-op
    # ``CREATE TABLE IF NOT EXISTS`` statements -- accounts for 0.33ms of that.
    # The DDL was never the cost; the header write was (polylogue-c1jgh).
    #
    target_version = spec.version if version is None else version
    if int(conn.execute("PRAGMA user_version").fetchone()[0]) != target_version:
        conn.execute(f"PRAGMA user_version = {target_version}")
    conn.commit()


def _ensure_user_annotation_schemas(conn: sqlite3.Connection) -> None:
    """Replay packaged schema rows without changing the user-tier DDL version."""

    from polylogue.storage.sqlite.archive_tiers.user_annotations import persist_builtin_annotation_schemas

    persist_builtin_annotation_schemas(conn, registered_at_ms=0)


def initialize_archive_database(
    path: Path,
    tier: ArchiveTier,
    *,
    allow_create: bool = True,
    expected_version: int | None = None,
    archive_root: Path | None = None,
    inactive_destination: InactiveTierDestination | None = None,
    page_size: int | None = None,
    inactive_generation: bool = False,
) -> None:
    """Create or initialize one archive tier database file.

    ``inactive_generation`` creates an inactive Index generation in
    rollback-journal mode: its bulk-build writer then opens without a header
    change, and promotion switches it to WAL.

    A path below ``.archive-tuples`` is an inactive whole-archive candidate,
    not an ordinary archive root.  Such a writer must carry the manifest-bound
    destination capability so a typo cannot silently open the active or a
    foreign generation.  Validation happens before ``sqlite3.connect``.

    Inactive Index generations also name their owning active archive explicitly:
    their database parent is a candidate directory, not the archive root whose
    write lease admits publication.
    """
    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    assert_population_admitted(path)
    from polylogue.storage.archive_identity import ArchiveLocation
    from polylogue.storage.archive_tuple_location import (
        ArchiveTupleError,
        InactiveTierDestination,
        is_archive_tuple_candidate_path,
        validate_inactive_destination,
    )

    spec = archive_tier_spec(tier)
    required_version = spec.version if expected_version is None else expected_version
    if is_archive_tuple_candidate_path(path):
        if not isinstance(inactive_destination, InactiveTierDestination):
            raise ArchiveTupleError("inactive archive tuple tier initialization requires a typed inactive_destination")
        validate_inactive_destination(
            inactive_destination,
            ArchiveLocation.resolve(inactive_destination.archive_root),
            path=path,
            expected_tier=tier,
        )
    elif inactive_destination is not None:
        raise ArchiveTupleError("inactive_destination does not match an archive tuple candidate path")
    if inactive_destination is not None:
        configured_root = inactive_destination.archive_root
        if archive_root is not None and Path(archive_root).resolve(strict=False) != Path(configured_root).resolve(
            strict=False
        ):
            raise ArchiveTupleError("archive_root does not match inactive_destination ownership")
    elif archive_root is not None:
        configured_root = archive_root
    elif inactive_generation:
        raise ValueError("inactive generation initialization requires its owning archive_root")
    else:
        configured_root = path.parent
    from polylogue.storage.sqlite.write_lease import require_write_lease

    require_write_lease("initialize archive tier", archive_root=configured_root)
    if allow_create:
        # Fresh bootstrap must not follow a pre-existing durable pathname out
        # of the archive root.  ``Path.exists()`` misses dangling symlinks,
        # so inspect the directory entry before SQLite gets a chance to
        # create or follow it.  Derived tuple destinations are validated by
        # their capability above and are intentionally not subject to the
        # active-root durable containment rule.
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            metadata = None
        if (
            metadata is not None
            and tier in DURABLE_MIGRATION_TIERS
            and (path.is_symlink() or not path.is_file() or metadata.st_nlink != 1)
        ):
            raise RuntimeError(f"durable tier is not a safe fresh file path; refusing initialization: {path}")
        if metadata is None and tier in DURABLE_MIGRATION_TIERS and required_version != spec.baseline_version:
            from polylogue.core.errors import SchemaSkew

            raise SchemaSkew(
                tier=tier.value,
                expected=required_version,
                found=0,
                remedy="initialize the canonical archive root to construct its baseline and admit declared trains",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = (
            _connect_archive_writer(path, profile=WRITE_CONNECTION_PROFILE, archive_root=configured_root)
            if tier is ArchiveTier.SOURCE
            else connect_measured(path)
        )
        owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=current_native_sql_lifetimes())
    else:
        if page_size is not None:
            raise ValueError("page_size is a creation-time choice; it cannot be applied to an existing tier")
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"durable tier is missing; refusing runtime initialization: {path}") from exc
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
            raise RuntimeError(f"durable tier is not a safe existing file; refusing runtime initialization: {path}")
        conn = (
            _connect_archive_writer(
                path, profile=WRITE_CONNECTION_PROFILE, archive_root=configured_root, existing_only=True
            )
            if tier is ArchiveTier.SOURCE
            else connect_measured(f"{path.resolve(strict=True).as_uri()}?mode=rw", uri=True)
        )
        owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=current_native_sql_lifetimes())
    primary: BaseException | None = None
    try:
        if page_size is not None:
            # Creation-only policy is SQL and shares the actual construction owner.
            apply_creation_page_size(conn, page_size)
        current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        # Both derived tiers refuse a stale identity before issuing DDL.
        # The daemon replaces disposable ops state at its startup seam.
        if current_version == required_version:
            converge_same_version_tier(conn, tier)
            return
        if current_version != 0:
            if current_version < required_version and tier in DURABLE_MIGRATION_TIERS:
                from polylogue.core.errors import SchemaSkew

                raise SchemaSkew(
                    tier=tier.value,
                    expected=required_version,
                    found=current_version,
                    remedy="admit the declared durable train through the canonical archive owner",
                )
            if current_version > required_version and tier in DURABLE_MIGRATION_TIERS:
                # Durable state ahead of the runtime is a stale runtime, never
                # a reason to move irreplaceable data aside.
                raise RuntimeError(
                    f"{path.name} schema version {current_version} is newer than this Polylogue runtime expects "
                    f"for the {tier.value} tier ({required_version}). Update the installed Polylogue runtime to "
                    "the build that created the database before opening it; do not move the database aside."
                )
            rebuild_command = (
                "polylogue ops reset --index, then restart polylogued"
                if tier is ArchiveTier.INDEX
                else f"mv {path} {path}.stale"
            )
            raise RuntimeError(
                f"{path} schema version {current_version} is not the current {tier.value} tier version "
                f"{required_version}; move it aside and rebuild the archive root, e.g.: {rebuild_command}"
            )
        if tier in DURABLE_MIGRATION_TIERS and required_version != spec.baseline_version:
            from polylogue.core.errors import SchemaSkew

            raise SchemaSkew(
                tier=tier.value,
                expected=required_version,
                found=current_version,
                remedy="initialize the canonical archive root before runtime tier opens",
            )
        # Database mode belongs to tier initialization, not each later writer
        # open: a mode pragma needs the tier's mode-transition lock and
        # rewrites the header under any reference seal prepared against it.
        from polylogue.storage.sqlite.connection_profile import initialize_tier_database_mode

        initialize_tier_database_mode(conn, rollback=tier is ArchiveTier.EMBEDDINGS or inactive_generation)
        initialize_archive_tier(conn, tier)
        if tier is ArchiveTier.INDEX:
            from polylogue.storage.sqlite.schema_manifest import assert_schema_manifest

            assert_schema_manifest(conn, tier)
    except BaseException as error:
        primary = error
        raise
    finally:
        if primary is None:
            owner.close()
        else:
            _close_failed_native_construction(owner, primary)


#: Bootstrap creates every durable tier together under one pending intent, so
#: an established archive without ``audit.db`` lost it outside Polylogue.
#: Durable evidence is never recreated in place: an empty audit tier would
#: silently claim a continuity history the archive no longer has.
_LOST_AUDIT_TIER_REFUSAL = (
    "established archive is missing audit.db; a lost durable tier is never recreated. "
    "Restore the archive root from a verified backup"
)


def _initialize_active_archive_root(root: Path, *, population_stage: _PopulationAdmission | None = None) -> None:
    """Create or initialize every tier database in an archive root."""
    from polylogue.storage.archive_identity import (
        ArchiveLocation,
        OwnedArchiveLocation,
        assert_owns_archive_location,
    )
    from polylogue.storage.sqlite.archive_tiers.archive_plan import (
        archive_format_marker_path,
        assert_archive_format_lineage,
        record_fresh_archive_format,
    )
    from polylogue.storage.sqlite.durable_change_train import (
        _record_fresh_durable_bootstrap,
        _record_fresh_durable_bootstrap_intent,
        _validate_fresh_durable_bootstrap_intent,
        durable_train_manifest_paths,
        execute_durable_change_train,
    )

    if population_stage is not None:
        from polylogue.storage.sqlite.population_admission import require_population_admission

        if require_population_admission(root) is not population_stage or population_stage.durable_versions is None:
            raise RuntimeError("population stage capability changed or lacks authenticated targets")

    # Ownership pins an existing directory descriptor. Fresh test and demo
    # archives legitimately arrive as a not-yet-created path, so create the
    # root before resolving and acquiring its identity. This is part of
    # bootstrap, not an authority bypass: the descriptor is still acquired
    # and checked before any tier is initialized.
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with OwnedArchiveLocation.acquire(
        ArchiveLocation.resolve(root),
        owner_id=f"bootstrap:{os.getpid()}",
        allow_reentrant=True,
    ) as owned:

        def assert_regular_audit_leaf() -> None:
            """Reject an audit pathname that could redirect durable authority outside this root."""

            audit_path = root / archive_tier_spec(ArchiveTier.AUDIT).filename
            try:
                audit_path.lstat()
            except FileNotFoundError:
                return
            except OSError as exc:
                raise RuntimeError(f"cannot inspect audit tier leaf: {audit_path}") from exc
            try:
                assert_verified_audit_leaf(audit_path)
            except AuditLeafError as exc:
                raise RuntimeError(str(exc)) from exc

        def assert_owned_root() -> None:
            """Refuse pathname writes after the owned root has been replaced."""
            assert_owns_archive_location(owned, ArchiveLocation.resolve(root))

        # Classify the archive after acquiring ownership. Another process may
        # publish a marker or durable train while the probe is in flight.
        assert_owned_root()
        assert_regular_audit_leaf()
        durable_tier_exists = any(
            (root / archive_tier_spec(tier).filename).exists() for tier in DURABLE_MIGRATION_TIERS
        )
        manifest_root = root / ".maintenance-state" / "durable-change-trains"
        has_durable_train_state = bool(durable_train_manifest_paths(manifest_root))
        has_bootstrap_marker = (manifest_root / ".bootstrap").is_file()
        pending_bootstrap_path = manifest_root / ".bootstrap.pending"
        has_pending_bootstrap = pending_bootstrap_path.is_file()

        # ``user_version == 1`` now belongs to a new format lineage.  Admit an
        # established root only through its marker before any tier initializer
        # gets a writable connection.  A historical v1 file therefore cannot
        # be restamped into apparent compatibility.
        format_marker = archive_format_marker_path(root)
        # A pending intent is the authenticated recovery authority for a
        # partially-created fresh archive.  Validate it before enforcing the
        # completed format marker, otherwise a crash between the first tier
        # and marker publication becomes unrecoverable.
        if has_pending_bootstrap:
            _validate_fresh_durable_bootstrap_intent(root)
            if has_durable_train_state:
                raise RuntimeError(
                    "fresh durable bootstrap intent conflicts with durable train state; "
                    "refusing to guess which authority is current"
                )
        any_durable_tier_exists = any(
            (root / archive_tier_spec(tier).filename).exists() or (root / archive_tier_spec(tier).filename).is_symlink()
            for tier in DURABLE_MIGRATION_TIERS
        )
        # A lineage member that has lost only ``audit.db`` gets the named
        # lost-tier refusal, not the generic marker text. Prove the surviving
        # durable pair belongs to this lineage -- which also reports a
        # symlinked or multiply-linked source/user tier first, because that is
        # the more severe finding -- and then refuse by name.
        established_pair_without_audit = (
            format_marker.is_file()
            and not (root / archive_tier_spec(ArchiveTier.AUDIT).filename).exists()
            and not (root / archive_tier_spec(ArchiveTier.AUDIT).filename).is_symlink()
            and all(
                (root / archive_tier_spec(tier).filename).is_file() for tier in (ArchiveTier.SOURCE, ArchiveTier.USER)
            )
        )
        if any_durable_tier_exists and not (has_pending_bootstrap and not has_bootstrap_marker):
            if established_pair_without_audit:
                assert_archive_format_lineage(root, tiers=frozenset({ArchiveTier.SOURCE, ArchiveTier.USER}))
                from polylogue.storage.sqlite.durable_change_train import assert_released_durable_tier_lineage

                for surviving in (ArchiveTier.SOURCE, ArchiveTier.USER):
                    with contextlib.closing(
                        open_readonly_connection(root / archive_tier_spec(surviving).filename, validate_schema=False)
                    ) as connection:
                        assert_released_durable_tier_lineage(root, surviving, connection)
                raise RuntimeError(_LOST_AUDIT_TIER_REFUSAL)
            assert_archive_format_lineage(root)
        elif format_marker.exists() and not any_durable_tier_exists:
            raise RuntimeError(f"archive format marker exists without a six-tier archive: {format_marker}")

        fresh_durable_bootstrap = (
            not durable_tier_exists
            and not has_durable_train_state
            and not has_bootstrap_marker
            and not has_pending_bootstrap
        )
        recovering_fresh_durable_bootstrap = fresh_durable_bootstrap or (
            has_pending_bootstrap and not has_bootstrap_marker
        )
        if population_stage is not None and not fresh_durable_bootstrap:
            raise RuntimeError("population stage requires a fresh destination durable core")
        if fresh_durable_bootstrap:
            assert_owned_root()
            _record_fresh_durable_bootstrap_intent(root)
        established_archive = has_bootstrap_marker or (
            (root / archive_tier_spec(ArchiveTier.SOURCE).filename).is_file()
            and (root / archive_tier_spec(ArchiveTier.USER).filename).is_file()
        )
        # A durable tier that is a symlink (or otherwise not a lone regular
        # file) is a tamper/containment finding, and the change-train
        # reconciliation below refuses startup for it by name. Announcing
        # "missing audit.db" first would misreport that condition as a lost
        # tier in an archive whose tiers were in fact replaced. Report the more
        # severe finding first by deferring to the barrier below.
        durable_tier_files_are_safe = all(
            not (path := root / archive_tier_spec(tier).filename).is_symlink() and (path.is_file() or not path.exists())
            for tier in (ArchiveTier.SOURCE, ArchiveTier.USER)
        )
        if (
            durable_tier_exists
            and not recovering_fresh_durable_bootstrap
            and established_archive
            and durable_tier_files_are_safe
            and not (root / archive_tier_spec(ArchiveTier.AUDIT).filename).is_file()
        ):
            raise RuntimeError(_LOST_AUDIT_TIER_REFUSAL)
        if not recovering_fresh_durable_bootstrap:
            assert_owned_root()
            reconcile_durable_change_trains_on_startup(root)
        location = ArchiveLocation.resolve(root)

        def advance_declared_trains() -> None:
            from polylogue.storage.sqlite.migration_runner import durable_migration_claims

            for tier in sorted(DURABLE_MIGRATION_TIERS, key=lambda item: item.value):
                path = root / archive_tier_spec(tier).filename
                if not path.exists():
                    continue
                while True:
                    assert_owned_root()
                    with contextlib.closing(open_readonly_connection(path, validate_schema=False)) as probe:
                        current = int(probe.execute("PRAGMA user_version").fetchone()[0])
                    target = (
                        dict(population_stage.durable_versions)[tier.value]
                        if population_stage is not None and population_stage.durable_versions is not None
                        else archive_tier_spec(tier).version
                    )
                    if current >= target:
                        break
                    claim = next(
                        (claim for claim in durable_migration_claims(tier) if claim.target_version == current + 1), None
                    )
                    if claim is None:
                        from polylogue.core.errors import SchemaSkew

                        raise SchemaSkew(
                            tier=tier.value,
                            expected=target,
                            found=current,
                            remedy="the next numbered migration must have a declared train",
                        )
                    backup_manifest = None
                    if claim.requires_backup:
                        from polylogue.storage.backup_package import create_pre_migration_backup

                        backup_manifest = create_pre_migration_backup(
                            root,
                            tier=tier.value,
                            current_version=current,
                            target_version=current + 1,
                            archive_owner=owned,
                        )
                    execute_durable_change_train(
                        root,
                        tier,
                        backup_manifest=backup_manifest,
                        daemon_stopped_evidence_ref="proof:bootstrap-before-runtime-open",
                        single_writer_evidence_ref="proof:bootstrap-owned-archive",
                        release_archive_ownership=lambda: None,
                    )
                    with contextlib.closing(open_readonly_connection(path, validate_schema=False)) as probe:
                        advanced = int(probe.execute("PRAGMA user_version").fetchone()[0])
                    if advanced != current + 1:
                        raise RuntimeError("declared bootstrap train did not advance exactly one version")

        if not recovering_fresh_durable_bootstrap:
            advance_declared_trains()
        for spec in ARCHIVE_TIER_SPECS.values():
            assert_owned_root()
            tier_path = location.active_index_path if spec.tier is ArchiveTier.INDEX else root / spec.filename
            initialize_archive_database(
                tier_path,
                spec.tier,
                expected_version=spec.baseline_version if recovering_fresh_durable_bootstrap else spec.version,
            )
        # Mutation composition performs source/audit reconciliation immediately
        # before it consumes authority. Ordinary archive opens stay read-only
        # with respect to continuity, including their steady-state path.
        if recovering_fresh_durable_bootstrap:
            assert_owned_root()
            # Keep the pending intent until both completed markers exist.
            # Publication may have succeeded just before a crash: validate that
            # same fresh bootstrap's marker instead of attempting to replace it.
            if format_marker.exists():
                assert_archive_format_lineage(root)
            else:
                record_fresh_archive_format(root)
            _record_fresh_durable_bootstrap(root)
            advance_declared_trains()
        elif has_pending_bootstrap:
            # A crash after publishing the completed marker but before
            # removing the intent is harmless. Keep the intent until the
            # completed marker has passed normal startup reconciliation.
            assert_owned_root()
            pending_bootstrap_path.unlink(missing_ok=True)


def _archive_generation_token(root: Path) -> tuple[object, ...]:
    """Identify the archive *generation* ``_initialize_active_archive_root`` validated.

    Tier components are file identities, never content mtimes or sizes: the
    daemon writes into its tiers constantly, so a token that moved on an
    ordinary write would identify a new generation on every batch. The format
    marker is immutable after publication, so its bytes identify the admitted
    lineage and must invalidate the memo if they change.

    What a difference here means, and why each term is present:

    * ``os.getpid`` -- a forked child inherits the parent's memo dict but not
      its open descriptors or its lease. It revalidates.
    * the root directory's ``(st_dev, st_ino)`` and each tier file's -- a
      replaced, relinked, restored-from-backup or newly created tier is a
      different file, so it is a different generation.
    * the active index pointer and its target identity -- an index
      *promotion* swaps which generation the archive reads and writes, while
      leaving the configured ``index.db`` pathname alone.
    * the durable-train manifest directory's entries with their sizes and
      mtimes -- a durable migration both adds manifest files and appends to
      them, and that is exactly the "schema change" case that must never be
      skipped. This directory is untouched in steady state.
    * the format marker's content digest and the other markers' presence --
      each one selects a different branch of the bootstrap body.

    A *code* schema change cannot move within a process (the derived identity
    is an import-time constant), and the tier-file identities above catch an
    archive swapped underneath a running process. Anything this token cannot
    see belongs to another writer, which the single-writer contract excludes.
    """
    from polylogue.storage.archive_identity import ArchiveLocation
    from polylogue.storage.sqlite.archive_tiers.archive_plan import archive_format_marker_path

    def file_identity(path: Path) -> tuple[object, ...]:
        try:
            info = path.stat()
        except OSError:
            return (None, None)
        return (info.st_dev, info.st_ino)

    def format_marker_digest(path: Path) -> str | None:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

    location = ArchiveLocation.resolve(root)
    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    manifest_entries: tuple[tuple[str, int, int], ...]
    try:
        with os.scandir(manifest_root) as entries:
            manifest_entries = tuple(
                sorted(
                    (entry.name, entry.stat().st_size, entry.stat().st_mtime_ns) for entry in entries if entry.is_file()
                )
            )
    except OSError:
        manifest_entries = ()

    return (
        os.getpid(),
        file_identity(root),
        tuple((tier.name, tier.stable_id) for tier in location.configured_tiers),
        location.active_index.stable_id,
        str(location.active_index.resolved_path),
        None if location.active_pointer is None else str(location.active_pointer),
        None if location.shadow_index is None else location.shadow_index.stable_id,
        format_marker_digest(archive_format_marker_path(root)),
        (manifest_root / ".bootstrap").is_file(),
        (manifest_root / ".bootstrap.pending").is_file(),
        manifest_entries,
    )


def invalidate_active_archive_bootstrap(root: Path | None = None) -> None:
    """Force the next bootstrap of ``root`` (or of every root) to revalidate.

    The generation token already notices a replaced tier, a promotion and a
    durable-train change. This is the explicit escape hatch for a caller that
    knows it has invalidated bootstrap state by some route the token cannot
    observe, and the hook tests use to prove the memo is a memo rather than a
    once-flag.
    """
    with _ACTIVE_ARCHIVE_BOOTSTRAP_LOCK:
        if root is None:
            _ACTIVE_ARCHIVE_BOOTSTRAP_GENERATIONS.clear()
        else:
            _ACTIVE_ARCHIVE_BOOTSTRAP_GENERATIONS.pop(str(root.absolute()), None)


def active_archive_bootstrap_validation_count() -> int:
    """How many times the six-tier validation body has actually run in this process.

    Diagnostic only -- nothing branches on it. A fresh-build cost assertion
    reads it to prove the per-batch bootstrap is paid once per generation
    rather than once per open (polylogue-q53j4 AC1).
    """
    return _ACTIVE_ARCHIVE_BOOTSTRAP_VALIDATIONS


def _initialize_population_archive_stage(root: Path) -> None:
    """Construct baseline and declared package targets under exact pending custody.

    The same constructor and train owner serve ordinary runtime initialization.
    Population targets are authenticated and bound into the held capability;
    this stage cannot create an independently usable partial runtime archive.
    """
    from polylogue.storage.sqlite.population_admission import require_population_admission

    admission = require_population_admission(root)
    if admission.durable_versions is None:
        from polylogue.storage.sqlite.population_admission import ArchivePopulationPendingError

        raise ArchivePopulationPendingError("population stage lacks authenticated durable targets")
    with _ACTIVE_ARCHIVE_BOOTSTRAP_LOCK:
        invalidate_active_archive_bootstrap(root)
        _initialize_active_archive_root(root, population_stage=admission)


def initialize_active_archive_root(root: Path) -> None:
    """Create or initialize every active archive tier under archive custody."""

    global _ACTIVE_ARCHIVE_BOOTSTRAP_VALIDATIONS

    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    assert_population_admitted(root)
    from polylogue.storage.archive_tuple_location import ArchiveTupleError, is_archive_tuple_candidate_path
    from polylogue.storage.sqlite.write_lease import require_write_lease, write_lease

    if is_archive_tuple_candidate_path(root):
        raise ArchiveTupleError(
            "inactive archive tuple roots require typed per-tier destinations; refusing active-root bootstrap"
        )

    require_write_lease("active archive bootstrap", archive_root=root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with write_lease("active archive bootstrap", archive_root=root):
        _initialize_active_archive_root_under_lease(root)


def _initialize_active_archive_root_under_lease(root: Path) -> None:
    """Materialize the active tiers after their writer has been admitted."""

    global _ACTIVE_ARCHIVE_BOOTSTRAP_VALIDATIONS

    # Active-root bootstrap creates or opens every writable tier.  It is a
    # daemon-owned operation when process-wide lease enforcement is armed;
    # inactive tuple destinations and scratch files use the lower-level
    # initializer directly and remain intentionally independent of this gate.
    # The authority check is never memoized: it decides whether *this* caller
    # may bootstrap, which is a fact about the caller, not about the archive.
    with _ACTIVE_ARCHIVE_BOOTSTRAP_LOCK:
        memo_key = str(root.absolute())
        observed = _archive_generation_token(root)
        if _ACTIVE_ARCHIVE_BOOTSTRAP_GENERATIONS.get(memo_key) == observed:
            return
        # Drop the stale record *before* the body runs: a failed validation
        # must leave the next call revalidating, not inherit a token from a
        # generation this process never finished bootstrapping.
        _ACTIVE_ARCHIVE_BOOTSTRAP_GENERATIONS.pop(memo_key, None)
        _ACTIVE_ARCHIVE_BOOTSTRAP_VALIDATIONS += 1
        _initialize_active_archive_root(root)
        # Recompute rather than storing ``observed``: bootstrap creates the
        # tiers and publishes the markers, so the generation it just
        # established is the one after the body, not the one before it.
        _ACTIVE_ARCHIVE_BOOTSTRAP_GENERATIONS[memo_key] = _archive_generation_token(root)


def reconcile_durable_change_trains_on_startup(root: Path) -> tuple[Path, ...]:
    """Reconcile persisted durable trains without executing migration SQL."""
    from polylogue.storage.sqlite.durable_change_train import reconcile_durable_change_train_startup

    return reconcile_durable_change_train_startup(root)


def open_initialized_tier_connection(
    path: Path | str,
    tier: ArchiveTier,
    *,
    timeout: float = 30.0,
    busy_timeout_ms: int | None = None,
    daemon: bool = True,
    archive_root: Path | str | None = None,
) -> sqlite3.Connection:
    """Open a tier database that may not exist yet, materialise it, and validate.

    Runtime creation is possible only when the immutable baseline equals the
    runtime target. Durable numbered trains belong to canonical root bootstrap;
    this connection owner never creates a baseline and calls it current.
    """
    from polylogue.storage.sqlite.connection_profile import (
        NativeSQLCustodyOwner,
        _close_failed_native_construction,
        assert_tier_schema_supported,
        open_connection,
        open_daemon_connection,
    )

    spec = archive_tier_spec(tier)
    if tier in DURABLE_MIGRATION_TIERS and spec.version != spec.baseline_version and not Path(path).exists():
        from polylogue.core.errors import SchemaSkew

        raise SchemaSkew(
            tier=tier.value,
            expected=spec.version,
            found=0,
            remedy="initialize the canonical archive root before runtime tier opens",
        )

    if daemon:
        conn = open_daemon_connection(
            path,
            timeout=timeout,
            busy_timeout_ms=busy_timeout_ms,
            tier=tier,
            validate_schema=False,
            archive_root=archive_root,
        )
    else:
        conn = open_connection(path, timeout=timeout, tier=tier, validate_schema=False, archive_root=archive_root)
    owner = NativeSQLCustodyOwner(conn)
    try:
        stored_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        required_version = archive_tier_spec(tier).version
        if stored_version == 0 and tier in DURABLE_MIGRATION_TIERS and required_version != spec.baseline_version:
            from polylogue.core.errors import SchemaSkew

            raise SchemaSkew(
                tier=tier.value,
                expected=required_version,
                found=0,
                remedy="initialize the canonical archive root before runtime tier opens",
            )
        if stored_version not in (0, required_version):
            from polylogue.core.errors import SchemaSkew

            raise SchemaSkew(
                tier=tier.value,
                expected=required_version,
                found=stored_version,
                remedy="rebuild or migrate the tier with the current runtime before retrying",
            )
        # A tier already stamped at the current version is materialised; the
        # declared same-version policy is what it needs, not a second full
        # initialization. Version 0 is the create-it case and keeps the DDL
        # route, which is what stamps the version this branch reads.
        if stored_version == required_version:
            converge_same_version_tier(conn, tier)
        else:
            initialize_archive_tier(conn, tier)
        assert_tier_schema_supported(conn, path, tier)
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner.handoff()


__all__ = [
    "ARCHIVE_TIER_SPECS",
    "DurabilityClass",
    "ArchiveTierSpec",
    "converge_same_version_tier",
    "active_archive_bootstrap_validation_count",
    "initialize_active_archive_root",
    "initialize_archive_database",
    "invalidate_active_archive_bootstrap",
    "initialize_archive_tier",
    "open_initialized_tier_connection",
    "reconcile_durable_change_trains_on_startup",
    "archive_tier_spec",
]
