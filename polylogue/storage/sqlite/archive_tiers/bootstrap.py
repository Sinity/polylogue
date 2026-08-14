"""Fresh bootstrap helpers for archive databases."""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.index_convergence import apply_index_benign_ddl_convergence
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.audit_leaf import AuditLeafError, assert_verified_audit_leaf
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

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


def archive_tier_spec(tier: ArchiveTier) -> ArchiveTierSpec:
    """Return the database-file spec for one durability tier."""
    return ARCHIVE_TIER_SPECS[tier]


def initialize_archive_tier(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    """Initialize a fresh archive tier database on an already-open connection."""
    spec = archive_tier_spec(tier)
    conn.execute("PRAGMA foreign_keys = ON")
    if tier is ArchiveTier.EMBEDDINGS:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            raise RuntimeError("archive embeddings initialization requires sqlite-vec") from error
    conn.executescript(spec.ddl)
    if tier is ArchiveTier.OPS:
        from polylogue.storage.sqlite.archive_tiers.ops_write import (
            ensure_embedding_catchup_run_outcome_columns,
            ensure_ops_status_checks,
        )

        _ensure_ops_runtime_columns(conn)
        _ensure_ops_cursor_lag_sample_columns(conn)
        _ensure_ops_ingest_attempt_outcome_columns(conn)
        ensure_embedding_catchup_run_outcome_columns(conn)
        ensure_ops_status_checks(conn)
        _ensure_schema_drift_samples_check(conn)
        _apply_ops_benign_ddl_convergence(conn)
    if tier is ArchiveTier.INDEX:
        # Fresh init never had a registered drop's target table, and any
        # additive registry entry lands identically to canonical DDL -- this
        # is a no-op today, kept for fresh-init/converged-live parity (see
        # index_convergence.py module docstring).
        apply_index_benign_ddl_convergence(conn)
    if tier is ArchiveTier.USER:
        _ensure_user_annotation_schemas(conn)
    conn.execute(f"PRAGMA user_version = {spec.version}")
    conn.commit()


def _ensure_schema_drift_samples_check(conn: sqlite3.Connection) -> None:
    """Repair a ``schema_drift_samples`` table bootstrapped with a stale CHECK.

    ``CREATE TABLE IF NOT EXISTS`` (the ``OPS_DDL`` reapply path every
    ``initialize_archive_tier(..., ArchiveTier.OPS)`` call goes through) never
    rewrites an *existing* table's constraints. An ops.db bootstrapped before
    the ``literal_check(DriftClassification)`` fix (#3451 / polylogue-u6tl)
    keeps a live CHECK naming only 3 of ``DriftClassification``'s 4 values --
    ``known_field_unread`` rows raise ``sqlite3.IntegrityError`` on every
    insert attempt against that archive forever, not just until the code
    ships. Detect the stale CHECK via ``sqlite_master.sql`` and drop+recreate:
    ``schema_drift_samples`` is disposable bounded telemetry (polylogue-da1),
    so losing its rows on repair is an accepted, documented cost -- there is
    no migration/version-bump ceremony for this tier.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'schema_drift_samples'"
    ).fetchone()
    if row is None or row[0] is None:
        return
    if "known_field_unread" in row[0]:
        return
    from polylogue.storage.sqlite.archive_tiers.ops import SCHEMA_DRIFT_SAMPLES_DDL

    conn.execute("DROP TABLE IF EXISTS schema_drift_samples")
    conn.executescript(SCHEMA_DRIFT_SAMPLES_DDL)


def _apply_ops_benign_ddl_convergence(conn: sqlite3.Connection) -> None:
    """Apply the declared idempotent OPS same-version fast-forward plan."""
    from polylogue.storage.sqlite.archive_tiers.ops import OPS_BENIGN_DDL_CONVERGENCE_PLAN

    for statement in OPS_BENIGN_DDL_CONVERGENCE_PLAN:
        conn.execute(statement)


def _ensure_user_annotation_schemas(conn: sqlite3.Connection) -> None:
    """Replay packaged schema rows without changing the user-tier DDL version."""

    from polylogue.storage.sqlite.archive_tiers.user_annotations import persist_builtin_annotation_schemas

    persist_builtin_annotation_schemas(conn, registered_at_ms=0)


def _ensure_ops_runtime_columns(conn: sqlite3.Connection) -> None:
    """Ensure disposable OPS-tier databases have the current cursor shape."""
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(ingest_cursor)")}
    additions = {
        "record_count": "INTEGER NOT NULL DEFAULT 0 CHECK(record_count >= 0)",
        "last_record_ts_ms": "INTEGER",
        "failure_count": "INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0)",
        "next_retry_at": "TEXT",
        "excluded": "INTEGER NOT NULL DEFAULT 0 CHECK(excluded IN (0, 1))",
        "deferred_end_offset": "INTEGER",
    }
    for name, definition in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE ingest_cursor ADD COLUMN {name} {definition}")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ingest_cursor_attention
        ON ingest_cursor(failure_count, excluded, source_path)
        """
    )


def _ensure_ops_ingest_attempt_outcome_columns(conn: sqlite3.Connection) -> None:
    """Ensure disposable OPS-tier ingest attempts carry a typed disposition (polylogue-cnu3).

    Pre-existing rows (written before this vocabulary existed) get
    ``outcome_code='legacy_unknown'`` via the column default -- never
    guess-classified into a real outcome (AC4).
    """
    from polylogue.core.enums import IngestOutcome
    from polylogue.storage.sqlite.archive_tiers.common import check

    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(ingest_attempts)")}
    additions = {
        "outcome_code": f"TEXT NOT NULL DEFAULT 'legacy_unknown' CHECK ({check('outcome_code', IngestOutcome)})",
        "retryable": "INTEGER CHECK(retryable IN (0, 1))",
        "evidence_ref": "TEXT",
        "diagnostic": "TEXT",
        "remediation": "TEXT",
    }
    for name, definition in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE ingest_attempts ADD COLUMN {name} {definition}")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ingest_attempts_outcome_code
        ON ingest_attempts(outcome_code, started_at_ms)
        """
    )


def _ensure_ops_cursor_lag_sample_columns(conn: sqlite3.Connection) -> None:
    """Ensure disposable OPS-tier cursor lag samples carry family rollups."""
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(cursor_lag_samples)")}
    additions = {
        "family": "TEXT",
        "stuck_file_count": "INTEGER NOT NULL DEFAULT 1 CHECK(stuck_file_count >= 0)",
        "p50_lag_ms": "INTEGER NOT NULL DEFAULT 0 CHECK(p50_lag_ms >= 0)",
        "p95_lag_ms": "INTEGER NOT NULL DEFAULT 0 CHECK(p95_lag_ms >= 0)",
    }
    for name, definition in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE cursor_lag_samples ADD COLUMN {name} {definition}")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cursor_lag_samples_family_time
        ON cursor_lag_samples(family, sampled_at_ms DESC)
        """
    )


def initialize_archive_database(
    path: Path,
    tier: ArchiveTier,
    *,
    allow_create: bool = True,
    expected_version: int | None = None,
) -> None:
    """Create or initialize one archive tier database file."""
    if allow_create:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    else:
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"durable tier is missing; refusing runtime initialization: {path}") from exc
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
            raise RuntimeError(f"durable tier is not a safe existing file; refusing runtime initialization: {path}")
        conn = sqlite3.connect(f"{path.resolve(strict=True).as_uri()}?mode=rw", uri=True)
    try:
        current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        required_version = archive_tier_spec(tier).version if expected_version is None else expected_version
        if current_version == required_version:
            # ops.db is disposable and evolves through idempotent additive DDL
            # without version bumps. Re-apply it so existing same-version
            # archives receive newly introduced tables and indexes.
            if tier is ArchiveTier.OPS:
                initialize_archive_tier(conn, tier)
            elif tier is ArchiveTier.USER:
                _ensure_user_annotation_schemas(conn)
                conn.commit()
            elif tier is ArchiveTier.INDEX:
                # index.db is rebuildable, but a *benign* registered DDL delta
                # (idempotent, data-non-transforming, zero-consumer at this
                # exact version -- see index_convergence.py) does not need the
                # full rebuild a schema-version bump would force. Re-apply the
                # registry on every same-version open so an already-populated
                # archive converges without touching INDEX_SCHEMA_VERSION.
                apply_index_benign_ddl_convergence(conn)
                conn.commit()
            return
        if current_version != 0:
            if current_version < required_version and tier in DURABLE_MIGRATION_TIERS:
                raise RuntimeError(
                    f"{path.name} schema version {current_version} is older than the current {tier.value} tier "
                    f"version {required_version}; run an explicit durable-tier migration with a verified backup "
                    "manifest"
                )
            if tier is ArchiveTier.INDEX and current_version < required_version:
                # index.db is rebuildable, but a bounded set of version gaps
                # are DECLARED clone-safe (polylogue.storage.sqlite.lifecycle)
                # -- no raw reparse, no consumer-visible semantic change. Apply
                # the declared plan instead of forcing the full rebuild the
                # declaration exists to avoid (polylogue-t3gk). Falls through
                # to the rebuild-required error below when no eligible plan
                # covers this exact gap (e.g. a SEMANTIC_REPARSE declaration
                # is in the span).
                from polylogue.storage.sqlite.archive_tiers.index_fast_forward_executor import (
                    apply_index_fast_forward,
                )
                from polylogue.storage.sqlite.lifecycle import index_fast_forward_plan

                plan = index_fast_forward_plan(current_version, required_version)
                if plan is not None:
                    apply_index_fast_forward(conn, plan)
                    return
            rebuild_command = (
                "polylogue ops reset --index && polylogued run"
                if tier is ArchiveTier.INDEX
                else f"mv {path} {path}.stale"
            )
            raise RuntimeError(
                f"{path} schema version {current_version} is not the current {tier.value} tier version "
                f"{required_version}; move it aside and rebuild the archive root, e.g.: {rebuild_command}"
            )
        initialize_archive_tier(conn, tier)
    finally:
        conn.close()


def _initialize_active_archive_root(root: Path) -> None:
    """Create or initialize every tier database in an archive root."""
    from polylogue.operations.durable_change_train import audit_adoption_receipt_path, recover_pending_audit_adoption
    from polylogue.storage.archive_identity import (
        ArchiveLocation,
        OwnedArchiveLocation,
        assert_owns_archive_location,
    )
    from polylogue.storage.sqlite.durable_change_train import (
        _record_fresh_durable_bootstrap,
        _record_fresh_durable_bootstrap_intent,
        _validate_fresh_durable_bootstrap_intent,
    )

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
        pending_audit_adoption = audit_adoption_receipt_path(root).exists()
        has_durable_train_state = any(
            path.name not in {"audit-adoption.json", "audit-continuity.json"}
            and not path.name.startswith("audit-restore.")
            for path in manifest_root.glob("*.json")
        )
        has_bootstrap_marker = (manifest_root / ".bootstrap").is_file()
        pending_bootstrap_path = manifest_root / ".bootstrap.pending"
        has_pending_bootstrap = pending_bootstrap_path.is_file()

        def classify_paths() -> tuple[bool, bool]:
            durable_exists = any((root / archive_tier_spec(tier).filename).exists() for tier in DURABLE_MIGRATION_TIERS)
            adoption = (
                (root / archive_tier_spec(ArchiveTier.SOURCE).filename).is_file()
                and all((root / archive_tier_spec(tier).filename).is_file() for tier in DURABLE_MIGRATION_TIERS)
                and manifest_root.is_dir()
                and not has_durable_train_state
                and not has_bootstrap_marker
                and not has_pending_bootstrap
            )
            return durable_exists, adoption

        durable_tier_exists, pre_marker_adoption = classify_paths()
        if has_pending_bootstrap:
            _validate_fresh_durable_bootstrap_intent(root)
            if has_durable_train_state:
                raise RuntimeError(
                    "fresh durable bootstrap intent conflicts with durable train state; "
                    "refusing to guess which authority is current"
                )
        fresh_durable_bootstrap = (
            not durable_tier_exists
            and not has_durable_train_state
            and not has_bootstrap_marker
            and not has_pending_bootstrap
        )
        recovering_fresh_durable_bootstrap = fresh_durable_bootstrap or (
            has_pending_bootstrap and not has_bootstrap_marker
        )
        if fresh_durable_bootstrap:
            assert_owned_root()
            _record_fresh_durable_bootstrap_intent(root)
        if pending_audit_adoption:
            assert_owned_root()
            recover_pending_audit_adoption(root)
            # Receipt-backed recovery can add audit.db to a legacy archive.
            # Recompute the path-sensitive classification before deciding
            # whether startup must create the missing bootstrap marker.
            durable_tier_exists, pre_marker_adoption = classify_paths()
        established_archive = has_bootstrap_marker or (
            (root / archive_tier_spec(ArchiveTier.SOURCE).filename).is_file()
            and (root / archive_tier_spec(ArchiveTier.USER).filename).is_file()
        )
        if (
            durable_tier_exists
            and not recovering_fresh_durable_bootstrap
            and established_archive
            and not (root / archive_tier_spec(ArchiveTier.AUDIT).filename).is_file()
        ):
            raise RuntimeError(
                "established archive is missing audit.db; use maintenance migrate-tier audit "
                "--adopt-established-audit with a verified full_evidence backup"
            )
        if not recovering_fresh_durable_bootstrap and not pre_marker_adoption:
            assert_owned_root()
            reconcile_durable_change_trains_on_startup(root)
        location = ArchiveLocation.resolve(root)
        for spec in ARCHIVE_TIER_SPECS.values():
            assert_owned_root()
            tier_path = location.active_index_path if spec.tier is ArchiveTier.INDEX else root / spec.filename
            initialize_archive_database(tier_path, spec.tier)
        # Mutation composition performs source/audit reconciliation immediately
        # before it consumes authority. Ordinary archive opens stay read-only
        # with respect to continuity, including their steady-state path.
        if recovering_fresh_durable_bootstrap:
            assert_owned_root()
            _record_fresh_durable_bootstrap(root)
        elif pre_marker_adoption:
            from polylogue.storage.sqlite.durable_change_train import _adopt_pre_marker_durable_bootstrap

            assert_owned_root()
            _adopt_pre_marker_durable_bootstrap(root)
            assert_owned_root()
            reconcile_durable_change_trains_on_startup(root)
        elif has_pending_bootstrap:
            # A crash after publishing the completed marker but before
            # removing the intent is harmless. Keep the intent until the
            # completed marker has passed normal startup reconciliation.
            assert_owned_root()
            pending_bootstrap_path.unlink(missing_ok=True)


def initialize_active_archive_root(root: Path) -> None:
    """Create or initialize every active archive tier under one local bootstrap owner."""

    with _ACTIVE_ARCHIVE_BOOTSTRAP_LOCK:
        _initialize_active_archive_root(root)


def reconcile_durable_change_trains_on_startup(root: Path) -> tuple[Path, ...]:
    """Reconcile persisted durable trains without executing migration SQL."""
    from polylogue.storage.sqlite.durable_change_train import reconcile_durable_change_train_startup

    return reconcile_durable_change_train_startup(root)


__all__ = [
    "ARCHIVE_TIER_SPECS",
    "DurabilityClass",
    "ArchiveTierSpec",
    "initialize_active_archive_root",
    "initialize_archive_database",
    "initialize_archive_tier",
    "reconcile_durable_change_trains_on_startup",
    "archive_tier_spec",
]
