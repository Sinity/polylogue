"""Clone-first archive schema fast-forward without raw replay.

This is deliberately an operator actuator rather than a runtime migration
path.  It accepts only the observed v35 file set, rejects any Beads evidence,
and works on staging clones.  The index and embeddings changes are derived-tier
copy-forwards; source/user continue to use the shipped durable migration
runner during the stopped-daemon activation window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_DDL, EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import migrate_archive_tier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

RECEIPT_SCHEMA: Final = "polylogue.archive-schema-fast-forward.v1"
_BEADS_ORIGIN: Final = "beads-issue"
_INDEX_COPY_FORWARD_TABLES: Final = ("sessions", "session_links")
_SQLITE_SIDECARS: Final = ("-wal", "-shm", "-journal")


class SchemaFastForwardError(RuntimeError):
    """Raised before a schema fast-forward can mutate an active archive."""


@dataclass(frozen=True, slots=True)
class DatabaseEvidence:
    path: str
    user_version: int
    sha256: str
    size_bytes: int
    table_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class CloneForwardResult:
    source: DatabaseEvidence
    clone: DatabaseEvidence
    foreign_key_declarations_preserved: bool
    foreign_key_check: tuple[str, ...]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise SchemaFastForwardError("receipt write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _require_no_sidecars(path: Path) -> None:
    present = [str(Path(f"{path}{suffix}")) for suffix in _SQLITE_SIDECARS if Path(f"{path}{suffix}").exists()]
    if present:
        raise SchemaFastForwardError(f"SQLite sidecars make clone proof ambiguous: {', '.join(present)}")


def _open_immutable_readonly(path: Path) -> sqlite3.Connection:
    """Open stable archive bytes without creating a WAL shared-memory sidecar.

    Clone proof deliberately rejects a database with a WAL, journal, or SHM
    sidecar: those bytes are not a complete, stable snapshot.  Check that
    invariant *before* opening SQLite, then use immutable mode so a census of
    a sidecar-free WAL-mode archive cannot create a new ``-shm`` file itself.
    """
    resolved = path.resolve(strict=True)
    _require_no_sidecars(resolved)
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro&immutable=1", uri=True)


def _table_names(conn: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    )


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        name: int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        for name in _table_names(conn)
        if not name.startswith("messages_fts")
    }


def _database_evidence(path: Path) -> DatabaseEvidence:
    with _open_immutable_readonly(path) as conn:
        _load_vec_if_required(conn)
        return DatabaseEvidence(
            path=str(path),
            user_version=int(conn.execute("PRAGMA user_version").fetchone()[0]),
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
            table_counts=_table_counts(conn),
        )


def _require_receipt_identity(payload: dict[str, object], key: str, path: Path) -> None:
    raw_evidence = payload.get(key)
    if key in {"index", "embeddings", "ops"} and isinstance(raw_evidence, dict):
        raw_evidence = raw_evidence.get("source")
    if not isinstance(raw_evidence, dict):
        raise SchemaFastForwardError(f"prepared receipt lacks {key} source evidence")
    expected = cast(dict[str, object], raw_evidence)
    actual = asdict(_database_evidence(path))
    fields = ("path", "user_version", "sha256", "size_bytes")
    if any(expected.get(field) != actual.get(field) for field in fields):
        raise SchemaFastForwardError(f"{key} changed since clone proof")


def _require_service_stopped(service: str | None) -> None:
    if service is None:
        return
    result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", service], check=False)
    if result.returncode == 0:
        raise SchemaFastForwardError(f"refusing activation while {service} is active; stop it explicitly first")


def _foreign_key_declarations(conn: sqlite3.Connection) -> dict[str, tuple[tuple[object, ...], ...]]:
    return {
        table: tuple(tuple(row) for row in conn.execute(f'PRAGMA foreign_key_list("{table}")'))
        for table in _table_names(conn)
    }


def _canonical_index_objects() -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    with sqlite3.connect(":memory:") as conn:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            raise SchemaFastForwardError(f"canonical index DDL requires sqlite-vec: {error}")
        conn.executescript(INDEX_DDL)
        tables = {
            name: str(row[0])
            for name in _INDEX_COPY_FORWARD_TABLES
            if (row := conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())
            and row[0]
        }
        indexes = {
            name: tuple(
                str(row[0])
                for row in conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL ORDER BY name",
                    (name,),
                )
            )
            for name in _INDEX_COPY_FORWARD_TABLES
        }
    if set(tables) != set(_INDEX_COPY_FORWARD_TABLES):
        raise SchemaFastForwardError("canonical index DDL is missing a v36 copy-forward table")
    return tables, indexes


def _load_vec_if_required(conn: sqlite3.Connection) -> None:
    needs_vec = conn.execute("SELECT 1 FROM sqlite_master WHERE sql LIKE '%vec0%' LIMIT 1").fetchone()
    if needs_vec is None:
        return
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        raise SchemaFastForwardError(f"database evidence requires sqlite-vec: {error}")


def _plain_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    # table_xinfo marks generated/hidden columns with non-zero `hidden`.
    return tuple(str(row[1]) for row in conn.execute(f'PRAGMA table_xinfo("{table}")') if int(row[6]) == 0)


def _copy_forward_index_table(
    conn: sqlite3.Connection,
    table: str,
    *,
    canonical_tables: dict[str, str],
    canonical_indexes: dict[str, tuple[str, ...]],
) -> None:
    old = f"__polylogue_v35_{table}"
    conn.execute(f'ALTER TABLE "{table}" RENAME TO "{old}"')
    # Index names survive a table rename.  Drop them before recreating the
    # canonical indexes on the replacement, while retaining all table rows.
    index_names = tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (old,)
        )
    )
    for index_name in index_names:
        conn.execute(f'DROP INDEX "{index_name}"')
    conn.execute(canonical_tables[table])
    columns = _plain_columns(conn, table)
    column_sql = ", ".join(f'"{column}"' for column in columns)
    conn.execute(f'INSERT INTO "{table}" ({column_sql}) SELECT {column_sql} FROM "{old}"')
    conn.execute(f'DROP TABLE "{old}"')
    for sql in canonical_indexes[table]:
        conn.execute(sql)


def _execute_script(conn: sqlite3.Connection, sql: str) -> None:
    """Execute DDL without sqlite3.executescript's implicit transaction commit."""
    statement = ""
    for line in sql.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                conn.execute(statement)
            statement = ""
    if statement.strip():
        raise SchemaFastForwardError("canonical DDL ended with an incomplete statement")


def beads_evidence(path: Path) -> dict[str, int]:
    """Return any Beads origin/path rows; non-empty evidence blocks execution."""
    findings: dict[str, int] = {}
    with _open_immutable_readonly(path) as conn:
        for table in _table_names(conn):
            columns = {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
            if "origin" in columns:
                count = int(
                    conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE origin=?', (_BEADS_ORIGIN,)).fetchone()[0]
                )
                if count:
                    findings[f"{table}.origin"] = count
            if "dst_origin" in columns:
                count = int(
                    conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE dst_origin=?', (_BEADS_ORIGIN,)).fetchone()[0]
                )
                if count:
                    findings[f"{table}.dst_origin"] = count
            for column in {"source_path", "path"} & columns:
                count = int(
                    conn.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" LIKE ? OR "{column}" = ?',
                        ("%/.beads/%", ".beads"),
                    ).fetchone()[0]
                )
                if count:
                    findings[f"{table}.{column}"] = count
    return findings


def require_no_beads_evidence(*paths: Path) -> None:
    findings = {str(path): beads_evidence(path) for path in paths}
    findings = {path: rows for path, rows in findings.items() if rows}
    if findings:
        raise SchemaFastForwardError(f"Beads evidence blocks schema fast-forward: {findings}")


def reflink_clone(source: Path, destination: Path) -> None:
    """Create a new clone without accepting SQLite sidecars or overwrites."""
    source = source.resolve(strict=True)
    if destination.exists():
        raise SchemaFastForwardError(f"clone destination already exists: {destination}")
    _require_no_sidecars(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # cp's reflink is a copy optimization, never a correctness dependency.
    try:
        import subprocess

        subprocess.run(
            ["cp", "--reflink=auto", "--preserve=mode,timestamps", str(source), str(destination)], check=True
        )
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise SchemaFastForwardError(f"could not clone {source}: {exc}") from exc
    _fsync_directory(destination.parent)


def _finalize_clone_database(path: Path) -> None:
    """Checkpoint a completed staging clone before immutable evidence reads.

    The copy-forward transaction can leave a WAL and shared-memory file even
    though the clone is complete.  They are safe to remove only after the
    writing connection has closed and a successful truncate checkpoint proves
    that the database file contains every committed page.
    """
    with sqlite3.connect(path) as conn:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise SchemaFastForwardError(f"could not checkpoint completed clone {path}: {checkpoint!r}")
    for suffix in _SQLITE_SIDECARS:
        sidecar = Path(f"{path}{suffix}")
        if not sidecar.exists():
            continue
        # A non-empty WAL or rollback journal after the completed checkpoint
        # would carry data the immutable database file does not prove.  SHM is
        # transient shared state and may remain non-empty after its last close.
        if suffix != "-shm" and sidecar.stat().st_size:
            raise SchemaFastForwardError(f"completed clone retains non-empty sidecar: {sidecar}")
        sidecar.unlink()
    _fsync_directory(path.parent)
    _require_no_sidecars(path)


def fast_forward_index_clone(source: Path, destination: Path) -> CloneForwardResult:
    """Copy-forward index v35→v36 while preserving rows and all FK declarations."""
    source_before = _database_evidence(source)
    if source_before.user_version != 35:
        raise SchemaFastForwardError(f"index clone requires v35, found v{source_before.user_version}")
    require_no_beads_evidence(source)
    reflink_clone(source, destination)
    canonical_tables, canonical_indexes = _canonical_index_objects()
    try:
        with sqlite3.connect(destination) as conn:
            before_fk = _foreign_key_declarations(conn)
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("PRAGMA legacy_alter_table = ON")
            conn.execute("BEGIN IMMEDIATE")
            for table in _INDEX_COPY_FORWARD_TABLES:
                _copy_forward_index_table(
                    conn,
                    table,
                    canonical_tables=canonical_tables,
                    canonical_indexes=canonical_indexes,
                )
            conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
            conn.commit()
            after_fk = _foreign_key_declarations(conn)
            foreign_key_check = tuple(str(row) for row in conn.execute("PRAGMA foreign_key_check"))
            if before_fk != after_fk or foreign_key_check:
                raise SchemaFastForwardError(
                    f"index copy-forward changed FK declarations or failed FK check: {foreign_key_check!r}"
                )
            quick_check = tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check"))
            if quick_check != ("ok",):
                raise SchemaFastForwardError(f"index clone quick_check failed: {quick_check!r}")
        _finalize_clone_database(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    clone_after = _database_evidence(destination)
    if clone_after.table_counts != source_before.table_counts:
        destination.unlink(missing_ok=True)
        raise SchemaFastForwardError("index copy-forward changed structural row counts")
    return CloneForwardResult(source_before, clone_after, True, ())


def fast_forward_embeddings_clone(source: Path, destination: Path) -> CloneForwardResult:
    """Clone embeddings v1→v2, adding failure lifecycle state without vector work."""
    source_before = _database_evidence(source)
    if source_before.user_version != 1:
        raise SchemaFastForwardError(f"embeddings clone requires v1, found v{source_before.user_version}")
    reflink_clone(source, destination)
    try:
        with sqlite3.connect(destination) as conn:
            loaded, error = try_load_sqlite_vec(conn)
            if not loaded:
                raise SchemaFastForwardError(f"embeddings clone requires sqlite-vec: {error}")
            before_fk = _foreign_key_declarations(conn)
            conn.execute("BEGIN IMMEDIATE")
            _execute_script(conn, EMBEDDINGS_DDL)
            conn.execute(f"PRAGMA user_version = {EMBEDDINGS_SCHEMA_VERSION}")
            conn.commit()
            foreign_key_check = tuple(str(row) for row in conn.execute("PRAGMA foreign_key_check"))
            quick_check = tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check"))
            after_fk = _foreign_key_declarations(conn)
            preserved_fk = all(after_fk.get(table) == declarations for table, declarations in before_fk.items())
            if not preserved_fk or foreign_key_check or quick_check != ("ok",):
                raise SchemaFastForwardError("embeddings clone integrity contract failed")
        _finalize_clone_database(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    clone_after = _database_evidence(destination)
    # The new lifecycle relation is intentionally empty. Existing vec/meta rows
    # must match exactly; a changed table-count set is therefore forbidden.
    comparable = {key: value for key, value in clone_after.table_counts.items() if key != "embedding_failures"}
    if comparable != source_before.table_counts:
        destination.unlink(missing_ok=True)
        raise SchemaFastForwardError("embeddings clone changed existing row counts")
    return CloneForwardResult(source_before, clone_after, True, ())


def reuse_index_clone(source: Path, staged_clone: Path, destination: Path) -> CloneForwardResult:
    """Prove one completed v36 index clone against a stable v35 active index.

    This deliberately accepts only a single already-completed staging clone.
    It does not resume arbitrary phases: the active source must still be the
    observed v35 snapshot, and the supplied clone must independently satisfy
    the same structural and integrity proof before its atomic move.
    """
    source_before = _database_evidence(source)
    if source_before.user_version != 35:
        raise SchemaFastForwardError(
            f"reused index clone requires active v35 source, found v{source_before.user_version}"
        )
    require_no_beads_evidence(source)
    staged_clone = staged_clone.resolve(strict=True)
    if not staged_clone.is_file() or staged_clone == source.resolve(strict=True):
        raise SchemaFastForwardError(f"invalid reused index clone: {staged_clone}")
    if destination.exists():
        raise SchemaFastForwardError(f"reused index destination already exists: {destination}")
    _finalize_clone_database(staged_clone)
    require_no_beads_evidence(staged_clone)
    clone_after = _database_evidence(staged_clone)
    if clone_after.user_version != INDEX_SCHEMA_VERSION:
        raise SchemaFastForwardError(
            f"reused index clone requires v{INDEX_SCHEMA_VERSION}, found v{clone_after.user_version}"
        )
    with _open_immutable_readonly(staged_clone) as conn:
        foreign_key_check = tuple(str(row) for row in conn.execute("PRAGMA foreign_key_check"))
        quick_check = tuple(str(row[0]) for row in conn.execute("PRAGMA quick_check"))
    if foreign_key_check or quick_check != ("ok",):
        raise SchemaFastForwardError(
            f"reused index clone integrity contract failed: fk={foreign_key_check!r}, quick={quick_check!r}"
        )
    if clone_after.table_counts != source_before.table_counts:
        raise SchemaFastForwardError("reused index clone changed structural row counts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_clone, destination)
    _fsync_directory(destination.parent)
    return CloneForwardResult(source_before, clone_after, True, foreign_key_check)


def atomic_promote(clone: Path, active: Path, rollback: Path) -> dict[str, str]:
    """Atomically publish one regular prepared clone and retain rollback bytes."""
    if rollback.exists() or clone == active:
        raise SchemaFastForwardError(f"rollback target is unavailable: {rollback}")
    if not clone.exists() or not active.exists():
        raise SchemaFastForwardError("atomic promotion requires both active and prepared clone")
    os.replace(active, rollback)
    try:
        os.replace(clone, active)
    except Exception:
        os.replace(rollback, active)
        raise
    _fsync_directory(active.parent)
    return {"kind": "file", "active": str(active), "rollback": str(rollback)}


def _swap_active_symlink(link: Path, target: Path, *, label: str) -> None:
    temporary = link.with_name(f".{link.name}.{label}-{uuid.uuid4().hex}.tmp")
    temporary.symlink_to(target)
    os.replace(temporary, link)
    _fsync_directory(link.parent)


def _promote_index_generation(clone: Path, active_link: Path) -> dict[str, str]:
    """Publish an index clone as a new generation without replacing its symlink."""
    if not active_link.is_symlink():
        return atomic_promote(clone, active_link, active_link.with_name(f"{active_link.name}.rollback"))
    previous_target = active_link.resolve(strict=True)
    generations_root = previous_target.parent.parent
    if generations_root.name != ".index-generations":
        raise SchemaFastForwardError(
            f"active index symlink has unknown generation layout: {active_link} -> {previous_target}"
        )
    generation = generations_root / f"gen-v{INDEX_SCHEMA_VERSION}-schema-forward-{uuid.uuid4().hex}"
    generation.mkdir(mode=0o700)
    target = generation / "index.db"
    os.replace(clone, target)
    _fsync_directory(generation)
    _swap_active_symlink(active_link, target, label="schema-forward")
    return {
        "kind": "symlink",
        "active": str(active_link),
        "rollback": str(previous_target),
        "generation_target": str(target),
    }


def _restore_promoted(item: dict[str, str], rollback_root: Path) -> None:
    active = Path(item["active"])
    rollback = Path(item["rollback"])
    if item.get("kind") == "symlink":
        _swap_active_symlink(active, rollback, label="schema-forward-rollback")
        return
    failed = rollback_root / f"failed-{active.name}"
    if rollback.exists():
        os.replace(active, failed)
        os.replace(rollback, active)


def plan_clone_forward(
    *,
    archive_root: Path,
    staging_root: Path,
    receipt_path: Path,
    backup_manifest: Path,
    reuse_index_clone_path: Path | None = None,
) -> dict[str, object]:
    """Prepare derived clones and a receipt; it never promotes or migrates durable tiers."""
    source = archive_root / "source.db"
    user = archive_root / "user.db"
    index = archive_root / "index.db"
    embeddings = archive_root / "embeddings.db"
    ops = archive_root / "ops.db"
    for path in (source, user, index, embeddings, ops):
        if not path.exists():
            raise SchemaFastForwardError(f"archive tier is missing: {path}")
    if reuse_index_clone_path is not None:
        _require_service_stopped("polylogued.service")
    require_no_beads_evidence(source, index)
    if not backup_manifest.exists():
        raise SchemaFastForwardError(f"verified backup manifest is missing: {backup_manifest}")
    run_root = staging_root / f"schema-forward-{uuid.uuid4().hex}"
    run_root.mkdir(parents=True, exist_ok=False)
    index_result = (
        fast_forward_index_clone(index, run_root / "index.db")
        if reuse_index_clone_path is None
        else reuse_index_clone(index, reuse_index_clone_path, run_root / "index.db")
    )
    embeddings_result = fast_forward_embeddings_clone(embeddings, run_root / "embeddings.db")
    initialize_archive_database(run_root / "ops.db", ArchiveTier.OPS)
    payload: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": "prepared",
        "prepared_at_ms": _now_ms(),
        "archive_root": str(archive_root),
        "backup_manifest": str(backup_manifest),
        "staging_root": str(run_root),
        "reused_index_clone": str(reuse_index_clone_path) if reuse_index_clone_path is not None else None,
        "source": asdict(_database_evidence(source)),
        "user": asdict(_database_evidence(user)),
        "index": asdict(index_result),
        "embeddings": asdict(embeddings_result),
        "ops": {
            "source": asdict(_database_evidence(ops)),
            "clone": asdict(_database_evidence(run_root / "ops.db")),
            "rotation": "disposable canonical reset",
        },
        "raw_reparse": False,
        "fts_rebuild": False,
        "vector_reembed": False,
        "durable_activation": "stopped-daemon shipped migration runner with retained reflink rollback clones",
    }
    _write_receipt(receipt_path, payload)
    return payload


def activate_prepared_forward(
    *,
    receipt_path: Path,
    backup_manifest: Path,
    service: str | None = None,
) -> dict[str, object]:
    """Migrate stopped durable tiers and atomically publish proven derived clones.

    The verified backup remains authoritative after a reflink because the
    shipped runner binds the stable active path plus bytes/version, not inode.
    Before migration this actuator retains byte-identical source/user rollback
    clones.  No second backup, raw reparse, FTS rebuild, or vector re-embed is
    required.
    """
    raw_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise SchemaFastForwardError(f"invalid prepared receipt: {receipt_path}")
    payload = cast(dict[str, object], raw_payload)
    if payload.get("schema") != RECEIPT_SCHEMA or payload.get("status") != "prepared":
        raise SchemaFastForwardError(f"invalid prepared receipt: {receipt_path}")
    prepared_manifest = Path(str(payload.get("backup_manifest", "")))
    if prepared_manifest.resolve(strict=False) != backup_manifest.resolve(strict=False):
        raise SchemaFastForwardError("activation backup manifest differs from the clone-proof manifest")
    archive_root = Path(str(payload["archive_root"]))
    staging_root = Path(str(payload["staging_root"]))
    source = archive_root / "source.db"
    user = archive_root / "user.db"
    index = archive_root / "index.db"
    embeddings = archive_root / "embeddings.db"
    ops = archive_root / "ops.db"
    _require_service_stopped(service)
    for key, path in (("source", source), ("user", user), ("index", index), ("embeddings", embeddings), ("ops", ops)):
        _require_receipt_identity(payload, key, path)
    require_no_beads_evidence(source, index)
    rollback_root = staging_root / "rollback"
    rollback_root.mkdir(exist_ok=False)
    promoted: list[dict[str, str]] = []
    try:
        # Keep exact old durable bytes locally before the runner commits.  The
        # pre-existing verified manifest still authenticates active paths
        # because these clones are not promoted before validation.
        for path in (source, user):
            clone = rollback_root / path.name
            reflink_clone(path, clone)
            promoted.append({"active": str(path), "rollback": str(clone)})
        with sqlite3.connect(source) as conn:
            migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=backup_manifest)
        with sqlite3.connect(user) as conn:
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=backup_manifest)
        promoted.append(_promote_index_generation(staging_root / index.name, index))
        promoted.append(atomic_promote(staging_root / embeddings.name, embeddings, rollback_root / embeddings.name))
        promoted.append(atomic_promote(staging_root / ops.name, ops, rollback_root / ops.name))
    except Exception as exc:
        for item in reversed(promoted):
            _restore_promoted(item, rollback_root)
        payload.update({"status": "rolled_back", "activation_error": f"{type(exc).__name__}: {exc}"})
        _write_receipt(receipt_path, payload)
        raise
    payload.update(
        {
            "status": "activated",
            "activated_at_ms": _now_ms(),
            "promoted": promoted,
            "versions": {
                "source": _database_evidence(source).user_version,
                "user": _database_evidence(user).user_version,
                "index": _database_evidence(index).user_version,
                "embeddings": _database_evidence(embeddings).user_version,
                "ops": _database_evidence(ops).user_version,
            },
        }
    )
    _write_receipt(receipt_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build and prove derived-tier clones")
    prepare.add_argument("--archive-root", type=Path, required=True)
    prepare.add_argument("--staging-root", type=Path, required=True)
    prepare.add_argument("--receipt", type=Path, required=True)
    prepare.add_argument("--backup-manifest", type=Path, required=True)
    prepare.add_argument(
        "--reuse-index-clone",
        type=Path,
        help="one completed v36 staging index clone to re-prove and move into this prepare run",
    )
    activate = commands.add_parser("activate", help="Run durable migrations and publish derived clones")
    activate.add_argument("--receipt", type=Path, required=True)
    activate.add_argument("--backup-manifest", type=Path, required=True)
    activate.add_argument("--service", default="polylogued.service")
    for owner in (parser, prepare, activate):
        owner.add_argument("--json", action="store_true", help="JSON output (always on)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the explicit stopped-daemon archive forward phases."""
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = plan_clone_forward(
            archive_root=args.archive_root,
            staging_root=args.staging_root,
            receipt_path=args.receipt,
            backup_manifest=args.backup_manifest,
            reuse_index_clone_path=args.reuse_index_clone,
        )
    else:
        result = activate_prepared_forward(
            receipt_path=args.receipt,
            backup_manifest=args.backup_manifest,
            service=args.service,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CloneForwardResult",
    "DatabaseEvidence",
    "RECEIPT_SCHEMA",
    "SchemaFastForwardError",
    "activate_prepared_forward",
    "atomic_promote",
    "beads_evidence",
    "fast_forward_embeddings_clone",
    "fast_forward_index_clone",
    "reuse_index_clone",
    "plan_clone_forward",
    "reflink_clone",
    "require_no_beads_evidence",
    "main",
]
