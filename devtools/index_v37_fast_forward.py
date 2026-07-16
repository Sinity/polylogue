"""Proof-gated clone-first index v36 -> v37 fast-forward.

Index v37 removes three derived run-projection cache tables and changes no
surviving schema object.  Replaying every raw blob is unnecessary for this
exact transition: clone the stopped active generation, prove that its only
schema surplus is the retired cache family, remove that family transactionally,
and promote through :class:`IndexGenerationStore` only after full postflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import cast

from devtools.archive_schema_fast_forward import reflink_clone
from polylogue.config import Config
from polylogue.maintenance.offline_guard import running_daemon_pid
from polylogue.storage.index_generation import IndexGenerationStore, RebuildLease, source_revision_snapshot
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION

FROM_VERSION = 36
TO_VERSION = 37
RECEIPT_SCHEMA = "polylogue.index-v37-fast-forward.v1"
RETIRED_TABLES = (
    "session_observed_events",
    "session_context_snapshots",
    "session_runs",
)


class IndexV37FastForwardError(RuntimeError):
    """The v36 clone could not be proven safe for v37 promotion."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_ddl(sql: str) -> str:
    normalized = re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", sql, flags=re.IGNORECASE)
    normalized = normalized.replace('"', "").replace("`", "").replace("[", "").replace("]", "")
    return re.sub(r"\s+", "", normalized).casefold()


def _schema_objects(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE type IN ('table', 'index', 'view', 'trigger')
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
        ORDER BY type, name
        """
    ).fetchall()
    return {f"{row[0]}:{row[1]}": _normalize_ddl(str(row[2])) for row in rows}


def _canonical_schema_objects() -> dict[str, str]:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(INDEX_DDL)
        return _schema_objects(conn)


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return {str(row[0]): int(conn.execute(f'SELECT COUNT(*) FROM "{row[0]}"').fetchone()[0]) for row in rows}


def _checks(conn: sqlite3.Connection) -> dict[str, object]:
    quick_check = [str(row[0]) for row in conn.execute("PRAGMA quick_check")]
    foreign_keys = [tuple(row) for row in conn.execute("PRAGMA foreign_key_check")]
    return {"quick_check": quick_check, "foreign_key_check": foreign_keys}


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "allocated_bytes": stat.st_blocks * 512,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def _receipt_hash(payload: dict[str, object]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    payload["receipt_sha256"] = _receipt_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_receipt(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RECEIPT_SCHEMA:
        raise IndexV37FastForwardError(f"invalid v37 fast-forward receipt: {path}")
    typed = cast(dict[str, object], payload)
    if typed.get("receipt_sha256") != _receipt_hash(typed):
        raise IndexV37FastForwardError(f"v37 fast-forward receipt hash mismatch: {path}")
    return typed


def _config(archive_root: Path) -> Config:
    return Config(
        archive_root=archive_root,
        render_root=archive_root / "render",
        sources=[],
        db_path=archive_root / "index.db",
    )


def _require_daemon_stopped(archive_root: Path) -> None:
    if (pid := running_daemon_pid(_config(archive_root))) is not None:
        raise IndexV37FastForwardError(f"polylogued PID {pid} is still running")


def _checkpoint_stopped_database(path: Path) -> None:
    """Consolidate a stopped writer's committed WAL before clone evidence."""
    resolved = path.resolve(strict=True)
    with closing(sqlite3.connect(resolved, timeout=120.0)) as conn:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is None or int(checkpoint[0]) != 0 or len(checkpoint) < 3 or int(checkpoint[1]) != int(checkpoint[2]):
        raise IndexV37FastForwardError(f"active index WAL checkpoint failed: {checkpoint}")
    for suffix in ("-wal", "-shm"):
        # A successful checkpoint under the stopped-daemon rebuild lease makes
        # both coordination files disposable.  SQLite commonly leaves a
        # non-empty SHM file behind after the final writer exits.
        sidecar = Path(f"{resolved}{suffix}")
        if sidecar.exists():
            sidecar.unlink()


def _require_clean_database(path: Path, *, expected_version: int) -> tuple[dict[str, str], dict[str, int]]:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{path.resolve(strict=True)}{suffix}")
        if sidecar.exists() and sidecar.stat().st_size:
            raise IndexV37FastForwardError(f"non-empty SQLite sidecar blocks fast-forward: {sidecar}")
    with closing(sqlite3.connect(f"file:{path.resolve(strict=True)}?mode=ro", uri=True)) as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version != expected_version:
            raise IndexV37FastForwardError(f"expected index v{expected_version}, found v{version}")
        checks = _checks(conn)
        if checks["quick_check"] != ["ok"] or checks["foreign_key_check"]:
            raise IndexV37FastForwardError(f"index preflight failed: {checks}")
        return _schema_objects(conn), _table_counts(conn)


def _prove_v36_delta(schema: dict[str, str]) -> dict[str, str]:
    canonical = _canonical_schema_objects()
    missing = sorted(set(canonical) - set(schema))
    surplus = {key: value for key, value in schema.items() if key not in canonical}
    expected_surplus = {
        key: value
        for key, value in schema.items()
        if key.split(":", 1)[1] in RETIRED_TABLES
        or any(key.split(":", 1)[1].startswith(f"idx_{table}") for table in RETIRED_TABLES)
    }
    changed = sorted(key for key in canonical.keys() & schema.keys() if canonical[key] != schema[key])
    if missing or surplus != expected_surplus or changed:
        raise IndexV37FastForwardError(
            f"v36 schema is not the exact v37-plus-retired-caches shape: "
            f"missing={missing}, unexpected_surplus={sorted(set(surplus) - set(expected_surplus))}, changed={changed}"
        )
    retired_tables = {f"table:{table}" for table in RETIRED_TABLES}
    if not retired_tables <= set(expected_surplus):
        raise IndexV37FastForwardError("v36 index is missing one or more retired cache tables")
    return canonical


def _transform_clone(path: Path, *, before_counts: dict[str, int], canonical: dict[str, str]) -> dict[str, object]:
    with closing(sqlite3.connect(path, timeout=120.0)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) != FROM_VERSION:
            raise IndexV37FastForwardError("clone version changed before transformation")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table in RETIRED_TABLES:
                conn.execute(f'DROP TABLE "{table}"')
            conn.execute(f"PRAGMA user_version = {TO_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise IndexV37FastForwardError(f"clone WAL checkpoint failed: {checkpoint}")
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists() and sidecar.stat().st_size == 0:
            sidecar.unlink()
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        checks = _checks(conn)
        after_schema = _schema_objects(conn)
        after_counts = _table_counts(conn)
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    expected_counts = {name: count for name, count in before_counts.items() if name not in RETIRED_TABLES}
    if version != INDEX_SCHEMA_VERSION or version != TO_VERSION:
        raise IndexV37FastForwardError(f"clone ended at unexpected index version {version}")
    if checks["quick_check"] != ["ok"] or checks["foreign_key_check"]:
        raise IndexV37FastForwardError(f"clone postflight failed: {checks}")
    if after_schema != canonical:
        raise IndexV37FastForwardError("clone schema does not exactly match canonical v37 DDL")
    if after_counts != expected_counts:
        raise IndexV37FastForwardError("one or more surviving table counts changed in the clone")
    return {"checks": checks, "table_counts": after_counts, "schema_object_count": len(after_schema)}


def prepare_forward(*, archive_root: Path, receipt_path: Path) -> dict[str, object]:
    """Create and prove an owned inactive v37 generation."""
    archive_root = archive_root.resolve(strict=True)
    _require_daemon_stopped(archive_root)
    store = IndexGenerationStore(archive_root)
    active_pointer = store.active_pointer
    with RebuildLease(archive_root):
        _require_daemon_stopped(archive_root)
        _checkpoint_stopped_database(active_pointer)
        source_snapshot = source_revision_snapshot(archive_root)
        active_identity = _file_identity(active_pointer)
        before_schema, before_counts = _require_clean_database(active_pointer, expected_version=FROM_VERSION)
        canonical = _prove_v36_delta(before_schema)
        generation = store.create(source_snapshot=source_snapshot)
        clone = Path(generation.index_path)
        try:
            clone.unlink()
            reflink_clone(active_pointer, clone)
            if _file_identity(active_pointer) != active_identity:
                raise IndexV37FastForwardError("active index changed while its clone was created")
            postflight = _transform_clone(clone, before_counts=before_counts, canonical=canonical)
            if source_revision_snapshot(archive_root) != source_snapshot:
                raise IndexV37FastForwardError("source evidence changed while preparing v37 clone")
            receipt: dict[str, object] = {
                "schema": RECEIPT_SCHEMA,
                "status": "prepared",
                "prepared_at_ms": _now_ms(),
                "archive_root": str(archive_root),
                "generation": asdict(generation),
                "source_snapshot": source_snapshot,
                "active_identity": active_identity,
                "before_table_counts": before_counts,
                "retired_tables": list(RETIRED_TABLES),
                "postflight": postflight,
                "raw_reparse": False,
            }
            _write_receipt(receipt_path, receipt)
            return receipt
        except Exception:
            store.discard_if_inactive(generation)
            raise


def activate_forward(*, receipt_path: Path) -> dict[str, object]:
    """Re-prove and atomically promote one prepared v37 generation."""
    receipt = _load_receipt(receipt_path)
    if receipt.get("status") != "prepared":
        raise IndexV37FastForwardError(f"receipt is not prepared: {receipt.get('status')}")
    archive_root = Path(str(receipt["archive_root"])).resolve(strict=True)
    _require_daemon_stopped(archive_root)
    store = IndexGenerationStore(archive_root)
    generation_payload = cast(dict[str, object], receipt["generation"])
    generation = store.load(str(generation_payload["generation_id"]))
    with RebuildLease(archive_root):
        _require_daemon_stopped(archive_root)
        if generation.owner_id != generation_payload["owner_id"] or generation.state != "inactive":
            raise IndexV37FastForwardError("prepared generation ownership/state changed")
        if source_revision_snapshot(archive_root) != receipt["source_snapshot"]:
            raise IndexV37FastForwardError("source evidence changed since v37 preparation")
        if _file_identity(store.active_pointer) != receipt["active_identity"]:
            raise IndexV37FastForwardError("active index changed since v37 preparation")
        clone = Path(generation.index_path)
        clone_schema, clone_counts = _require_clean_database(clone, expected_version=TO_VERSION)
        canonical = _canonical_schema_objects()
        if clone_schema != canonical:
            raise IndexV37FastForwardError("prepared clone no longer matches canonical v37 DDL")
        expected_counts = cast(dict[str, int], cast(dict[str, object], receipt["postflight"])["table_counts"])
        if clone_counts != expected_counts:
            raise IndexV37FastForwardError("prepared clone table counts changed before activation")
        promoted = store.promote(generation)
        receipt.update(
            {
                "status": "activated",
                "activated_at_ms": _now_ms(),
                "generation": asdict(promoted),
                "active_identity_after": _file_identity(store.active_pointer),
            }
        )
        _write_receipt(receipt_path, receipt)
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--archive-root", type=Path, required=True)
    prepare.add_argument("--receipt", type=Path, required=True)
    activate = subparsers.add_parser("activate")
    activate.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        prepare_forward(archive_root=args.archive_root, receipt_path=args.receipt)
        if args.command == "prepare"
        else activate_forward(receipt_path=args.receipt)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IndexV37FastForwardError",
    "activate_forward",
    "main",
    "prepare_forward",
]
