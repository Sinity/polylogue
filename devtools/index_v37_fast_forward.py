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
from contextlib import closing, suppress
from dataclasses import asdict
from pathlib import Path
from typing import cast

from devtools.archive_schema_fast_forward import reflink_clone
from polylogue.config import Config
from polylogue.maintenance.offline_guard import running_daemon_pid
from polylogue.storage.index_generation import IndexGenerationStore, RebuildLease, source_revision_snapshot
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

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
        ensure_runtime_indexes_sync(conn)
        return _schema_objects(conn)


def _schema_rootpages(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT type, name, rootpage
        FROM sqlite_master
        WHERE type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
          AND rootpage > 0
        ORDER BY type, name
        """
    ).fetchall()
    return {f"{row[0]}:{row[1]}": int(row[2]) for row in rows}


def _checks(conn: sqlite3.Connection) -> dict[str, object]:
    quick_check = [str(row[0]) for row in conn.execute("PRAGMA quick_check")]
    attachment_native_ids_foreign_keys = [
        tuple(row) for row in conn.execute("PRAGMA foreign_key_check(attachment_native_ids)")
    ]
    return {
        "quick_check": quick_check,
        "attachment_native_ids_foreign_key_check": attachment_native_ids_foreign_keys,
    }


def _require_retired_tables_unreferenced(conn: sqlite3.Connection) -> None:
    """Prove dropping the caches cannot remove a surviving FK parent."""
    tables = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    references = {
        (table, str(row[2]))
        for table in tables
        if table not in RETIRED_TABLES
        for row in conn.execute(f'PRAGMA foreign_key_list("{table}")')
        if str(row[2]) in RETIRED_TABLES
    }
    if references:
        raise IndexV37FastForwardError(f"surviving tables reference retired cache parents: {sorted(references)}")


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


def _proven_clone_identity(path: Path) -> dict[str, object]:
    """Bind a prepared proof to exact clone bytes, not only row counts."""
    identity = _file_identity(path)
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    identity["sha256"] = digest.hexdigest()
    return identity


def _canonical_schema_sha256(schema: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _receipt_hash(payload: dict[str, object]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require_receipt_destination_writable(path: Path) -> None:
    """Fail before expensive preparation when an atomic receipt cannot land."""
    probe = path.with_name(f".{path.name}.{uuid.uuid4().hex}.probe")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not path.is_file():
            raise OSError(f"receipt destination is not a regular file: {path}")
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        probe.unlink()
    except OSError as exc:
        with suppress(OSError):
            probe.unlink(missing_ok=True)
        raise IndexV37FastForwardError(f"receipt destination is not writable: {path}: {exc}") from exc


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


def _checkpoint_stopped_database(path: Path, *, label: str = "active index") -> None:
    """Consolidate a stopped writer's committed WAL before clone evidence."""
    resolved = path.resolve(strict=True)
    with closing(sqlite3.connect(resolved, timeout=120.0)) as conn:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is None or int(checkpoint[0]) != 0 or len(checkpoint) < 3 or int(checkpoint[1]) != int(checkpoint[2]):
        raise IndexV37FastForwardError(f"{label} WAL checkpoint failed: {checkpoint}")
    for suffix in ("-wal", "-shm"):
        # A successful checkpoint under the stopped-daemon rebuild lease makes
        # both coordination files disposable.  SQLite commonly leaves a
        # non-empty SHM file behind after the final writer exits.
        sidecar = Path(f"{resolved}{suffix}")
        if sidecar.exists():
            sidecar.unlink()


def _inspect_clean_database(path: Path, *, expected_version: int) -> tuple[dict[str, str], dict[str, int]]:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{path.resolve(strict=True)}{suffix}")
        if sidecar.exists() and sidecar.stat().st_size:
            raise IndexV37FastForwardError(f"non-empty SQLite sidecar blocks fast-forward: {sidecar}")
    with closing(sqlite3.connect(f"file:{path.resolve(strict=True)}?mode=ro&immutable=1", uri=True)) as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version != expected_version:
            raise IndexV37FastForwardError(f"expected index v{expected_version}, found v{version}")
        return _schema_objects(conn), _schema_rootpages(conn)


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


def _transform_clone(path: Path, *, before_rootpages: dict[str, int], canonical: dict[str, str]) -> dict[str, object]:
    with closing(sqlite3.connect(path, timeout=120.0)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) != FROM_VERSION:
            raise IndexV37FastForwardError("clone version changed before transformation")
        _require_retired_tables_unreferenced(conn)
        changes_before = conn.total_changes
        conn.execute("BEGIN IMMEDIATE")
        try:
            repaired_orphan_native_ids = conn.execute(
                """
                DELETE FROM attachment_native_ids
                WHERE NOT EXISTS (
                    SELECT 1 FROM attachment_refs WHERE attachment_refs.ref_id = attachment_native_ids.ref_id
                )
                """
            ).rowcount
            for table in RETIRED_TABLES:
                conn.execute(f'DROP TABLE "{table}"')
            conn.execute(f"PRAGMA user_version = {TO_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        if conn.total_changes - changes_before != repaired_orphan_native_ids:
            raise IndexV37FastForwardError("v37 clone transformation changed rows outside the declared repair")
    _checkpoint_stopped_database(path, label="prepared clone")
    with closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)) as conn:
        checks = _checks(conn)
        after_schema = _schema_objects(conn)
        after_rootpages = _schema_rootpages(conn)
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    expected_rootpages = {
        key: rootpage
        for key, rootpage in before_rootpages.items()
        if key.split(":", 1)[1] not in RETIRED_TABLES
        and not any(key.split(":", 1)[1].startswith(f"idx_{table}") for table in RETIRED_TABLES)
    }
    if version != INDEX_SCHEMA_VERSION or version != TO_VERSION:
        raise IndexV37FastForwardError(f"clone ended at unexpected index version {version}")
    if checks["quick_check"] != ["ok"] or checks["attachment_native_ids_foreign_key_check"]:
        raise IndexV37FastForwardError(f"clone postflight failed: {checks}")
    if after_schema != canonical:
        raise IndexV37FastForwardError("clone schema does not exactly match canonical v37 DDL")
    if after_rootpages != expected_rootpages:
        raise IndexV37FastForwardError("one or more surviving schema root pages changed in the clone")
    return {
        "checks": checks,
        "repaired_orphan_attachment_native_ids": repaired_orphan_native_ids,
        "schema_rootpages": after_rootpages,
        "schema_object_count": len(after_schema),
    }


def prepare_forward(*, archive_root: Path, receipt_path: Path) -> dict[str, object]:
    """Create and prove an owned inactive v37 generation."""
    prepare_started_ns = time.monotonic_ns()
    phase_timings_ms: dict[str, int] = {}
    archive_root = archive_root.resolve(strict=True)
    _require_daemon_stopped(archive_root)
    _require_receipt_destination_writable(receipt_path)
    store = IndexGenerationStore(archive_root)
    active_pointer = store.active_pointer
    with RebuildLease(archive_root):
        _require_daemon_stopped(archive_root)
        _checkpoint_stopped_database(active_pointer)
        phase_started_ns = time.monotonic_ns()
        source_snapshot = source_revision_snapshot(archive_root)
        active_identity = _file_identity(active_pointer)
        before_schema, before_rootpages = _inspect_clean_database(active_pointer, expected_version=FROM_VERSION)
        canonical = _prove_v36_delta(before_schema)
        phase_timings_ms["active_evidence"] = (time.monotonic_ns() - phase_started_ns) // 1_000_000
        generation = store.create(source_snapshot=source_snapshot)
        clone = Path(generation.index_path)
        try:
            phase_started_ns = time.monotonic_ns()
            clone.unlink()
            reflink_clone(active_pointer, clone)
            if _file_identity(active_pointer) != active_identity:
                raise IndexV37FastForwardError("active index changed while its clone was created")
            phase_timings_ms["reflink_clone"] = (time.monotonic_ns() - phase_started_ns) // 1_000_000
            phase_started_ns = time.monotonic_ns()
            postflight = _transform_clone(clone, before_rootpages=before_rootpages, canonical=canonical)
            phase_timings_ms["transform_and_postflight"] = (time.monotonic_ns() - phase_started_ns) // 1_000_000
            if source_revision_snapshot(archive_root) != source_snapshot:
                raise IndexV37FastForwardError("source evidence changed while preparing v37 clone")
            phase_started_ns = time.monotonic_ns()
            clone_identity = _proven_clone_identity(clone)
            phase_timings_ms["clone_sha256"] = (time.monotonic_ns() - phase_started_ns) // 1_000_000
            phase_timings_ms["total"] = (time.monotonic_ns() - prepare_started_ns) // 1_000_000
            receipt: dict[str, object] = {
                "schema": RECEIPT_SCHEMA,
                "status": "prepared",
                "prepared_at_ms": _now_ms(),
                "archive_root": str(archive_root),
                "generation": asdict(generation),
                "source_snapshot": source_snapshot,
                "active_identity": active_identity,
                "clone_identity": clone_identity,
                "canonical_schema_sha256": _canonical_schema_sha256(canonical),
                "before_schema_rootpages": before_rootpages,
                "retired_tables": list(RETIRED_TABLES),
                "postflight": postflight,
                "phase_timings_ms": phase_timings_ms,
                "raw_reparse": False,
            }
            _write_receipt(receipt_path, receipt)
            return receipt
        except Exception:
            store.discard_if_inactive(generation)
            raise


def activate_forward(*, receipt_path: Path) -> dict[str, object]:
    """Reconcile or atomically promote one prepared v37 generation."""
    receipt = _load_receipt(receipt_path)
    status = receipt.get("status")
    if status not in {"prepared", "activating", "activated"}:
        raise IndexV37FastForwardError(f"receipt is not prepared: {receipt.get('status')}")
    archive_root = Path(str(receipt["archive_root"])).resolve(strict=True)
    _require_daemon_stopped(archive_root)
    store = IndexGenerationStore(archive_root)
    generation_payload = cast(dict[str, object], receipt["generation"])
    generation = store.load(str(generation_payload["generation_id"]))
    with RebuildLease(archive_root):
        _require_daemon_stopped(archive_root)
        if status in {"activating", "activated"}:
            generation = store.recover_promotion(generation.generation_id)
        if generation.owner_id != generation_payload["owner_id"]:
            raise IndexV37FastForwardError("prepared generation ownership changed")
        clone = Path(generation.index_path)
        if generation.state == "active":
            if store.active_pointer.resolve(strict=True) != clone.resolve(strict=True):
                raise IndexV37FastForwardError("active generation does not own the active index pointer")
            receipt.update(
                {
                    "status": "activated",
                    "activated_at_ms": receipt.get("activated_at_ms", _now_ms()),
                    "generation": asdict(generation),
                    "active_identity_after": _file_identity(store.active_pointer),
                }
            )
            _write_receipt(receipt_path, receipt)
            return receipt
        if generation.state != "inactive":
            raise IndexV37FastForwardError(f"prepared generation has unrecoverable state {generation.state}")
        if source_revision_snapshot(archive_root) != receipt["source_snapshot"]:
            raise IndexV37FastForwardError("source evidence changed since v37 preparation")
        if _file_identity(store.active_pointer) != receipt["active_identity"]:
            raise IndexV37FastForwardError("active index changed since v37 preparation")
        canonical = _canonical_schema_objects()
        if _canonical_schema_sha256(canonical) != receipt.get("canonical_schema_sha256"):
            raise IndexV37FastForwardError("canonical v37 schema changed since clone preparation")
        if _proven_clone_identity(clone) != receipt.get("clone_identity"):
            raise IndexV37FastForwardError("prepared clone bytes changed before activation")
        if status == "prepared":
            receipt.update({"status": "activating", "activation_started_at_ms": _now_ms()})
            _write_receipt(receipt_path, receipt)
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
