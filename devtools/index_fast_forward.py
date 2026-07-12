"""Clone-first index.db v32 -> v35 fast-forward without raw replay.

This operator actuator is intentionally derived-tier-only.  It reflink-clones
the quiesced active index generation, applies the exact canonical schema deltas
to the clone, structurally rebuilds the three FTS surfaces from normalized
index tables, and advances ``user_version`` only after every clone invariant is
green.  Activation is a separate atomic symlink swap with an explicit rollback
target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import unicodedata
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync
from polylogue.storage.fts.fts_lifecycle import (
    fts_invariant_snapshot_sync,
    insert_missing_message_rows_batched_sync,
)
from polylogue.storage.fts.pl_fold import pl_fold, pl_fold_sql_expr
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL

FAST_FORWARD_FROM_VERSION = 32
FAST_FORWARD_TO_VERSION = 35
RECEIPT_SCHEMA = "polylogue.index-fast-forward.v1"
DEFAULT_MESSAGE_BATCH_ROWS = 50_000
DEFAULT_MAX_IO_FULL_AVG10 = 45.0
DEFAULT_MAX_MEMORY_FULL_AVG10 = 5.0

_CANONICAL_OBJECTS = (
    ("table", "insight_materialization"),
    ("index", "idx_web_constructs_message"),
    ("view", "delegations"),
    ("table", "messages_fts"),
    ("trigger", "messages_fts_ai"),
    ("trigger", "messages_fts_ad"),
    ("trigger", "messages_fts_au"),
    ("table", "session_work_events_fts"),
    ("trigger", "session_work_events_fts_ai"),
    ("trigger", "session_work_events_fts_ad"),
    ("trigger", "session_work_events_fts_au"),
    ("table", "threads_fts"),
    ("trigger", "threads_fts_ai"),
    ("trigger", "threads_fts_ad"),
    ("trigger", "threads_fts_au"),
)

_STRUCTURAL_TABLES = (
    "sessions",
    "messages",
    "blocks",
    "web_content_constructs",
    "raw_revision_applications",
    "raw_revision_heads",
    "session_links",
    "insight_materialization",
    "session_work_events",
    "threads",
)


@dataclass(frozen=True, slots=True)
class PressureSample:
    observed_at_ms: int
    io_some_avg10: float
    io_full_avg10: float
    memory_some_avg10: float
    memory_full_avg10: float


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    resolved_path: str
    size_bytes: int
    allocated_bytes: int
    inode: int
    mtime_ns: int
    wal_size_bytes: int
    shm_size_bytes: int


@dataclass(slots=True)
class StageReceipt:
    name: str
    started_at_ms: int
    completed_at_ms: int | None = None
    duration_ms: int | None = None
    detail: dict[str, object] = field(default_factory=dict)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _parse_pressure(path: Path) -> tuple[float, float]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0, 0.0
    values: dict[str, float] = {}
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        prefix = fields[0]
        for field_value in fields[1:]:
            key, _, raw = field_value.partition("=")
            if key == "avg10":
                values[prefix] = float(raw)
    return values.get("some", 0.0), values.get("full", 0.0)


def pressure_sample() -> PressureSample:
    io_some, io_full = _parse_pressure(Path("/proc/pressure/io"))
    memory_some, memory_full = _parse_pressure(Path("/proc/pressure/memory"))
    return PressureSample(_now_ms(), io_some, io_full, memory_some, memory_full)


def _assert_pressure(sample: PressureSample, *, max_io_full_avg10: float, max_memory_full_avg10: float) -> None:
    if sample.io_full_avg10 > max_io_full_avg10:
        raise RuntimeError(
            f"I/O pressure gate refused work: full avg10={sample.io_full_avg10:.2f} > {max_io_full_avg10:.2f}"
        )
    if sample.memory_full_avg10 > max_memory_full_avg10:
        raise RuntimeError(
            f"memory pressure gate refused work: full avg10={sample.memory_full_avg10:.2f} "
            f"> {max_memory_full_avg10:.2f}"
        )


def file_identity(path: Path) -> FileIdentity:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    wal = resolved.with_name(resolved.name + "-wal")
    shm = resolved.with_name(resolved.name + "-shm")
    return FileIdentity(
        path=str(path),
        resolved_path=str(resolved),
        size_bytes=stat.st_size,
        allocated_bytes=stat.st_blocks * 512,
        inode=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        wal_size_bytes=wal.stat().st_size if wal.exists() else 0,
        shm_size_bytes=shm.stat().st_size if shm.exists() else 0,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    if payload.get("schema") == RECEIPT_SCHEMA:
        payload["receipt_payload_sha256"] = _receipt_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RuntimeError("receipt write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _receipt_hash(payload: dict[str, object]) -> str:
    hashable = {key: value for key, value in payload.items() if key != "receipt_payload_sha256"}
    return hashlib.sha256(json.dumps(hashable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_receipt(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RECEIPT_SCHEMA:
        raise RuntimeError(f"invalid index fast-forward receipt: {path}")
    typed = cast(dict[str, object], payload)
    expected_hash = typed.get("receipt_payload_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _receipt_hash(typed):
        raise RuntimeError(f"index fast-forward receipt hash mismatch: {path}")
    return typed


def _canonical_schema() -> dict[tuple[str, str], str]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(INDEX_DDL)
        result: dict[tuple[str, str], str] = {}
        for object_type, name in _CANONICAL_OBJECTS:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
                (object_type, name),
            ).fetchone()
            if row is None or row[0] is None:
                raise RuntimeError(f"canonical INDEX_DDL is missing {object_type} {name}")
            result[(object_type, name)] = str(row[0])
        return result
    finally:
        conn.close()


def _normalize_ddl(sql: str) -> str:
    normalized = re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", sql, flags=re.IGNORECASE)
    normalized = normalized.replace('"', "").replace("`", "").replace("[", "").replace("]", "")
    return re.sub(r"\s+", "", normalized).casefold()


def _ddl_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for object_type, name in _CANONICAL_OBJECTS:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, name),
        ).fetchone()
        sql = str(row[0]) if row is not None and row[0] is not None else ""
        hashes[f"{object_type}:{name}"] = hashlib.sha256(_normalize_ddl(sql).encode()).hexdigest()
    return hashes


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _STRUCTURAL_TABLES:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if exists is not None:
            counts[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    return counts


def _database_metrics(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "user_version": int(conn.execute("PRAGMA user_version").fetchone()[0]),
        "page_count": int(conn.execute("PRAGMA page_count").fetchone()[0]),
        "page_size": int(conn.execute("PRAGMA page_size").fetchone()[0]),
        "freelist_count": int(conn.execute("PRAGMA freelist_count").fetchone()[0]),
    }


def plan_index(path: Path) -> dict[str, object]:
    identity = file_identity(path)
    with sqlite3.connect(f"file:{identity.resolved_path}?mode=ro", uri=True, timeout=30.0) as conn:
        metrics = _database_metrics(conn)
        counts = _table_counts(conn)
        fts = {
            "messages_source": int(conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0]),
            "messages_indexed": int(conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]),
            "work_events_source": int(conn.execute("SELECT COUNT(*) FROM session_work_events").fetchone()[0]),
            "work_events_indexed": int(conn.execute("SELECT COUNT(*) FROM session_work_events_fts").fetchone()[0]),
            "threads_source": int(conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0]),
            "threads_indexed": int(conn.execute("SELECT COUNT(*) FROM threads_fts").fetchone()[0]),
        }
    return {
        "identity": asdict(identity),
        "metrics": metrics,
        "counts": counts,
        "fts": fts,
        "pressure": asdict(pressure_sample()),
        "eligible": metrics["user_version"] == FAST_FORWARD_FROM_VERSION,
        "target_version": FAST_FORWARD_TO_VERSION,
        "raw_reparse": False,
    }


def _run_stage(
    receipt: dict[str, object],
    receipt_path: Path,
    name: str,
    operation: Callable[[], dict[str, object]],
) -> dict[str, object]:
    stage = StageReceipt(name=name, started_at_ms=_now_ms())
    stages = cast(list[dict[str, object]], receipt.setdefault("stages", []))
    stages.append(asdict(stage))
    _write_receipt(receipt_path, receipt)
    started = time.monotonic()
    detail = operation()
    completed = _now_ms()
    stages[-1] = asdict(
        StageReceipt(
            name=name,
            started_at_ms=stage.started_at_ms,
            completed_at_ms=completed,
            duration_ms=int((time.monotonic() - started) * 1000),
            detail=detail,
        )
    )
    _write_receipt(receipt_path, receipt)
    return detail


def _replace_insight_materialization(conn: sqlite3.Connection, canonical_sql: str) -> None:
    columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(insight_materialization)").fetchall()]
    if not columns:
        raise RuntimeError("insight_materialization table is missing")
    temporary_name = f"insight_materialization_v35_{uuid.uuid4().hex[:8]}"
    create_temporary = re.sub(
        r"(?i)^CREATE\s+TABLE\s+insight_materialization",
        f"CREATE TABLE {temporary_name}",
        canonical_sql,
        count=1,
    )
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(create_temporary)
        conn.execute(
            f'INSERT INTO "{temporary_name}" ({quoted_columns}) SELECT {quoted_columns} FROM "insight_materialization"'
        )
        conn.execute('DROP TABLE "insight_materialization"')
        conn.execute(f'ALTER TABLE "{temporary_name}" RENAME TO "insight_materialization"')
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _replace_view(conn: sqlite3.Connection, name: str, canonical_sql: str) -> None:
    conn.execute(f'DROP VIEW IF EXISTS "{name}"')
    conn.execute(canonical_sql)
    conn.commit()


def _replace_fts_surface(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    trigger_names: Sequence[str],
    canonical: dict[tuple[str, str], str],
) -> None:
    for trigger_name in trigger_names:
        conn.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(canonical[("table", table_name)])
    for trigger_name in trigger_names:
        conn.execute(canonical[("trigger", trigger_name)])
    conn.commit()


def _rebuild_messages_fts(
    conn: sqlite3.Connection,
    canonical: dict[tuple[str, str], str],
    *,
    batch_rows: int,
    pressure_guard: Callable[[], None],
) -> dict[str, object]:
    triggers = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")
    _replace_fts_surface(conn, table_name="messages_fts", trigger_names=triggers, canonical=canonical)
    windows: list[dict[str, int]] = []

    def progress(lower: int, upper: int, inserted: int) -> None:
        pressure_guard()
        windows.append({"lower": lower, "upper": upper, "inserted": inserted})

    inserted = insert_missing_message_rows_batched_sync(
        conn,
        batch_rows=batch_rows,
        measure_counts=True,
        progress_callback=progress,
    )
    conn.commit()
    return {"inserted": inserted, "windows": len(windows), "last_window": windows[-1] if windows else None}


def _rebuild_small_fts(
    conn: sqlite3.Connection,
    canonical: dict[tuple[str, str], str],
    *,
    table_name: str,
    trigger_names: Sequence[str],
    insert_sql: str,
) -> dict[str, object]:
    _replace_fts_surface(conn, table_name=table_name, trigger_names=trigger_names, canonical=canonical)
    before = conn.total_changes
    conn.execute(insert_sql)
    conn.commit()
    return {"inserted": max(0, conn.total_changes - before)}


def _ascii_fold_token(value: str) -> str:
    folded = pl_fold(value) or value
    return "".join(
        character for character in unicodedata.normalize("NFKD", folded) if not unicodedata.combining(character)
    )


def _fold_smoke(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    source_column: str,
    fts_table: str,
) -> dict[str, object]:
    row = conn.execute(
        f'SELECT "{source_column}" FROM "{source_table}" '
        f'WHERE "{source_column}" LIKE ? OR "{source_column}" LIKE ? LIMIT 1',
        ("%ł%", "%Ł%"),
    ).fetchone()
    if row is None or not isinstance(row[0], str):
        return {"candidate": None, "query": None, "matched": None, "skipped": True}
    tokens = re.findall(r"[^\W_]+", row[0], flags=re.UNICODE)
    token = next((candidate for candidate in tokens if "ł" in candidate.casefold()), None)
    if token is None:
        return {"candidate": None, "query": None, "matched": None, "skipped": True}
    query = _ascii_fold_token(token)
    matched = conn.execute(
        f'SELECT 1 FROM "{fts_table}" WHERE "{fts_table}" MATCH ? LIMIT 1',
        (f'"{query.replace(chr(34), chr(34) * 2)}"',),
    ).fetchone()
    return {"candidate": token, "query": query, "matched": matched is not None, "skipped": False}


def validate_clone(
    db_path: Path,
    *,
    expected_counts: dict[str, int],
    expected_version: int,
    run_quick_check: bool = True,
) -> dict[str, object]:
    canonical = _canonical_schema()
    with sqlite3.connect(db_path, timeout=120.0) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        counts = _table_counts(conn)
        count_drift = {
            table: {"expected": expected, "actual": counts.get(table)}
            for table, expected in expected_counts.items()
            if counts.get(table) != expected
        }
        ddl_drift: dict[str, dict[str, str]] = {}
        for key, expected_sql in canonical.items():
            object_type, name = key
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type=? AND name=?",
                (object_type, name),
            ).fetchone()
            actual_sql = str(row[0]) if row is not None and row[0] is not None else ""
            if _normalize_ddl(actual_sql) != _normalize_ddl(expected_sql):
                ddl_drift[f"{object_type}:{name}"] = {
                    "expected_sha256": hashlib.sha256(_normalize_ddl(expected_sql).encode()).hexdigest(),
                    "actual_sha256": hashlib.sha256(_normalize_ddl(actual_sql).encode()).hexdigest(),
                }
        foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
        quick_check = [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()] if run_quick_check else []
        invariant = fts_invariant_snapshot_sync(conn)
        fts_counts = {
            surface.name: {
                "source": surface.source_rows,
                "indexed": surface.indexed_rows,
                "missing": surface.missing_rows,
                "excess": surface.excess_rows,
                "duplicates": surface.duplicate_rows,
                "triggers": surface.triggers_present,
                "ready": surface.ready,
            }
            for surface in invariant.surfaces
        }
        fold_smoke = {
            "messages_fts": _fold_smoke(
                conn, source_table="blocks", source_column="search_text", fts_table="messages_fts"
            ),
            "session_work_events_fts": _fold_smoke(
                conn,
                source_table="session_work_events",
                source_column="search_text",
                fts_table="session_work_events_fts",
            ),
            "threads_fts": _fold_smoke(
                conn, source_table="threads", source_column="search_text", fts_table="threads_fts"
            ),
        }
        fold_failures = [
            name
            for name, smoke in fold_smoke.items()
            if not bool(smoke.get("skipped")) and not bool(smoke.get("matched"))
        ]
        ready = (
            version == expected_version
            and not count_drift
            and not ddl_drift
            and not foreign_keys
            and (not run_quick_check or quick_check == ["ok"])
            and invariant.ready
            and not fold_failures
        )
        return {
            "ready": ready,
            "user_version": version,
            "count_drift": count_drift,
            "ddl_drift": ddl_drift,
            "foreign_key_violations": len(foreign_keys),
            "foreign_key_samples": [list(row) for row in foreign_keys[:20]],
            "quick_check": quick_check,
            "fts": fts_counts,
            "fold_smoke": fold_smoke,
            "fold_failures": fold_failures,
            "ddl_hashes": _ddl_hashes(conn),
            "metrics": _database_metrics(conn),
        }


def fast_forward_clone(
    db_path: Path,
    receipt_path: Path,
    *,
    max_io_full_avg10: float = DEFAULT_MAX_IO_FULL_AVG10,
    max_memory_full_avg10: float = DEFAULT_MAX_MEMORY_FULL_AVG10,
    batch_rows: int = DEFAULT_MESSAGE_BATCH_ROWS,
    fail_after_stage: str | None = None,
) -> dict[str, object]:
    canonical = _canonical_schema()
    identity = file_identity(db_path)
    with sqlite3.connect(db_path, timeout=120.0) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        metrics = _database_metrics(conn)
        if metrics["user_version"] != FAST_FORWARD_FROM_VERSION:
            raise RuntimeError(
                f"clone must start at index user_version {FAST_FORWARD_FROM_VERSION}, found {metrics['user_version']}"
            )
        before_counts = _table_counts(conn)
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": "upgrading",
        "started_at_ms": _now_ms(),
        "source_version": FAST_FORWARD_FROM_VERSION,
        "target_version": FAST_FORWARD_TO_VERSION,
        "clone_identity_before": asdict(identity),
        "structural_counts_before": before_counts,
        "raw_reparse": False,
        "stages": [],
        "pressure_samples": [],
    }
    _write_receipt(receipt_path, receipt)

    def guard() -> None:
        sample = pressure_sample()
        cast(list[dict[str, object]], receipt["pressure_samples"]).append(asdict(sample))
        _assert_pressure(
            sample,
            max_io_full_avg10=max_io_full_avg10,
            max_memory_full_avg10=max_memory_full_avg10,
        )

    def apply_v33(conn: sqlite3.Connection) -> dict[str, object]:
        _replace_insight_materialization(conn, canonical[("table", "insight_materialization")])
        rows = int(conn.execute("SELECT COUNT(*) FROM insight_materialization").fetchone()[0])
        return {"rows": rows}

    def apply_v34(conn: sqlite3.Connection) -> dict[str, object]:
        conn.execute(canonical[("index", "idx_web_constructs_message")])
        _replace_view(conn, "delegations", canonical[("view", "delegations")])
        conn.commit()
        return {"index": "idx_web_constructs_message", "view": "delegations"}

    try:
        guard()
        with sqlite3.connect(db_path, timeout=120.0) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            _run_stage(
                receipt,
                receipt_path,
                "v33-insight-check",
                lambda: apply_v33(conn),
            )
            if fail_after_stage == "v33-insight-check":
                raise RuntimeError("injected failure after v33-insight-check")
            guard()

            _run_stage(
                receipt,
                receipt_path,
                "v34-index-and-delegations",
                lambda: apply_v34(conn),
            )
            if fail_after_stage == "v34-index-and-delegations":
                raise RuntimeError("injected failure after v34-index-and-delegations")
            guard()

            _run_stage(
                receipt,
                receipt_path,
                "v35-messages-fts",
                lambda: _rebuild_messages_fts(
                    conn,
                    canonical,
                    batch_rows=batch_rows,
                    pressure_guard=guard,
                ),
            )
            if fail_after_stage == "v35-messages-fts":
                raise RuntimeError("injected failure after v35-messages-fts")
            guard()

            _run_stage(
                receipt,
                receipt_path,
                "v35-insight-fts",
                lambda: {
                    "session_work_events_fts": _rebuild_small_fts(
                        conn,
                        canonical,
                        table_name="session_work_events_fts",
                        trigger_names=(
                            "session_work_events_fts_ai",
                            "session_work_events_fts_ad",
                            "session_work_events_fts_au",
                        ),
                        insert_sql=f"""
                            INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
                            SELECT event_id, session_id, work_event_type, {pl_fold_sql_expr("search_text")}
                            FROM session_work_events
                        """,
                    ),
                    "threads_fts": _rebuild_small_fts(
                        conn,
                        canonical,
                        table_name="threads_fts",
                        trigger_names=("threads_fts_ai", "threads_fts_ad", "threads_fts_au"),
                        insert_sql=f"""
                            INSERT INTO threads_fts (thread_id, root_id, text)
                            SELECT thread_id, thread_id, {pl_fold_sql_expr("search_text")}
                            FROM threads
                        """,
                    ),
                },
            )
            record_fts_invariant_snapshot_sync(conn, fts_invariant_snapshot_sync(conn))
            conn.commit()
            if fail_after_stage == "v35-insight-fts":
                raise RuntimeError("injected failure after v35-insight-fts")
            guard()

        pre_version_validation = _run_stage(
            receipt,
            receipt_path,
            "clone-validation-before-version",
            lambda: validate_clone(
                db_path,
                expected_counts=before_counts,
                expected_version=FAST_FORWARD_FROM_VERSION,
                run_quick_check=True,
            ),
        )
        if not bool(pre_version_validation["ready"]):
            raise RuntimeError("clone invariants failed before user_version advance")
        if fail_after_stage == "clone-validation-before-version":
            raise RuntimeError("injected failure after clone-validation-before-version")

        with sqlite3.connect(db_path, timeout=120.0) as conn:
            conn.execute(f"PRAGMA user_version = {FAST_FORWARD_TO_VERSION}")
            conn.commit()
        final_validation = validate_clone(
            db_path,
            expected_counts=before_counts,
            expected_version=FAST_FORWARD_TO_VERSION,
            run_quick_check=False,
        )
        if not bool(final_validation["ready"]):
            raise RuntimeError("clone invariants failed after user_version advance")
        receipt.update(
            {
                "status": "clone_ready",
                "completed_at_ms": _now_ms(),
                "clone_identity_after": asdict(file_identity(db_path)),
                "validation": final_validation,
            }
        )
        receipt["receipt_payload_sha256"] = _receipt_hash(receipt)
        _write_receipt(receipt_path, receipt)
        return receipt
    except Exception as exc:
        receipt.update(
            {
                "status": "failed",
                "failed_at_ms": _now_ms(),
                "error": f"{type(exc).__name__}: {exc}",
                "clone_identity_after": asdict(file_identity(db_path)),
            }
        )
        _write_receipt(receipt_path, receipt)
        raise


def _reflink_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        ["cp", "--reflink=always", "--preserve=mode,timestamps", str(source), str(destination)],
        check=True,
    )
    source_wal = source.with_name(source.name + "-wal")
    if source_wal.exists() and source_wal.stat().st_size:
        shutil.copyfile(source_wal, destination.with_name(destination.name + "-wal"))
    _fsync_directory(destination.parent)


def create_and_fast_forward_generation(
    source_link: Path,
    receipt_path: Path,
    *,
    generation_root: Path | None = None,
    max_io_full_avg10: float = DEFAULT_MAX_IO_FULL_AVG10,
    max_memory_full_avg10: float = DEFAULT_MAX_MEMORY_FULL_AVG10,
    batch_rows: int = DEFAULT_MESSAGE_BATCH_ROWS,
) -> dict[str, object]:
    if not source_link.is_symlink():
        raise RuntimeError("active index path must be a symlink for atomic generation activation")
    source = source_link.resolve(strict=True)
    archive_root = source_link.parent
    root = generation_root or archive_root / ".index-generations"
    generation_id = f"gen-v35-fastforward-{_now_ms()}-{uuid.uuid4().hex[:8]}"
    generation_dir = root / generation_id
    clone = generation_dir / "index.db"
    before = file_identity(source_link)
    _reflink_copy(source, clone)
    with sqlite3.connect(clone, timeout=120.0) as conn:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        conn.commit()
    after_source = file_identity(source_link)
    if before != after_source:
        raise RuntimeError("active source identity changed while the clone was created")
    metadata = {
        "generation_id": generation_id,
        "owner_id": str(uuid.uuid4()),
        "archive_root": str(archive_root),
        "index_path": str(clone),
        "state": "inactive",
        "created_at_ms": _now_ms(),
        "source_snapshot": hashlib.sha256(
            json.dumps(asdict(before), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _write_receipt(generation_dir / "generation.json", metadata)
    receipt = fast_forward_clone(
        clone,
        receipt_path,
        max_io_full_avg10=max_io_full_avg10,
        max_memory_full_avg10=max_memory_full_avg10,
        batch_rows=batch_rows,
    )
    receipt.update(
        {
            "source_link": str(source_link),
            "source_identity": asdict(before),
            "clone_generation_id": generation_id,
            "clone_path": str(clone),
            "clone_checkpoint": list(checkpoint) if checkpoint is not None else None,
        }
    )
    receipt["receipt_payload_sha256"] = _receipt_hash(receipt)
    _write_receipt(receipt_path, receipt)
    return receipt


def _service_active(service: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", service],
        check=False,
    )
    return result.returncode == 0


def _service_state(service: str) -> dict[str, object]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "NRestarts",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    fields: dict[str, object] = {"returncode": result.returncode}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = int(value) if key in {"MainPID", "NRestarts"} and value.isdigit() else value
    return fields


def _swap_active_symlink(source_link: Path, target: Path, *, label: str) -> None:
    temporary = source_link.with_name(f".{source_link.name}.{label}-{uuid.uuid4().hex}.tmp")
    temporary.symlink_to(target)
    os.replace(temporary, source_link)
    _fsync_directory(source_link.parent)


def _require_unchanged_identity(path: Path, expected: object, *, label: str) -> dict[str, object]:
    expected_identity = cast(dict[str, object], expected)
    actual_identity = cast(dict[str, object], asdict(file_identity(path)))
    if actual_identity != expected_identity:
        raise RuntimeError(f"{label} file identity changed since clone proof")
    return actual_identity


def _quick_check_only(db_path: Path) -> list[str]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120.0) as conn:
        conn.execute("PRAGMA query_only = ON")
        return [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]


def _runtime_version_sanity(db_path: Path) -> dict[str, object]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120.0) as conn:
        conn.execute("PRAGMA query_only = ON")
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        schema_probe = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE (type='table' AND name='messages_fts') "
            "OR (type='view' AND name='delegations')"
        ).fetchone()
    return {
        "ready": version == FAST_FORWARD_TO_VERSION and int(schema_probe[0]) == 2,
        "user_version": version,
        "required_objects": int(schema_probe[0]),
    }


def _restart_and_verify_or_rollback(
    receipt: dict[str, object],
    receipt_path: Path,
    *,
    service: str,
    source_link: Path,
    rollback_target: Path,
) -> None:
    try:
        subprocess.run(["systemctl", "--user", "restart", service], check=True)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not _service_active(service):
            time.sleep(0.5)
        state = _service_state(service)
        validation = _runtime_version_sanity(source_link)
        if not _service_active(service) or not bool(validation["ready"]):
            raise RuntimeError(f"post-restart contract failed: service={state}, ready={validation['ready']}")
        receipt.update(
            {
                "service_restarted_at_ms": _now_ms(),
                "post_restart_service": state,
                "post_restart_validation": validation,
            }
        )
        _write_receipt(receipt_path, receipt)
    except Exception as exc:
        subprocess.run(["systemctl", "--user", "stop", service], check=False)
        _swap_active_symlink(source_link, rollback_target, label="auto-rollback")
        rollback_restart = subprocess.run(
            ["systemctl", "--user", "restart", service],
            check=False,
            capture_output=True,
            text=True,
        )
        receipt.update(
            {
                "status": "rolled_back_after_failed_activation",
                "rolled_back_at_ms": _now_ms(),
                "activation_failure": f"{type(exc).__name__}: {exc}",
                "rollback_restart_returncode": rollback_restart.returncode,
                "rollback_service": _service_state(service),
            }
        )
        _write_receipt(receipt_path, receipt)
        raise RuntimeError("v35 activation failed and the v32 symlink was restored") from exc


def activate_generation(receipt_path: Path, *, service: str | None = None, restart: bool = False) -> dict[str, object]:
    receipt = _load_receipt(receipt_path)
    if receipt.get("status") != "clone_ready":
        raise RuntimeError("activation requires a clone_ready receipt")
    if service and _service_active(service):
        raise RuntimeError(f"service must be stopped before activation: {service}")
    source_link = Path(str(receipt["source_link"]))
    source_target = Path(str(cast(dict[str, object], receipt["source_identity"])["resolved_path"]))
    clone = Path(str(receipt["clone_path"])).resolve(strict=True)
    if source_link.resolve(strict=True) != source_target.resolve(strict=True):
        raise RuntimeError("active generation changed since clone proof")
    source_identity = _require_unchanged_identity(
        source_link,
        receipt["source_identity"],
        label="source",
    )
    clone_identity = _require_unchanged_identity(
        clone,
        receipt["clone_identity_after"],
        label="clone",
    )
    quick_check = _quick_check_only(clone)
    if quick_check != ["ok"]:
        raise RuntimeError("clone quick_check failed before activation")
    validation: dict[str, object] = {
        "ready": True,
        "quick_check": quick_check,
        "source_identity": source_identity,
        "clone_identity": clone_identity,
        "receipt_validation_reused": True,
    }
    _swap_active_symlink(source_link, clone, label="v35")
    receipt.update(
        {
            "status": "activated",
            "activated_at_ms": _now_ms(),
            "rollback_target": str(source_target),
            "activation_validation": validation,
        }
    )
    _write_receipt(receipt_path, receipt)
    if service and restart:
        _restart_and_verify_or_rollback(
            receipt,
            receipt_path,
            service=service,
            source_link=source_link,
            rollback_target=source_target,
        )
    return receipt


def rollback_generation(receipt_path: Path, *, service: str | None = None, restart: bool = False) -> dict[str, object]:
    receipt = _load_receipt(receipt_path)
    if receipt.get("status") != "activated":
        raise RuntimeError("rollback requires an activated receipt")
    if service and _service_active(service):
        raise RuntimeError(f"service must be stopped before rollback: {service}")
    source_link = Path(str(receipt["source_link"]))
    rollback_target = Path(str(receipt["rollback_target"])).resolve(strict=True)
    _swap_active_symlink(source_link, rollback_target, label="rollback")
    receipt.update({"status": "rolled_back", "rolled_back_at_ms": _now_ms()})
    _write_receipt(receipt_path, receipt)
    if service and restart:
        subprocess.run(["systemctl", "--user", "restart", service], check=True)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Read-only v32 source census")
    plan.add_argument("--source", type=Path, required=True)
    plan.add_argument("--receipt", type=Path)

    clone = subparsers.add_parser("clone-upgrade", help="Reflink and fast-forward an inactive clone")
    clone.add_argument("--source", type=Path, required=True)
    clone.add_argument("--receipt", type=Path, required=True)
    clone.add_argument("--generation-root", type=Path)
    clone.add_argument("--service", default="polylogued.service")
    clone.add_argument("--batch-rows", type=int, default=DEFAULT_MESSAGE_BATCH_ROWS)
    clone.add_argument("--max-io-full-avg10", type=float, default=DEFAULT_MAX_IO_FULL_AVG10)
    clone.add_argument("--max-memory-full-avg10", type=float, default=DEFAULT_MAX_MEMORY_FULL_AVG10)

    validate = subparsers.add_parser("validate", help="Revalidate a clone receipt")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--quick-check", action="store_true")

    activate = subparsers.add_parser("activate", help="Atomically activate a proven clone")
    activate.add_argument("--receipt", type=Path, required=True)
    activate.add_argument("--service", default="polylogued.service")
    activate.add_argument("--restart", action="store_true")

    rollback = subparsers.add_parser("rollback", help="Atomically restore the retained v32 target")
    rollback.add_argument("--receipt", type=Path, required=True)
    rollback.add_argument("--service", default="polylogued.service")
    rollback.add_argument("--restart", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = plan_index(args.source)
        if args.receipt:
            payload = {"schema": RECEIPT_SCHEMA, "status": "planned", **result}
            payload["receipt_payload_sha256"] = _receipt_hash(payload)
            _write_receipt(args.receipt, payload)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if bool(result["eligible"]) else 1
    if args.command == "clone-upgrade":
        if _service_active(args.service):
            raise SystemExit(f"refusing clone while {args.service} is active; stop it explicitly first")
        result = create_and_fast_forward_generation(
            args.source,
            args.receipt,
            generation_root=args.generation_root,
            max_io_full_avg10=args.max_io_full_avg10,
            max_memory_full_avg10=args.max_memory_full_avg10,
            batch_rows=args.batch_rows,
        )
    elif args.command == "validate":
        receipt = _load_receipt(args.receipt)
        result = validate_clone(
            Path(str(receipt["clone_path"])),
            expected_counts=cast(dict[str, int], receipt["structural_counts_before"]),
            expected_version=FAST_FORWARD_TO_VERSION,
            run_quick_check=args.quick_check,
        )
    elif args.command == "activate":
        result = activate_generation(args.receipt, service=args.service, restart=args.restart)
    else:
        result = rollback_generation(args.receipt, service=args.service, restart=args.restart)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
