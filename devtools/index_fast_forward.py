"""Plan-driven, source-proven fast-forward for rebuildable index generations.

The lifecycle declaration is the only version authority.  This actuator owns
the clone, source-backed proof, receipt, and promotion boundary.  The proof
replays a deterministic sample of retained raw evidence through the production
parser and session writer into a scratch current-schema index, then compares
canonical sessions/messages/blocks/FTS rows with the transformed clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import closing, suppress
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import cast

from devtools.clone_support import reflink_clone
from polylogue.config import Config
from polylogue.maintenance.archive_verification import (
    REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS,
    passes_strict_acceptance,
    strict_acceptance_failures,
    verify_archive,
)
from polylogue.maintenance.offline_guard import running_daemon_pid
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.index_generation import IndexGenerationStore, RebuildLease, source_revision_snapshot
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.index_fast_forward_executor import apply_index_fast_forward
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.lifecycle import FastForwardOperationKind, IndexFastForwardPlan, index_fast_forward_plan
from polylogue.storage.sqlite.runtime_indexes import ensure_runtime_indexes_sync

RECEIPT_SCHEMA = "polylogue.index-fast-forward.v1"
DEFAULT_SAMPLE_SIZE = 8
IN_QUERY_CHUNK_SIZE = 500


class IndexFastForwardError(RuntimeError):
    """The declared index transition could not be proven safe to activate."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_ddl(sql: str) -> str:
    """Normalize canonical DDL without collapsing literal or identifier boundaries."""
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in sql:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            if current:
                tokens.append("".join(current).lower())
                current.clear()
            quote = char
            current.append(char)
        elif char.isalnum() or char in {"_", "$", "."}:
            current.append(char)
        else:
            if current:
                tokens.append("".join(current).lower())
                current.clear()
            if not char.isspace():
                tokens.append(char)
    if current:
        tokens.append("".join(current).lower())
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index : index + 3] == ["if", "not", "exists"]:
            index += 3
            continue
        normalized.append(tokens[index])
        index += 1
    return json.dumps(normalized, separators=(",", ":"))


def _schema_objects(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT type, name, sql FROM sqlite_master
        WHERE type IN ('table', 'index', 'view', 'trigger')
          AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
        ORDER BY type, name
        """
    )
    return {f"{row[0]}:{row[1]}": _normalize_ddl(str(row[2])) for row in rows}


def _canonical_schema_objects() -> dict[str, str]:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX])
        ensure_runtime_indexes_sync(conn)
        return _schema_objects(conn)


def _canonical_schema_sha256(schema: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    identity = _file_identity(path)
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    identity["sha256"] = digest.hexdigest()
    return identity


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
        raise IndexFastForwardError(f"invalid fast-forward receipt: {path}")
    typed = cast(dict[str, object], payload)
    if typed.get("receipt_sha256") != _receipt_hash(typed):
        raise IndexFastForwardError(f"fast-forward receipt hash mismatch: {path}")
    return typed


def _config(archive_root: Path) -> Config:
    return Config(
        archive_root=archive_root, render_root=archive_root / "render", sources=[], db_path=archive_root / "index.db"
    )


def _require_daemon_stopped(archive_root: Path) -> None:
    if (pid := running_daemon_pid(_config(archive_root))) is not None:
        raise IndexFastForwardError(f"polylogued PID {pid} is still running")


def _require_receipt_destination_writable(path: Path) -> None:
    probe = path.with_name(f".{path.name}.{uuid.uuid4().hex}.probe")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not path.is_file():
            raise OSError(f"receipt destination is not a regular file: {path}")
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        probe.unlink()
    except OSError as exc:
        with suppress(OSError):
            probe.unlink(missing_ok=True)
        raise IndexFastForwardError(f"receipt destination is not writable: {path}: {exc}") from exc


def _checkpoint_stopped_database(path: Path, *, label: str = "active index") -> None:
    resolved = path.resolve(strict=True)
    with closing(sqlite3.connect(resolved, timeout=120.0)) as conn:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is None or int(checkpoint[0]) != 0 or int(checkpoint[1]) != int(checkpoint[2]):
        raise IndexFastForwardError(f"{label} WAL checkpoint failed: {checkpoint}")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{resolved}{suffix}")
        if sidecar.exists():
            sidecar.unlink()


def _inspect_clean_database(path: Path) -> tuple[int, dict[str, str]]:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{path.resolve(strict=True)}{suffix}")
        if sidecar.exists() and sidecar.stat().st_size:
            raise IndexFastForwardError(f"non-empty SQLite sidecar blocks fast-forward: {sidecar}")
    with closing(open_readonly_connection(path.resolve(strict=True), immutable=True)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0]), _schema_objects(conn)


def _plan_for_database(version: int) -> IndexFastForwardPlan:
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    target = int(ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX])
    plan = index_fast_forward_plan(version, target)
    if plan is None or not plan.eligible_for_sql_fast_forward:
        raise IndexFastForwardError(
            f"index v{version} to v{target} is not SQL-fast-forwardable; route semantic work to replay/rebuild"
        )
    return plan


def _expected_surplus(plan: IndexFastForwardPlan) -> set[str]:
    return {
        f"{kind}:{name}"
        for declaration in plan.declarations
        for operation in declaration.operations
        if operation.kind is FastForwardOperationKind.DROP_TABLE
        for kind, name in operation.objects
    }


def _transform_clone(path: Path, *, plan: IndexFastForwardPlan, before_schema: dict[str, str]) -> dict[str, object]:
    with closing(sqlite3.connect(path, timeout=120.0)) as conn:
        apply_index_fast_forward(conn, plan)
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        checks = [str(row[0]) for row in conn.execute("PRAGMA quick_check")]
        after_schema = _schema_objects(conn)
    canonical = _canonical_schema_objects()
    actual_surplus = set(after_schema) - set(canonical)
    expected_surplus = _expected_surplus(plan)
    before_surplus = set(before_schema) - set(canonical)
    if version != plan.target_version or checks != ["ok"] or after_schema != canonical:
        raise IndexFastForwardError(
            f"fast-forward postflight failed: version={version}, checks={checks}, "
            f"missing={sorted(set(canonical) - set(after_schema))}, surplus={sorted(actual_surplus)}"
        )
    if before_surplus != expected_surplus:
        raise IndexFastForwardError(
            f"active schema does not match declared plan surplus: expected={sorted(expected_surplus)}, "
            f"found={sorted(before_surplus)}"
        )
    return {
        "quick_check": checks,
        "schema_object_count": len(after_schema),
        "removed_objects": sorted(expected_surplus),
    }


def _materializer_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "polylogue/storage/sqlite/archive_tiers/write.py",
        root / "polylogue/storage/insights/session/rebuild.py",
        root / "polylogue/storage/runtime/store_constants.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    digest.update(str(SESSION_INSIGHT_MATERIALIZER_VERSION).encode())
    return digest.hexdigest()


def _fingerprints(origins: tuple[str, ...]) -> dict[str, object]:
    return {
        "parser": {origin: parser_fingerprint_for_origin(origin) for origin in origins},
        "lowering": lowering_fingerprint(),
        "materializer": _materializer_fingerprint(),
        "materializer_version": SESSION_INSIGHT_MATERIALIZER_VERSION,
    }


def _chunks(values: tuple[str, ...] | list[str], *, size: int | None = None) -> Iterator[tuple[str, ...]]:
    effective_size = IN_QUERY_CHUNK_SIZE if size is None else size
    if effective_size <= 0:
        raise ValueError("SQLite IN-query chunk size must be positive")
    for offset in range(0, len(values), effective_size):
        yield tuple(values[offset : offset + effective_size])


def _query_rows_in_chunks(
    conn: sqlite3.Connection,
    sql_for_marks: Callable[[str], str],
    values: tuple[str, ...] | list[str],
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for chunk in _chunks(values):
        marks = ", ".join("?" for _ in chunk)
        rows.extend(conn.execute(sql_for_marks(marks), chunk).fetchall())
    return rows


def _sample_manifest(archive_root: Path, index_path: Path, *, limit: int) -> list[dict[str, object]]:
    with closing(sqlite3.connect(archive_root / "source.db")) as source_conn:
        raw_ids = [str(row[0]) for row in source_conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id")]
    if not raw_ids:
        raise IndexFastForwardError("source-backed fast-forward proof requires retained raw evidence")
    with closing(open_readonly_connection(index_path.resolve(strict=True), immutable=True)) as index_conn:
        rows = _query_rows_in_chunks(
            index_conn,
            lambda marks: (
                f"SELECT raw_id, session_id, origin FROM sessions WHERE raw_id IN ({marks}) ORDER BY raw_id, session_id"
            ),
            raw_ids,
        )
    grouped: dict[str, dict[str, object]] = {}
    for raw_id, session_id, origin in rows:
        entry = grouped.setdefault(str(raw_id), {"raw_id": str(raw_id), "session_ids": [], "origins": []})
        cast(list[str], entry["session_ids"]).append(str(session_id))
        if str(origin) not in cast(list[str], entry["origins"]):
            cast(list[str], entry["origins"]).append(str(origin))
    sample = [grouped[raw_id] for raw_id in sorted(grouped)[:limit]]
    if not sample:
        raise IndexFastForwardError("source-backed fast-forward proof requires a raw-backed indexed session")
    return sample


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return value


def _hash_rows(rows: list[tuple[object, ...]]) -> str:
    payload = [[_json_value(value) for value in row] for row in rows]
    payload.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash_query_in_chunks(
    conn: sqlite3.Connection,
    sql_for_marks: Callable[[str], str],
    values: tuple[str, ...],
) -> str:
    return _hash_rows(_query_rows_in_chunks(conn, sql_for_marks, values))


def _scoped_table_sql(marks: str, *, table_name: str, key_column: str, order_by: str) -> str:
    return f'SELECT * FROM "{table_name}" WHERE "{key_column}" IN ({marks}) ORDER BY {order_by}'


def _canonical_hashes(conn: sqlite3.Connection, session_ids: tuple[str, ...]) -> dict[str, object]:
    if not session_ids:
        return {"sessions": "", "messages": "", "blocks": "", "fts": "", "scoped": {}}
    messages = _query_rows_in_chunks(
        conn,
        lambda marks: f"SELECT message_id FROM messages WHERE session_id IN ({marks}) ORDER BY message_id",
        session_ids,
    )
    message_ids = tuple(sorted(str(row[0]) for row in messages))
    scoped: dict[str, str] = {}
    for table_name, key_column in (
        (str(row[0]), key_column)
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        for key_column in ("session_id", "src_session_id")
        if key_column and key_column in {str(info[1]) for info in conn.execute(f'PRAGMA table_info("{row[0]}")')}
    ):
        columns = [str(info[1]) for info in conn.execute(f'PRAGMA table_info("{table_name}")')]
        order_by = ", ".join(str(index) for index in range(1, len(columns) + 1))

        scoped[table_name] = _hash_query_in_chunks(
            conn,
            partial(_scoped_table_sql, table_name=table_name, key_column=key_column, order_by=order_by),
            session_ids,
        )
    return {
        "sessions": _hash_query_in_chunks(
            conn, lambda marks: f"SELECT * FROM sessions WHERE session_id IN ({marks}) ORDER BY session_id", session_ids
        ),
        "messages": _hash_query_in_chunks(
            conn,
            lambda marks: f"SELECT * FROM messages WHERE session_id IN ({marks}) ORDER BY message_id",
            session_ids,
        ),
        "blocks": _hash_query_in_chunks(
            conn,
            lambda marks: f"SELECT * FROM blocks WHERE message_id IN ({marks}) ORDER BY block_id",
            message_ids,
        ),
        "fts": _hash_query_in_chunks(
            conn,
            lambda marks: (
                "SELECT block_id, message_id, session_id, block_type, text "
                f"FROM messages_fts WHERE session_id IN ({marks}) ORDER BY session_id, message_id, block_id"
            ),
            session_ids,
        ),
        "scoped": scoped,
    }


def _replay_sample(archive_root: Path, candidate_index: Path, manifest: list[dict[str, object]]) -> dict[str, object]:
    from polylogue.sources.revision_backfill import parse_retained_raw_sessions

    expected_ids = tuple(session_id for entry in manifest for session_id in cast(list[str], entry["session_ids"]))
    with closing(sqlite3.connect(":memory:")) as replay_conn:
        replay_conn.executescript(INDEX_DDL)
        ensure_runtime_indexes_sync(replay_conn)
        # ``parse_retained_raw_sessions`` is semantically read-only, but the
        # shared raw-revision descriptor requires the production blob
        # publisher to be present.  Attach its read-only filesystem facade to
        # the read-only archive store.  No pending blob is queued or flushed.
        with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
            archive._blob_publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
            replayed_ids: list[str] = []
            for entry in manifest:
                raw_id = str(entry["raw_id"])
                for session in parse_retained_raw_sessions(archive, raw_id):
                    replayed_ids.append(
                        write_parsed_session_to_archive(
                            replay_conn,
                            session,
                            content_hash=session_content_hash(session),
                            raw_id=raw_id,
                            force_replace=True,
                        )
                    )
        actual_ids = tuple(sorted(set(replayed_ids)))
        canonical = _canonical_hashes(replay_conn, tuple(sorted(set(expected_ids) | set(actual_ids))))
    with closing(open_readonly_connection(candidate_index.resolve(strict=True), immutable=True)) as candidate_conn:
        fast_forward = _canonical_hashes(candidate_conn, tuple(sorted(set(expected_ids) | set(actual_ids))))
    mismatches = [key for key in canonical if canonical[key] != fast_forward[key]]
    if tuple(sorted(set(expected_ids))) != actual_ids:
        mismatches.append("sample_session_ids")
    return {
        "fast_forward_hashes": fast_forward,
        "canonical_replay_hashes": canonical,
        "replayed_session_ids": list(actual_ids),
        "mismatch_details": mismatches,
        "verdict": "equivalent" if not mismatches else "mismatch",
    }


def _require_complete_proof(proof: dict[str, object]) -> None:
    required_hashes = {"sessions", "messages", "blocks", "fts"}
    if proof.get("verdict") != "equivalent":
        raise IndexFastForwardError(f"source replay proof failed: {proof.get('mismatch_details')}")
    if proof.get("mismatch_details"):
        raise IndexFastForwardError(f"source replay proof has mismatches: {proof['mismatch_details']}")
    if not proof.get("replayed_session_ids"):
        raise IndexFastForwardError("source replay proof has no replayed session ids")
    for key in ("fast_forward_hashes", "canonical_replay_hashes"):
        hashes = proof.get(key)
        if (
            not isinstance(hashes, dict)
            or not required_hashes <= set(hashes)
            or not all(hashes[key] for key in required_hashes)
        ):
            raise IndexFastForwardError(f"source replay proof is incomplete: {key}")
        scoped = hashes.get("scoped")
        if not isinstance(scoped, dict) or not scoped:
            raise IndexFastForwardError(f"source replay proof is incomplete: {key}.scoped")
    if proof["fast_forward_hashes"] != proof["canonical_replay_hashes"]:
        raise IndexFastForwardError("source replay proof hashes disagree between clone and canonical replay")


def _require_candidate_strict_acceptance(archive_root: Path, candidate_index: Path) -> None:
    report = verify_archive(
        archive_root,
        checks=REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS,
        index_path_override=candidate_index,
    )
    if not passes_strict_acceptance(report, required_checks=REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS):
        failing = "; ".join(strict_acceptance_failures(report, required_checks=REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS))
        raise IndexFastForwardError(f"candidate strict acceptance gate failed: {failing}")


def prepare_forward(
    *, archive_root: Path, receipt_path: Path, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> dict[str, object]:
    """Create an inactive plan-driven candidate and prove it against retained raw replay."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    archive_root = archive_root.resolve(strict=True)
    _require_daemon_stopped(archive_root)
    _require_receipt_destination_writable(receipt_path)
    store = IndexGenerationStore.for_archive_root(archive_root)
    with RebuildLease(archive_root):
        _require_daemon_stopped(archive_root)
        active_pointer = store.active_pointer
        _checkpoint_stopped_database(active_pointer)
        source_snapshot = source_revision_snapshot(archive_root)
        active_identity = _file_identity(active_pointer)
        source_version, before_schema = _inspect_clean_database(active_pointer)
        plan = _plan_for_database(source_version)
        manifest = _sample_manifest(archive_root, active_pointer, limit=sample_size)
        origins = tuple(sorted({str(origin) for entry in manifest for origin in cast(list[str], entry["origins"])}))
        fingerprints = _fingerprints(origins)
        generation = store.create(source_snapshot=source_snapshot)
        clone = Path(generation.index_path)
        try:
            clone.unlink()
            reflink_clone(active_pointer, clone)
            if _file_identity(active_pointer) != active_identity:
                raise IndexFastForwardError("active index changed while its clone was created")
            postflight = _transform_clone(clone, plan=plan, before_schema=before_schema)
            proof = _replay_sample(archive_root, clone, manifest)
            _require_complete_proof(proof)
            if source_revision_snapshot(archive_root) != source_snapshot:
                raise IndexFastForwardError("source evidence changed while preparing fast-forward")
            receipt: dict[str, object] = {
                "schema": RECEIPT_SCHEMA,
                "status": "prepared",
                "prepared_at_ms": _now_ms(),
                "archive_root": str(archive_root),
                "generation": asdict(generation),
                "source_snapshot": source_snapshot,
                "active_identity": active_identity,
                "clone_identity": _proven_clone_identity(clone),
                "source_version": source_version,
                "target_version": plan.target_version,
                "stage_names": list(plan.stage_names),
                "canonical_schema_sha256": _canonical_schema_sha256(_canonical_schema_objects()),
                "sample_manifest": manifest,
                "fingerprints": fingerprints,
                "proof": proof,
                "postflight": postflight,
                "raw_reparse": False,
            }
            _write_receipt(receipt_path, receipt)
            return receipt
        except Exception:
            store.discard_if_inactive(generation)
            raise


def activate_forward(*, receipt_path: Path) -> dict[str, object]:
    """Verify the prepared proof and atomically promote its inactive generation."""
    receipt = _load_receipt(receipt_path)
    status = receipt.get("status")
    if status not in {"prepared", "activating", "activated"}:
        raise IndexFastForwardError(f"receipt is not prepared: {status}")
    if status == "activated":
        return receipt
    archive_root = Path(str(receipt["archive_root"])).resolve(strict=True)
    _require_daemon_stopped(archive_root)
    store = IndexGenerationStore.for_archive_root(archive_root)
    generation_payload = cast(dict[str, object], receipt["generation"])
    generation = store.load(str(generation_payload["generation_id"]))
    with RebuildLease(archive_root):
        _require_daemon_stopped(archive_root)
        if status == "activating":
            generation = store.recover_promotion(generation.generation_id)
        if generation.owner_id != generation_payload["owner_id"]:
            raise IndexFastForwardError("prepared generation ownership changed")
        clone = Path(generation.index_path)
        if generation.state == "active":
            if store.active_pointer.resolve(strict=True) != clone.resolve(strict=True):
                raise IndexFastForwardError("active generation does not own the active index pointer")
            receipt.update(
                {
                    "status": "activated",
                    "activated_at_ms": receipt.get("activated_at_ms", _now_ms()),
                    "generation": asdict(generation),
                }
            )
            _write_receipt(receipt_path, receipt)
            return receipt
        if generation.state != "inactive":
            raise IndexFastForwardError(f"prepared generation has unrecoverable state {generation.state}")
        if source_revision_snapshot(archive_root) != receipt["source_snapshot"]:
            raise IndexFastForwardError("source evidence changed since fast-forward preparation")
        if _file_identity(store.active_pointer) != receipt["active_identity"]:
            raise IndexFastForwardError("active index changed since fast-forward preparation")
        canonical_sha = _canonical_schema_sha256(_canonical_schema_objects())
        if canonical_sha != receipt.get("canonical_schema_sha256"):
            raise IndexFastForwardError("canonical index schema changed since preparation")
        if _proven_clone_identity(clone) != receipt.get("clone_identity"):
            raise IndexFastForwardError("prepared clone bytes changed before activation")
        proof = cast(dict[str, object], receipt.get("proof", {}))
        _require_complete_proof(proof)
        manifest = cast(list[dict[str, object]], receipt.get("sample_manifest", []))
        origins = tuple(sorted({str(origin) for entry in manifest for origin in cast(list[str], entry["origins"])}))
        if _fingerprints(origins) != receipt.get("fingerprints"):
            raise IndexFastForwardError("parser/materializer fingerprints changed since preparation")
        _require_candidate_strict_acceptance(archive_root, clone)
        if source_revision_snapshot(archive_root) != receipt["source_snapshot"]:
            raise IndexFastForwardError("source evidence changed immediately before promotion")
        if _proven_clone_identity(clone) != receipt.get("clone_identity"):
            raise IndexFastForwardError("prepared clone bytes changed immediately before promotion")
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
    prepare.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    activate = subparsers.add_parser("activate")
    activate.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        prepare_forward(archive_root=args.archive_root, receipt_path=args.receipt, sample_size=args.sample_size)
        if args.command == "prepare"
        else activate_forward(receipt_path=args.receipt)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["IndexFastForwardError", "activate_forward", "main", "prepare_forward"]
