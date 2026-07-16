#!/usr/bin/env python3
"""Invalidate only census-proven disposable cursors for yla8.6 recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

_CURSOR_COLUMNS = (
    "source_path",
    "stat_size",
    "byte_offset",
    "last_complete_newline",
    "content_fingerprint",
    "failure_count",
    "next_retry_at",
    "excluded",
    "updated_at_ms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assert_daemon_stopped() -> None:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", "polylogued.service"],
        check=False,
    )
    if result.returncode == 0:
        raise RuntimeError("polylogued.service is active; cursor repair requires a stopped daemon")


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"backup timestamp has no timezone: {value}")
    return parsed.astimezone(UTC)


def _validate_durable_backup(
    path: Path,
    *,
    archive_root: Path,
    census_generated_at: str,
) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("format") != "polylogue-backup-v1":
        raise ValueError("durable backup manifest has the wrong format")
    included = set(manifest.get("included_tiers") or ())
    if not {"source.db", "user.db"}.issubset(included):
        raise ValueError("durable backup must include source.db and user.db")
    warnings = manifest.get("warnings") or []
    if warnings:
        raise ValueError(f"durable backup manifest has warnings: {warnings}")
    blob_debt = manifest.get("blob_reference_debt")
    if not isinstance(blob_debt, dict) or blob_debt.get("ok") is not True:
        raise ValueError("durable backup does not prove referenced-blob completeness")
    created_at = str(manifest.get("created_at") or "")
    if not created_at or _timestamp(created_at) < _timestamp(census_generated_at):
        raise ValueError("durable backup predates the repair census")
    for filename in ("source.db", "user.db", "manifest.json", "blob-inventory.json"):
        if not (path.parent / filename).exists():
            raise ValueError(f"durable backup is missing {filename}")
    expected_receipt = path.parent / "verification-receipt.json"
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER):
        live_path = archive_root / f"{tier.value}.db"
        with sqlite3.connect(f"file:{live_path.resolve()}?mode=ro", uri=True) as conn:
            validated = validate_migration_backup_manifest(path, tier, connection=conn)
        if validated.resolve() != expected_receipt.resolve():
            raise ValueError(f"durable backup resolved to an unexpected verification receipt: {validated}")
    return manifest


def _candidate_snapshots(census: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    snapshots: dict[str, dict[str, Any]] = {}
    broken_head_paths: set[str] = set()
    for key in ("cursor_ahead_rows", "broken_head_cursors"):
        rows = census.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"census field {key} is not a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"census field {key} contains a non-object row")
            source_path = str(row.get("source_path") or "")
            if not source_path:
                raise ValueError(f"census field {key} contains an empty source path")
            if key == "broken_head_cursors":
                broken_head_paths.add(source_path)
            snapshot = {column: row.get(column) for column in _CURSOR_COLUMNS}
            prior = snapshots.get(source_path)
            if prior is not None and prior != snapshot:
                raise ValueError(f"conflicting census cursor snapshots for {source_path}")
            snapshots[source_path] = snapshot
    if not snapshots:
        raise ValueError("census contains no cursor repair candidates")
    return snapshots, broken_head_paths


def _read_cursor(conn: sqlite3.Connection, source_path: str) -> dict[str, Any] | None:
    row = conn.execute(
        f"SELECT {', '.join(_CURSOR_COLUMNS)} FROM ingest_cursor WHERE source_path = ?",
        (source_path,),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(_CURSOR_COLUMNS, row, strict=True))


def _validate_live_rows(
    conn: sqlite3.Connection,
    snapshots: dict[str, dict[str, Any]],
    *,
    allow_excluded_paths: set[str],
) -> list[dict[str, Any]]:
    live: list[dict[str, Any]] = []
    for source_path, expected in sorted(snapshots.items()):
        actual = _read_cursor(conn, source_path)
        if actual is None:
            raise RuntimeError(f"census cursor disappeared before repair: {source_path}")
        if actual != expected:
            raise RuntimeError(f"census cursor changed before repair: {source_path}")
        if int(actual["excluded"] or 0) != 0 and source_path not in allow_excluded_paths:
            raise RuntimeError(f"refusing to repair an excluded cursor: {source_path}")
        if not Path(source_path).is_file():
            raise RuntimeError(f"source file is unavailable: {source_path}")
        live.append(actual)
    return live


def _backup_ops(ops_db: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    with (
        sqlite3.connect(f"file:{ops_db.resolve()}?mode=ro", uri=True) as source,
        sqlite3.connect(destination) as target,
    ):
        source.backup(target)


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def repair(
    *,
    archive_root: Path,
    census_path: Path,
    durable_backup_manifest: Path | None,
    receipt_dir: Path | None,
    apply_changes: bool,
) -> dict[str, Any]:
    _assert_daemon_stopped()
    census = _load_json(census_path)
    if census.get("schema") != "polylogue.yla8-production-census.v1":
        raise ValueError("unsupported census schema")
    if Path(str(census.get("archive_root") or "")).resolve() != archive_root.resolve():
        raise ValueError("census archive root does not match the requested archive root")
    snapshots, broken_head_paths = _candidate_snapshots(census)
    ops_db = archive_root / "ops.db"
    if not ops_db.is_file():
        raise FileNotFoundError(ops_db)

    with sqlite3.connect(f"file:{ops_db.resolve()}?mode=ro", uri=True) as conn:
        live_rows = _validate_live_rows(conn, snapshots, allow_excluded_paths=broken_head_paths)
        before_count = int(conn.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0])

    summary: dict[str, Any] = {
        "schema": "polylogue.yla8-cursor-repair.v1",
        "mode": "apply" if apply_changes else "dry-run",
        "archive_root": str(archive_root.resolve()),
        "census_path": str(census_path.resolve()),
        "census_sha256": _sha256(census_path),
        "census_generated_at": census.get("generated_at"),
        "candidate_count": len(snapshots),
        "broken_head_candidate_count": len(broken_head_paths),
        "excluded_broken_head_candidate_count": sum(1 for row in live_rows if int(row["excluded"] or 0) != 0),
        "before_cursor_count": before_count,
        "candidate_source_bytes": sum(Path(path).stat().st_size for path in snapshots),
        "candidates": live_rows,
    }
    if not apply_changes:
        return summary
    if durable_backup_manifest is None or receipt_dir is None:
        raise ValueError("--apply requires --durable-backup-manifest and --receipt-dir")
    if receipt_dir.exists() and any(receipt_dir.iterdir()):
        raise FileExistsError(f"receipt directory is not empty: {receipt_dir}")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    manifest = _validate_durable_backup(
        durable_backup_manifest,
        archive_root=archive_root,
        census_generated_at=str(census.get("generated_at") or ""),
    )

    _assert_daemon_stopped()
    ops_backup = receipt_dir / "ops-before.db"
    _backup_ops(ops_db, ops_backup)
    with sqlite3.connect(ops_db) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _validate_live_rows(conn, snapshots, allow_excluded_paths=broken_head_paths)
            deleted = 0
            for source_path, expected in sorted(snapshots.items()):
                deleted += conn.execute(
                    """DELETE FROM ingest_cursor
                       WHERE source_path = ? AND updated_at_ms = ? AND byte_offset = ?""",
                    (source_path, expected["updated_at_ms"], expected["byte_offset"]),
                ).rowcount
            if deleted != len(snapshots):
                raise RuntimeError(f"deleted {deleted} cursors, expected {len(snapshots)}")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        after_count = int(conn.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0])
        remaining = sum(1 for source_path in snapshots if _read_cursor(conn, source_path) is not None)
    if remaining:
        raise RuntimeError(f"{remaining} repaired cursors remain after commit")

    summary.update(
        {
            "applied_at": datetime.now(UTC).isoformat(),
            "deleted_cursor_count": deleted,
            "after_cursor_count": after_count,
            "remaining_candidate_count": remaining,
            "ops_backup": str(ops_backup),
            "ops_backup_sha256": _sha256(ops_backup),
            "durable_backup_manifest": str(durable_backup_manifest.resolve()),
            "durable_backup_created_at": manifest.get("created_at"),
        }
    )
    receipt_path = receipt_dir / "cursor-repair.json"
    _write_receipt(receipt_path, summary)
    summary["receipt_path"] = str(receipt_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--durable-backup-manifest", type=Path)
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = repair(
        archive_root=args.archive_root,
        census_path=args.census,
        durable_backup_manifest=args.durable_backup_manifest,
        receipt_dir=args.receipt_dir,
        apply_changes=args.apply,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
