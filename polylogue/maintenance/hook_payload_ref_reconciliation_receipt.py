"""Crash-recoverable receipts for hook-payload reference reconciliation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

from polylogue.storage.hook_payload_ref_reconciliation import (
    HookPayloadRefReconciliationCandidate,
    HookPayloadRefReconciliationPlan,
    plan_hook_payload_ref_reconciliation,
)

RECEIPT_KIND = "hook_payload_ref_reconciliation"


class HookPayloadRefReconciliationReceiptError(RuntimeError):
    """Raised when a reconciliation receipt cannot be safely created or recovered."""


def classification_counts(plan: HookPayloadRefReconciliationPlan) -> dict[str, int]:
    """Return the complete count projection recorded before and after apply."""
    return {
        "scanned_count": plan.scanned_count,
        "matched_count": len(plan.matched),
        "matched_bytes": plan.matched_bytes,
        "unmatched_count": plan.unmatched_count,
    }


def _candidate_dict(candidate: HookPayloadRefReconciliationCandidate) -> dict[str, object]:
    return {
        "blob_hash": candidate.blob_hash.hex(),
        "orphaned_ref_id": candidate.orphaned_ref_id,
        "source_path": candidate.source_path,
        "size_bytes": candidate.size_bytes,
        "acquired_at_ms": candidate.acquired_at_ms,
        "hook_event_id": candidate.hook_event_id,
    }


def _candidate_digest(candidates: tuple[HookPayloadRefReconciliationCandidate, ...]) -> str:
    payload = sorted(
        (_candidate_dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            str(candidate["hook_event_id"]),
            str(candidate["orphaned_ref_id"]),
            str(candidate["blob_hash"]),
        ),
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reconciled_ids_digest(hook_event_ids: tuple[str, ...] | list[str]) -> str:
    """Return a stable digest for the exact ids reconciled by one transaction."""
    encoded = json.dumps(sorted(hook_event_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def backup_manifest_identity(backup_manifest: Path) -> dict[str, object]:
    """Return the byte identity of the manifest just accepted by validation."""
    manifest_path = backup_manifest / "manifest.json" if backup_manifest.is_dir() else backup_manifest
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise HookPayloadRefReconciliationReceiptError(
            f"could not read validated backup manifest identity: {manifest_path}"
        ) from exc
    return {
        "path": str(manifest_path.resolve(strict=True)),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size_bytes": len(manifest_bytes),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_prepared_receipt(
    receipt_path: Path,
    *,
    source_db: Path,
    tool_version: str,
    backup_manifest: dict[str, object],
    plan: HookPayloadRefReconciliationPlan,
) -> None:
    """Exclusively write and fsync the immutable plan before database mutation."""
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "kind": RECEIPT_KIND,
        "phase": "prepared",
        "tool_version": tool_version,
        "source_db": str(source_db.resolve(strict=True)),
        "backup_manifest": backup_manifest,
        "prepared_at_ms": int(time.time() * 1000),
        "pre_classification": classification_counts(plan),
        "candidate_count": len(plan.matched),
        "candidate_digest": _candidate_digest(plan.matched),
    }
    try:
        with receipt_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(header, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            for candidate in plan.matched:
                handle.write(
                    json.dumps(
                        {"kind": "candidate", **_candidate_dict(candidate)}, sort_keys=True, separators=(",", ":")
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(receipt_path.parent)
    except FileExistsError as exc:
        raise HookPayloadRefReconciliationReceiptError(f"receipt already exists: {receipt_path}") from exc


def append_terminal_receipt(
    receipt_path: Path,
    *,
    terminal_state: str,
    post_plan: HookPayloadRefReconciliationPlan | None = None,
    reconciled_hook_event_ids: tuple[str, ...] | None = None,
    error: str | None = None,
) -> None:
    """Append the only terminal receipt record and sync it to durable storage."""
    payload: dict[str, object] = {
        "kind": RECEIPT_KIND,
        "phase": terminal_state,
        "terminal_state": terminal_state,
        "completed_at_ms": int(time.time() * 1000),
    }
    if post_plan is not None:
        payload["post_classification"] = classification_counts(post_plan)
    if reconciled_hook_event_ids is not None:
        payload["reconciled_hook_event_ids"] = list(reconciled_hook_event_ids)
        payload["reconciled_ids_digest"] = reconciled_ids_digest(reconciled_hook_event_ids)
    if error is not None:
        payload["error"] = error
    with receipt_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(receipt_path.parent)


def _load_prepared_candidates(receipt_path: Path) -> list[dict[str, object]]:
    try:
        rows = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise HookPayloadRefReconciliationReceiptError(f"could not read receipt: {receipt_path}") from exc
    if not rows or rows[0].get("kind") != RECEIPT_KIND or rows[0].get("phase") != "prepared":
        raise HookPayloadRefReconciliationReceiptError(
            f"receipt is not a prepared hook reconciliation receipt: {receipt_path}"
        )
    terminal_states = {"committed", "aborted", "recovered_committed", "recovered_rolled_back", "indeterminate"}
    if any(row.get("terminal_state") in terminal_states for row in rows[1:]):
        raise HookPayloadRefReconciliationReceiptError(f"receipt is already terminal: {receipt_path}")
    candidates = [row for row in rows[1:] if row.get("kind") == "candidate"]
    if len(candidates) != rows[0].get("candidate_count"):
        raise HookPayloadRefReconciliationReceiptError(f"receipt candidate count is invalid: {receipt_path}")
    return candidates


def recover_prepared_receipt(source_db: Path, receipt_path: Path) -> str:
    """Determine the outcome of a prepared receipt without making DB changes."""
    candidates = _load_prepared_candidates(receipt_path)
    committed = 0
    rolled_back = 0
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
        for candidate in candidates:
            blob_hash = bytes.fromhex(str(candidate["blob_hash"]))
            hook_event_id = str(candidate["hook_event_id"])
            orphaned_ref_id = str(candidate["orphaned_ref_id"])
            canonical = conn.execute(
                "SELECT 1 FROM blob_refs WHERE blob_hash = ? AND ref_type = 'hook_payload' AND ref_id = ?",
                (blob_hash, hook_event_id),
            ).fetchone()
            stale = conn.execute(
                "SELECT 1 FROM blob_refs WHERE blob_hash = ? AND ref_type = 'raw_payload' AND ref_id = ?",
                (blob_hash, orphaned_ref_id),
            ).fetchone()
            event = conn.execute(
                "SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = ?", (hook_event_id,)
            ).fetchone()
            if (
                canonical is not None
                and stale is None
                and event is not None
                and event[0] is not None
                and bytes(event[0]) == blob_hash
            ):
                committed += 1
            elif canonical is None and stale is not None:
                rolled_back += 1
    if committed == len(candidates):
        terminal_state = "recovered_committed"
    elif rolled_back == len(candidates):
        terminal_state = "recovered_rolled_back"
    else:
        terminal_state = "indeterminate"
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as conn:
        post_plan = plan_hook_payload_ref_reconciliation(conn)
    reconciled_ids = (
        tuple(str(candidate["hook_event_id"]) for candidate in candidates)
        if terminal_state == "recovered_committed"
        else None
    )
    append_terminal_receipt(
        receipt_path,
        terminal_state=terminal_state,
        post_plan=post_plan,
        reconciled_hook_event_ids=reconciled_ids,
    )
    return terminal_state


__all__ = [
    "HookPayloadRefReconciliationReceiptError",
    "append_terminal_receipt",
    "backup_manifest_identity",
    "classification_counts",
    "reconciled_ids_digest",
    "recover_prepared_receipt",
    "write_prepared_receipt",
]
