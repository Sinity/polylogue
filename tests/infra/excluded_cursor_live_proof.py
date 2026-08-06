"""Proof harness for parser-fingerprint revival of excluded live cursors.

The harness uses the production watcher and live batch processor against a
candidate archive fixture. It deliberately reports candidate-fixture
coverage separately from a live census.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash, session_id
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import LiveWatcher, WatchSource
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.reindex_campaign import _codex_records, _write_jsonl

RECEIPT_SCHEMA = "polylogue.excluded-cursor-live-proof.v1"
FIXTURE_VERSION = "candidate-codex-live-compatible-2026-08-06"
OLD_PARSER_FINGERPRINT = "live-batched-v1"
NEW_PARSER_FINGERPRINT = "live-batched-v2"


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_excluded(cursor: CursorStore, path: Path, *, parser_fingerprint: str) -> None:
    stat = path.stat()
    cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint=parser_fingerprint,
        content_fingerprint=_sha256_file(path),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=5,
        excluded=True,
    )


def _seed_byte_authority(root: Path, path: Path, *, native_id: str) -> None:
    """Seed source evidence plus a byte head without materializing a session."""
    payload = path.read_bytes()
    logical_source_key = f"codex:{native_id}"
    [parsed] = parse_payload(
        Provider.CODEX,
        [json.loads(line) for line in payload.splitlines()],
        native_id,
        source_path=str(path),
    )
    source_revision = "excluded-cursor-proof-authority-0"
    accepted_content_hash = bytes.fromhex(session_content_hash(parsed))
    accepted_session_id = str(session_id(parsed.source_name, parsed.provider_session_id))
    revision = RawRevisionEnvelope(
        logical_source_key,
        RawRevisionKind.FULL,
        source_revision,
        0,
        authority=RawRevisionAuthority.BYTE_PROVEN,
    )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=str(path),
            acquired_at_ms=1,
            native_id=native_id,
            revision=revision,
        )
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_revision_heads (
                logical_source_key, session_id, accepted_raw_id,
                accepted_source_revision, accepted_content_hash,
                accepted_frontier_kind, accepted_frontier,
                acquisition_generation, append_end_offset, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, 'byte', ?, 0, NULL, 1)
            """,
            (
                logical_source_key,
                accepted_session_id,
                raw_id,
                source_revision,
                accepted_content_hash,
                len(payload),
            ),
        )
        conn.commit()


def _attempts_for_path(root: Path, path: Path) -> list[dict[str, object]]:
    conn = sqlite3.connect(root / "ops.db")
    try:
        rows = conn.execute(
            """
            SELECT outcome_code, retryable, evidence_ref, status, source_paths_json
            FROM ingest_attempts
            ORDER BY started_at_ms DESC, attempt_id DESC
            """
        ).fetchall()
    finally:
        conn.close()
    attempts: list[dict[str, object]] = []
    for outcome_code, retryable, evidence_ref, status, source_paths_json in rows:
        try:
            source_paths = json.loads(str(source_paths_json or "[]"))
        except json.JSONDecodeError:
            source_paths = []
        if str(path) not in source_paths:
            continue
        attempts.append(
            {
                "outcome_code": outcome_code,
                "retryable": None if retryable is None else bool(retryable),
                "evidence_ref": evidence_ref,
                "status": status,
            }
        )
    return attempts


def _retry_state(cursor: CursorStore, path: Path) -> dict[str, object]:
    record = cursor.get_record(path)
    if record is None:
        raise AssertionError(f"proof cursor disappeared for {path}")
    retry_paths = {record.source_path for record in cursor.list_retry_records()}
    failed_paths = set(cursor.list_failed_with_retry())
    return {
        "excluded": bool(record.excluded),
        "failure_count": record.failure_count,
        "retry_due": record.source_path in retry_paths,
        "failed_with_retry": record.source_path in failed_paths,
        "parser_fingerprint": record.parser_fingerprint,
    }


def _indexed_counts(root: Path, path: Path) -> dict[str, int]:
    with sqlite3.connect(root / "source.db") as source_conn:
        raw_rows = source_conn.execute(
            "SELECT raw_id FROM raw_sessions WHERE source_path = ? AND parse_error IS NULL",
            (str(path),),
        ).fetchall()
    raw_ids = tuple(str(row[0]) for row in raw_rows)
    if not raw_ids:
        return {"parsed_raw": 0, "indexed_sessions": 0}
    placeholders = ",".join("?" for _ in raw_ids)
    with sqlite3.connect(root / "index.db") as index_conn:
        indexed = index_conn.execute(
            f"SELECT COUNT(*) FROM sessions WHERE raw_id IN ({placeholders})",
            raw_ids,
        ).fetchone()
    return {"parsed_raw": len(raw_ids), "indexed_sessions": int(indexed[0]) if indexed else 0}


def _terminal_evidence(root: Path, path: Path) -> dict[str, object] | None:
    with sqlite3.connect(root / "source.db") as conn:
        row = conn.execute(
            """
            SELECT a.artifact_kind, a.support_status, r.parse_error
            FROM raw_artifacts AS a
            JOIN raw_sessions AS r USING (raw_id)
            WHERE r.source_path = ? AND a.artifact_kind LIKE 'terminal_%'
            ORDER BY r.acquired_at_ms DESC, r.raw_id DESC
            LIMIT 1
            """,
            (str(path),),
        ).fetchone()
    if row is None:
        return None
    return {
        "artifact_kind": str(row[0]),
        "support_status": str(row[1]),
        "parse_error_present": row[2] is not None,
    }


def _case_summary(
    *,
    case_id: str,
    path: Path,
    cursor: CursorStore,
    fingerprint_changed_before_catch_up: bool,
    metrics: object | None,
    root: Path,
    attempts_before: int,
) -> dict[str, object]:
    retry_state = _retry_state(cursor, path)
    attempts = _attempts_for_path(root, path)
    proof_attempts = attempts[: max(0, len(attempts) - attempts_before)]
    attempt = proof_attempts[0] if proof_attempts else None
    return {
        "case_id": case_id,
        "source_content_sha256": _sha256_file(path),
        "fingerprint_changed_before_catch_up": fingerprint_changed_before_catch_up,
        "metrics": {
            "succeeded_file_count": int(getattr(metrics, "succeeded_file_count", 0)) if metrics else 0,
            "failed_file_count": int(getattr(metrics, "failed_file_count", 0)) if metrics else 0,
            "full_file_count": int(getattr(metrics, "full_file_count", 0)) if metrics else 0,
        },
        "indexed": _indexed_counts(root, path),
        "terminal_evidence": _terminal_evidence(root, path),
        "attempt": attempt,
        "proof_attempt_count": len(proof_attempts),
        "retry_state": retry_state,
        "attempt_present": attempt is not None,
    }


def _run_case(
    *,
    root: Path,
    source_root: Path,
    cursor: CursorStore,
    case_id: str,
    path: Path,
    parser_fingerprint: str,
    attempts_before: int,
    bypass_frontier_gate: bool = False,
) -> dict[str, Any]:
    polylogue = SimpleNamespace(archive_root=root, backend=SimpleNamespace(db_path=root / "index.db"))
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="codex", root=source_root),), cursor=cursor)
    try:
        record = cursor.get_record(path)
        fingerprint_changed_before_catch_up = (
            record is not None and bool(record.excluded) and record.parser_fingerprint != parser_fingerprint
        )
        metrics_holder: list[object] = []
        original_ingest = watcher._ingest_files

        async def capture_ingest(*args: Any, **kwargs: Any) -> object:
            metrics = await original_ingest(*args, **kwargs)
            metrics_holder.append(metrics)
            return metrics

        with patch("polylogue.sources.live.watcher._PARSER_FINGERPRINT", parser_fingerprint):
            frontier_patch = (
                patch("polylogue.readiness.capability.raw_frontier_source_selection_block_reason", lambda _root: None)
                if bypass_frontier_gate
                else nullcontext()
            )
            with frontier_patch, patch.object(watcher, "_ingest_files", capture_ingest):
                asyncio.run(watcher._catch_up([source_root]))
    finally:
        watcher.stop()
    return _case_summary(
        case_id=case_id,
        path=path,
        cursor=cursor,
        fingerprint_changed_before_catch_up=fingerprint_changed_before_catch_up,
        metrics=metrics_holder[-1] if metrics_holder else None,
        root=root,
        attempts_before=attempts_before,
    )


def run_excluded_cursor_live_proof(root: Path, receipt_path: Path) -> dict[str, Any]:
    """Run the real cursor/fingerprint route and write a self-hashed receipt."""
    case_roots = {case_id: root / case_id for case_id in ("indexed", "still-excluded", "typed-terminal")}

    def prepare_case(case_id: str, native_id: str, texts: tuple[str, ...]) -> tuple[Path, Path, CursorStore, int]:
        case_root = case_roots[case_id]
        source_root = case_root / "wire" / "excluded-cursor-proof"
        path = _write_jsonl(source_root / f"{case_id}.jsonl", _codex_records(native_id, texts))
        initialize_active_archive_root(case_root)
        cursor = CursorStore(case_root / "ops.db")
        return case_root, source_root, cursor, len(_attempts_for_path(case_root, path))

    indexed_root, indexed_source_root, indexed_cursor, indexed_attempts_before = prepare_case(
        "indexed", "excluded-proof-indexed", ("revived", "indexed")
    )
    indexed_path = indexed_source_root / "indexed.jsonl"
    _seed_byte_authority(indexed_root, indexed_path, native_id="excluded-proof-indexed")
    indexed_before = _indexed_counts(indexed_root, indexed_path)
    if indexed_before["indexed_sessions"] != 0:
        raise AssertionError(f"indexed case was not empty before catch-up: {indexed_before}")
    _seed_excluded(indexed_cursor, indexed_path, parser_fingerprint=OLD_PARSER_FINGERPRINT)
    indexed = _run_case(
        root=indexed_root,
        source_root=indexed_source_root,
        cursor=indexed_cursor,
        case_id="indexed",
        path=indexed_path,
        parser_fingerprint=NEW_PARSER_FINGERPRINT,
        attempts_before=indexed_attempts_before,
    )
    indexed["indexed_before"] = indexed_before

    unchanged_root, unchanged_source_root, unchanged_cursor, unchanged_attempts_before = prepare_case(
        "still-excluded", "excluded-proof-still-excluded", ("unchanged", "poison")
    )
    unchanged_path = unchanged_source_root / "still-excluded.jsonl"
    _seed_excluded(unchanged_cursor, unchanged_path, parser_fingerprint=NEW_PARSER_FINGERPRINT)
    still_excluded = _run_case(
        root=unchanged_root,
        source_root=unchanged_source_root,
        cursor=unchanged_cursor,
        case_id="still-excluded",
        path=unchanged_path,
        parser_fingerprint=NEW_PARSER_FINGERPRINT,
        attempts_before=unchanged_attempts_before,
    )

    terminal_root = case_roots["typed-terminal"]
    terminal_source_root = terminal_root / "wire" / "excluded-cursor-proof"
    terminal_path = terminal_source_root / "typed-terminal.jsonl"
    _write_jsonl(
        terminal_path,
        _codex_records("excluded-proof-terminal", ("valid prefix", "terminal corruption")),
    )
    with terminal_path.open("ab") as handle:
        handle.write(b'{"type":"response_item","payload":{"type":"message","content":[')
    initialize_active_archive_root(terminal_root)
    terminal_cursor = CursorStore(terminal_root / "ops.db")
    terminal_attempts_before = len(_attempts_for_path(terminal_root, terminal_path))
    _seed_excluded(terminal_cursor, terminal_path, parser_fingerprint=OLD_PARSER_FINGERPRINT)
    typed_terminal = _run_case(
        root=terminal_root,
        source_root=terminal_source_root,
        cursor=terminal_cursor,
        case_id="typed-terminal",
        path=terminal_path,
        parser_fingerprint=NEW_PARSER_FINGERPRINT,
        attempts_before=terminal_attempts_before,
        bypass_frontier_gate=True,
    )

    cases = [indexed, still_excluded, typed_terminal]
    indexed_attempt = indexed["attempt"]
    terminal_evidence = typed_terminal["terminal_evidence"]
    outcomes = {
        "indexed": indexed["indexed_before"]["indexed_sessions"] == 0
        and indexed["indexed"]["indexed_sessions"] == 1
        and indexed["retry_state"]["excluded"] is False
        and indexed_attempt is not None
        and indexed_attempt["outcome_code"] == "success",
        "still_excluded": still_excluded["retry_state"]["excluded"] is True
        and still_excluded["attempt_present"] is False
        and still_excluded["retry_state"]["retry_due"] is False,
        "typed_terminal": terminal_evidence is not None
        and terminal_evidence["artifact_kind"] == "terminal_corrupt_input"
        and terminal_evidence["support_status"] == "decode_failed"
        and typed_terminal["retry_state"]["excluded"] is False
        and typed_terminal["retry_state"]["retry_due"] is False,
    }
    if not all(outcomes.values()):
        raise AssertionError(f"excluded-cursor proof outcomes failed: {outcomes}")

    body: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "fixture_version": FIXTURE_VERSION,
        "execution": {
            "mode": "candidate_fixture",
            "live_census": "not_run",
            "live_residual": "Historical excluded population and current live file states were not accessed.",
            "terminal_frontier_residual": "The typed-terminal candidate has no accepted byte head, so its readiness gate was injected for this case only.",
            "residual_successor": "polylogue-excluded-cursor-live-proof",
        },
        "production_route": {
            "cursor_gate": "LiveWatcher._needs_work",
            "transition": "CursorStore.revive_replaced_exclusion",
            "catch_up": (
                "LiveWatcher._catch_up -> _scan_catch_up_candidates -> _catch_up_candidates -> "
                "_plan_catch_up -> coordinated chunk ingest"
            ),
            "ingest": "LiveWatcher._ingest_files -> LiveBatchProcessor.ingest_files",
            "terminal_evidence": "source.raw_artifacts",
            "retry_state": "ops.ingest_cursor and ops.ingest_attempts",
        },
        "outcomes": outcomes,
        "cases": cases,
        "fairness": {
            "planner": "_interleave_by_source",
            "property": (
                "browser-capture drains first; among non-browser-capture families, one candidate from each "
                "present family reaches the first round"
            ),
        },
        "anti_vacuity": {
            "indexed_authority": "byte_proven_source_raw_and_revision_head",
            "indexed_session_count_before": indexed["indexed_before"]["indexed_sessions"],
            "indexed_session_count": indexed["indexed"]["indexed_sessions"],
            "typed_terminal_artifact": terminal_evidence["artifact_kind"] if terminal_evidence else None,
            "unchanged_excluded_attempt_present": still_excluded["attempt_present"],
        },
    }
    digest = sha256(_canonical_json(body)).hexdigest()
    receipt = {**body, "receipt_sha256": digest}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary_path.write_bytes(_canonical_json(receipt))
    os.replace(temporary_path, receipt_path)
    return receipt


def verify_receipt(receipt_path: Path) -> dict[str, Any]:
    """Load and verify the immutable self-hash carried by a proof receipt."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise AssertionError("proof receipt must be a JSON object")
    recorded = receipt.pop("receipt_sha256", None)
    if not isinstance(recorded, str):
        raise AssertionError("proof receipt has no receipt_sha256")
    actual = sha256(_canonical_json(receipt)).hexdigest()
    if recorded != actual:
        raise AssertionError(f"proof receipt hash mismatch: recorded={recorded}, actual={actual}")
    receipt["receipt_sha256"] = recorded
    return receipt


__all__ = ["RECEIPT_SCHEMA", "run_excluded_cursor_live_proof", "verify_receipt"]
