"""Real candidate-promotion proof for semantic stamp acceptance."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import polylogue.maintenance.archive_verification as archive_verification
import polylogue.storage.sqlite.archive_tiers.revision_governance as revision_governance
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.rebuild_index import RebuildIndexReceipt, RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.rebuild_receipt import write_current_rebuild_receipt


def _codex_session(native_id: str, text: str) -> bytes:
    rows = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-08-05T10:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{native_id}-m0",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        },
    ]
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _seed_raw(root: Path, native_id: str, text: str) -> None:
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session(native_id, text),
            source_path=f"stamp-regression/{native_id}.jsonl",
            acquired_at_ms=1,
            native_id=native_id,
        )
        archive.bind_raw_revision(
            raw_id,
            RawRevisionEnvelope(
                logical_source_key=f"codex-session:{native_id}",
                kind=RawRevisionKind.FULL,
                source_revision=f"stamp-regression:{native_id}",
                acquisition_generation=0,
                baseline_raw_id=raw_id,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )


def _active_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    with sqlite3.connect(root / "index.db") as conn:
        return (
            tuple(conn.execute("SELECT session_id, content_hash FROM sessions ORDER BY session_id")),
            tuple(conn.execute("SELECT message_id, text FROM blocks ORDER BY block_id")),
        )


def _rebuild_with_fresh_receipt(root: Path, receipt_path: Path) -> RebuildIndexReceipt:
    receipt = write_current_rebuild_receipt(root, receipt_path)
    return rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, promote=True, schema_inference_receipt_path=receipt)
    )


def test_stamp_corruption_blocks_real_candidate_promotion_without_touching_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production rebuild route rejects an unstamped candidate before swap."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _seed_raw(root, "active-session", "active generation remains exact")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    initial = _rebuild_with_fresh_receipt(root, tmp_path / "initial-receipt.json")
    assert initial.status == "replayed"

    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    snapshot_before = _active_snapshot(root)

    _seed_raw(root, "candidate-session", "candidate must never become active")
    candidate_receipt = write_current_rebuild_receipt(root, tmp_path / "candidate-receipt.json")
    original_writer = cast(Callable[..., str], revision_governance.__dict__["write_parsed_session_to_archive"])
    corruption_calls = 0

    def bypass_stamps(conn: sqlite3.Connection, *args: object, **kwargs: object) -> str:
        nonlocal corruption_calls
        result = original_writer(conn, *args, **kwargs)
        conn.execute("UPDATE sessions SET parser_fingerprint = NULL, lowering_fingerprint = NULL")
        corruption_calls += 1
        return result

    monkeypatch.setattr(revision_governance, "write_parsed_session_to_archive", bypass_stamps)

    with pytest.raises(RuntimeError, match="reindex acceptance gate failed.*session-fingerprint-stamps"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, promote=True, schema_inference_receipt_path=candidate_receipt)
        )

    assert corruption_calls > 0
    assert store.active_pointer.resolve(strict=True) == active_before
    assert _active_snapshot(root) == snapshot_before


def test_waived_embedding_orphan_blocks_full_rebuild_candidate_promotion(
    tmp_path: Path,
) -> None:
    """The full rebuild route must not treat the feu0 waiver as acceptance."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _seed_raw(root, "active-session", "active generation remains exact")

    initial = _rebuild_with_fresh_receipt(root, tmp_path / "initial-receipt.json")
    assert initial.status == "replayed"

    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    _seed_raw(root, "candidate-session", "candidate must never become active")
    candidate_receipt = write_current_rebuild_receipt(root, tmp_path / "candidate-receipt.json")
    with sqlite3.connect(root / "embeddings.db") as conn:
        conn.execute(
            """
            INSERT INTO message_embedding_refs(message_id, session_id, origin, embedding_input_hash)
            VALUES ('codex-session:active-session:no-such-message', 'codex-session:active-session', 'codex-session', ?)
            """,
            (b"o" * 32,),
        )
        conn.commit()

    with pytest.raises(RuntimeError, match=r"embeddings-refs-liveness.*waived by polylogue-feu0"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, promote=True, schema_inference_receipt_path=candidate_receipt)
        )

    assert store.active_pointer.resolve(strict=True) == active_before


def test_cross_tier_user_reference_blocks_full_rebuild_candidate_promotion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _seed_raw(root, "active-session", "active generation remains exact")
    initial = _rebuild_with_fresh_receipt(root, tmp_path / "initial-receipt.json")
    assert initial.status == "replayed"

    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute(
            """
            INSERT INTO assertions(assertion_id, target_ref, kind, body_text, created_at_ms, updated_at_ms)
            VALUES ('dangling-candidate-assertion', 'session:codex-session:no-such-session', 'note', 'orphaned', 1, 1)
            """
        )
        conn.commit()
    _seed_raw(root, "candidate-session", "candidate must never become active")
    candidate_receipt = write_current_rebuild_receipt(root, tmp_path / "candidate-receipt.json")

    with pytest.raises(RuntimeError, match=r"user-tier-refs \[error\]"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, promote=True, schema_inference_receipt_path=candidate_receipt)
        )

    assert store.active_pointer.resolve(strict=True) == active_before


@pytest.mark.parametrize(
    ("mutation", "failure_pattern"),
    (
        ("missing", "rebuild schema-inference preflight gate failed"),
        ("corrupt", "rebuild schema-inference preflight gate failed"),
        ("orphan", "reindex source preflight gate failed:.*blob-integrity"),
    ),
)
def test_rebuild_preflight_rejects_each_physical_blob_failure_before_candidate_creation(
    tmp_path: Path, mutation: str, failure_pattern: str
) -> None:
    """The source preflight catches physical debt before any new generation exists."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _seed_raw(root, "active-session", "active generation remains exact")
    initial = _rebuild_with_fresh_receipt(root, tmp_path / "initial-receipt.json")
    assert initial.status == "replayed"

    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    _seed_raw(root, "candidate-session", "candidate must never become active")
    candidate_receipt = write_current_rebuild_receipt(root, tmp_path / "candidate-receipt.json")
    with sqlite3.connect(root / "source.db") as conn:
        blob_hash = bytes(conn.execute("SELECT blob_hash FROM raw_sessions LIMIT 1").fetchone()[0]).hex()
    blob_store = BlobStore(root / "blob")
    if mutation == "missing":
        blob_store.blob_path(blob_hash).unlink()
    elif mutation == "corrupt":
        blob_store.blob_path(blob_hash).write_bytes(b"corrupt raw bytes")
    else:
        blob_store.write_from_bytes(b"orphan physical bytes")
    with pytest.raises(RuntimeError, match=failure_pattern):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, promote=True, schema_inference_receipt_path=candidate_receipt)
        )

    assert store.active_pointer.resolve(strict=True) == active_before
    assert len(list(store.generations_root.glob("gen-*/index.db"))) == 1


def test_rebuild_preflight_rejects_acquired_unreachable_attachment_before_candidate_creation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _seed_raw(root, "active-session", "active generation remains exact")
    initial = _rebuild_with_fresh_receipt(root, tmp_path / "initial-receipt.json")
    assert initial.status == "replayed"

    _seed_raw(root, "candidate-session", "candidate must never become active")
    candidate_receipt = write_current_rebuild_receipt(root, tmp_path / "candidate-receipt.json")
    blob_hash, size = BlobStore(root / "blob").write_from_bytes(b"unreachable attachment")
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO attachments(attachment_id, blob_hash, byte_count, acquisition_status, ref_count)
            VALUES ('unreachable-attachment', ?, ?, 'acquired', 0)
            """,
            (bytes.fromhex(blob_hash), size),
        )
        conn.commit()
    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    with pytest.raises(RuntimeError, match="reindex source preflight gate failed:.*attachment-acquisition-debt"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, promote=True, schema_inference_receipt_path=candidate_receipt)
        )

    assert store.active_pointer.resolve(strict=True) == active_before
    assert len(list(store.generations_root.glob("gen-*/index.db"))) == 1


@pytest.mark.parametrize(
    ("status", "check_name"),
    (
        (OutcomeStatus.WARNING, "fts-parity"),
        (OutcomeStatus.SKIP, "lineage-sanity"),
        (None, "session-fingerprint-stamps"),
    ),
)
def test_full_rebuild_promotion_rejects_non_ok_or_missing_required_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: OutcomeStatus | None,
    check_name: str,
) -> None:
    """The production promotion route requires every strict result to be OK."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _seed_raw(root, "active-session", "active generation remains exact")
    initial = _rebuild_with_fresh_receipt(root, tmp_path / "initial-receipt.json")
    assert initial.status == "replayed"

    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    _seed_raw(root, "candidate-session", "candidate must never become active")
    candidate_receipt = write_current_rebuild_receipt(root, tmp_path / "candidate-receipt.json")

    def mutated_verifier(*args: object, **kwargs: object) -> archive_verification.ArchiveVerificationReport:
        checks = cast(tuple[str, ...], kwargs["checks"])
        return archive_verification.ArchiveVerificationReport(
            checks=[
                archive_verification.ArchiveVerificationCheck(
                    name=name,
                    status=(status if name == check_name and status is not None else OutcomeStatus.OK),
                )
                for name in checks
                if name != check_name or status is not None
            ]
        )

    monkeypatch.setattr(archive_verification, "verify_archive", mutated_verifier)
    with pytest.raises(RuntimeError, match=f"reindex acceptance gate failed.*{check_name}"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, promote=True, schema_inference_receipt_path=candidate_receipt)
        )

    assert store.active_pointer.resolve(strict=True) == active_before
