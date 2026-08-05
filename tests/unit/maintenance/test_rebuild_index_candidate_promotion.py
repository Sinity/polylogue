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
from polylogue.core.enums import Provider
from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


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
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=_codex_session(native_id, text),
            source_path=f"stamp-regression/{native_id}.jsonl",
            acquired_at_ms=1,
        )


def _active_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    with sqlite3.connect(root / "index.db") as conn:
        return (
            tuple(conn.execute("SELECT session_id, content_hash FROM sessions ORDER BY session_id")),
            tuple(conn.execute("SELECT message_id, text FROM blocks ORDER BY block_id")),
        )


def test_stamp_corruption_blocks_real_candidate_promotion_without_touching_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production rebuild route rejects an unstamped candidate before swap."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _seed_raw(root, "active-session", "active generation remains exact")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    initial = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
    assert initial.status == "replayed"

    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    snapshot_before = _active_snapshot(root)

    _seed_raw(root, "candidate-session", "candidate must never become active")
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
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))

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

    initial = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
    assert initial.status == "replayed"

    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    with sqlite3.connect(root / "embeddings.db") as conn:
        conn.execute(
            """
            INSERT INTO message_embedding_refs(message_id, session_id, origin, embedding_input_hash)
            VALUES ('codex-session:active-session:no-such-message', 'codex-session:active-session', 'codex-session', ?)
            """,
            (b"o" * 32,),
        )
        conn.commit()
    _seed_raw(root, "candidate-session", "candidate must never become active")

    with pytest.raises(RuntimeError, match=r"embeddings-refs-liveness.*waived by polylogue-feu0"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))

    assert store.active_pointer.resolve(strict=True) == active_before


def test_cross_tier_user_reference_blocks_full_rebuild_candidate_promotion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    _seed_raw(root, "active-session", "active generation remains exact")
    initial = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
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

    with pytest.raises(RuntimeError, match=r"user-tier-refs \[error\]"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))

    assert store.active_pointer.resolve(strict=True) == active_before


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
    initial = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
    assert initial.status == "replayed"

    store = IndexGenerationStore.for_archive_root(root)
    active_before = store.active_pointer.resolve(strict=True)
    _seed_raw(root, "candidate-session", "candidate must never become active")

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
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))

    assert store.active_pointer.resolve(strict=True) == active_before
