"""Real raw-replay recovery proof for ``polylogue-b5l.1``.

The route under test is the production offline rebuild orchestrator.  The
fixture interrupts immediately after the first replay page is durably
checkpointed, then resumes the same candidate and compares its promoted
semantic output with an independently clean rebuild.  It deliberately checks
rows, topology, public FTS reads, and insight materialization rather than a
test-only cursor counter.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


class InjectedInterruptError(RuntimeError):
    """Test-only process-death analogue raised after a durable checkpoint."""


def _payload(native_id: str, text: str, *, parent_native_id: str | None = None) -> bytes:
    cwd = "/realm/project/resume-fixture"
    rows: list[dict[str, object]] = [
        {
            "type": "session_meta",
            "payload": {"id": native_id, "timestamp": "2026-08-04T12:00:00Z", "cwd": cwd},
        }
    ]
    if parent_native_id is not None:
        rows.append(
            {
                "type": "session_meta",
                "payload": {"id": parent_native_id, "timestamp": "2026-08-04T11:00:00Z", "cwd": cwd},
            }
        )
    rows.append(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{native_id}-m0",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        },
    )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _seed(root: Path) -> None:
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index, native_id, parent_native_id in (
            (0, "resume-parent", None),
            (1, "resume-child", "resume-parent"),
            (2, "resume-standalone", None),
        ):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_payload(native_id, f"resume-token-{index}", parent_native_id=parent_native_id),
                source_path=f"resume/{index}.jsonl",
                acquired_at_ms=index + 1,
            )


def _semantic_snapshot(root: Path) -> tuple[object, ...]:
    with sqlite3.connect(root / "index.db") as conn:
        rows = tuple(
            tuple(conn.execute(query))
            for query in (
                "SELECT session_id, message_count, content_hash FROM sessions ORDER BY session_id",
                "SELECT message_id, session_id, role, position FROM messages ORDER BY message_id",
                "SELECT block_id, message_id, block_type, text FROM blocks ORDER BY block_id",
                "SELECT src_session_id, dst_origin, dst_native_id, link_type, resolved_dst_session_id "
                "FROM session_links ORDER BY src_session_id, dst_origin, dst_native_id, link_type",
                "SELECT session_id, message_count, materializer_version FROM session_profiles ORDER BY session_id",
                "SELECT insight_type, session_id, materializer_version "
                "FROM insight_materialization ORDER BY insight_type, session_id",
            )
        )
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        fts = tuple(
            (f"resume-token-{index}", tuple(archive.search_blocks(f"resume-token-{index}"))) for index in range(3)
        )
    return (*rows, fts)


def test_committed_page_interrupt_resumes_only_suffix_and_matches_clean_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "resumed"
    clean_root = tmp_path / "clean"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    _seed(root)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")

    original_checkpoint = IndexGenerationStore.checkpoint_transaction
    interrupted = False

    def interrupt_after_committed_page(self: IndexGenerationStore, transaction: object, **kwargs: object) -> object:
        nonlocal interrupted
        checkpointed = original_checkpoint(self, transaction, **kwargs)  # type: ignore[arg-type]
        if not interrupted and kwargs.get("processed_raw_count") == 1:
            interrupted = True
            raise InjectedInterruptError("simulated process interruption after committed page")
        return checkpointed

    with monkeypatch.context() as scoped:
        scoped.setattr(IndexGenerationStore, "checkpoint_transaction", interrupt_after_committed_page)
        with pytest.raises(InjectedInterruptError, match="after committed page"):
            rebuild_index_from_source_sync(
                RebuildIndexRequest(archive_root=root, raw_batch_size=1, schema_inference_receipt_path=receipt_path)
            )

    store = IndexGenerationStore.for_archive_root(root)
    operation_id = next(path.stem for path in store.transactions_root.glob("*.json"))
    transaction = store.load_transaction(operation_id)
    assert transaction.processed_raw_count == 1
    assert transaction.cursor is not None
    with sqlite3.connect(Path(store.load(transaction.generation_id).index_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    # Each resumed pass can schedule only rows beyond the committed source
    # cursor.  Recording the real replay call catches a cursor that merely
    # reports progress while restarting from page one.
    import polylogue.maintenance.replay as replay_module

    replayed_raw_pages: list[tuple[str, ...]] = []
    real_replay = replay_module.rebuild_index_from_source

    async def recording_replay(*args: object, **kwargs: object) -> dict[str, object]:
        replayed_raw_pages.append(tuple(cast(list[str], kwargs["raw_ids"])))
        return await real_replay(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(replay_module, "rebuild_index_from_source", recording_replay)
    while True:
        receipt = rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                operation_id=operation_id,
                raw_batch_size=1,
                schema_inference_receipt_path=receipt_path,
            )
        )
        if receipt.status == "replayed":
            break
        assert receipt.status == "paused"

    assert [len(page) for page in replayed_raw_pages] == [1, 1]
    assert len({raw_id for page in replayed_raw_pages for raw_id in page}) == 2
    assert receipt.operation["cursor"] is not None
    assert receipt.operation["heartbeat"]["at_ms"] is not None  # type: ignore[index]
    assert receipt.operation["recovery_state"] == "promoted"

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(clean_root))
    _seed(clean_root)
    clean_receipt_path = write_valid_rebuild_receipt(
        clean_root, tmp_path / "clean-receipt" / "schema-inference-gate-receipt.json"
    )
    clean = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=clean_root, raw_batch_size=1, schema_inference_receipt_path=clean_receipt_path)
    )
    assert clean.status == "paused"
    assert clean.transaction is not None
    clean_operation = clean.transaction["operation_id"]
    assert isinstance(clean_operation, str)
    while clean.status != "replayed":
        clean = rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=clean_root,
                operation_id=clean_operation,
                raw_batch_size=1,
                schema_inference_receipt_path=clean_receipt_path,
            )
        )

    resumed_snapshot = _semantic_snapshot(root)
    assert resumed_snapshot[3], "the real replay fixture must exercise a lineage link"
    assert resumed_snapshot == _semantic_snapshot(clean_root)
