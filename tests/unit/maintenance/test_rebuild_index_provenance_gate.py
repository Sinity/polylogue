"""Real-route tests for the schema-inference rebuild hard gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.rebuild_receipt import write_valid_rebuild_receipt


def _payload(native_id: str, text: str) -> bytes:
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


def _seed(root: Path, count: int = 2) -> None:
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(count):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_payload(f"gate-session-{index}", f"gate text {index}"),
                source_path=f"current/{index}.jsonl",
                acquired_at_ms=index + 1,
            )


def _active_bytes(root: Path) -> bytes:
    return root.joinpath("index.db").read_bytes()


def test_missing_receipt_fails_before_lease_and_candidate_mutation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    active_before = _active_bytes(root)
    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
    assert _active_bytes(root) == active_before
    assert not (root / ".index-generations").exists()


def test_forged_top_level_pass_cannot_bypass_a_failing_subgate(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["verdict"] = "PASS"
    payload["query_results"]["zero-surviving-quarantine"]["passed"] = False
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    active_before = _active_bytes(root)

    with pytest.raises(RuntimeError, match="zero-surviving-quarantine is not PASS"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )
    assert _active_bytes(root) == active_before
    assert not (root / ".index-generations").exists()


def test_valid_receipt_allows_real_candidate_acceptance_and_promotion(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")

    result = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path, promote=True)
    )

    assert result.status == "replayed"
    assert result.transaction is not None
    assert result.transaction["status"] == "promoted"
    assert result.consumed_evidence["receipt_path"] == str(receipt_path)
    assert result.consumed_evidence["source_snapshot"]
    assert result.consumed_evidence["external_ground_truth_digest"]
    assert IndexGenerationStore.for_archive_root(root).active_pointer.resolve().exists()


def test_resume_revalidates_external_mapping_before_more_replay(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    first = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            raw_batch_size=1,
            promote=True,
        )
    )
    assert first.status == "paused"
    assert first.transaction is not None
    operation_id = str(first.transaction["operation_id"])
    processed_value = first.transaction["processed_raw_count"]
    assert isinstance(processed_value, int)
    processed_before = processed_value
    active_before = _active_bytes(root)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    origin = receipt["ground_truth_inputs"]["origins"]["codex-session"]
    external_path = Path(origin["declared_roots"][0]) / origin["external_inventory"][0]["relative_path"]
    external_path.write_bytes(b"changed external corpus")

    with pytest.raises(RuntimeError, match="external ground-truth corpus changed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                operation_id=operation_id,
                raw_batch_size=1,
                promote=True,
            )
        )
    transaction = IndexGenerationStore.for_archive_root(root).load_transaction(operation_id)
    assert transaction.processed_raw_count == processed_before
    assert _active_bytes(root) == active_before


def test_mapping_mutation_is_rejected_without_relying_on_aggregate_counts(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    mapping = payload["ground_truth_inputs"]["origins"]["codex-session"]["raw_external_mapping"]
    mapping[0]["source_path"] = "stale/source/path.jsonl"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="raw external mapping or inventory"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )
    assert not (root / ".index-generations").exists()
