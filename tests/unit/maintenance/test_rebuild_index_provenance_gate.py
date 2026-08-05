"""Real-route tests for the schema-inference rebuild hard gate."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

import polylogue.maintenance.rebuild_index as rebuild_index_module
import polylogue.maintenance.sharded_rebuild as sharded_rebuild_module
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.maintenance.sharded_rebuild import shard_raw_ids
from polylogue.sources.revision_backfill import RebuildDeadlineExceededError
from polylogue.storage.archive_identity import OwnedArchiveLocation
from polylogue.storage.index_generation import (
    IndexGeneration,
    IndexGenerationStore,
    IndexRebuildTransaction,
    rebuild_source_evidence_snapshot,
)
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
                revision=RawRevisionEnvelope(
                    logical_source_key=f"codex-session:gate-session-{index}",
                    kind=RawRevisionKind.FULL,
                    source_revision=f"seed-revision-{index}",
                    acquisition_generation=0,
                    authority=RawRevisionAuthority.ASSERTED,
                ),
            )


def _active_bytes(root: Path) -> bytes:
    return root.joinpath("index.db").read_bytes()


def _generation_ids(root: Path) -> set[str]:
    generations = root / ".index-generations"
    return {path.name for path in generations.glob("gen-*")} if generations.exists() else set()


def _same_shard_raw_ids(root: Path) -> tuple[str, ...]:
    with sqlite3.connect(root / "source.db") as conn:
        raw_ids = [str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id")]
    return tuple(next(bucket for bucket in shard_raw_ids(root, raw_ids, 2) if len(bucket) >= 2))


def _interpose_receipt_mutation(monkeypatch: pytest.MonkeyPatch, receipt_path: Path, *, expire: bool) -> None:
    original_acquire = OwnedArchiveLocation.acquire

    def acquire_after_mutation(
        cls: type[OwnedArchiveLocation], /, location: object, **kwargs: object
    ) -> OwnedArchiveLocation:
        owned = original_acquire(location, **kwargs)  # type: ignore[arg-type]
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if expire:
            payload["generated_at"] = "2000-01-01T00:00:00Z"
        else:
            payload["source_snapshot"] = "post-preflight-source-drift"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        return owned

    monkeypatch.setattr(OwnedArchiveLocation, "acquire", classmethod(acquire_after_mutation))


def test_missing_receipt_fails_before_lease_and_candidate_mutation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    active_before = _active_bytes(root)
    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
    assert _active_bytes(root) == active_before
    assert not (root / ".index-generations").exists()


@pytest.mark.parametrize("expire", [False, True], ids=["external-drift", "receipt-expiry"])
def test_offline_post_preflight_receipt_change_fails_before_lease_or_candidate_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expire: bool
) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    _interpose_receipt_mutation(monkeypatch, receipt_path, expire=expire)
    lease_path = root / ".index-rebuild.lock"
    lease_before = lease_path.read_bytes() if lease_path.exists() else None

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    assert (lease_path.read_bytes() if lease_path.exists() else None) == lease_before
    assert not (root / ".index-generations").exists()
    assert not (root / ".index-rebuild-transactions").exists()


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


@pytest.mark.parametrize("expire", [False, True], ids=["external-drift", "receipt-expiry"])
def test_deadline_checkpoint_revalidates_receipt_before_persisting_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expire: bool
) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    transaction = store.create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root), operation_id="deadline-checkpoint"
    )
    transaction = store.checkpoint_transaction(transaction, status="running", derived_stores_cleared=True)
    transaction_path = root / ".index-rebuild-transactions" / "deadline-checkpoint.json"
    before = transaction_path.read_bytes()

    async def fail_after_receipt_drift(*args: object, **kwargs: object) -> dict[str, object]:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if expire:
            payload["generated_at"] = "2000-01-01T00:00:00Z"
        else:
            payload["source_snapshot"] = "post-preflight-source-drift"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        raise RebuildDeadlineExceededError("synthetic deadline")

    monkeypatch.setattr("polylogue.maintenance.replay.rebuild_index_from_source", fail_after_receipt_drift)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                operation_id="deadline-checkpoint",
                schema_inference_receipt_path=receipt_path,
                raw_batch_size=1,
            )
        )

    assert transaction_path.read_bytes() == before


def test_sharded_replay_revalidates_expired_receipt_before_candidate_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shard replay must fail before its first index write and clean scratch generations.

    Anti-vacuity: the real sharded rebuild route calls the real replay engine. The
    wrapper expires the receipt immediately before that engine starts. Removing
    the shard replay provenance guard lets the engine write the shard candidate,
    which makes ``replayed_rows`` nonzero before the route fails later.
    """
    root = tmp_path / "archive"
    _seed(root, count=4)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    raw_ids = _same_shard_raw_ids(root)
    replayed_rows: list[int] = []

    from polylogue.maintenance import replay as replay_module

    original_replay = cast(Callable[..., Any], replay_module.rebuild_index_from_source)

    async def expire_before_replay(*args: Any, **kwargs: Any) -> dict[str, object]:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["generated_at"] = "2000-01-01T00:00:00Z"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            return cast(dict[str, object], await original_replay(*args, **kwargs))
        finally:
            config = cast(Config, args[0])
            index_path = Path(config.db_path)
            with sqlite3.connect(index_path) as conn:
                replayed_rows.append(int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]))

    monkeypatch.setattr(replay_module, "rebuild_index_from_source", expire_before_replay)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                raw_ids=raw_ids,
                promote=False,
                shard_count=2,
                schema_inference_receipt_path=receipt_path,
            )
        )

    assert replayed_rows == [0]
    assert _generation_ids(root) == set()


def test_sharded_target_creation_cleans_candidate_when_receipt_expires_after_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipt failure immediately after target creation cannot strand the target."""
    root = tmp_path / "archive"
    _seed(root, count=4)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    raw_ids = _same_shard_raw_ids(root)
    original_create = IndexGenerationStore.create
    create_count = 0

    def expire_after_target_create(
        store: IndexGenerationStore, *, owner_id: str | None = None, source_snapshot: str
    ) -> IndexGeneration:
        nonlocal create_count
        generation = original_create(store, owner_id=owner_id, source_snapshot=source_snapshot)
        create_count += 1
        if create_count == 1:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            payload["generated_at"] = "2000-01-01T00:00:00Z"
            receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        return generation

    monkeypatch.setattr(IndexGenerationStore, "create", expire_after_target_create)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                raw_ids=raw_ids,
                promote=False,
                shard_count=2,
                schema_inference_receipt_path=receipt_path,
            )
        )

    assert create_count == 1
    assert _generation_ids(root) == set()


def test_full_source_transaction_creation_cleans_candidate_after_post_create_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full-source setup cleans both durable records after post-create validation fails.

    Anti-vacuity: the production full-source route creates both the inactive
    candidate and its transaction before the receipt is expired. Removing the
    post-create validation or either cleanup leaves the captured generation or
    transaction record behind.
    """
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    original_create_transaction = IndexGenerationStore.create_transaction
    created_records: list[tuple[str, str]] = []

    def expire_after_transaction(
        store: IndexGenerationStore,
        *,
        source_snapshot: str,
        operation_id: str | None = None,
        pass_byte_budget: int | None = None,
        pass_deadline_ms: int | None = None,
    ) -> IndexRebuildTransaction:
        transaction = original_create_transaction(
            store,
            source_snapshot=source_snapshot,
            operation_id=operation_id,
            pass_byte_budget=pass_byte_budget,
            pass_deadline_ms=pass_deadline_ms,
        )
        assert store.load(transaction.generation_id).state == "inactive"
        assert store.load_transaction(transaction.operation_id).operation_id == transaction.operation_id
        created_records.append((transaction.generation_id, transaction.operation_id))
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["generated_at"] = "2000-01-01T00:00:00Z"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        return transaction

    monkeypatch.setattr(IndexGenerationStore, "create_transaction", expire_after_transaction)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    assert len(created_records) == 1
    assert _generation_ids(root) == set()
    assert not list((root / ".index-rebuild-transactions").glob("*.json"))


def test_full_source_snapshot_mismatch_after_transaction_creation_cleans_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh transaction cannot be retained when its source snapshot drifts."""
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    original_create = rebuild_index_module._create_rebuild_transaction_after_receipt_validation

    def create_then_drift(*args: Any, **kwargs: Any) -> IndexRebuildTransaction:
        transaction = original_create(*args, **kwargs)
        with sqlite3.connect(root / "source.db") as conn:
            conn.execute("UPDATE raw_sessions SET source_path = source_path || '.drifted'")
        write_valid_rebuild_receipt(root, receipt_path)
        return transaction

    monkeypatch.setattr(rebuild_index_module, "_create_rebuild_transaction_after_receipt_validation", create_then_drift)

    with pytest.raises(RuntimeError, match="source evidence changed since this rebuild was planned"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    assert _generation_ids(root) == set()
    assert not list((root / ".index-rebuild-transactions").glob("*.json"))


def test_full_source_snapshot_mismatch_after_replay_cleans_transaction_and_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source drift detected after replay uses the fresh-transaction cleanup path."""
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")

    def planner_boundary_then_drift(*, processed_before: int | None, processed_after: int) -> bool:
        with sqlite3.connect(root / "source.db") as conn:
            conn.execute("UPDATE raw_sessions SET source_path = source_path || '.drifted'")
        write_valid_rebuild_receipt(root, receipt_path)
        return False

    monkeypatch.setattr(
        rebuild_index_module,
        "_should_refresh_generation_planner_statistics",
        planner_boundary_then_drift,
    )

    with pytest.raises(RuntimeError, match="source evidence changed during this bounded rebuild pass"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    assert _generation_ids(root) == set()
    assert not list((root / ".index-rebuild-transactions").glob("*.json"))


def test_full_source_provenance_cleanup_reports_failed_discards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleanup actuator failure is diagnostic while the mismatch stays primary."""
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    original_create = rebuild_index_module._create_rebuild_transaction_after_receipt_validation

    def create_then_drift(*args: Any, **kwargs: Any) -> IndexRebuildTransaction:
        transaction = original_create(*args, **kwargs)
        with sqlite3.connect(root / "source.db") as conn:
            conn.execute("UPDATE raw_sessions SET source_path = source_path || '.drifted'")
        write_valid_rebuild_receipt(root, receipt_path)
        return transaction

    monkeypatch.setattr(rebuild_index_module, "_create_rebuild_transaction_after_receipt_validation", create_then_drift)
    monkeypatch.setattr(IndexGenerationStore, "discard_if_inactive", lambda *args, **kwargs: False)
    monkeypatch.setattr(IndexGenerationStore, "discard_transaction", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="source evidence changed since this rebuild was planned") as raised:
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

    notes = "\n".join(raised.value.__notes__ or ())
    assert "rebuild transaction cleanup also failed" in notes
    assert "was not discarded" in notes


def test_sharded_graph_drift_blocks_derived_stages_and_cleans_resumable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipt drift after graph resolution cannot reach derived-state writers."""
    root = tmp_path / "archive"
    _seed(root, count=4)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    original_graph_resolution = sharded_rebuild_module.resolve_cross_shard_session_graph
    original_create = IndexGenerationStore.create
    original_discard = IndexGenerationStore.discard_if_inactive
    created_generation_ids: list[str] = []
    discard_calls: list[str] = []
    repopulate_calls: list[Path] = []
    insight_calls: list[object] = []
    source_write_lock = threading.Lock()

    from polylogue.sources import revision_backfill as revision_backfill_module

    original_backfill = revision_backfill_module.backfill_historical_revision_evidence

    def serialize_backfill(*args: Any, **kwargs: Any) -> object:
        # Each replay keeps its archive source transaction open while it
        # records parser census and revision evidence on source.db. Serialize
        # the existing real route here so the regression remains about the
        # provenance boundary instead of SQLite's unrelated concurrent-writer
        # behavior.
        with source_write_lock:
            return original_backfill(*args, **kwargs)

    monkeypatch.setattr(revision_backfill_module, "backfill_historical_revision_evidence", serialize_backfill)

    def drift_after_graph_resolution(*args: Any, **kwargs: Any) -> float:
        graph_elapsed_s = original_graph_resolution(*args, **kwargs)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["generated_at"] = "2000-01-01T00:00:00Z"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        return graph_elapsed_s

    def record_create(
        store: IndexGenerationStore, *, owner_id: str | None = None, source_snapshot: str
    ) -> IndexGeneration:
        generation = original_create(store, owner_id=owner_id, source_snapshot=source_snapshot)
        created_generation_ids.append(generation.generation_id)
        return generation

    def record_discard(store: IndexGenerationStore, generation: IndexGeneration) -> bool:
        discard_calls.append(generation.generation_id)
        return original_discard(store, generation)

    def unexpected_repopulate(index_path: Path) -> dict[str, float]:
        repopulate_calls.append(index_path)
        raise AssertionError("stale provenance reached bulk-derived repopulation")

    def unexpected_insight_repair(*args: Any, **kwargs: Any) -> object:
        insight_calls.append((args, kwargs))
        raise AssertionError("stale provenance reached session insight repair")

    monkeypatch.setattr(sharded_rebuild_module, "resolve_cross_shard_session_graph", drift_after_graph_resolution)
    monkeypatch.setattr(IndexGenerationStore, "create", record_create)
    monkeypatch.setattr(IndexGenerationStore, "discard_if_inactive", record_discard)
    monkeypatch.setattr(rebuild_index_module, "_repopulate_bulk_build_derived_state", unexpected_repopulate)
    monkeypatch.setattr("polylogue.storage.repair.repair_session_insights", unexpected_insight_repair)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                shard_count=2,
            )
        )

    assert repopulate_calls == []
    assert insight_calls == []
    assert not list((root / ".index-rebuild-transactions").glob("*.json"))
    shard_ids = set(created_generation_ids[1:])
    assert shard_ids
    assert shard_ids <= set(discard_calls)
    assert len([generation_id for generation_id in discard_calls if generation_id in shard_ids]) == len(shard_ids)
    assert _generation_ids(root) == set()
