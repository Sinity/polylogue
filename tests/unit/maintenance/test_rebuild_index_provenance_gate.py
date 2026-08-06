"""Real-route tests for the schema-inference rebuild hard gate."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

import polylogue.maintenance.rebuild_index as rebuild_index_module
import polylogue.maintenance.schema_inference_gate as schema_gate_module
import polylogue.maintenance.sharded_rebuild as sharded_rebuild_module
import polylogue.sources.origin_specs as origin_specs_module
import polylogue.storage.index_generation as index_generation_module
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.daemon import bulk_rebuild as bulk_rebuild_module
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.maintenance.sharded_rebuild import shard_raw_ids
from polylogue.sources.revision_backfill import RebuildDeadlineExceededError
from polylogue.storage.archive_identity import ArchiveLocation, ArchiveOwnershipError, OwnedArchiveLocation
from polylogue.storage.blob_store import BlobStore
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
    # The receipt is intentionally taken from a source tier whose authority
    # classification is already settled.  Replay must not manufacture a
    # baseline or rewrite asserted authority before the first checkpoint, so
    # any later mutation of these bindings is a real stale-pass condition.
    with sqlite3.connect(root / "source.db") as source:
        source.execute("UPDATE raw_sessions SET baseline_raw_id = raw_id, revision_authority = 'byte_proven'")
        source.commit()


def _active_bytes(root: Path) -> bytes:
    return root.joinpath("index.db").read_bytes()


def _generation_ids(root: Path) -> set[str]:
    generations = root / ".index-generations"
    return {path.name for path in generations.glob("gen-*")} if generations.exists() else set()


def _same_shard_raw_ids(root: Path) -> tuple[str, ...]:
    with sqlite3.connect(root / "source.db") as conn:
        raw_ids = [str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id")]
    return tuple(next(bucket for bucket in shard_raw_ids(root, raw_ids, 2) if len(bucket) >= 2))


def _raw_ids(root: Path) -> tuple[str, ...]:
    with sqlite3.connect(root / "source.db") as conn:
        return tuple(str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id"))


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


def test_receipt_reference_policy_fails_before_candidate_mutation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = root / "schema-inference-gate-receipt.json"
    write_valid_rebuild_receipt(root, receipt_path)
    active_before = _active_bytes(root)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path)
        )

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


def test_rebuild_context_reuses_refreshed_inventory_token_after_detector_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful metadata-only rehash updates the pass token for later checkpoints.

    Anti-vacuity: this runs the real offline rebuild entry point and validator.
    Touching the external corpus after the first context validation forces one
    successful full inventory refresh. Every later context validation must use
    the refreshed token instead of scanning that corpus again.
    """
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    origin = receipt["ground_truth_inputs"]["origins"]["codex-session"]
    external_path = Path(origin["declared_roots"][0]) / origin["external_inventory"][0]["relative_path"]

    full_inventory_calls = 0
    original_inventory = schema_gate_module._external_inventory

    def counted_inventory(roots: object) -> object:
        nonlocal full_inventory_calls
        full_inventory_calls += 1
        return original_inventory(roots)  # type: ignore[arg-type]

    monkeypatch.setattr(schema_gate_module, "_external_inventory", counted_inventory)
    result = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path, promote=False)
    )

    assert result.status == "replayed"
    consumed_evidence = result.consumed_evidence
    inventory_token = cast(dict[str, object], consumed_evidence["external_ground_truth_inventory_token"])
    provenance = rebuild_index_module.RebuildProvenanceContext(
        root=root,
        receipt_path=receipt_path,
        source_snapshot=str(consumed_evidence["source_snapshot"]),
        consumed_evidence=consumed_evidence,
        external_inventory_token=inventory_token,
    )
    stat = external_path.stat()
    os.utime(external_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    inventory_calls_before_refresh = full_inventory_calls

    rebuild_index_module._validate_before_derived_state(provenance)
    inventory_calls_after_refresh = full_inventory_calls
    rebuild_index_module._validate_before_derived_state(provenance)

    assert inventory_calls_after_refresh == inventory_calls_before_refresh + 1
    assert full_inventory_calls == inventory_calls_after_refresh, (
        "the refreshed pass token must prevent a second full inventory scan"
    )


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


@pytest.mark.parametrize("failure_kind", ["replay", "readiness"], ids=["replay-failure", "readiness-failure"])
def test_nonresumable_failure_discards_inactive_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    """One-shot rebuild failures cannot leave an inactive candidate behind.

    Anti-vacuity: both cases enter the production nonresumable raw-id route and
    create a real SQLite generation. The replay case fails inside the real
    replay call, while the readiness case completes replay and fails at the
    terminal readiness gate. Removing the explicit cleanup leaves ``gen-*``
    metadata behind.
    """
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    if failure_kind == "replay":

        async def fail_replay(*args: Any, **kwargs: Any) -> dict[str, object]:
            raise RuntimeError("synthetic nonresumable replay failure")

        monkeypatch.setattr("polylogue.maintenance.replay.rebuild_index_from_source", fail_replay)
        expected = "synthetic nonresumable replay failure"
    else:
        monkeypatch.setattr(
            "polylogue.storage.archive_readiness.archive_readiness_status",
            lambda root: {
                "checked": True,
                "blocked_surface_count": 1,
                "surfaces": {"synthetic": {"ready": False}},
            },
        )
        expected = "is not exact-ready"

    with pytest.raises(RuntimeError, match=expected):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                raw_ids=_raw_ids(root),
                promote=False,
                schema_inference_receipt_path=receipt_path,
            )
        )

    assert _generation_ids(root) == set()


def test_capture_evidence_mutation_after_receipt_rejects_before_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture evidence drift after receipt production cannot be promoted.

    Anti-vacuity: the real full-source promote route replays into an inactive
    generation before the patched production replay boundary mutates capture
    mode, capture index, file metadata, and capture observations in source.db.
    The canonical source-evidence snapshot rejects the candidate before the
    activation swap and cleans its transaction and generation.
    """
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    active_before = _active_bytes(root)
    from polylogue.maintenance import replay as replay_module

    real_replay = cast(Callable[..., Any], replay_module.rebuild_index_from_source)

    async def mutate_capture_evidence(*args: object, **kwargs: object) -> dict[str, object]:
        replay = await real_replay(*args, **kwargs)
        with sqlite3.connect(root / "source.db") as conn:
            raw_id = str(conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id LIMIT 1").fetchone()[0])
            conn.execute(
                "UPDATE raw_sessions SET capture_mode = 'gemini', source_index = source_index + 1, "
                "file_mtime_ms = COALESCE(file_mtime_ms, 0) + 1 WHERE raw_id = ?",
                (raw_id,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO raw_capture_observations "
                "(raw_id, capture_mode, first_observed_at_ms) VALUES (?, 'gemini', 999999)",
                (raw_id,),
            )
        return cast(dict[str, object], replay)

    monkeypatch.setattr(replay_module, "rebuild_index_from_source", mutate_capture_evidence)

    with pytest.raises(RuntimeError, match="schema-inference preflight gate failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                promote=True,
                schema_inference_receipt_path=receipt_path,
            )
        )

    assert _active_bytes(root) == active_before
    assert _generation_ids(root) == set()
    assert not list((root / ".index-rebuild-transactions").glob("*.json"))


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


def test_revision_authority_binding_stales_a_resumed_real_rebuild(
    tmp_path: Path,
) -> None:
    """A replay-affecting raw authority mutation cannot cross a page boundary.

    Anti-vacuity: removing the authority fields from the source binding leaves
    the transaction paused instead of stale, failing the terminal assertion.
    """
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    first = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            raw_batch_size=1,
            promote=False,
        )
    )
    assert first.status == "paused"
    assert first.transaction is not None
    operation_id = str(first.transaction["operation_id"])
    before_evidence_digest = rebuild_source_evidence_snapshot(root)
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            "UPDATE raw_sessions SET revision_authority_evidence = 'live_source_verification_v1' "
            "WHERE raw_id = (SELECT raw_id FROM raw_sessions ORDER BY raw_id LIMIT 1)"
        )
        source.commit()
    assert rebuild_source_evidence_snapshot(root) != before_evidence_digest

    with pytest.raises(RuntimeError, match="source snapshot does not match"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                operation_id=operation_id,
                raw_batch_size=1,
                promote=False,
            )
        )
    transaction = IndexGenerationStore.for_archive_root(root).load_transaction(operation_id)
    assert transaction.status == "stale"


def test_blob_bytes_changed_after_replay_are_rejected_before_candidate_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final real readiness route verifies bytes, not only source.db hashes.

    Anti-vacuity: removing the readiness blob binding removes the dedicated
    integrity failure asserted here, even though source.db still names a
    parseable row.
    """
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    original_repopulate = rebuild_index_module._repopulate_bulk_build_derived_state

    def corrupt_after_replay(index_path: Path) -> dict[str, float]:
        timings = original_repopulate(index_path)
        with sqlite3.connect(root / "source.db") as source:
            blob_hash = bytes(source.execute("SELECT blob_hash FROM raw_sessions LIMIT 1").fetchone()[0]).hex()
        BlobStore(root / "blob").blob_path(blob_hash).write_bytes(b"parseable but corrupted")
        return timings

    monkeypatch.setattr(rebuild_index_module, "_repopulate_bulk_build_derived_state", corrupt_after_replay)
    with pytest.raises(RuntimeError, match="referenced source blob integrity verification failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                promote=False,
            )
        )
    assert not list((root / ".index-generations").glob("gen-*"))


def test_pointer_flip_records_post_promotion_attestation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-flip checkpoint fault leaves an active, terminally classified operation.

    Anti-vacuity: removing the terminal attestation transition makes the
    injected post-flip failure escape instead of returning an active failed
    attestation.
    """
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    original_checkpoint = IndexGenerationStore.checkpoint_transaction

    def fail_promoted_checkpoint(self: IndexGenerationStore, transaction: object, **kwargs: object) -> object:
        if kwargs.get("status") == "promoted":
            raise OSError("simulated post-promotion attestation failure")
        return original_checkpoint(self, transaction, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(IndexGenerationStore, "checkpoint_transaction", fail_promoted_checkpoint)
    with pytest.raises(OSError, match="simulated post-promotion attestation failure"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path, promote=True)
        )

    store = IndexGenerationStore.for_archive_root(root)
    operation_id = next(path.stem for path in store.transactions_root.glob("*.json"))
    transaction = store.load_transaction(operation_id)
    assert transaction.status == "promoted-attestation-failed"
    attestation = transaction.post_promotion_attestation
    assert isinstance(attestation, dict)
    assert attestation["status"] == "failed"
    assert store.load(transaction.generation_id).state == "active"
    assert store.active_pointer.resolve(strict=True) == Path(store.load(transaction.generation_id).index_path).resolve(
        strict=True
    )


def test_daemon_reconciles_active_generation_after_both_attestation_checkpoints_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    seeded_transaction = store.create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root),
        operation_id=bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID,
    )
    original_checkpoint = IndexGenerationStore.checkpoint_transaction

    def fail_attestation_checkpoint(self: IndexGenerationStore, transaction: object, **kwargs: object) -> object:
        if kwargs.get("status") in {"promoted", "promoted-attestation-failed"}:
            raise OSError("simulated double attestation checkpoint failure")
        return original_checkpoint(self, transaction, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(IndexGenerationStore, "checkpoint_transaction", fail_attestation_checkpoint)
    with pytest.raises(OSError, match="simulated double attestation checkpoint failure"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                operation_id=bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID,
                promote=True,
            )
        )

    transaction = store.load_transaction(bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID)
    assert transaction.status == "ready"
    assert transaction.generation_id == seeded_transaction.generation_id
    assert store.load(transaction.generation_id).state == "active"

    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="promoted-attestation-failed; start a new operation"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                operation_id=bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID,
                promote=True,
            )
        )

    offline_terminal = store.load_transaction(bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID)
    assert offline_terminal.status == "promoted-attestation-failed"
    assert store.load(offline_terminal.generation_id).state == "active"

    reconciled = bulk_rebuild_module.resolve_or_start_daemon_bulk_rebuild_transaction(
        root,
        schema_inference_receipt_path=receipt_path,
    )

    assert reconciled.status == "promoted-attestation-failed"
    assert reconciled.post_promotion_attestation == {
        "status": "reconciled-after-restart",
        "generation_id": transaction.generation_id,
        "generation_state": "active",
    }


def test_active_generation_reconciliation_requires_transaction_owner_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale transaction cannot checkpoint a generation owned by another pass."""
    root = tmp_path / "archive"
    _seed(root, count=1)
    store = IndexGenerationStore.for_archive_root(root)
    generation = IndexGeneration(
        generation_id="gen-active",
        owner_id="current-owner",
        archive_root=str(root),
        index_path=str(root / "index.db"),
        state="active",
        created_at_ms=1,
    )
    transaction = IndexRebuildTransaction(
        operation_id="rebuild-stale-owner",
        generation_id=generation.generation_id,
        generation_owner_id="stale-owner",
        source_snapshot="source-snapshot",
        status="ready",
        created_at_ms=1,
        updated_at_ms=1,
    )
    monkeypatch.setattr(store, "load", lambda _generation_id: generation)

    def fail_checkpoint(*args: object, **kwargs: object) -> object:
        raise AssertionError("owner-mismatched transaction must not checkpoint")

    monkeypatch.setattr(store, "checkpoint_transaction", fail_checkpoint)

    reconciled = rebuild_index_module._reconcile_active_generation_transaction(store, transaction)

    assert reconciled == transaction


def test_daemon_does_not_route_promoted_attestation_failure_back_to_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal active operation is not handed to the rebuild engine again."""
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    store.create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root),
        operation_id=bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID,
    )
    original_checkpoint = IndexGenerationStore.checkpoint_transaction

    def fail_promoted_checkpoint(self: IndexGenerationStore, transaction: object, **kwargs: object) -> object:
        if kwargs.get("status") == "promoted":
            raise OSError("simulated post-promotion attestation failure")
        return original_checkpoint(self, transaction, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(IndexGenerationStore, "checkpoint_transaction", fail_promoted_checkpoint)
    with pytest.raises(OSError, match="simulated post-promotion attestation failure"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                operation_id=bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID,
                promote=True,
            )
        )
    terminal = store.load_transaction(bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID)
    assert terminal.status == "promoted-attestation-failed"
    assert bulk_rebuild_module.has_resumable_daemon_bulk_rebuild_transaction(root) is False

    monkeypatch.setenv("POLYLOGUE_SCHEMA_INFERENCE_RECEIPT", str(receipt_path))
    rebuild_called = False

    def unexpected_rebuild(*args: object, **kwargs: object) -> object:
        nonlocal rebuild_called
        rebuild_called = True
        raise AssertionError("terminal daemon operation was routed back to rebuild")

    monkeypatch.setattr(rebuild_index_module, "rebuild_index_from_source_sync", unexpected_rebuild)
    result = asyncio.run(
        bulk_rebuild_module.run_daemon_bulk_rebuild_pass(
            config=Config(archive_root=root, render_root=root / "render", sources=[]),
            parse_stage=cast(Any, object()),
            max_payload_bytes=1,
        )
    )
    assert result is None
    assert rebuild_called is False
    active_generation = store.load(terminal.generation_id)
    assert store.active_pointer.resolve(strict=True) == Path(active_generation.index_path).resolve(strict=True)


def test_daemon_retires_attestation_failure_after_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed source gets a fresh daemon transaction without replacing the active index."""
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    store = IndexGenerationStore.for_archive_root(root)
    terminal = store.create_transaction(
        source_snapshot=rebuild_source_evidence_snapshot(root),
        operation_id=bulk_rebuild_module.DAEMON_BULK_REBUILD_OPERATION_ID,
    )
    store.promote(store.load(terminal.generation_id))
    store.checkpoint_transaction(terminal, status="promoted-attestation-failed")
    monkeypatch.setattr(bulk_rebuild_module, "rebuild_source_evidence_snapshot", lambda _root: "changed-source")

    replacement = bulk_rebuild_module.resolve_or_start_daemon_bulk_rebuild_transaction(
        root, schema_inference_receipt_path=receipt_path
    )

    assert replacement.status == "running"
    assert replacement.generation_id != terminal.generation_id
    assert replacement.source_snapshot == "changed-source"
    assert store.load(terminal.generation_id).state == "active"


def test_validation_rejection_cannot_stale_transaction_before_ownership(
    tmp_path: Path,
) -> None:
    """A live archive owner rejects the invocation before transaction mutation."""
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    first = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            raw_batch_size=1,
            promote=False,
        )
    )
    assert first.transaction is not None
    operation_id = str(first.transaction["operation_id"])
    store = IndexGenerationStore.for_archive_root(root)
    before = store.load_transaction(operation_id)

    owner = OwnedArchiveLocation.acquire(ArchiveLocation.resolve(root))
    try:
        with pytest.raises(ArchiveOwnershipError):
            rebuild_index_from_source_sync(
                RebuildIndexRequest(
                    archive_root=root,
                    schema_inference_receipt_path=receipt_path,
                    operation_id=operation_id,
                    raw_batch_size=1,
                    promote=False,
                )
            )
    finally:
        owner.release()

    assert store.load_transaction(operation_id) == before


@pytest.mark.parametrize(
    "exception_type",
    [KeyboardInterrupt, asyncio.CancelledError, SystemExit],
    ids=["keyboard-interrupt", "cancelled", "control-flow-base-exception"],
)
def test_validation_control_flow_does_not_change_resumability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception_type: type[BaseException]
) -> None:
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    first = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            raw_batch_size=1,
            promote=False,
        )
    )
    assert first.transaction is not None
    operation_id = str(first.transaction["operation_id"])

    def raise_control_flow(*args: object, **kwargs: object) -> object:
        raise exception_type("validation interrupted")

    monkeypatch.setattr(rebuild_index_module, "_validate_rebuild_provenance_receipt", raise_control_flow)
    with pytest.raises(exception_type, match="validation interrupted"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                operation_id=operation_id,
                raw_batch_size=1,
                promote=False,
            )
        )
    assert IndexGenerationStore.for_archive_root(root).load_transaction(operation_id).status == "paused"


def test_external_rewrite_with_preserved_size_inode_and_mtime_is_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    _seed(root, count=2)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    first = rebuild_index_from_source_sync(
        RebuildIndexRequest(
            archive_root=root,
            schema_inference_receipt_path=receipt_path,
            raw_batch_size=1,
            promote=False,
        )
    )
    assert first.transaction is not None
    operation_id = str(first.transaction["operation_id"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    origin = receipt["ground_truth_inputs"]["origins"]["codex-session"]
    external_path = Path(origin["declared_roots"][0]) / origin["external_inventory"][0]["relative_path"]
    before = external_path.stat()
    original = external_path.read_bytes()
    replacement = bytes(byte ^ 0xFF for byte in original)
    external_path.write_bytes(replacement)
    os.utime(external_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = external_path.stat()
    assert after.st_ino == before.st_ino
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert replacement != original

    with pytest.raises(RuntimeError, match="external ground-truth corpus changed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(
                archive_root=root,
                schema_inference_receipt_path=receipt_path,
                operation_id=operation_id,
                raw_batch_size=1,
                promote=False,
            )
        )


def test_full_blob_verification_supplies_referenced_snapshot_without_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    _seed(root, count=1)
    with sqlite3.connect(root / "source.db") as source:
        referenced_hashes = {bytes(row[0]).hex() for row in source.execute("SELECT blob_hash FROM raw_sessions")}
    verify_calls = 0
    original_verify = BlobStore.verify

    def count_verify(self: BlobStore, blob_hash: str) -> bool:
        nonlocal verify_calls
        verify_calls += 1
        return original_verify(self, blob_hash)

    monkeypatch.setattr(BlobStore, "verify", count_verify)
    evidence = schema_gate_module._full_blob_hash_evidence(root, referenced_hashes=referenced_hashes)
    assert evidence["passed"] is True
    assert verify_calls == 0
    snapshot = cast(dict[str, object], evidence["referenced_blob_integrity_snapshot"])
    assert snapshot["verifier"] == "polylogue.storage.blob_store.BlobStore.verify_all"
    assert snapshot["passed"] is True


def test_rebuild_reuses_verified_blob_snapshot_across_readiness_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two production readiness checks share one byte-verification result."""
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    calls = 0
    original_snapshot = schema_gate_module._referenced_blob_integrity_snapshot

    def count_snapshot(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original_snapshot(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(schema_gate_module, "_referenced_blob_integrity_snapshot", count_snapshot)
    result = rebuild_index_from_source_sync(
        RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path, promote=False)
    )
    assert result.status == "replayed"
    assert calls == 2


def test_rebuild_rechecks_blob_bytes_at_final_readiness_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corruption after insight materialization is caught before readiness."""
    root = tmp_path / "archive"
    _seed(root, count=1)
    receipt_path = write_valid_rebuild_receipt(root, tmp_path / "receipt.json")
    from polylogue.storage import repair as repair_module

    original_repair = repair_module.repair_session_insights

    def corrupt_after_insights(*args: object, **kwargs: object) -> object:
        result = cast(Any, original_repair)(*args, **kwargs)
        with sqlite3.connect(root / "source.db") as source:
            blob_hash = bytes(source.execute("SELECT blob_hash FROM raw_sessions LIMIT 1").fetchone()[0]).hex()
        BlobStore(root / "blob").blob_path(blob_hash).write_bytes(b"corrupted after readiness precursor")
        return result

    monkeypatch.setattr("polylogue.storage.repair.repair_session_insights", corrupt_after_insights)
    with pytest.raises(RuntimeError, match="referenced source blob integrity verification failed"):
        rebuild_index_from_source_sync(
            RebuildIndexRequest(archive_root=root, schema_inference_receipt_path=receipt_path, promote=False)
        )


def test_replay_closure_caches_fingerprints_per_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay evidence computes parser and lowering fingerprints once per closure."""
    root = tmp_path / "archive"
    _seed(root, count=2)
    lower_calls = 0
    parser_calls: list[str] = []
    original_lowering = origin_specs_module.lowering_fingerprint
    original_parser = origin_specs_module.parser_fingerprint_for_origin

    def count_lowering() -> str:
        nonlocal lower_calls
        lower_calls += 1
        return original_lowering()

    def count_parser(origin: str) -> str:
        parser_calls.append(origin)
        return original_parser(origin)

    monkeypatch.setattr(origin_specs_module, "lowering_fingerprint", count_lowering)
    monkeypatch.setattr(origin_specs_module, "parser_fingerprint_for_origin", count_parser)
    evidence = rebuild_index_module._rebuild_replay_closure_evidence(root, _raw_ids(root))
    assert evidence["raw_session_evidence"]
    assert lower_calls == 1
    assert parser_calls == ["codex-session"]


def test_referenced_blob_snapshot_ignores_volatile_filesystem_metadata(
    tmp_path: Path,
) -> None:
    """Identical content remains valid when a blob's mtime changes.

    Anti-vacuity: restoring the old inode/mtime fields to the content snapshot
    would make this metadata-only touch change the recorded evidence.
    """
    root = tmp_path / "archive"
    _seed(root, count=1)
    with sqlite3.connect(root / "source.db") as source:
        referenced_hashes = {bytes(row[0]).hex() for row in source.execute("SELECT blob_hash FROM raw_sessions")}
    before = schema_gate_module._referenced_blob_integrity_snapshot(root, referenced_hashes=referenced_hashes)
    blob_path = BlobStore(root / "blob").blob_path(next(iter(referenced_hashes)))
    stat = blob_path.stat()
    os.utime(blob_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    after = schema_gate_module._referenced_blob_integrity_snapshot(root, referenced_hashes=referenced_hashes)
    assert all(
        "inode" not in entry and "mtime_ns" not in entry for entry in cast(list[dict[str, object]], before["entries"])
    )
    assert after == before


def test_source_evidence_snapshot_streams_raw_session_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "archive"
    _seed(root, count=2)
    sqlite_module = cast(Any, index_generation_module).sqlite3
    real_connect = sqlite_module.connect

    class GuardedCursor:
        def __init__(self, cursor: sqlite3.Cursor, sql: str) -> None:
            self._cursor = cursor
            self._sql = sql

        def __iter__(self) -> GuardedCursor:
            return self

        def __next__(self) -> tuple[object, ...]:
            return next(self._cursor)

        def fetchall(self) -> list[tuple[object, ...]]:
            if "FROM RAW_SESSIONS" in self._sql.upper():
                raise AssertionError("raw_sessions evidence must be consumed as a stream")
            return self._cursor.fetchall()

        def __getattr__(self, name: str) -> object:
            return getattr(self._cursor, name)

    class GuardedConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def execute(self, sql: str, parameters: object = ()) -> GuardedCursor:
            return GuardedCursor(self._connection.execute(sql, cast(Any, parameters)), sql)

        def __getattr__(self, name: str) -> object:
            return getattr(self._connection, name)

    def guarded_connect(*args: object, **kwargs: object) -> GuardedConnection:
        return GuardedConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(sqlite_module, "connect", guarded_connect)
    assert index_generation_module.rebuild_source_evidence_snapshot(root)
