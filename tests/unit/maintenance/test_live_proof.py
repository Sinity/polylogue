"""Tests for the fixed live-proof protocol.

The production dependency exercised here is the archive-verification registry
and the source/index binding readers.  The red mutations alter one captured
binding or candidate metadata after collection; validation must reject them,
which a receipt-only serializer would incorrectly accept.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from polylogue.core.hashing import hash_payload
from polylogue.maintenance.live_proof import (
    EXISTING_APPLY_RECEIPT_SCHEMA,
    LIVE_PROOF_SPECS,
    LiveProofBindings,
    LiveProofError,
    LiveProofId,
    LiveProofMode,
    capture_live_proof_bindings,
    collect_live_proof,
    validate_candidate_proof_receipts,
    validate_live_operation_aggregate,
    validate_live_proof_receipt,
    validate_live_proof_registry,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


@pytest.fixture
def archive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "private-archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_CODE_SHA", "a" * 40)
    return root


def _apply_receipt(bindings: LiveProofBindings, *, status: str = "applied") -> dict[str, object]:
    document = {
        "receipt_schema": EXISTING_APPLY_RECEIPT_SCHEMA,
        "operation_id": "known-source-remediation",
        "bindings": bindings.to_document(),
        "result": {"status": status, "changed_count": 1},
    }
    return {**document, "receipt_sha256": hash_payload(document)}


def _candidate(root: Path) -> str:
    generation_id = "gen-live-proof"
    generation = root / ".index-generations" / generation_id
    generation.mkdir(parents=True)
    candidate_index = generation / "index.db"
    shutil.copy2(root / "index.db", candidate_index)
    (generation / "generation.json").write_text(
        json.dumps(
            {
                "generation_id": generation_id,
                "owner_id": "proof-owner",
                "archive_root": str(root),
                "index_path": str(candidate_index),
                "state": "inactive",
                "source_snapshot": "candidate-source-snapshot",
            }
        ),
        encoding="utf-8",
    )
    return generation_id


def test_fixed_registry_has_exactly_the_three_supported_modes() -> None:
    validate_live_proof_registry()

    assert {spec.mode for spec in LIVE_PROOF_SPECS} == set(LiveProofMode)
    assert {spec.proof_id for spec in LIVE_PROOF_SPECS} == set(LiveProofId)
    assert all(not callable(spec.producer) for spec in LIVE_PROOF_SPECS)


def test_read_only_receipt_is_deterministic_self_hashed_and_private_path_safe(archive_root: Path) -> None:
    first = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)
    second = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)

    assert first.to_document() == second.to_document()
    assert first.to_document()["receipt_sha256"] == first.receipt_sha256
    assert str(archive_root) not in json.dumps(first.to_document())
    assert {name for name, _version in first.bindings.schema_versions} == {
        "audit",
        "embeddings",
        "index",
        "ops",
        "source",
        "user",
    }
    assert validate_live_proof_receipt(first.to_document(), archive_root) == first


def test_mode_inputs_are_isolated(archive_root: Path, tmp_path: Path) -> None:
    apply_path = tmp_path / "apply.json"
    apply_path.write_text(json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root))), encoding="utf-8")

    with pytest.raises(LiveProofError, match="read-only proof"):
        collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root, apply_receipt_path=apply_path)
    with pytest.raises(LiveProofError, match="candidate proof"):
        collect_live_proof(LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION.value, archive_root)
    with pytest.raises(LiveProofError, match="existing-apply proof"):
        collect_live_proof(LiveProofId.EXISTING_APPLY_RECEIPT.value, archive_root)


def test_existing_apply_receipt_is_bound_and_rejects_controlled_binding_mutation(
    archive_root: Path, tmp_path: Path
) -> None:
    apply_path = tmp_path / "private-apply-receipt.json"
    apply_path.write_text(json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root))), encoding="utf-8")

    receipt = collect_live_proof(
        LiveProofId.EXISTING_APPLY_RECEIPT.value,
        archive_root,
        apply_receipt_path=apply_path,
    )

    assert receipt.input_receipt_digests
    assert str(apply_path) not in json.dumps(receipt.to_document())
    mutated = receipt.to_document()
    binding_value = mutated["bindings"]
    assert isinstance(binding_value, dict)
    bindings = dict(binding_value)
    bindings["source_snapshot"] = "0" * 64
    mutated["bindings"] = bindings
    unsigned = dict(mutated)
    unsigned.pop("receipt_sha256")
    mutated["receipt_sha256"] = hash_payload(unsigned)
    with pytest.raises(LiveProofError, match="bindings are stale"):
        validate_live_proof_receipt(mutated, archive_root)


def test_existing_apply_receipt_rejects_non_successful_result(archive_root: Path, tmp_path: Path) -> None:
    apply_path = tmp_path / "failed-apply-receipt.json"
    apply_path.write_text(
        json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root), status="unknown")), encoding="utf-8"
    )

    with pytest.raises(LiveProofError, match="result status is not successful"):
        collect_live_proof(
            LiveProofId.EXISTING_APPLY_RECEIPT.value,
            archive_root,
            apply_receipt_path=apply_path,
        )


def test_candidate_receipt_binds_exact_inactive_generation_and_detects_content_mutation(archive_root: Path) -> None:
    generation_id = _candidate(archive_root)
    receipt = collect_live_proof(
        LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION.value,
        archive_root,
        candidate_generation_id=generation_id,
    )

    assert receipt.bindings.candidate_generation_id == generation_id
    assert receipt.bindings.candidate_index_sha256 is not None
    candidate_index = archive_root / ".index-generations" / generation_id / "index.db"
    with candidate_index.open("ab") as stream:
        stream.write(b"binding mutation")
    with pytest.raises(LiveProofError, match="bindings are stale"):
        validate_candidate_proof_receipts((receipt.to_document(),), archive_root, candidate_generation_id=generation_id)


def test_aggregate_rejects_a_self_hashed_failed_proof_result(archive_root: Path) -> None:
    receipt = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)
    failed = receipt.to_document()
    result = failed["result"]
    assert isinstance(result, dict)
    result["status"] = "failed"
    unsigned = dict(failed)
    unsigned.pop("receipt_sha256")
    failed["receipt_sha256"] = hash_payload(unsigned)

    with pytest.raises(LiveProofError, match="not acceptable"):
        validate_live_operation_aggregate((failed,), archive_root)
