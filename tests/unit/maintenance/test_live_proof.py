"""Tests for the fixed live-proof protocol.

The production dependencies exercised here are the archive-verification
profiles and archive generation store. Red mutations alter a captured binding,
generation record, result status, or output write so consumers must reject
evidence a receipt-only serializer would otherwise accept.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDocument
from polylogue.maintenance import live_proof
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
    validate_final_proof_receipts,
    validate_live_operation_aggregate,
    validate_live_proof_receipt,
    validate_live_proof_registry,
    write_live_proof_receipt,
)
from polylogue.maintenance.schema_inference_gate import rebuild_source_revision_snapshot
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.version import VERSION_INFO


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
    store = IndexGenerationStore(ArchiveLocation.resolve(root))
    generation = store.create(source_snapshot=rebuild_source_revision_snapshot(root))
    return generation.generation_id


def _rehash(document: JSONDocument) -> JSONDocument:
    unsigned = dict(document)
    unsigned.pop("receipt_sha256")
    document["receipt_sha256"] = hash_payload(unsigned)
    return document


def test_fixed_registry_has_exactly_the_three_supported_modes() -> None:
    validate_live_proof_registry()

    assert {spec.mode for spec in LIVE_PROOF_SPECS} == set(LiveProofMode)
    assert {spec.proof_id for spec in LIVE_PROOF_SPECS} == set(LiveProofId)
    assert all(not callable(spec.producer) for spec in LIVE_PROOF_SPECS)


def test_read_only_receipt_preserves_full_redacted_canonical_evidence(archive_root: Path) -> None:
    receipt = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)
    repeated = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)

    assert receipt.to_document() == repeated.to_document()
    evidence = receipt.result["archive_verification"]
    assert isinstance(evidence, dict)
    profiles = evidence["profiles"]
    assert isinstance(profiles, dict)
    assert list(profiles) == ["active-archive"]
    active = profiles["active-archive"]
    assert isinstance(active, dict)
    checks = active["checks"]
    assert isinstance(checks, list)
    assert len(checks) > 2
    assert str(archive_root) not in json.dumps(receipt.to_document())
    assert receipt.to_document()["receipt_sha256"] == receipt.receipt_sha256
    assert {name for name, _version in receipt.bindings.schema_versions} == {
        "audit",
        "embeddings",
        "index",
        "ops",
        "source",
        "user",
    }
    assert receipt.bindings.active_index_sha256
    assert validate_live_proof_receipt(receipt.to_document(), archive_root) == receipt


def test_read_only_receipt_redacts_an_external_active_index_path(archive_root: Path) -> None:
    external_root = archive_root.parent / "private-active-index"
    external_root.mkdir()
    external_index = external_root / "index.db"
    os.link(archive_root / "index.db", external_index)
    (archive_root / ".index-active-pointer").write_text(str(external_index), encoding="utf-8")

    receipt = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)

    assert str(external_root) not in json.dumps(receipt.to_document())


def test_read_only_receipt_redacts_external_verification_evidence(archive_root: Path) -> None:
    external_source = archive_root.parent / "private source" / "session.jsonl"
    with sqlite3.connect(archive_root / "ops.db") as connection:
        connection.execute(
            """
            INSERT INTO ingest_cursor(source_path, excluded, stat_size, byte_offset, updated_at_ms)
            VALUES (?, 0, 100000000, 5000000, 0)
            """,
            (str(external_source),),
        )

    receipt = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)

    serialized = json.dumps(receipt.to_document())
    assert str(external_source) not in serialized
    assert "[private-path:" in serialized


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
    with pytest.raises(LiveProofError, match="bindings are stale"):
        validate_live_proof_receipt(_rehash(mutated), archive_root)


@pytest.mark.parametrize(
    ("apply_status", "proof_status"),
    [("applied", "passed"), ("already_satisfied", "passed"), ("not_applicable", "not_applicable")],
)
def test_existing_apply_receipt_preserves_terminal_result_status(
    archive_root: Path, tmp_path: Path, apply_status: str, proof_status: str
) -> None:
    apply_path = tmp_path / "apply-receipt.json"
    apply_path.write_text(
        json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root), status=apply_status)), encoding="utf-8"
    )

    receipt = collect_live_proof(LiveProofId.EXISTING_APPLY_RECEIPT.value, archive_root, apply_receipt_path=apply_path)

    assert receipt.result["status"] == proof_status
    apply_receipt = receipt.result["apply_receipt"]
    assert isinstance(apply_receipt, dict)
    apply_result = apply_receipt["result"]
    assert isinstance(apply_result, dict)
    assert apply_result["status"] == apply_status
    assert validate_live_operation_aggregate((receipt.to_document(),), archive_root) == (receipt,)


def test_existing_apply_receipt_embeds_the_digest_bound_input(archive_root: Path, tmp_path: Path) -> None:
    apply_path = tmp_path / "apply-receipt.json"
    apply_path.write_text(json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root))), encoding="utf-8")
    receipt = collect_live_proof(LiveProofId.EXISTING_APPLY_RECEIPT.value, archive_root, apply_receipt_path=apply_path)
    mutated = receipt.to_document()
    result = mutated["result"]
    assert isinstance(result, dict)
    embedded = result["apply_receipt"]
    assert isinstance(embedded, dict)
    embedded_result = embedded["result"]
    assert isinstance(embedded_result, dict)
    embedded_result["status"] = "failed"
    _rehash(embedded)
    result["status"] = "failed"
    mutated["residues"] = [{"kind": "check_failed", "code": "apply-result-failed"}]

    with pytest.raises(LiveProofError, match="input digest"):
        validate_live_proof_receipt(_rehash(mutated), archive_root)


def test_existing_apply_receipt_keeps_failure_as_failed_evidence(archive_root: Path, tmp_path: Path) -> None:
    apply_path = tmp_path / "failed-apply-receipt.json"
    apply_path.write_text(
        json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root), status="failed")), encoding="utf-8"
    )

    receipt = collect_live_proof(LiveProofId.EXISTING_APPLY_RECEIPT.value, archive_root, apply_receipt_path=apply_path)

    assert receipt.result["status"] == "failed"
    with pytest.raises(LiveProofError, match="not acceptable"):
        validate_live_operation_aggregate((receipt.to_document(),), archive_root)


def test_existing_apply_receipt_rejects_unknown_result(archive_root: Path, tmp_path: Path) -> None:
    apply_path = tmp_path / "unknown-apply-receipt.json"
    apply_path.write_text(
        json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root), status="unknown")), encoding="utf-8"
    )

    with pytest.raises(LiveProofError, match="result status is not successful"):
        collect_live_proof(LiveProofId.EXISTING_APPLY_RECEIPT.value, archive_root, apply_receipt_path=apply_path)


def test_candidate_receipt_binds_canonical_generation_and_all_profiles(archive_root: Path) -> None:
    generation_id = _candidate(archive_root)
    receipt = collect_live_proof(
        LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION.value,
        archive_root,
        candidate_generation_id=generation_id,
    )

    assert receipt.bindings.candidate_generation_id == generation_id
    assert receipt.bindings.candidate_index_sha256 is not None
    verification = receipt.result["archive_verification"]
    assert isinstance(verification, dict)
    profiles = verification["profiles"]
    assert isinstance(profiles, dict)
    assert set(profiles) == {"candidate-index", "candidate-cross-tier"}
    candidate_path = Path(IndexGenerationStore(ArchiveLocation.resolve(archive_root)).load(generation_id).index_path)
    with candidate_path.open("ab") as stream:
        stream.write(b"binding mutation")
    with pytest.raises(LiveProofError, match="bindings are stale"):
        validate_candidate_proof_receipts((receipt.to_document(),), archive_root, candidate_generation_id=generation_id)


def test_candidate_rejects_source_snapshot_drift(archive_root: Path) -> None:
    generation_id = _candidate(archive_root)
    store = IndexGenerationStore(ArchiveLocation.resolve(archive_root))
    generation_path = store.generations_root / generation_id / "generation.json"
    payload = json.loads(generation_path.read_text(encoding="utf-8"))
    payload["source_snapshot"] = "outdated"
    generation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LiveProofError, match="candidate generation binding"):
        collect_live_proof(
            LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION.value,
            archive_root,
            candidate_generation_id=generation_id,
        )


def test_candidate_rejects_symlink_index(archive_root: Path) -> None:
    generation_id = _candidate(archive_root)
    store = IndexGenerationStore(ArchiveLocation.resolve(archive_root))
    candidate_path = Path(store.load(generation_id).index_path)
    candidate_path.unlink()
    candidate_path.symlink_to(archive_root / "index.db")

    with pytest.raises(LiveProofError, match="candidate generation binding"):
        collect_live_proof(
            LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION.value,
            archive_root,
            candidate_generation_id=generation_id,
        )


def test_candidate_rejects_poisoned_active_pointer_without_repairing_it(archive_root: Path) -> None:
    generation_id = _candidate(archive_root)
    store = IndexGenerationStore(ArchiveLocation.resolve(archive_root))
    candidate_path = Path(store.load(generation_id).index_path)
    pointer = archive_root / ".index-active-pointer"
    pointer.write_text(str(candidate_path), encoding="utf-8")

    with pytest.raises(LiveProofError, match="canonical active-index pointer"):
        collect_live_proof(
            LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION.value,
            archive_root,
            candidate_generation_id=generation_id,
        )

    assert pointer.read_text(encoding="utf-8") == str(candidate_path)


def test_receipt_rejects_invalid_residue_code(archive_root: Path) -> None:
    receipt = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)
    mutated = receipt.to_document()
    residues = mutated["residues"]
    assert isinstance(residues, list) and residues
    residue = residues[0]
    assert isinstance(residue, dict)
    residue["code"] = "not a valid code"

    with pytest.raises(LiveProofError, match="residues are malformed"):
        validate_live_proof_receipt(_rehash(mutated), archive_root)


def test_receipt_rejects_residues_mismatched_to_check_evidence(archive_root: Path) -> None:
    receipt = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)
    mutated = receipt.to_document()
    result = mutated["result"]
    assert isinstance(result, dict)
    result["status"] = "failed"
    verification = result["archive_verification"]
    assert isinstance(verification, dict)
    profiles = verification["profiles"]
    assert isinstance(profiles, dict)
    profile = profiles["active-archive"]
    assert isinstance(profile, dict)
    checks = profile["checks"]
    assert isinstance(checks, list) and checks
    check = checks[0]
    assert isinstance(check, dict)
    check["status"] = "error"

    with pytest.raises(LiveProofError, match="status and residues are inconsistent"):
        validate_live_proof_receipt(_rehash(mutated), archive_root)


def test_aggregate_rejects_a_self_hashed_failed_proof_result(archive_root: Path) -> None:
    bindings = capture_live_proof_bindings(archive_root)
    document = _apply_receipt(bindings, status="failed")
    apply_path = archive_root.parent / "apply.json"
    apply_path.write_text(json.dumps(document), encoding="utf-8")
    failed = collect_live_proof(LiveProofId.EXISTING_APPLY_RECEIPT.value, archive_root, apply_receipt_path=apply_path)

    with pytest.raises(LiveProofError, match="not acceptable"):
        validate_live_operation_aggregate((failed.to_document(),), archive_root)


def test_final_proof_requires_every_route_exactly_once(archive_root: Path, tmp_path: Path) -> None:
    apply_path = tmp_path / "apply.json"
    apply_path.write_text(json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root))), encoding="utf-8")
    receipt = collect_live_proof(LiveProofId.EXISTING_APPLY_RECEIPT.value, archive_root, apply_receipt_path=apply_path)

    with pytest.raises(LiveProofError, match="complete route coverage"):
        validate_final_proof_receipts((receipt.to_document(),), archive_root)


def test_aggregate_captures_candidate_and_active_bindings_once(
    archive_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_id = _candidate(archive_root)
    apply_path = tmp_path / "apply.json"
    apply_path.write_text(json.dumps(_apply_receipt(capture_live_proof_bindings(archive_root))), encoding="utf-8")
    receipts = (
        collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root).to_document(),
        collect_live_proof(
            LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION.value,
            archive_root,
            candidate_generation_id=generation_id,
        ).to_document(),
        collect_live_proof(
            LiveProofId.EXISTING_APPLY_RECEIPT.value,
            archive_root,
            apply_receipt_path=apply_path,
        ).to_document(),
    )
    original = live_proof._capture_expected_bindings
    calls = 0

    def capture_once(root: Path, candidate_id: str | None) -> LiveProofBindings:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("aggregate captured another active snapshot")
        return original(root, candidate_id)

    monkeypatch.setattr(live_proof, "_capture_expected_bindings", capture_once)

    assert live_proof._validate_aggregate(receipts, archive_root)
    assert calls == 1


def test_readonly_uri_encodes_sqlite_metacharacters(tmp_path: Path) -> None:
    database = tmp_path / "archive?name#fragment.db"
    sqlite3.connect(database).close()

    uri = live_proof._readonly_uri(database)

    assert "%3F" in uri and "%23" in uri
    with sqlite3.connect(uri, uri=True) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)


def test_capture_rejects_archive_paths_unsafe_for_dependency_uris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive?name#fragment"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_CODE_SHA", "a" * 40)

    with pytest.raises(LiveProofError, match="SQLite URI query characters"):
        capture_live_proof_bindings(root)


def test_git_fallback_rejects_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_CODE_SHA", raising=False)
    monkeypatch.setattr(
        live_proof,
        "_run_git",
        lambda *_args: subprocess.CompletedProcess(args=(), returncode=0, stdout=" M live_proof.py\n", stderr=""),
    )

    with pytest.raises(LiveProofError, match="clean git worktree"):
        live_proof._code_sha()


def test_installed_code_identity_uses_version_info_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_CODE_SHA", raising=False)
    monkeypatch.setattr(VERSION_INFO, "commit", "B" * 40)
    monkeypatch.setattr(live_proof, "__file__", "/installed/polylogue/maintenance/live_proof.py")

    assert live_proof._code_sha() == "b" * 40


def test_receipt_write_removes_partial_output_after_os_error(
    archive_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = collect_live_proof(LiveProofId.ARCHIVE_VERIFICATION.value, archive_root)
    target = tmp_path / "new-receipt.json"

    def fail_write(_descriptor: int, _payload: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(os, "write", fail_write)
    with pytest.raises(LiveProofError, match="could not be written"):
        write_live_proof_receipt(target, receipt)

    assert not target.exists()
