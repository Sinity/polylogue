"""Production-route regression tests for the schema-inference gate."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict, cast

import pytest

import polylogue.maintenance.schema_inference_gate as gate
from polylogue.maintenance.schema_inference_gate import (
    RECEIPT_FILENAME,
    SchemaInferenceGateError,
    run_schema_inference_gate,
    schema_inference_gate_receipt_digest,
    validate_schema_inference_gate_receipt,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import RebuildLease
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.schema_inference import seed_schema_inference_archive


class _RawGroundTruthProvenance(TypedDict):
    source_path: str
    matched_external_relative_path: str | None


class _OriginGroundTruthReceipt(TypedDict):
    external_files: int
    unverified_source_blob_hashes: int
    provenance: list[_RawGroundTruthProvenance]
    unmatched_external_files: list[object]
    cross_origin_mismatches: list[object]
    count_discrepancy: bool
    byte_discrepancy: bool


class _GroundTruthInputs(TypedDict):
    origins: dict[str, _OriginGroundTruthReceipt]


class _BlobVerifier(TypedDict):
    identity: str


class _BlobSnapshot(TypedDict):
    digest: str


class _BlobFailure(TypedDict):
    reason: str


class _FullBlobHashVerification(TypedDict):
    passed: bool
    verifier: _BlobVerifier
    before_snapshot: _BlobSnapshot
    after_snapshot: _BlobSnapshot
    failures: list[_BlobFailure]


class _GateQueryResult(TypedDict):
    passed: bool
    invalid_receipt_count: int
    resolved_by_rule: dict[str, int]


class _GateReceipt(TypedDict):
    verdict: str
    ground_truth_inputs: _GroundTruthInputs
    full_blob_hash_verification: _FullBlobHashVerification
    query_results: dict[str, _GateQueryResult]
    pass_fail_reasons: list[str]


_seed_archive = seed_schema_inference_archive


def _run(
    root: Path,
    tmp_path: Path,
    *,
    ground_truth: Path | None = None,
    ground_truth_roots: dict[str, tuple[Path, ...]] | None = None,
) -> _GateReceipt:
    external_root = ground_truth or root.parent / f"{root.name}-codex-ground-truth"
    receipt_count = sum(1 for child in tmp_path.iterdir() if child.name.startswith("schema-gate-receipt-"))
    result = run_schema_inference_gate(
        root,
        receipt_path=tmp_path / f"schema-gate-receipt-{receipt_count}" / RECEIPT_FILENAME,
        ground_truth_roots=({"codex-session": (external_root,)} if ground_truth_roots is None else ground_truth_roots),
    )
    return cast(_GateReceipt, result.payload)


def test_clean_archive_runs_actual_blobstore_verifier_and_external_reconciliation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("source.db", "index.db")}

    payload = _run(root, tmp_path, ground_truth=ground_truth)

    assert payload["verdict"] == "PASS"
    assert payload["ground_truth_inputs"]["origins"]["codex-session"]["external_files"] == 1
    assert payload["ground_truth_inputs"]["origins"]["codex-session"]["unverified_source_blob_hashes"] == 0
    provenance = payload["ground_truth_inputs"]["origins"]["codex-session"]["provenance"]
    expected_hash = hashlib.sha256(b"actual external codex raw").hexdigest()
    assert provenance == [
        {
            "raw_id": "raw-1",
            "origin": "codex-session",
            "native_id": "session",
            "logical_source_key": "codex:session",
            "source_path": str(ground_truth / "session.jsonl"),
            "blob_hash": expected_hash,
            "blob_size": len(b"actual external codex raw"),
            "matched_external_relative_path": "session.jsonl",
            "matched_external_hash": expected_hash,
            "matched_external_size": len(b"actual external codex raw"),
            "disposition": "materialized",
        }
    ]
    full = payload["full_blob_hash_verification"]
    assert full["passed"] is True
    assert full["verifier"]["identity"] == "polylogue.storage.blob_store.BlobStore.verify_all"
    assert full["before_snapshot"]["digest"] == full["after_snapshot"]["digest"]
    assert {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("source.db", "index.db")
    } == before


def test_readiness_accepts_equivalent_blob_evidence_from_another_verifier(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    receipt_path = tmp_path / "receipt" / RECEIPT_FILENAME
    run_schema_inference_gate(root, receipt_path=receipt_path, ground_truth_roots={"codex-session": (ground_truth,)})
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    recorded = dict(payload["full_blob_hash_verification"]["referenced_blob_integrity_snapshot"])
    recorded["verifier"] = "polylogue.storage.blob_store.BlobStore.verify"

    validated = gate.validate_schema_inference_receipt(
        root,
        receipt_path,
        verify_blob_integrity=True,
        verified_blob_integrity_snapshot=recorded,
    )

    assert validated["_verified_blob_integrity_snapshot"] == recorded


def test_empty_archive_cannot_authorize_schema_inference(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)

    payload = _run(root, tmp_path, ground_truth_roots={})

    assert payload["verdict"] == "FAIL"
    assert "schema inference requires at least one reconciled source raw" in payload["pass_fail_reasons"]


def test_gate_rejects_stale_non_source_tier_schema_identity(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute("PRAGMA user_version = 0")

    payload = _run(root, tmp_path, ground_truth=ground_truth)

    assert payload["verdict"] == "FAIL"
    assert any("user.db schema identity is stale" in reason for reason in payload["pass_fail_reasons"])


def test_gate_receipt_path_is_immutable(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    receipt = tmp_path / RECEIPT_FILENAME
    run_schema_inference_gate(root, receipt_path=receipt, ground_truth_roots={"codex-session": (ground_truth,)})

    with pytest.raises(SchemaInferenceGateError, match="immutable"):
        run_schema_inference_gate(root, receipt_path=receipt, ground_truth_roots={"codex-session": (ground_truth,)})


def test_authoritative_receipt_expires_from_its_signed_generation_time(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    receipt = tmp_path / RECEIPT_FILENAME
    payload = run_schema_inference_gate(
        root,
        receipt_path=receipt,
        ground_truth_roots={"codex-session": (ground_truth,)},
    ).payload
    generated_at = datetime.fromisoformat(str(payload["generated_at"])).astimezone(UTC)

    with pytest.raises(SchemaInferenceGateError, match="stale or from the future"):
        validate_schema_inference_gate_receipt(
            receipt,
            archive_root=root,
            now=generated_at + timedelta(hours=24, seconds=1),
        )


def test_gate_refuses_concurrent_writer_lease(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    with RebuildLease(root):
        with pytest.raises(SchemaInferenceGateError, match="exclusive offline archive ownership"):
            run_schema_inference_gate(
                root,
                receipt_path=tmp_path / RECEIPT_FILENAME,
                ground_truth_roots={"codex-session": (ground_truth,)},
            )


def test_receipt_target_can_never_replace_a_live_tier(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    source_before = hashlib.sha256((root / "source.db").read_bytes()).hexdigest()

    with pytest.raises(SchemaInferenceGateError, match="receipt filename"):
        run_schema_inference_gate(
            root,
            receipt_path=root / "source.db",
            ground_truth_roots={"codex-session": (ground_truth,)},
        )
    with pytest.raises(SchemaInferenceGateError, match="outside the archive root"):
        run_schema_inference_gate(
            root,
            receipt_path=root / RECEIPT_FILENAME,
            ground_truth_roots={"codex-session": (ground_truth,)},
        )

    assert hashlib.sha256((root / "source.db").read_bytes()).hexdigest() == source_before


def test_unavailable_or_unverified_external_ground_truth_is_a_hard_failure(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed_archive(root)

    unavailable = _run(root, tmp_path, ground_truth=tmp_path / "missing-root")
    assert unavailable["verdict"] == "FAIL"
    assert any("unavailable or undeclared" in reason for reason in unavailable["pass_fail_reasons"])

    wrong_root = tmp_path / "wrong-root"
    wrong_root.mkdir()
    (wrong_root / "session.jsonl").write_bytes(b"wrong bytes")
    unverified = _run(root, tmp_path, ground_truth=wrong_root)
    assert unverified["verdict"] == "FAIL"
    assert unverified["ground_truth_inputs"]["origins"]["codex-session"]["unverified_source_blob_hashes"] == 1


def test_extra_external_file_is_rejected_by_bidirectional_reconciliation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    (ground_truth / "extra-session.jsonl").write_bytes(b"external-only")

    payload = _run(root, tmp_path, ground_truth=ground_truth)

    origin = payload["ground_truth_inputs"]["origins"]["codex-session"]
    assert payload["verdict"] == "FAIL"
    assert origin["unmatched_external_files"] == [
        {
            "root_index": 0,
            "relative_path": "extra-session.jsonl",
            "hash": hashlib.sha256(b"external-only").hexdigest(),
            "size": len(b"external-only"),
            "disposition": "unmatched-external-file",
        }
    ]
    assert origin["count_discrepancy"] is True
    assert origin["byte_discrepancy"] is True
    assert any("external file(s) have no source raw match" in reason for reason in payload["pass_fail_reasons"])


def test_archive_owned_blob_cannot_pose_as_external_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed_archive(root)
    blob_path = next(path for path in (root / "blob").rglob("*") if path.is_file())

    payload = _run(root, tmp_path, ground_truth=blob_path)

    assert payload["verdict"] == "FAIL"
    assert any(
        "must be external to the archive and blob namespace" in reason for reason in payload["pass_fail_reasons"]
    )


def test_cross_origin_external_source_is_rejected_and_recorded(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    with sqlite3.connect(root / "source.db") as conn:
        blob_hash, blob_size = conn.execute(
            "SELECT blob_hash, blob_size FROM raw_sessions WHERE raw_id = 'raw-1'"
        ).fetchone()
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, logical_source_key, revision_authority
            ) VALUES ('raw-cross-origin', 'claude-code-session', 'session', ?, ?, ?, 101,
                      'claude:session', 'byte_proven')
            """,
            (str(ground_truth / "session.jsonl"), blob_hash, blob_size),
        )

    payload = _run(
        root,
        tmp_path,
        ground_truth_roots={
            "codex-session": (ground_truth,),
            "claude-code-session": (ground_truth,),
        },
    )

    claude = payload["ground_truth_inputs"]["origins"]["claude-code-session"]
    assert payload["verdict"] == "FAIL"
    assert claude["cross_origin_mismatches"] == [
        {
            "root_index": 0,
            "relative_path": "session.jsonl",
            "hash": hashlib.sha256(b"actual external codex raw").hexdigest(),
            "size": len(b"actual external codex raw"),
            "disposition": "cross-origin-source",
            "other_origins": ["codex-session"],
        }
    ]
    assert any("claimed by multiple origins" in reason for reason in payload["pass_fail_reasons"])


def test_stale_acquisition_path_is_preserved_while_content_matches_external_file(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    stale_path = "/retired/archive-root/session.jsonl"
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = ? WHERE raw_id = 'raw-1'", (stale_path,))

    payload = _run(root, tmp_path, ground_truth=ground_truth)
    provenance = payload["ground_truth_inputs"]["origins"]["codex-session"]["provenance"][0]

    assert payload["verdict"] == "PASS"
    assert provenance["source_path"] == stale_path
    assert provenance["matched_external_relative_path"] == "session.jsonl"
    with sqlite3.connect(root / "source.db") as conn:
        assert conn.execute("SELECT source_path FROM raw_sessions WHERE raw_id = 'raw-1'").fetchone()[0] == stale_path


def test_receipt_rerun_keeps_deterministic_external_provenance(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    first = cast(
        _GateReceipt,
        run_schema_inference_gate(
            root,
            receipt_path=tmp_path / "first" / RECEIPT_FILENAME,
            ground_truth_roots={"codex-session": (ground_truth,)},
        ).payload,
    )
    second = cast(
        _GateReceipt,
        run_schema_inference_gate(
            root,
            receipt_path=tmp_path / "second" / RECEIPT_FILENAME,
            ground_truth_roots={"codex-session": (ground_truth,)},
        ).payload,
    )

    first_origin = first["ground_truth_inputs"]["origins"]["codex-session"]
    second_origin = second["ground_truth_inputs"]["origins"]["codex-session"]
    assert first_origin["provenance"] == second_origin["provenance"]
    assert first_origin["unmatched_external_files"] == second_origin["unmatched_external_files"] == []


def test_full_blob_hash_evidence_is_not_a_caller_claim(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    with sqlite3.connect(root / "source.db") as conn:
        blob_hash = bytes(conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = 'raw-1'").fetchone()[0]).hex()
    BlobStore(root / "blob").blob_path(blob_hash).write_bytes(b"corrupted")

    payload = _run(root, tmp_path, ground_truth=ground_truth)

    assert payload["verdict"] == "FAIL"
    full = payload["full_blob_hash_verification"]
    assert full["passed"] is False
    assert full["verifier"]["identity"] == "polylogue.storage.blob_store.BlobStore.verify_all"
    assert any(finding["reason"] == "hash_mismatch" for finding in full["failures"])


def test_non_session_duplicate_requires_durable_content_bound_twin_receipt(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    with sqlite3.connect(root / "source.db") as conn:
        blob_hash, blob_size = conn.execute(
            "SELECT blob_hash, blob_size FROM raw_sessions WHERE raw_id = 'raw-1'"
        ).fetchone()
        conn.execute(
            """
            INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size,
                                     acquired_at_ms, logical_source_key, revision_authority)
            VALUES ('raw-excluded', 'codex-session', 'excluded', ?, ?, ?, 101,
                    'codex:excluded', 'byte_proven')
            """,
            (str(ground_truth / "session.jsonl"), blob_hash, blob_size),
        )
        conn.execute(
            """
            INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail)
            VALUES ('raw-excluded', 'fixture/1', 'non_session', 0, 101, 'fixture')
            """
        )

    bare_status = _run(root, tmp_path, ground_truth=ground_truth)
    assert bare_status["verdict"] == "FAIL"
    duplicate = bare_status["query_results"]["zero-unexplained-byte-duplicates"]
    assert duplicate["invalid_receipt_count"] == 1

    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_non_session_duplicate_exclusion_receipts(
                raw_id, blob_hash, blob_size, indexed_twin_raw_id, indexed_twin_session_id,
                parser_fingerprint, excluded_at_ms, tool_version
            ) VALUES ('raw-excluded', ?, ?, 'raw-1', 'codex-session:session', 'fixture/1', 101, 'fixture/1')
            """,
            (blob_hash, blob_size),
        )

    receipted = _run(root, tmp_path, ground_truth=ground_truth)
    assert receipted["verdict"] == "PASS"
    assert receipted["query_results"]["zero-unexplained-byte-duplicates"]["resolved_by_rule"] == {
        "legitimately-excluded-non-conversation": 1
    }


@pytest.mark.parametrize(
    ("mutation", "expected_gates"),
    (
        ("quarantine", {"zero-surviving-quarantine"}),
        ("orphan-quarantine", {"zero-surviving-quarantine", "zero-quarantine-without-logical-source-key"}),
        ("blocker", {"zero-unresolved-raw-authority-blockers"}),
    ),
)
def test_each_source_hard_gate_has_a_real_sqlite_red_twin(
    tmp_path: Path,
    mutation: str,
    expected_gates: set[str],
) -> None:
    root = tmp_path / mutation
    ground_truth = _seed_archive(root)
    with sqlite3.connect(root / "source.db") as conn:
        if mutation == "quarantine":
            conn.execute(
                "INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, logical_source_key, revision_authority) VALUES ('raw-quarantine', 'codex-session', 'q', ?, ?, 1, 101, 'codex:q', 'quarantined')",
                (str(ground_truth / "session.jsonl"), b"q" * 32),
            )
        elif mutation == "orphan-quarantine":
            conn.execute(
                "INSERT INTO raw_sessions(raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms, revision_authority) VALUES ('raw-orphan', 'codex-session', ?, ?, 1, 101, 'quarantined')",
                (str(ground_truth / "session.jsonl"), b"o" * 32),
            )
        else:
            conn.execute(
                "INSERT INTO raw_authority_blockers(blocker_id, plan_id, census_id, reason, expected_json, observed_json, created_at_ms) VALUES ('blocker-1', 'missing-plan', 'missing-census', 'fixture', '{}', '{}', 101)"
            )

    payload = _run(root, tmp_path, ground_truth=ground_truth)
    assert payload["verdict"] == "FAIL"
    results = payload["query_results"]
    assert {name for name in expected_gates if not results[name]["passed"]} == expected_gates


def test_untyped_fidelity_residual_fails_even_when_aggregate_looks_green(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "false-favorable"
    ground_truth = _seed_archive(root)

    class Report:
        def to_json(self) -> dict[str, object]:
            return {
                "checks": [
                    {
                        "name": "corpus-absences",
                        "status": "ok",
                        "evidence": {
                            "absent_total": 0,
                            "documents_known": 1,
                            "documents_present": 1,
                            "documents_known_by_origin": {"codex-session": 1},
                            "documents_present_by_origin": {"codex-session": 1},
                            "absent_by_origin_cause": {"codex-session/settled-yet-absent": 1},
                            "raws_without_attributable_identity": 0,
                        },
                    },
                    {"name": "corpus-attachment-fidelity", "status": "ok", "evidence": {"refs_unfetched": 0}},
                    {
                        "name": "corpus-revision-fidelity",
                        "status": "ok",
                        "evidence": {"unexplained_shortfall": 0, "unexplained_by_origin": {}, "worst": []},
                    },
                ]
            }

    monkeypatch.setattr(gate, "verify_archive", lambda *args, **kwargs: Report())
    payload = _run(root, tmp_path, ground_truth=ground_truth)
    assert payload["verdict"] == "FAIL"
    assert any("untyped corpus residual" in reason for reason in payload["pass_fail_reasons"])


def test_gate_receipt_digest_is_canonical_and_content_bound() -> None:
    first = {"verdict": "PASS", "schema": "polylogue.schema-inference-gate.v1", "nested": {"b": 2, "a": 1}}
    reordered = {"nested": {"a": 1, "b": 2}, "schema": first["schema"], "verdict": first["verdict"]}

    assert schema_inference_gate_receipt_digest(first) == schema_inference_gate_receipt_digest(reordered)
    altered = dict(first)
    altered["verdict"] = "FAIL"
    assert schema_inference_gate_receipt_digest(first) != schema_inference_gate_receipt_digest(altered)


def test_external_inventory_token_reuses_metadata_detector_and_rejects_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real receipt validator hashes once, then detects corpus drift cheaply.

    Anti-vacuity: removing the pass token binding makes the second validation
    hash the full corpus again, so the call-count assertions fail.
    """
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    receipt_path = tmp_path / "receipt" / RECEIPT_FILENAME
    run_schema_inference_gate(root, receipt_path=receipt_path, ground_truth_roots={"codex-session": (ground_truth,)})

    full_inventory_calls = 0
    original_inventory = gate._external_inventory

    def counted_inventory(roots: object) -> object:
        nonlocal full_inventory_calls
        full_inventory_calls += 1
        return original_inventory(roots)  # type: ignore[arg-type]

    monkeypatch.setattr(gate, "_external_inventory", counted_inventory)
    first = gate.validate_schema_inference_receipt(root, receipt_path)
    token = cast(dict[str, object], first["external_ground_truth_inventory_token"])
    assert full_inventory_calls == 1, "the first validation establishes the authoritative full inventory"
    gate.validate_schema_inference_receipt(root, receipt_path, inventory_token=token)
    assert full_inventory_calls == 1, "unchanged pass validation must reuse the bound detector token"

    (ground_truth / "session.jsonl").write_bytes(b"changed external codex raw")
    with pytest.raises(SchemaInferenceGateError, match="external ground-truth corpus changed"):
        gate.validate_schema_inference_receipt(root, receipt_path, inventory_token=token)
    assert full_inventory_calls == 2, "a changed detector must trigger one authoritative inventory recalculation"


def test_inventory_change_detector_triggers_rehash_without_changing_content_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    receipt_path = tmp_path / "receipt" / RECEIPT_FILENAME
    run_schema_inference_gate(root, receipt_path=receipt_path, ground_truth_roots={"codex-session": (ground_truth,)})
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    origins = payload["ground_truth_inputs"]["origins"]
    baseline = gate._canonical_external_ground_truth_digest(origins)
    altered = json.loads(json.dumps(origins))
    altered["codex-session"]["inventory_change_detector"] = {"forced": "changed"}
    assert gate._canonical_external_ground_truth_digest(altered) == baseline

    validated = gate.validate_schema_inference_receipt(root, receipt_path)
    token = cast(dict[str, object], validated["external_ground_truth_inventory_token"])
    (ground_truth / "session.jsonl").touch()
    inventory_calls = 0
    original_inventory = gate._external_inventory

    def counted_inventory(roots: tuple[Path, ...]) -> list[object]:
        nonlocal inventory_calls
        inventory_calls += 1
        return original_inventory(roots)  # type: ignore[return-value]

    monkeypatch.setattr(gate, "_external_inventory", counted_inventory)
    refreshed = gate.validate_schema_inference_receipt(root, receipt_path, inventory_token=token)
    assert inventory_calls == 1
    refreshed_token = cast(dict[str, object], refreshed["external_ground_truth_inventory_token"])
    original_origin_token = cast(dict[str, object], token["origins"])["codex-session"]
    refreshed_origin_token = cast(dict[str, object], refreshed_token["origins"])["codex-session"]
    assert (
        cast(dict[str, object], refreshed_origin_token)["inventory_change_detector"]
        != cast(dict[str, object], original_origin_token)["inventory_change_detector"]
    )
    gate.validate_schema_inference_receipt(root, receipt_path, inventory_token=refreshed_token)
    assert inventory_calls == 1, "a successful rehash must refresh the detector token"


def test_inventory_token_rejects_foreign_nonce_and_empty_token_falls_back_to_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    receipt_path = tmp_path / "receipt" / RECEIPT_FILENAME
    run_schema_inference_gate(root, receipt_path=receipt_path, ground_truth_roots={"codex-session": (ground_truth,)})
    validated = gate.validate_schema_inference_receipt(root, receipt_path)
    token = cast(dict[str, object], validated["external_ground_truth_inventory_token"])
    foreign = json.loads(json.dumps(token))
    foreign["receipt_nonce"] = "foreign-pass"
    with pytest.raises(SchemaInferenceGateError, match="token is not bound"):
        gate.validate_schema_inference_receipt(root, receipt_path, inventory_token=foreign)
    gate.validate_schema_inference_receipt(root, receipt_path, inventory_token={})
