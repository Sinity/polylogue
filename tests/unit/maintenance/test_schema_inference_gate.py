"""Production-route regression tests for the schema-inference gate."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

import polylogue.maintenance.schema_inference_gate as gate
from polylogue.maintenance.schema_inference_gate import (
    RECEIPT_FILENAME,
    SchemaInferenceGateError,
    run_schema_inference_gate,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _seed_archive(root: Path) -> Path:
    """Create one source raw whose actual external file and blob agree."""

    initialize_active_archive_root(root)
    ground_truth = root.parent / f"{root.name}-codex-ground-truth"
    ground_truth.mkdir()
    payload = b"actual external codex raw"
    source_file = ground_truth / "session.jsonl"
    source_file.write_bytes(payload)
    blob_hash, blob_size = BlobStore(root / "blob").write_from_bytes(payload)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, logical_source_key, revision_authority
            ) VALUES ('raw-1', 'codex-session', 'session', ?, ?, ?, 100,
                      'codex:session', 'byte_proven')
            """,
            (str(source_file), bytes.fromhex(blob_hash), blob_size),
        )
        conn.execute(
            """
            INSERT INTO raw_session_memberships(
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, decision, decided_at_ms
            ) VALUES ('raw-1', 'codex:session', 'session', 'rev-1', ?, 1, 'applied', 100)
            """,
            (b"m" * 32,),
        )
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions(native_id, origin, raw_id, content_hash, message_count)
            VALUES ('session', 'codex-session', 'raw-1', ?, 1)
            """,
            (b"s" * 32,),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, position, role, material_origin, content_hash)
            VALUES ('codex-session:session', 0, 'user', 'human_authored', ?)
            """,
            (b"n" * 32,),
        )
        conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, text)
            VALUES ('codex-session:session:0.0', 'codex-session:session', 0, 'text', 'hello')
            """
        )
        conn.execute("ANALYZE blocks")
        conn.execute("ANALYZE messages")
        conn.execute("ANALYZE action_pairs")
    return ground_truth


def _run(root: Path, tmp_path: Path, *, ground_truth: Path | None = None) -> dict[str, Any]:
    external_root = ground_truth or root.parent / f"{root.name}-codex-ground-truth"
    result = run_schema_inference_gate(
        root,
        receipt_path=tmp_path / RECEIPT_FILENAME,
        ground_truth_roots={"codex-session": (external_root,)},
    )
    return cast(dict[str, Any], result.payload)


def test_clean_archive_runs_actual_blobstore_verifier_and_external_reconciliation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    ground_truth = _seed_archive(root)
    before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("source.db", "index.db")}

    payload = _run(root, tmp_path, ground_truth=ground_truth)

    assert payload["verdict"] == "PASS"
    assert payload["ground_truth_inputs"]["origins"]["codex-session"]["external_files"] == 1
    assert payload["ground_truth_inputs"]["origins"]["codex-session"]["unverified_source_blob_hashes"] == 0
    full = payload["full_blob_hash_verification"]
    assert full["passed"] is True
    assert full["verifier"]["identity"] == "polylogue.storage.blob_store.BlobStore.verify_all"
    assert full["before_snapshot"]["digest"] == full["after_snapshot"]["digest"]
    assert {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("source.db", "index.db")
    } == before


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
