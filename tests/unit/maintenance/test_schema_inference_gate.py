"""Acceptance and anti-vacuity tests for the schema-inference gate."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

import polylogue.maintenance.schema_inference_gate as gate
from polylogue.maintenance.schema_inference_gate import (
    BLOB_HASH_RECEIPT_SCHEMA,
    run_schema_inference_gate,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _seed_archive(root: Path) -> None:
    initialize_active_archive_root(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, logical_source_key, revision_authority
            ) VALUES ('raw-1', 'codex-session', 'session', '/fixture/session.jsonl', ?, 10, 100,
                      'codex:session', 'byte_proven')
            """,
            (b"a" * 32,),
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


def _write_hash_receipt(root: Path, path: Path, *, mode: str = "full", verdict: str = "PASS") -> None:
    with sqlite3.connect(root / "source.db") as conn:
        source_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        source_count = int(conn.execute("SELECT COUNT(DISTINCT blob_hash) FROM raw_sessions").fetchone()[0])
    path.write_text(
        json.dumps(
            {
                "schema": BLOB_HASH_RECEIPT_SCHEMA,
                "archive_root": str(root.absolute()),
                "source_schema_version": source_version,
                "mode": mode,
                "verdict": verdict,
                "tool_version": "fixture-hash-verifier/1",
                "input_paths": {"source_db": str(root / "source.db"), "blob_root": str(root / "blob")},
                "counts": {
                    "scanned_blobs": source_count,
                    "total_blobs_seen": source_count,
                    "scanned_references": source_count,
                    "total_references_seen": source_count,
                },
                "findings": [],
            }
        ),
        encoding="utf-8",
    )


def _run(root: Path, tmp_path: Path, *, hash_receipt: Path | None = None) -> dict[str, Any]:
    hash_path = hash_receipt or tmp_path / "blob-hash.json"
    if hash_receipt is None:
        _write_hash_receipt(root, hash_path)
    result = run_schema_inference_gate(
        root,
        blob_hash_receipt=hash_path,
        receipt_path=tmp_path / "schema-inference-gate.json",
    )
    return cast(dict[str, Any], result.payload)


def test_clean_archive_writes_pass_receipt_and_does_not_mutate_sqlite(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _seed_archive(root)
    hash_receipt = tmp_path / "blob-hash.json"
    _write_hash_receipt(root, hash_receipt)
    before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("source.db", "index.db")}

    payload = _run(root, tmp_path, hash_receipt=hash_receipt)

    assert payload["verdict"] == "PASS"
    assert payload["schema"] == "polylogue.schema-inference-gate.v1"
    assert (
        payload["source_schema_identity"]["actual_user_version"]
        == payload["source_schema_identity"]["expected_user_version"]
    )
    assert payload["ground_truth_inputs"]["hooks"]["exempt"] is True
    assert payload["ground_truth_inputs"]["browser-capture"]["exempt"] is True
    assert payload["ground_truth_denominators"]["documents_known_by_origin"] == {"codex-session": 1}
    assert payload["ground_truth_denominators"]["documents_present_by_origin"] == {"codex-session": 1}
    assert payload["full_blob_hash_verification"]["passed"] is True
    assert payload["tool_versions"]["polylogue"]
    assert {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("source.db", "index.db")
    } == before


@pytest.mark.parametrize(
    ("mutation", "expected_gates"),
    (
        ("quarantine", {"zero-surviving-quarantine"}),
        ("orphan-quarantine", {"zero-surviving-quarantine", "zero-quarantine-without-logical-source-key"}),
        ("blocker", {"zero-unresolved-raw-authority-blockers"}),
        ("duplicate", {"zero-unexplained-byte-duplicates"}),
    ),
)
def test_each_source_hard_gate_has_a_red_twin(
    tmp_path: Path,
    mutation: str,
    expected_gates: set[str],
) -> None:
    root = tmp_path / mutation
    _seed_archive(root)
    with sqlite3.connect(root / "source.db") as conn:
        if mutation == "quarantine":
            conn.execute(
                "INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, logical_source_key, revision_authority) "
                "VALUES ('raw-quarantine', 'codex-session', 'q', '/q', ?, 1, 101, 'codex:q', 'quarantined')",
                (b"q" * 32,),
            )
        elif mutation == "orphan-quarantine":
            conn.execute(
                "INSERT INTO raw_sessions(raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms, revision_authority) "
                "VALUES ('raw-orphan-quarantine', 'codex-session', '/q', ?, 1, 101, 'quarantined')",
                (b"o" * 32,),
            )
        elif mutation == "blocker":
            conn.execute(
                "INSERT INTO raw_authority_blockers(blocker_id, plan_id, census_id, reason, expected_json, observed_json, created_at_ms) "
                "VALUES ('blocker-1', 'missing-plan', 'missing-census', 'fixture', '{}', '{}', 101)"
            )
        else:
            conn.execute(
                "INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, logical_source_key, revision_authority) "
                "VALUES ('raw-duplicate', 'codex-session', 'duplicate', '/duplicate', ?, 10, 101, 'codex:duplicate', 'byte_proven')",
                (b"a" * 32,),
            )

    payload = _run(root, tmp_path)
    results = payload["query_results"]
    assert payload["verdict"] == "FAIL"
    assert {name for name in expected_gates if not results[name]["passed"]} == expected_gates
    for name in expected_gates:
        assert results[name]["count"] > 0


def test_valid_supersession_receipt_and_typed_exclusion_are_explicit_duplicate_rules(tmp_path: Path) -> None:
    root = tmp_path / "resolved-duplicates"
    _seed_archive(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            "INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, logical_source_key, revision_authority) "
            "VALUES ('raw-receipted', 'codex-session', 'duplicate', '/duplicate', ?, 10, 101, 'codex:duplicate', 'byte_proven')",
            (b"a" * 32,),
        )
        conn.execute(
            "INSERT INTO raw_byte_duplicate_supersession_receipts(raw_id, blob_hash, blob_size, duplicate_of_raw_id, duplicate_of_session_id, previous_revision_authority, promoted_at_ms, tool_version, backup_manifest_path) "
            "VALUES ('raw-receipted', ?, 10, 'raw-1', 'codex-session:session', 'byte_proven', 101, 'fixture/1', '/fixture/backup.json')",
            (b"a" * 32,),
        )
        conn.execute(
            "INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, logical_source_key, revision_authority) "
            "VALUES ('raw-excluded', 'codex-session', 'excluded', '/excluded', ?, 10, 102, 'codex:excluded', 'byte_proven')",
            (b"a" * 32,),
        )
        conn.execute(
            "INSERT INTO raw_membership_census(raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail) "
            "VALUES ('raw-excluded', 'fixture/1', 'non_session', 0, 102, 'non-conversation fixture')"
        )

    payload = _run(root, tmp_path)

    assert payload["verdict"] == "PASS"
    duplicate = payload["query_results"]["zero-unexplained-byte-duplicates"]
    assert duplicate["resolved_by_rule"] == {
        "legitimately-excluded-non-conversation": 1,
        "superseded-duplicate": 1,
    }


def test_untyped_fidelity_residual_fails_even_when_existing_aggregate_is_false_green(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "false-favorable"
    _seed_archive(root)

    class _Report:
        def to_json(self) -> dict[str, object]:
            return {
                "archive_root": str(root),
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
                ],
            }

    monkeypatch.setattr(gate, "verify_archive", lambda *args, **kwargs: _Report())
    payload = _run(root, tmp_path)

    assert payload["verdict"] == "FAIL"
    assert any("aggregate" in reason or "untyped corpus residual" in reason for reason in payload["pass_fail_reasons"])


def test_missing_or_partial_hash_receipt_can_never_become_a_pass(tmp_path: Path) -> None:
    root = tmp_path / "hash-required"
    _seed_archive(root)
    missing = run_schema_inference_gate(
        root,
        blob_hash_receipt=tmp_path / "missing.json",
        receipt_path=tmp_path / "missing-gate.json",
    )
    assert missing.payload["verdict"] == "FAIL"
    missing_hash = missing.payload["full_blob_hash_verification"]
    assert isinstance(missing_hash, dict)
    assert missing_hash["passed"] is False

    partial = tmp_path / "partial.json"
    _write_hash_receipt(root, partial, mode="sample", verdict="PASS")
    sampled = run_schema_inference_gate(
        root,
        blob_hash_receipt=partial,
        receipt_path=tmp_path / "partial-gate.json",
    )
    assert sampled.payload["verdict"] == "FAIL"
    sampled_hash = sampled.payload["full_blob_hash_verification"]
    assert isinstance(sampled_hash, dict)
    assert "mode must be 'full'" in sampled_hash["reason"]

    uncovered = tmp_path / "uncovered.json"
    _write_hash_receipt(root, uncovered)
    uncovered_payload = json.loads(uncovered.read_text(encoding="utf-8"))
    uncovered_payload["counts"] = {
        "scanned_blobs": 0,
        "total_blobs_seen": 0,
        "scanned_references": 0,
        "total_references_seen": 0,
    }
    uncovered.write_text(json.dumps(uncovered_payload), encoding="utf-8")
    uncovered_result = run_schema_inference_gate(
        root,
        blob_hash_receipt=uncovered,
        receipt_path=tmp_path / "uncovered-gate.json",
    )
    assert uncovered_result.payload["verdict"] == "FAIL"
    uncovered_hash = uncovered_result.payload["full_blob_hash_verification"]
    assert isinstance(uncovered_hash, dict)
    assert "does not match the source-tier" in uncovered_hash["reason"]


def test_unresolved_duplicate_with_invalid_receipt_is_not_hidden_by_receipt_presence(tmp_path: Path) -> None:
    root = tmp_path / "invalid-duplicate-receipt"
    _seed_archive(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            "INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, logical_source_key, revision_authority) "
            "VALUES ('raw-invalid-receipt', 'codex-session', 'duplicate', '/duplicate', ?, 10, 101, 'codex:duplicate', 'byte_proven')",
            (b"a" * 32,),
        )
        conn.execute(
            "INSERT INTO raw_sessions(raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, logical_source_key, revision_authority) "
            "VALUES ('raw-unrelated', 'codex-session', 'unrelated', '/unrelated', ?, 10, 101, 'codex:unrelated', 'byte_proven')",
            (b"b" * 32,),
        )
        conn.execute(
            "INSERT INTO raw_byte_duplicate_supersession_receipts(raw_id, blob_hash, blob_size, duplicate_of_raw_id, duplicate_of_session_id, previous_revision_authority, promoted_at_ms, tool_version, backup_manifest_path) "
            "VALUES ('raw-invalid-receipt', ?, 10, 'raw-unrelated', 'codex-session:unrelated', 'byte_proven', 101, 'fixture/1', '/fixture/backup.json')",
            (b"a" * 32,),
        )
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(
            "INSERT INTO sessions(native_id, origin, raw_id, content_hash, message_count) "
            "VALUES ('unrelated', 'codex-session', 'raw-unrelated', ?, 1)",
            (b"u" * 32,),
        )

    payload = _run(root, tmp_path)

    assert payload["verdict"] == "FAIL"
    duplicate = payload["query_results"]["zero-unexplained-byte-duplicates"]
    assert duplicate["invalid_receipt_count"] == 1
