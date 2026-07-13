from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.revision_backfill import _parse_one
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import RebuildLease
from polylogue.storage.repair import repair_quarantined_accepted_raws
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    RevisionApplicationReceipt,
    record_revision_application_sync,
)


def _chatgpt_session(native_id: str, text: str) -> dict[str, object]:
    return {
        "id": native_id,
        "conversation_id": native_id,
        "title": native_id,
        "create_time": 1_700_000_000,
        "update_time": 1_700_000_001,
        "current_node": "node-1",
        "mapping": {
            "node-1": {
                "id": "node-1",
                "parent": None,
                "children": [],
                "message": {
                    "id": "message-1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [text]},
                    "create_time": 1_700_000_000,
                },
            }
        },
    }


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")


def _seed_invalid_head(
    root: Path,
    native_id: str = "repair-one",
    *,
    multi_session: bool = False,
    typed_quarantined: bool = True,
) -> str:
    initialize_active_archive_root(root)
    records = [_chatgpt_session(native_id, "proof text")]
    if multi_session:
        records.append(_chatgpt_session(f"{native_id}-other", "other text"))
    payload = json.dumps(records if multi_session else records[0], sort_keys=True).encode()
    parsed = _parse_one(Provider.CHATGPT, payload, f"{native_id}.json")
    assert parsed
    session = parsed[0]
    source_revision = hashlib.sha256(payload).hexdigest()
    content_hash = bytes.fromhex(session_content_hash(session))
    logical_source_key = f"chatgpt:{native_id}"
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path=f"{native_id}.json",
            acquired_at_ms=1,
        )
        _raw_id, session_id = archive.write_parsed_for_retained_raw(
            session,
            raw_id=raw_id,
            source_path=f"{native_id}.json",
            acquired_at_ms=1,
            revision_authoritative=True,
        )
        assert session_id == f"chatgpt-export:{native_id}"
        decided_at_ms = 2
        record_revision_application_sync(
            archive._conn,
            RevisionApplicationReceipt(
                raw_id=raw_id,
                session_id=session_id,
                logical_source_key=logical_source_key,
                source_revision=source_revision,
                acquisition_generation=0,
                decision=ApplicationDecision.SELECTED_BASELINE,
                accepted_raw_id=raw_id,
                accepted_source_revision=source_revision,
                accepted_content_hash=content_hash,
                accepted_frontier_kind="byte",
                accepted_frontier=len(payload),
                baseline_raw_id=raw_id,
                detail="newest unique byte-proven full baseline",
            ),
            decided_at_ms=decided_at_ms,
        )
        archive.commit()
    with sqlite3.connect(root / "source.db") as source:
        if typed_quarantined:
            source.execute(
                """
                UPDATE raw_sessions
                SET logical_source_key = ?, revision_kind = 'full', source_revision = ?,
                    baseline_raw_id = NULL, acquisition_generation = 0,
                    revision_authority = 'quarantined'
                WHERE raw_id = ?
                """,
                (logical_source_key, source_revision, raw_id),
            )
        source.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 'quarantined')
            """,
            (raw_id, logical_source_key, native_id, content_hash.hex(), content_hash, len(session.messages)),
        )
        source.execute(
            """
            INSERT INTO raw_membership_census (
                raw_id, parser_fingerprint, status, member_count, censused_at_ms
            ) VALUES (?, 'revision-membership-v1', 'complete', 1, 0)
            """,
            (raw_id,),
        )
        source.commit()
    return raw_id


def _logical_state(root: Path, raw_id: str) -> dict[str, list[tuple[object, ...]]]:
    result: dict[str, list[tuple[object, ...]]] = {}
    for tier, tables in {
        "source": ("raw_sessions", "blob_refs", "raw_artifacts", "raw_session_memberships", "raw_membership_census"),
        "index": ("sessions", "messages", "blocks", "raw_revision_heads", "raw_revision_applications"),
    }.items():
        with sqlite3.connect(root / f"{tier}.db") as conn:
            for table in tables:
                rows = conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                result[f"{tier}.{table}"] = [tuple(row) for row in rows]
    result["raw_id"] = [(raw_id,)]
    return result


def _raw_session_row(root: Path, raw_id: str) -> dict[str, object]:
    with sqlite3.connect(root / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
    assert row is not None
    return dict(row)


def _retarget_fixture_raw_id(root: Path, old_raw_id: str, new_raw_id: str) -> None:
    with sqlite3.connect(root / "source.db") as source:
        source.execute("PRAGMA foreign_keys = OFF")
        source.execute("UPDATE raw_sessions SET raw_id = ? WHERE raw_id = ?", (new_raw_id, old_raw_id))
        source.execute("UPDATE blob_refs SET ref_id = ? WHERE ref_id = ?", (new_raw_id, old_raw_id))
        source.execute("UPDATE raw_session_memberships SET raw_id = ? WHERE raw_id = ?", (new_raw_id, old_raw_id))
        source.execute("UPDATE raw_membership_census SET raw_id = ? WHERE raw_id = ?", (new_raw_id, old_raw_id))
        source.execute("UPDATE raw_artifacts SET raw_id = ? WHERE raw_id = ?", (new_raw_id, old_raw_id))
        source.commit()
    with sqlite3.connect(root / "index.db") as index:
        index.execute("PRAGMA foreign_keys = OFF")
        index.execute("UPDATE sessions SET raw_id = ? WHERE raw_id = ?", (new_raw_id, old_raw_id))
        index.execute(
            "UPDATE raw_revision_heads SET accepted_raw_id = ? WHERE accepted_raw_id = ?",
            (new_raw_id, old_raw_id),
        )
        index.execute(
            """
            UPDATE raw_revision_applications
            SET raw_id = ?, accepted_raw_id = ?, baseline_raw_id = ?
            WHERE raw_id = ?
            """,
            (new_raw_id, new_raw_id, new_raw_id, old_raw_id),
        )
        index.commit()


@pytest.mark.parametrize("typed_quarantined", [False, True])
def test_quarantined_accepted_raw_repair_roundtrip_is_receipted_and_idempotent(
    tmp_path: Path, typed_quarantined: bool
) -> None:
    raw_id = _seed_invalid_head(tmp_path, typed_quarantined=typed_quarantined)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            """
            INSERT INTO raw_artifacts (
                artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                support_status, classification_reason, parse_as_session, schema_eligible,
                malformed_jsonl_lines, first_observed_at_ms, last_observed_at_ms
            ) VALUES ('artifact-witness', ?, 'chatgpt-export', 'repair-one.json', 0,
                      'session', 'supported_parseable', 'witness', 1, 1, 0, 1, 2)
            """,
            (raw_id,),
        )
    before = _logical_state(tmp_path, raw_id)
    before_raw = _raw_session_row(tmp_path, raw_id)

    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])

    assert dry_run.eligible_count == 1, dry_run.items[0].reason
    assert dry_run.items[0].proof_digest
    assert dry_run.items[0].application_decision_id
    assert dry_run.items[0].origin == "chatgpt-export"
    assert dry_run.items[0].source_index == 0
    assert dry_run.items[0].blob_hash == dry_run.items[0].accepted_source_revision
    assert dry_run.items[0].blob_ref_hash == dry_run.items[0].blob_hash
    assert dry_run.items[0].accepted_frontier_kind == "byte"
    assert dry_run.items[0].accepted_frontier == dry_run.items[0].blob_size
    assert dry_run.items[0].head_decided_at_ms == 2
    assert [artifact.artifact_id for artifact in dry_run.items[0].artifact_witnesses] == ["artifact-witness"]
    assert dry_run.items[0].application_witness is not None
    assert dry_run.items[0].application_witness.detail == "newest unique byte-proven full baseline"
    assert _logical_state(tmp_path, raw_id) == before

    receipt_path = tmp_path / "recovery" / "quarantined-raw-repair.jsonl"
    receipt_path.parent.mkdir()
    applied = repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt_path,
        proof_digest=dry_run.proof_digest,
    )

    assert applied.repaired_count == 1
    assert applied.items[0].status == "already_repaired"
    records = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    assert [record["state"] for record in records] == ["planned", "applied"]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        envelope = conn.execute(
            """
            SELECT logical_source_key, revision_kind, source_revision, baseline_raw_id,
                   acquisition_generation, revision_authority
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchone()
    assert envelope == (
        "chatgpt:repair-one",
        "full",
        dry_run.items[0].accepted_source_revision,
        raw_id,
        0,
        "byte_proven",
    )
    after = _logical_state(tmp_path, raw_id)
    for key in before:
        if key != "source.raw_sessions":
            assert after[key] == before[key]
    after_raw = _raw_session_row(tmp_path, raw_id)
    expected_updates = {
        "logical_source_key": "chatgpt:repair-one",
        "revision_kind": "full",
        "source_revision": dry_run.items[0].accepted_source_revision,
        "baseline_raw_id": raw_id,
        "acquisition_generation": 0,
        "revision_authority": "byte_proven",
    }
    if typed_quarantined:
        expected_updates = {"baseline_raw_id": raw_id, "revision_authority": "byte_proven"}
    assert {key: after_raw[key] for key in expected_updates} == expected_updates
    assert {key for key in before_raw if before_raw[key] != after_raw[key]} == set(expected_updates)

    reapplied = repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt_path,
        proof_digest=dry_run.proof_digest,
    )
    assert reapplied.repaired_count == 0
    assert receipt_path.read_text().count("\n") == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "head_raw",
        "missing_blob",
        "blob_ref",
        "artifact",
        "frontier",
        "session_hash",
        "origin",
        "capture_mode",
        "application",
        "receipt_time",
        "typed_competitor",
        "second_indexed_session",
        "membership",
        "envelope",
        "multi_session",
    ],
)
def test_quarantined_accepted_raw_repair_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    raw_id = _seed_invalid_head(tmp_path, multi_session=mutation == "multi_session")
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "index.db") as index:
        if mutation == "head_raw":
            index.execute("UPDATE raw_revision_heads SET accepted_raw_id = ?", ("0" * 64,))
        elif mutation == "missing_blob":
            blob_hash = source.execute(
                "SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (raw_id,)
            ).fetchone()[0]
            BlobStore(tmp_path / "blob").blob_path(str(blob_hash).lower()).unlink()
        elif mutation == "blob_ref":
            source.execute("UPDATE blob_refs SET size_bytes = size_bytes + 1 WHERE ref_id = ?", (raw_id,))
        elif mutation == "artifact":
            source.execute(
                """
                INSERT INTO raw_artifacts (
                    artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                    support_status, classification_reason, parse_as_session, schema_eligible,
                    malformed_jsonl_lines, first_observed_at_ms, last_observed_at_ms
                ) VALUES ('artifact', ?, 'chatgpt-export', 'wrong.json', 0, 'session',
                          'supported_parseable', 'test', 1, 1, 0, 1, 1)
                """,
                (raw_id,),
            )
        elif mutation == "frontier":
            index.execute("UPDATE raw_revision_heads SET accepted_frontier = accepted_frontier + 1")
        elif mutation == "session_hash":
            index.execute("UPDATE sessions SET content_hash = zeroblob(32)")
        elif mutation == "origin":
            source.execute("UPDATE raw_sessions SET origin = 'claude-ai-export' WHERE raw_id = ?", (raw_id,))
        elif mutation == "capture_mode":
            source.execute(
                "UPDATE raw_sessions SET origin = 'aistudio-drive', capture_mode = NULL WHERE raw_id = ?", (raw_id,)
            )
        elif mutation == "application":
            index.execute(
                """
                INSERT INTO raw_revision_applications (
                    decision_id, raw_id, session_id, logical_source_key, source_revision,
                    acquisition_generation, decision, detail, decided_at_ms
                ) SELECT 'competing', raw_id, session_id, logical_source_key, source_revision,
                         acquisition_generation, 'ambiguous', 'test', decided_at_ms + 1
                  FROM raw_revision_applications LIMIT 1
                """
            )
        elif mutation == "receipt_time":
            index.execute("UPDATE raw_revision_applications SET decided_at_ms = decided_at_ms + 1")
        elif mutation == "typed_competitor":
            source.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms,
                    logical_source_key, revision_kind, source_revision, baseline_raw_id,
                    acquisition_generation, revision_authority
                ) SELECT ?, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms,
                         'chatgpt:repair-one', 'full', ?, ?, 1, 'byte_proven'
                  FROM raw_sessions WHERE raw_id = ?
                """,
                ("1" * 64, "2" * 64, "1" * 64, raw_id),
            )
        elif mutation == "second_indexed_session":
            index.execute(
                """
                INSERT INTO sessions (native_id, origin, raw_id, content_hash)
                SELECT 'unexpected-second-session', origin, raw_id, content_hash
                FROM sessions LIMIT 1
                """
            )
        elif mutation == "membership":
            source.execute(
                "UPDATE raw_membership_census SET status = 'failed', member_count = 0, detail = 'ambiguous' WHERE raw_id = ?",
                (raw_id,),
            )
        elif mutation == "envelope":
            source.execute("UPDATE raw_sessions SET logical_source_key = 'partial' WHERE raw_id = ?", (raw_id,))
        source.commit()
        index.commit()
    before = _logical_state(tmp_path, raw_id)

    report = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert _logical_state(tmp_path, raw_id) == before


def test_quarantined_accepted_raw_repair_preserves_parallel_provenance_context(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "index.db") as index:
        source.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms,
                logical_source_key, revision_kind, source_revision,
                acquisition_generation, revision_authority
            ) SELECT ?, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms,
                     logical_source_key, 'full', ?, 0, 'quarantined'
              FROM raw_sessions WHERE raw_id = ?
            """,
            ("1" * 64, "2" * 64, raw_id),
        )
        index.execute(
            """
            INSERT INTO raw_revision_heads (
                logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                acquisition_generation, append_end_offset, decided_at_ms
            ) SELECT 'chatgpt:parallel-provenance', session_id, ?, ?,
                     accepted_content_hash, 'semantic', 1, 0, NULL, decided_at_ms - 1
              FROM raw_revision_heads LIMIT 1
            """,
            ("3" * 64, "4" * 64),
        )
        source.commit()
        index.commit()

    report = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])

    assert report.eligible_count == 1, report.items[0].reason
    assert report.items[0].parallel_session_head_count == 1
    assert report.items[0].quarantined_sibling_raw_count == 1
    assert report.items[0].authority_context_digest


def test_quarantined_accepted_raw_repair_source_v7_requires_injective_origin(tmp_path: Path) -> None:
    injective_raw = _seed_invalid_head(tmp_path, "injective")
    noninjective_raw = _seed_invalid_head(tmp_path, "noninjective")
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("ALTER TABLE raw_sessions DROP COLUMN capture_mode")
        source.execute("PRAGMA user_version = 7")
        source.execute("UPDATE raw_sessions SET origin = 'aistudio-drive' WHERE raw_id = ?", (noninjective_raw,))
        source.commit()

    eligible = repair_quarantined_accepted_raws(_config(tmp_path), [injective_raw])
    refused = repair_quarantined_accepted_raws(_config(tmp_path), [noninjective_raw])

    assert eligible.eligible_count == 1, eligible.items[0].reason
    assert eligible.items[0].capture_mode is None
    assert refused.ineligible_count == 1
    assert refused.items[0].reason == "source-v7 origin is not injective without capture-mode authority"


def test_quarantined_accepted_raw_repair_stages_source_v7_census_before_target_cas(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path, "staged")
    sibling_raw_id = "e" * 64
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, file_mtime_ms, logical_source_key, revision_kind, source_revision,
                acquisition_generation, revision_authority
            ) SELECT ?, origin, native_id, source_path, source_index, blob_hash, blob_size,
                     acquired_at_ms, file_mtime_ms, logical_source_key, revision_kind, source_revision,
                     acquisition_generation, revision_authority
              FROM raw_sessions WHERE raw_id = ?
            """,
            (sibling_raw_id, raw_id),
        )
        source.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            SELECT blob_hash, ?, ref_type, source_path, size_bytes, acquired_at_ms
              FROM blob_refs WHERE ref_id = ?
            """,
            (sibling_raw_id, raw_id),
        )
        source.execute("DELETE FROM raw_session_memberships WHERE raw_id = ?", (raw_id,))
        source.execute("DELETE FROM raw_membership_census WHERE raw_id = ?", (raw_id,))
        source.execute("ALTER TABLE raw_sessions DROP COLUMN capture_mode")
        source.execute("PRAGMA user_version = 7")
        source.commit()

    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])

    assert dry_run.eligible_count == 1, dry_run.items[0].reason
    assert dry_run.items[0].census_stage_raw_ids == tuple(sorted((raw_id, sibling_raw_id)))
    receipt = tmp_path / "source-v7-stage.jsonl"
    applied = repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )

    assert applied.repaired_count == 1
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT revision_authority, baseline_raw_id FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone() == ("byte_proven", raw_id)
        assert source.execute(
            "SELECT revision_authority, baseline_raw_id FROM raw_sessions WHERE raw_id = ?", (sibling_raw_id,)
        ).fetchone() == ("quarantined", None)
        assert source.execute(
            "SELECT COUNT(*) FROM raw_session_memberships WHERE raw_id IN (?, ?)", (raw_id, sibling_raw_id)
        ).fetchone() == (2,)
        assert source.execute(
            "SELECT COUNT(*) FROM raw_membership_census WHERE raw_id IN (?, ?)", (raw_id, sibling_raw_id)
        ).fetchone() == (2,)


def test_quarantined_accepted_raw_repair_refuses_source_v7_census_staging_with_mixed_cohort(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path, "mixed")
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("DELETE FROM raw_session_memberships WHERE raw_id = ?", (raw_id,))
        source.execute("DELETE FROM raw_membership_census WHERE raw_id = ?", (raw_id,))
        source.execute("UPDATE raw_sessions SET origin = 'aistudio-drive' WHERE raw_id = ?", (raw_id,))
        source.execute("ALTER TABLE raw_sessions DROP COLUMN capture_mode")
        source.commit()

    report = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert report.items[0].reason == "source-v7 origin is not injective without capture-mode authority"


def test_quarantined_accepted_raw_repair_rejects_duplicates_and_rolls_back_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _seed_invalid_head(tmp_path, "first")
    second = _seed_invalid_head(tmp_path, "second")
    with pytest.raises(ValueError, match="duplicate"):
        repair_quarantined_accepted_raws(_config(tmp_path), [first, first])
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [first, second])
    before = _logical_state(tmp_path, first)
    receipt = tmp_path / "rollback-receipt.jsonl"

    from polylogue.storage import repair as repair_module

    original = repair_module._inspect_quarantined_accepted_raw
    calls = 0

    def fail_postproof(archive_root: Path, raw_id: str, *, conn: sqlite3.Connection) -> Any:
        nonlocal calls
        calls += 1
        item = original(archive_root, raw_id, conn=conn)
        if calls == 5:
            return repair_module._quarantined_raw_item(item.raw_id, "injected post-proof failure")
        return item

    monkeypatch.setattr(repair_module, "_inspect_quarantined_accepted_raw", fail_postproof)
    with pytest.raises(RuntimeError, match="terminal state"):
        repair_quarantined_accepted_raws(
            _config(tmp_path),
            [first, second],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )

    assert _logical_state(tmp_path, first) == before
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned"]


def test_quarantined_accepted_raw_repair_resumes_planned_receipt_after_committed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "planned-resume.jsonl"
    from polylogue.storage import repair as repair_module

    original_finish = repair_module._finish_quarantined_raw_repair_receipt
    monkeypatch.setattr(
        repair_module,
        "_finish_quarantined_raw_repair_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected terminal append crash")),
    )
    with pytest.raises(RuntimeError, match="terminal append crash"):
        repair_quarantined_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned"]
    assert _raw_session_row(tmp_path, raw_id)["revision_authority"] == "byte_proven"

    monkeypatch.setattr(repair_module, "_finish_quarantined_raw_repair_receipt", original_finish)
    resumed = repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    assert resumed.already_repaired_count == 1
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [record["state"] for record in records] == ["planned", "applied"]
    assert records[1]["repaired_raw_ids"] == []


def test_quarantined_accepted_raw_repair_does_not_claim_a_competing_receipt_commit(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    interrupted_receipt = tmp_path / "interrupted.jsonl"
    competing_receipt = tmp_path / "competing.jsonl"
    from polylogue.storage import repair as repair_module

    planned = repair_module._lock_quarantined_raw_repair_receipt(interrupted_receipt, list(dry_run.items))
    planned.close()
    competing = repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=competing_receipt,
        proof_digest=dry_run.proof_digest,
    )
    resumed = repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=interrupted_receipt,
        proof_digest=dry_run.proof_digest,
    )

    assert competing.repaired_count == 1
    assert resumed.repaired_count == 0
    assert json.loads(competing_receipt.read_text().splitlines()[-1])["repaired_raw_ids"] == [raw_id]
    assert json.loads(interrupted_receipt.read_text().splitlines()[-1])["repaired_raw_ids"] == []


def test_quarantined_accepted_raw_repair_attributes_from_locked_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "pre-lease-competitor.jsonl"
    from polylogue.storage import index_generation

    class CompetingRepairLease:
        def __init__(self, archive_root: Path) -> None:
            self.archive_root = archive_root

        def __enter__(self) -> CompetingRepairLease:
            with sqlite3.connect(self.archive_root / "source.db") as source:
                source.execute(
                    """
                    UPDATE raw_sessions
                    SET logical_source_key = ?, revision_kind = 'full', source_revision = ?,
                        baseline_raw_id = raw_id, acquisition_generation = 0,
                        revision_authority = 'byte_proven'
                    WHERE raw_id = ?
                    """,
                    ("chatgpt:repair-one", dry_run.items[0].accepted_source_revision, raw_id),
                )
                source.commit()
            return self

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr(index_generation, "RebuildLease", CompetingRepairLease)
    result = repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )

    assert result.repaired_count == 0
    assert result.already_repaired_count == 1
    assert json.loads(receipt.read_text().splitlines()[-1])["repaired_raw_ids"] == []


def test_quarantined_accepted_raw_repair_acquires_exclusive_archive_lock_before_receipt(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "blocked.jsonl"

    with RebuildLease(tmp_path), pytest.raises(RuntimeError, match="rebuild lease is already held"):
        repair_quarantined_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    assert not receipt.exists()


def test_quarantined_accepted_raw_repair_writes_every_receipt_record_in_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "short-write.jsonl"
    from polylogue.storage import repair as repair_module

    original_write = repair_module._receipt_write

    def short_write(descriptor: int, payload: bytes) -> int:
        return original_write(descriptor, payload[: max(1, min(11, len(payload)))])

    monkeypatch.setattr(repair_module, "_receipt_write", short_write)
    repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [record["state"] for record in records] == ["planned", "applied"]


def test_quarantined_accepted_raw_repair_recovers_preserved_torn_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "torn-terminal.jsonl"
    from polylogue.storage import repair as repair_module

    original_finish = repair_module._finish_quarantined_raw_repair_receipt

    def tear_terminal(locked: Any, *, items: list[Any]) -> None:
        del items
        # A complete JSON prefix without its newline must remain distinguishable
        # from the eventual applied record after recovery seals the torn line.
        repair_module._write_receipt_all(locked.descriptor, b"{}")
        os.fsync(locked.descriptor)
        raise RuntimeError("injected torn terminal")

    monkeypatch.setattr(repair_module, "_finish_quarantined_raw_repair_receipt", tear_terminal)
    with pytest.raises(RuntimeError, match="torn terminal"):
        repair_quarantined_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    assert _raw_session_row(tmp_path, raw_id)["revision_authority"] == "byte_proven"
    assert not receipt.read_bytes().endswith(b"\n")

    def tear_recovery(locked: Any, *, items: list[Any]) -> None:
        del items
        if locked.torn_terminals and not locked.receipt_terminated:
            repair_module._write_receipt_all(locked.descriptor, b"\xff\n")
        repair_module._write_receipt_all(locked.descriptor, b'{"state":')
        os.fsync(locked.descriptor)
        raise RuntimeError("injected torn recovery terminal")

    monkeypatch.setattr(repair_module, "_finish_quarantined_raw_repair_receipt", tear_recovery)
    with pytest.raises(RuntimeError, match="torn recovery terminal"):
        repair_quarantined_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )

    monkeypatch.setattr(repair_module, "_finish_quarantined_raw_repair_receipt", original_finish)
    repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    lines = receipt.read_bytes().splitlines()
    assert len(lines) == 4
    assert lines[1] == b"{}\xff"
    assert lines[2] == b'{"state":\xff'
    recovered = json.loads(lines[3])
    assert recovered["state"] == "applied"
    assert recovered["torn_terminals"] == [
        {"bytes": len(fragment), "sha256": hashlib.sha256(fragment).hexdigest()} for fragment in lines[1:3]
    ]

    reapplied = repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    assert reapplied.already_repaired_count == 1


def test_quarantined_accepted_raw_repair_rejects_torn_terminal_without_source_commit(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    receipt_path = tmp_path / "false-torn-terminal.jsonl"
    from polylogue.storage import repair as repair_module

    locked = repair_module._lock_quarantined_raw_repair_receipt(receipt_path, list(dry_run.items))
    repair_module._write_receipt_all(locked.descriptor, b'{"state":')
    os.fsync(locked.descriptor)
    locked.close()

    with pytest.raises(RuntimeError, match="no matching committed source refinement"):
        repair_quarantined_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt_path,
            proof_digest=dry_run.proof_digest,
        )
    assert _raw_session_row(tmp_path, raw_id)["revision_authority"] == "quarantined"


@pytest.mark.parametrize(
    ("record_index", "field", "value"),
    [
        (0, "state", "applied"),
        (0, "planned_at_ms", "not-an-int"),
        (1, "schema", "wrong"),
        (1, "target_hash", "0" * 64),
        (1, "proven_raw_ids", []),
        (1, "repaired_raw_ids", ["0" * 64]),
    ],
)
def test_quarantined_accepted_raw_repair_rejects_corrupt_receipt_records(
    tmp_path: Path,
    record_index: int,
    field: str,
    value: object,
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "corrupt.jsonl"
    repair_quarantined_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    records[record_index][field] = value
    receipt.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(RuntimeError, match="receipt"):
        repair_quarantined_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )


def test_quarantined_accepted_raw_repair_serializes_one_receipt_inode(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [raw_id])
    receipt_path = tmp_path / "locked.jsonl"
    from polylogue.storage import repair as repair_module

    locked = repair_module._lock_quarantined_raw_repair_receipt(receipt_path, list(dry_run.items))
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            repair_quarantined_accepted_raws(
                _config(tmp_path),
                [raw_id],
                apply=True,
                receipt_path=receipt_path,
                proof_digest=dry_run.proof_digest,
            )
        assert _raw_session_row(tmp_path, raw_id)["revision_authority"] == "quarantined"
    finally:
        locked.close()


def test_quarantined_accepted_raw_repair_cas_failure_rolls_back_entire_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _seed_invalid_head(tmp_path, "cas-first")
    second = _seed_invalid_head(tmp_path, "cas-second")
    dry_run = repair_quarantined_accepted_raws(_config(tmp_path), [first, second])
    before = {raw_id: _raw_session_row(tmp_path, raw_id) for raw_id in (first, second)}
    from polylogue.storage import repair as repair_module

    original = repair_module._cas_refine_quarantined_accepted_raw
    calls = 0

    def fail_second(conn: sqlite3.Connection, item: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected CAS failure")
        original(conn, item)

    monkeypatch.setattr(repair_module, "_cas_refine_quarantined_accepted_raw", fail_second)
    with pytest.raises(RuntimeError, match="injected CAS failure"):
        repair_quarantined_accepted_raws(
            _config(tmp_path),
            [first, second],
            apply=True,
            receipt_path=tmp_path / "cas-rollback.jsonl",
            proof_digest=dry_run.proof_digest,
        )
    assert {raw_id: _raw_session_row(tmp_path, raw_id) for raw_id in (first, second)} == before


@pytest.mark.parametrize("limit_kind", ["target", "aggregate"])
def test_quarantined_accepted_raw_repair_checks_blob_budget_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    raw_ids = [_seed_invalid_head(tmp_path, "budget-first")]
    if limit_kind == "aggregate":
        raw_ids.append(_seed_invalid_head(tmp_path, "budget-second"))
    from polylogue.storage import repair as repair_module

    blob_sizes = [int(str(_raw_session_row(tmp_path, raw_id)["blob_size"])) for raw_id in raw_ids]
    if limit_kind == "target":
        monkeypatch.setattr(repair_module, "_QUARANTINED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES", int(blob_sizes[0]) - 1)
    else:
        monkeypatch.setattr(
            repair_module, "_QUARANTINED_ACCEPTED_RAW_REPAIR_TOTAL_BLOB_LIMIT_BYTES", sum(blob_sizes) - 1
        )
    monkeypatch.setattr(BlobStore, "read_all", lambda *args, **kwargs: pytest.fail("blob was read before budget check"))

    with pytest.raises(RuntimeError, match="blob limit"):
        repair_quarantined_accepted_raws(_config(tmp_path), raw_ids)
