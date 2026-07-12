from __future__ import annotations

import hashlib
import json
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
from polylogue.storage.repair import repair_untyped_accepted_raws
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


def _seed_invalid_head(root: Path, native_id: str = "repair-one", *, multi_session: bool = False) -> str:
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


def test_untyped_accepted_raw_repair_roundtrip_is_receipted_and_idempotent(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
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

    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])

    assert dry_run.eligible_count == 1
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

    receipt_path = tmp_path / "recovery" / "untyped-raw-repair.jsonl"
    receipt_path.parent.mkdir()
    applied = repair_untyped_accepted_raws(
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
    assert {key: after_raw[key] for key in expected_updates} == expected_updates
    assert {key for key in before_raw if before_raw[key] != after_raw[key]} == set(expected_updates)

    reapplied = repair_untyped_accepted_raws(
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
        "application",
        "receipt_time",
        "typed_competitor",
        "typed_quarantined_competitor",
        "other_session_head",
        "other_session_application",
        "second_indexed_session",
        "membership",
        "envelope",
        "multi_session",
    ],
)
def test_untyped_accepted_raw_repair_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
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
        elif mutation == "typed_quarantined_competitor":
            source.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms,
                    logical_source_key, revision_kind, source_revision, baseline_raw_id,
                    acquisition_generation, revision_authority
                ) SELECT ?, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms,
                         'chatgpt:repair-one', 'full', ?, ?, 1, 'quarantined'
                  FROM raw_sessions WHERE raw_id = ?
                """,
                ("1" * 64, "2" * 64, "1" * 64, raw_id),
            )
        elif mutation == "other_session_head":
            index.execute(
                """
                INSERT INTO raw_revision_heads (
                    logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                    accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                    acquisition_generation, append_end_offset, decided_at_ms
                ) SELECT 'chatgpt:different-key', session_id, accepted_raw_id,
                         accepted_source_revision, accepted_content_hash, accepted_frontier_kind,
                         accepted_frontier, acquisition_generation, append_end_offset, decided_at_ms
                  FROM raw_revision_heads LIMIT 1
                """
            )
        elif mutation == "other_session_application":
            index.execute(
                """
                INSERT INTO raw_revision_applications (
                    decision_id, raw_id, session_id, logical_source_key, source_revision,
                    acquisition_generation, decision, detail, decided_at_ms
                ) SELECT 'other-session-application', ?, session_id, 'chatgpt:different-key', ?,
                         acquisition_generation + 1, 'ambiguous', 'test', decided_at_ms + 1
                  FROM raw_revision_applications LIMIT 1
                """,
                ("3" * 64, "4" * 64),
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
                """
                INSERT INTO raw_membership_census (
                    raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
                ) VALUES (?, 'test', 'failed', 0, 1, 'ambiguous')
                """,
                (raw_id,),
            )
        elif mutation == "envelope":
            source.execute("UPDATE raw_sessions SET logical_source_key = 'partial' WHERE raw_id = ?", (raw_id,))
        source.commit()
        index.commit()
    before = _logical_state(tmp_path, raw_id)

    report = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert _logical_state(tmp_path, raw_id) == before


def test_untyped_accepted_raw_repair_rejects_duplicates_and_rolls_back_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _seed_invalid_head(tmp_path, "first")
    second = _seed_invalid_head(tmp_path, "second")
    with pytest.raises(ValueError, match="duplicate"):
        repair_untyped_accepted_raws(_config(tmp_path), [first, first])
    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [first, second])
    before = _logical_state(tmp_path, first)
    receipt = tmp_path / "rollback-receipt.jsonl"

    from polylogue.storage import repair as repair_module

    original = repair_module._inspect_untyped_accepted_raw
    calls = 0

    def fail_postproof(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        item = original(*args, **kwargs)
        if calls == 5:
            return repair_module._untyped_raw_item(item.raw_id, "injected post-proof failure")
        return item

    monkeypatch.setattr(repair_module, "_inspect_untyped_accepted_raw", fail_postproof)
    with pytest.raises(RuntimeError, match="terminal state"):
        repair_untyped_accepted_raws(
            _config(tmp_path),
            [first, second],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )

    assert _logical_state(tmp_path, first) == before
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned"]


def test_untyped_accepted_raw_repair_resumes_planned_receipt_after_committed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "planned-resume.jsonl"
    from polylogue.storage import repair as repair_module

    original_finish = repair_module._finish_untyped_raw_repair_receipt
    monkeypatch.setattr(
        repair_module,
        "_finish_untyped_raw_repair_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected terminal append crash")),
    )
    with pytest.raises(RuntimeError, match="terminal append crash"):
        repair_untyped_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned"]
    assert _raw_session_row(tmp_path, raw_id)["revision_authority"] == "byte_proven"

    monkeypatch.setattr(repair_module, "_finish_untyped_raw_repair_receipt", original_finish)
    resumed = repair_untyped_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    assert resumed.already_repaired_count == 1
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [record["state"] for record in records] == ["planned", "applied"]
    assert records[1]["repaired_raw_ids"] == [raw_id]


def test_untyped_accepted_raw_repair_writes_every_receipt_record_in_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "short-write.jsonl"
    from polylogue.storage import repair as repair_module

    original_write = repair_module._receipt_write

    def short_write(descriptor: int, payload: bytes) -> int:
        return original_write(descriptor, payload[: max(1, min(11, len(payload)))])

    monkeypatch.setattr(repair_module, "_receipt_write", short_write)
    repair_untyped_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [record["state"] for record in records] == ["planned", "applied"]


def test_untyped_accepted_raw_repair_recovers_preserved_torn_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "torn-terminal.jsonl"
    from polylogue.storage import repair as repair_module

    original_finish = repair_module._finish_untyped_raw_repair_receipt

    def tear_terminal(locked: Any, *, items: list[Any]) -> None:
        del items
        repair_module._write_receipt_all(locked.descriptor, b'{"schema":')
        repair_module.os.fsync(locked.descriptor)
        raise RuntimeError("injected torn terminal")

    monkeypatch.setattr(repair_module, "_finish_untyped_raw_repair_receipt", tear_terminal)
    with pytest.raises(RuntimeError, match="torn terminal"):
        repair_untyped_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    assert _raw_session_row(tmp_path, raw_id)["revision_authority"] == "byte_proven"
    assert not receipt.read_bytes().endswith(b"\n")

    monkeypatch.setattr(repair_module, "_finish_untyped_raw_repair_receipt", original_finish)
    repair_untyped_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    lines = receipt.read_bytes().splitlines()
    assert len(lines) == 3
    assert lines[1] == b'{"schema":'
    recovered = json.loads(lines[2])
    assert recovered["state"] == "applied"
    assert recovered["torn_terminal_bytes"] == len(lines[1])
    assert recovered["torn_terminal_sha256"] == hashlib.sha256(lines[1]).hexdigest()

    reapplied = repair_untyped_accepted_raws(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    assert reapplied.already_repaired_count == 1


def test_untyped_accepted_raw_repair_rejects_torn_terminal_without_source_commit(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])
    receipt_path = tmp_path / "false-torn-terminal.jsonl"
    from polylogue.storage import repair as repair_module

    locked = repair_module._lock_untyped_raw_repair_receipt(receipt_path, list(dry_run.items))
    repair_module._write_receipt_all(locked.descriptor, b'{"state":')
    repair_module.os.fsync(locked.descriptor)
    locked.close()

    with pytest.raises(RuntimeError, match="no matching committed source refinement"):
        repair_untyped_accepted_raws(
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
        (1, "repaired_raw_ids", []),
    ],
)
def test_untyped_accepted_raw_repair_rejects_corrupt_receipt_records(
    tmp_path: Path,
    record_index: int,
    field: str,
    value: object,
) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])
    receipt = tmp_path / "corrupt.jsonl"
    repair_untyped_accepted_raws(
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
        repair_untyped_accepted_raws(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )


def test_untyped_accepted_raw_repair_serializes_one_receipt_inode(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])
    receipt_path = tmp_path / "locked.jsonl"
    from polylogue.storage import repair as repair_module

    locked = repair_module._lock_untyped_raw_repair_receipt(receipt_path, list(dry_run.items))
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            repair_untyped_accepted_raws(
                _config(tmp_path),
                [raw_id],
                apply=True,
                receipt_path=receipt_path,
                proof_digest=dry_run.proof_digest,
            )
        assert _raw_session_row(tmp_path, raw_id)["revision_authority"] == "quarantined"
    finally:
        locked.close()


def test_untyped_accepted_raw_repair_cas_failure_rolls_back_entire_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _seed_invalid_head(tmp_path, "cas-first")
    second = _seed_invalid_head(tmp_path, "cas-second")
    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [first, second])
    before = {raw_id: _raw_session_row(tmp_path, raw_id) for raw_id in (first, second)}
    from polylogue.storage import repair as repair_module

    original = repair_module._cas_refine_untyped_accepted_raw
    calls = 0

    def fail_second(conn: sqlite3.Connection, item: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected CAS failure")
        original(conn, item)

    monkeypatch.setattr(repair_module, "_cas_refine_untyped_accepted_raw", fail_second)
    with pytest.raises(RuntimeError, match="injected CAS failure"):
        repair_untyped_accepted_raws(
            _config(tmp_path),
            [first, second],
            apply=True,
            receipt_path=tmp_path / "cas-rollback.jsonl",
            proof_digest=dry_run.proof_digest,
        )
    assert {raw_id: _raw_session_row(tmp_path, raw_id) for raw_id in (first, second)} == before


@pytest.mark.parametrize("limit_kind", ["target", "aggregate"])
def test_untyped_accepted_raw_repair_checks_blob_budget_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_kind: str,
) -> None:
    raw_ids = [_seed_invalid_head(tmp_path, "budget-first")]
    if limit_kind == "aggregate":
        raw_ids.append(_seed_invalid_head(tmp_path, "budget-second"))
    from polylogue.storage import repair as repair_module

    blob_sizes = [_raw_session_row(tmp_path, raw_id)["blob_size"] for raw_id in raw_ids]
    if limit_kind == "target":
        monkeypatch.setattr(repair_module, "_UNTYPED_ACCEPTED_RAW_REPAIR_BLOB_LIMIT_BYTES", int(blob_sizes[0]) - 1)
    else:
        monkeypatch.setattr(
            repair_module, "_UNTYPED_ACCEPTED_RAW_REPAIR_TOTAL_BLOB_LIMIT_BYTES", sum(map(int, blob_sizes)) - 1
        )
    monkeypatch.setattr(BlobStore, "read_all", lambda *args, **kwargs: pytest.fail("blob was read before budget check"))

    with pytest.raises(RuntimeError, match="blob limit"):
        repair_untyped_accepted_raws(_config(tmp_path), raw_ids)
