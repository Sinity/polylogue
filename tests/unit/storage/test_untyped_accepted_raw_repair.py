from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

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


def test_untyped_accepted_raw_repair_roundtrip_is_receipted_and_idempotent(tmp_path: Path) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    before = _logical_state(tmp_path, raw_id)

    dry_run = repair_untyped_accepted_raws(_config(tmp_path), [raw_id])

    assert dry_run.eligible_count == 1
    assert dry_run.items[0].proof_digest
    assert dry_run.items[0].application_decision_id
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
