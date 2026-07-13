from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.revision_backfill import _parse_one, backfill_historical_revision_evidence
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.repair import (
    repair_browser_capture_origin_mismatches,
    repair_quarantined_accepted_raws,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    RevisionApplicationReceipt,
    record_revision_application_sync,
)


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")


def _browser_payload(native_id: str = "browser-origin-one") -> bytes:
    payload = {
        "capture_id": f"chatgpt:{native_id}",
        "polylogue_capture_kind": "browser_llm_session",
        "provenance": {
            "adapter_name": "chatgpt-native-v1",
            "capture_mode": "snapshot",
            "captured_at": "2026-07-13T00:00:00Z",
            "source_url": f"https://chatgpt.com/c/{native_id}",
        },
        "raw_provider_payload": {
            "id": native_id,
            "title": "Historical browser capture",
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
                        "content": {"content_type": "text", "parts": ["preserved text"]},
                        "create_time": 1_700_000_000,
                    },
                }
            },
            "preserved_padding": "x" * 16_000,
        },
        "schema_version": 1,
        "session": {
            "provider": "chatgpt",
            "provider_session_id": native_id,
            "title": "DOM fallback",
            "turns": [{"provider_turn_id": "dom-1", "role": "user", "text": "fallback"}],
        },
        "source": "browser-extension",
    }
    return json.dumps(payload, sort_keys=True).encode()


def _seed_mismatched_browser_head(root: Path) -> str:
    initialize_active_archive_root(root)
    payload = _browser_payload()
    session = _parse_one(Provider.CHATGPT, payload, "browser-capture/chatgpt/one.json")[0]
    accepted_hash = bytes.fromhex(session_content_hash(session))
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.UNKNOWN,
            payload=payload,
            source_path="browser-capture/chatgpt/one.json",
            acquired_at_ms=1,
        )
        blob_hash = (
            archive._ensure_source_conn()
            .execute("SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (raw_id,))
            .fetchone()[0]
            .lower()
        )
        _stored_raw, session_id = archive.write_parsed_for_retained_raw(
            session,
            raw_id=raw_id,
            source_path="browser-capture/chatgpt/one.json",
            acquired_at_ms=1,
            revision_authoritative=True,
        )
        record_revision_application_sync(
            archive._conn,
            RevisionApplicationReceipt(
                raw_id=raw_id,
                session_id=session_id,
                logical_source_key="unknown:browser-origin-one",
                source_revision=blob_hash,
                acquisition_generation=0,
                decision=ApplicationDecision.SELECTED_BASELINE,
                accepted_raw_id=raw_id,
                accepted_source_revision=blob_hash,
                accepted_content_hash=accepted_hash,
                accepted_frontier_kind="byte",
                accepted_frontier=len(payload),
                baseline_raw_id=raw_id,
                detail="historical mismatched browser head",
            ),
            decided_at_ms=2,
        )
        archive.commit()
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """
            UPDATE raw_sessions
            SET origin = 'unknown-export',
                logical_source_key = 'unknown:browser-origin-one', revision_kind = 'full',
                source_revision = lower(hex(blob_hash)), acquisition_generation = 0,
                revision_authority = 'quarantined'
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        source.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            ) VALUES (?, 'chatgpt:browser-origin-one', 'browser-origin-one', ?, ?, 1, 0,
                      'quarantined', NULL, NULL)
            """,
            (raw_id, accepted_hash.hex(), accepted_hash),
        )
        source.execute(
            """
            INSERT INTO raw_membership_census (
                raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
            ) VALUES (?, 'historical-parser', 'complete', 1, 2, 'historical mismatch')
            """,
            (raw_id,),
        )
    return raw_id


def _seed_equivalent_canonical_head(root: Path, mismatched_raw_id: str) -> str:
    payload = _browser_payload()
    session = _parse_one(Provider.CHATGPT, payload, "browser-capture/chatgpt/canonical.json")[0]
    accepted_hash = bytes.fromhex(session_content_hash(session))
    projection = session_revision_projection(session)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="browser-capture/chatgpt/canonical.json",
            acquired_at_ms=3,
        )
        blob_hash = (
            archive._ensure_source_conn()
            .execute("SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (raw_id,))
            .fetchone()[0]
            .lower()
        )
        _stored_raw, session_id = archive.write_parsed_for_retained_raw(
            session,
            raw_id=raw_id,
            source_path="browser-capture/chatgpt/canonical.json",
            acquired_at_ms=3,
            revision_authoritative=True,
        )
        record_revision_application_sync(
            archive._conn,
            RevisionApplicationReceipt(
                raw_id=raw_id,
                session_id=session_id,
                logical_source_key="chatgpt:browser-origin-one",
                source_revision=blob_hash,
                acquisition_generation=0,
                decision=ApplicationDecision.SELECTED_BASELINE,
                accepted_raw_id=raw_id,
                accepted_source_revision=blob_hash,
                accepted_content_hash=accepted_hash,
                accepted_frontier_kind="byte",
                accepted_frontier=len(payload),
                baseline_raw_id=raw_id,
                detail="pre-existing canonical head",
            ),
            decided_at_ms=4,
        )
        archive._conn.execute(
            "UPDATE sessions SET raw_id = ? WHERE session_id = ?",
            (mismatched_raw_id, session_id),
        )
        archive.commit()
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """
            UPDATE raw_sessions
            SET logical_source_key = 'chatgpt:browser-origin-one', revision_kind = 'full',
                source_revision = lower(hex(blob_hash)), baseline_raw_id = raw_id,
                acquisition_generation = 0, revision_authority = 'byte_proven'
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        source.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            ) VALUES (?, 'chatgpt:browser-origin-one', 'browser-origin-one', ?, ?, ?, 0,
                      'byte_proven', 'applied', 4)
            """,
            (raw_id, blob_hash, accepted_hash, len(projection.message_hashes)),
        )
        source.execute(
            """
            INSERT INTO raw_membership_census (
                raw_id, parser_fingerprint, status, member_count, censused_at_ms, detail
            ) VALUES (?, 'canonical-parser', 'complete', 1, 4, 'canonical evidence')
            """,
            (raw_id,),
        )
    return raw_id


def _seed_semantic_canonical_head(root: Path, mismatched_raw_id: str) -> str:
    raw_id = _seed_equivalent_canonical_head(root, mismatched_raw_id)
    with sqlite3.connect(root / "source.db") as source, sqlite3.connect(root / "index.db") as index:
        source.execute(
            """
            UPDATE raw_sessions
            SET logical_source_key = NULL, revision_kind = 'unknown', source_revision = NULL,
                baseline_raw_id = NULL, acquisition_generation = NULL,
                revision_authority = 'quarantined'
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        source.execute(
            """
            UPDATE raw_session_memberships
            SET source_revision = lower(hex(normalized_content_hash)), revision_authority = 'quarantined',
                decision = NULL, decided_at_ms = NULL
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        index.execute(
            """
            UPDATE raw_revision_heads
            SET accepted_source_revision = lower(hex(accepted_content_hash)),
                accepted_frontier_kind = 'semantic', accepted_frontier = 1
            WHERE accepted_raw_id = ?
            """,
            (raw_id,),
        )
        index.execute(
            """
            UPDATE raw_revision_applications
            SET source_revision = lower(hex(accepted_content_hash)),
                accepted_source_revision = lower(hex(accepted_content_hash)),
                baseline_raw_id = NULL
            WHERE raw_id = ? AND decision = 'selected_baseline'
            """,
            (raw_id,),
        )
        accepted_hash = bytes(
            index.execute(
                "SELECT accepted_content_hash FROM raw_revision_heads WHERE accepted_raw_id = ?", (raw_id,)
            ).fetchone()[0]
        )
        semantic_revision = accepted_hash.hex()
        decision_id = RevisionApplicationReceipt(
            raw_id=raw_id,
            session_id="chatgpt-export:browser-origin-one",
            logical_source_key="chatgpt:browser-origin-one",
            source_revision=semantic_revision,
            acquisition_generation=0,
            decision=ApplicationDecision.SELECTED_BASELINE,
            accepted_raw_id=raw_id,
            accepted_source_revision=semantic_revision,
            accepted_content_hash=accepted_hash,
        ).decision_id
        index.execute(
            "UPDATE raw_revision_applications SET decision_id = ? WHERE raw_id = ? AND decision = 'selected_baseline'",
            (decision_id, raw_id),
        )
        source.commit()
        index.commit()
    return raw_id


def _seed_semantic_superseded_sibling(root: Path, semantic_raw_id: str) -> str:
    sibling_raw_id = "a" * 64
    with sqlite3.connect(root / "source.db") as source:
        for table, identity in (
            ("raw_sessions", "raw_id"),
            ("blob_refs", "ref_id"),
            ("raw_session_memberships", "raw_id"),
            ("raw_membership_census", "raw_id"),
        ):
            columns = [str(row[1]) for row in source.execute(f"PRAGMA table_info({table})")]
            projection = ["?" if column == identity else column for column in columns]
            source.execute(
                f"INSERT INTO {table} ({', '.join(columns)}) "
                f"SELECT {', '.join(projection)} FROM {table} WHERE {identity} = ?",
                (sibling_raw_id, semantic_raw_id),
            )
        source.execute(
            "UPDATE raw_sessions SET native_id = 'browser-origin-one' WHERE raw_id = ?",
            (sibling_raw_id,),
        )
        original_blob_hash = source.execute(
            "SELECT lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id = ?",
            (semantic_raw_id,),
        ).fetchone()[0]
        store = BlobStore(root / "blob")
        sibling_blob_hash, sibling_blob_size = store.write_from_bytes(store.read_all(original_blob_hash) + b"\n")
        source.execute(
            "UPDATE raw_sessions SET blob_hash = ?, blob_size = ? WHERE raw_id = ?",
            (bytes.fromhex(sibling_blob_hash), sibling_blob_size, sibling_raw_id),
        )
        source.execute(
            "UPDATE blob_refs SET blob_hash = ?, size_bytes = ? WHERE ref_id = ?",
            (bytes.fromhex(sibling_blob_hash), sibling_blob_size, sibling_raw_id),
        )
        source.commit()
    with sqlite3.connect(root / "index.db") as index:
        accepted_hash = index.execute(
            "SELECT accepted_content_hash FROM raw_revision_heads WHERE accepted_raw_id = ?",
            (semantic_raw_id,),
        ).fetchone()[0]
        record_revision_application_sync(
            index,
            RevisionApplicationReceipt(
                raw_id=sibling_raw_id,
                session_id="chatgpt-export:browser-origin-one",
                logical_source_key="chatgpt:browser-origin-one",
                source_revision=bytes(accepted_hash).hex(),
                acquisition_generation=1,
                decision=ApplicationDecision.SUPERSEDED,
                accepted_raw_id=semantic_raw_id,
                accepted_source_revision=bytes(accepted_hash).hex(),
                accepted_content_hash=bytes(accepted_hash),
                detail="membership:superseded_equivalent",
            ),
            decided_at_ms=3,
        )
        index.commit()
    return sibling_raw_id


def _rows(root: Path, tier: str, table: str, where: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
    with sqlite3.connect(root / f"{tier}.db") as conn:
        return [tuple(row) for row in conn.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY 1", params)]


def test_browser_capture_origin_copy_forward_preserves_old_evidence_and_is_idempotent(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    old_evidence = {
        "raw": _rows(tmp_path, "source", "raw_sessions", "raw_id = ?", (raw_id,)),
        "blob": _rows(tmp_path, "source", "blob_refs", "ref_id = ?", (raw_id,)),
        "membership": _rows(tmp_path, "source", "raw_session_memberships", "raw_id = ?", (raw_id,)),
        "census": _rows(tmp_path, "source", "raw_membership_census", "raw_id = ?", (raw_id,)),
        "head": _rows(tmp_path, "index", "raw_revision_heads", "accepted_raw_id = ?", (raw_id,)),
        "application": _rows(tmp_path, "index", "raw_revision_applications", "raw_id = ?", (raw_id,)),
    }

    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])

    assert dry_run.eligible_count == 1, dry_run.items[0].reason
    item = dry_run.items[0]
    assert item.canonical_origin == "chatgpt-export"
    assert item.canonical_logical_source_key == "chatgpt:browser-origin-one"
    assert item.copy_forward_raw_id not in {None, raw_id}
    assert repair_quarantined_accepted_raws(_config(tmp_path), [raw_id]).ineligible_count == 1

    receipt = tmp_path / "recovery" / "browser-origin.jsonl"
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )

    assert applied.repaired_count == 1
    assert applied.items[0].status == "already_repaired"
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned", "applied"]
    for name, before in old_evidence.items():
        tier = "index" if name in {"head", "application"} else "source"
        table = {
            "raw": "raw_sessions",
            "blob": "blob_refs",
            "membership": "raw_session_memberships",
            "census": "raw_membership_census",
            "head": "raw_revision_heads",
            "application": "raw_revision_applications",
        }[name]
        key = "accepted_raw_id" if name == "head" else "raw_id" if name not in {"blob"} else "ref_id"
        assert _rows(tmp_path, tier, table, f"{key} = ?", (raw_id,)) == before
    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert copy_raw_id is not None
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT origin, logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (copy_raw_id,),
        ).fetchone() == ("chatgpt-export", "chatgpt:browser-origin-one", "byte_proven")
    with sqlite3.connect(tmp_path / "index.db") as index:
        index.execute("ATTACH DATABASE ? AS source", (str(tmp_path / "source.db"),))
        assert index.execute(
            "SELECT raw_id FROM sessions WHERE session_id = 'chatgpt-export:browser-origin-one'"
        ).fetchone() == (copy_raw_id,)
        assert index.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:browser-origin-one'"
        ).fetchone() == (copy_raw_id,)
        assert index.execute(
            """
            SELECT COUNT(*) FROM raw_revision_heads AS h
            JOIN sessions AS s ON s.session_id = h.session_id AND s.raw_id = h.accepted_raw_id
            JOIN source.raw_sessions AS r ON r.raw_id = h.accepted_raw_id
            WHERE r.origin = 'unknown-export'
            """
        ).fetchone() == (0,)
    reapplied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    assert reapplied.repaired_count == 0
    assert receipt.read_text().count("\n") == 2


def test_browser_capture_origin_rejects_decided_quarantined_membership(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            """
            UPDATE raw_session_memberships
            SET decision = 'ambiguous', decided_at_ms = 2
            WHERE raw_id = ?
            """,
            (raw_id,),
        )

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert report.items[0].reason == "membership census does not exactly reproduce the accepted session"


def test_browser_capture_origin_rebuild_keeps_copy_forward_head(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=tmp_path / "rebuild-repair.jsonl",
        proof_digest=dry_run.proof_digest,
    )
    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert copy_raw_id is not None

    backfill_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id, copy_raw_id])

    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = 'chatgpt:browser-origin-one'"
        ).fetchone() == (copy_raw_id,)


@pytest.mark.parametrize(
    "mutation", ["blob", "head", "origin", "application", "generation", "authority", "canonical_head"]
)
def test_browser_capture_origin_copy_forward_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "index.db") as index:
        if mutation == "blob":
            source.execute("UPDATE blob_refs SET size_bytes = size_bytes + 1 WHERE ref_id = ?", (raw_id,))
        elif mutation == "head":
            index.execute("UPDATE raw_revision_heads SET accepted_frontier = accepted_frontier + 1")
        elif mutation == "origin":
            source.execute("UPDATE raw_sessions SET origin = 'chatgpt-export' WHERE raw_id = ?", (raw_id,))
        elif mutation == "application":
            index.execute(
                "DELETE FROM raw_revision_applications WHERE raw_id = ? AND decision = 'selected_baseline'",
                (raw_id,),
            )
        elif mutation == "generation":
            source.execute("UPDATE raw_sessions SET acquisition_generation = 1 WHERE raw_id = ?", (raw_id,))
            source.execute("UPDATE raw_session_memberships SET acquisition_generation = 1 WHERE raw_id = ?", (raw_id,))
            index.execute(
                "UPDATE raw_revision_heads SET acquisition_generation = 1 WHERE accepted_raw_id = ?", (raw_id,)
            )
            index.execute(
                "UPDATE raw_revision_applications SET acquisition_generation = 1 WHERE raw_id = ?",
                (raw_id,),
            )
        elif mutation == "authority":
            source.execute("UPDATE raw_sessions SET revision_authority = 'byte_proven' WHERE raw_id = ?", (raw_id,))
        else:
            index.execute(
                """
                INSERT INTO raw_revision_heads (
                    logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                    accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                    acquisition_generation, decided_at_ms
                ) SELECT 'chatgpt:browser-origin-one', session_id, accepted_raw_id,
                         accepted_source_revision, accepted_content_hash, accepted_frontier_kind,
                         accepted_frontier, acquisition_generation, decided_at_ms
                  FROM raw_revision_heads
                """
            )

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    assert report.ineligible_count == 1


def test_browser_capture_origin_rejects_unresolved_source_membership(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            "UPDATE raw_session_memberships SET decision = 'ambiguous', decided_at_ms = 2 WHERE raw_id = ?",
            (raw_id,),
        )

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1


def test_browser_capture_origin_copy_forward_accepts_source_v7_without_capture_mode(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("ALTER TABLE raw_sessions DROP COLUMN capture_mode")
        source.execute("PRAGMA user_version = 7")
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    receipt = tmp_path / "source-v7-receipt.jsonl"

    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )

    assert applied.repaired_count == 1
    assert applied.items[0].status == "already_repaired"


def test_browser_capture_origin_repair_restores_equivalent_canonical_head(tmp_path: Path) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    canonical_raw_id = _seed_equivalent_canonical_head(tmp_path, mismatched_raw_id)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source_count = source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])

    assert dry_run.items[0].repair_strategy == "restore_canonical_head"
    assert dry_run.items[0].replacement_raw_id == canonical_raw_id
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [mismatched_raw_id],
        apply=True,
        receipt_path=tmp_path / "restore-receipt.jsonl",
        proof_digest=dry_run.proof_digest,
    )

    assert applied.repaired_count == 1
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == source_count
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute(
            "SELECT raw_id FROM sessions WHERE session_id = 'chatgpt-export:browser-origin-one'"
        ).fetchone() == (canonical_raw_id,)
        assert index.execute(
            """
            SELECT accepted_raw_id, detail FROM raw_revision_applications
            WHERE raw_id = ? AND logical_source_key = 'chatgpt:browser-origin-one'
              AND decision = 'superseded'
            """,
            (mismatched_raw_id,),
        ).fetchone() == (
            canonical_raw_id,
            f"browser_capture_origin_supersession:{mismatched_raw_id}",
        )


def test_browser_capture_origin_repair_copy_forwards_from_semantic_canonical_witness(tmp_path: Path) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, mismatched_raw_id)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source_count = source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]

    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])

    assert dry_run.eligible_count == 1, dry_run.items[0].reason
    assert dry_run.items[0].repair_strategy == "copy_forward"
    assert dry_run.items[0].replacement_raw_id != semantic_raw_id
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [mismatched_raw_id],
        apply=True,
        receipt_path=tmp_path / "semantic-copy-receipt.jsonl",
        proof_digest=dry_run.proof_digest,
    )

    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert copy_raw_id is not None
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (source_count + 1,)
        assert source.execute(
            "SELECT revision_authority FROM raw_sessions WHERE raw_id = ?", (semantic_raw_id,)
        ).fetchone() == ("quarantined",)
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute(
            "SELECT raw_id FROM sessions WHERE session_id = 'chatgpt-export:browser-origin-one'"
        ).fetchone() == (copy_raw_id,)


def test_browser_origin_repair_accepts_exact_semantic_superseded_sibling(tmp_path: Path) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, mismatched_raw_id)
    sibling_raw_id = _seed_semantic_superseded_sibling(tmp_path, semantic_raw_id)
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert (
            source.execute(
                "SELECT blob_hash FROM raw_sessions WHERE raw_id = ?",
                (sibling_raw_id,),
            ).fetchone()
            != source.execute(
                "SELECT blob_hash FROM raw_sessions WHERE raw_id = ?",
                (semantic_raw_id,),
            ).fetchone()
        )
    historical_before = _rows(
        tmp_path,
        "index",
        "raw_revision_applications",
        "raw_id = ?",
        (sibling_raw_id,),
    )

    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])

    assert dry_run.eligible_count == 1, dry_run.items[0].reason
    item = dry_run.items[0]
    assert item.semantic_canonical_raw_id == semantic_raw_id
    assert item.semantic_historical_raw_ids == (sibling_raw_id,)
    assert item.semantic_witness_digest is not None
    assert (
        _rows(
            tmp_path,
            "index",
            "raw_revision_applications",
            "raw_id = ?",
            (sibling_raw_id,),
        )
        == historical_before
    )
    receipt = tmp_path / "semantic-sibling-receipt.jsonl"
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [mismatched_raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    assert applied.repaired_count == 1
    assert applied.items[0].status == "already_repaired"
    assert applied.items[0].semantic_canonical_raw_id == semantic_raw_id
    assert applied.items[0].semantic_historical_raw_ids == (sibling_raw_id,)
    assert applied.items[0].semantic_witness_digest == item.semantic_witness_digest
    assert applied.items[0].terminal_byte_witness_digest is not None

    reapplied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [mismatched_raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )

    assert reapplied.repaired_count == 0
    assert reapplied.items[0].semantic_witness_digest == item.semantic_witness_digest
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned", "applied"]


@pytest.mark.parametrize("mutation", ["head_frontier", "head_revision", "receipt_id", "receipt_time"])
def test_browser_origin_repair_refuses_mutated_terminal_byte_authority(tmp_path: Path, mutation: str) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, mismatched_raw_id)
    _seed_semantic_superseded_sibling(tmp_path, semantic_raw_id)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])
    receipt = tmp_path / "terminal-byte-authority.jsonl"
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [mismatched_raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )
    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert copy_raw_id is not None
    with sqlite3.connect(tmp_path / "source.db") as source:
        source_count = source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
    with sqlite3.connect(tmp_path / "index.db") as index:
        if mutation == "head_frontier":
            index.execute(
                "UPDATE raw_revision_heads SET accepted_frontier = accepted_frontier + 1 WHERE accepted_raw_id = ?",
                (copy_raw_id,),
            )
        elif mutation == "head_revision":
            index.execute(
                "UPDATE raw_revision_heads SET accepted_source_revision = ? WHERE accepted_raw_id = ?",
                ("0" * 64, copy_raw_id),
            )
        elif mutation == "receipt_id":
            index.execute(
                "UPDATE raw_revision_applications SET decision_id = ? WHERE raw_id = ?",
                ("e" * 64, copy_raw_id),
            )
        else:
            index.execute(
                "UPDATE raw_revision_applications SET decided_at_ms = decided_at_ms + 1 WHERE raw_id = ?",
                (copy_raw_id,),
            )

    with pytest.raises(RuntimeError, match="ineligible"):
        repair_browser_capture_origin_mismatches(
            _config(tmp_path),
            [mismatched_raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (source_count,)
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned", "applied"]


@pytest.mark.parametrize("mutation", ["receipt", "revision", "membership", "selected", "blob"])
def test_browser_origin_repair_rejects_underproven_semantic_supersession(tmp_path: Path, mutation: str) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, mismatched_raw_id)
    sibling_raw_id = _seed_semantic_superseded_sibling(tmp_path, semantic_raw_id)
    if mutation == "receipt":
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "UPDATE raw_revision_applications SET accepted_raw_id = ? WHERE raw_id = ?",
                (sibling_raw_id, sibling_raw_id),
            )
    elif mutation == "revision":
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "UPDATE raw_revision_applications SET accepted_source_revision = ? WHERE raw_id = ?",
                ("0" * 64, sibling_raw_id),
            )
    elif mutation == "membership":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                "UPDATE raw_session_memberships SET decision = 'ambiguous', decided_at_ms = 4 WHERE raw_id = ?",
                (sibling_raw_id,),
            )
    elif mutation == "selected":
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "UPDATE raw_revision_applications SET decision = 'selected_baseline' WHERE raw_id = ?",
                (sibling_raw_id,),
            )
    else:
        different_payload = _browser_payload("browser-origin-two")
        blob_hash, blob_size = BlobStore(tmp_path / "blob").write_from_bytes(different_payload)
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                "UPDATE raw_sessions SET native_id = 'browser-origin-two', blob_hash = ?, blob_size = ? WHERE raw_id = ?",
                (bytes.fromhex(blob_hash), blob_size, sibling_raw_id),
            )
            source.execute(
                "UPDATE blob_refs SET blob_hash = ?, size_bytes = ? WHERE ref_id = ?",
                (bytes.fromhex(blob_hash), blob_size, sibling_raw_id),
            )

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])

    assert report.ineligible_count == 1


@pytest.mark.parametrize("field", ["decision_id", "detail", "decided_at_ms", "frontier"])
def test_browser_origin_repair_refuses_changed_semantic_receipt_before_apply(tmp_path: Path, field: str) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, mismatched_raw_id)
    sibling_raw_id = _seed_semantic_superseded_sibling(tmp_path, semantic_raw_id)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])
    with sqlite3.connect(tmp_path / "index.db") as index:
        if field == "decision_id":
            index.execute(
                "UPDATE raw_revision_applications SET decision_id = ? WHERE raw_id = ?",
                ("f" * 64, sibling_raw_id),
            )
        elif field == "detail":
            index.execute(
                "UPDATE raw_revision_applications SET detail = 'tampered' WHERE raw_id = ?",
                (sibling_raw_id,),
            )
        elif field == "decided_at_ms":
            index.execute(
                "UPDATE raw_revision_applications SET decided_at_ms = decided_at_ms + 1 WHERE raw_id = ?",
                (sibling_raw_id,),
            )
        else:
            index.execute(
                "UPDATE raw_revision_heads SET accepted_frontier = accepted_frontier + 1 WHERE accepted_raw_id = ?",
                (semantic_raw_id,),
            )

    with pytest.raises(RuntimeError, match="(ineligible|proof digest)"):
        repair_browser_capture_origin_mismatches(
            _config(tmp_path),
            [mismatched_raw_id],
            apply=True,
            receipt_path=tmp_path / f"semantic-{field}.jsonl",
            proof_digest=dry_run.proof_digest,
        )
    assert (
        _rows(tmp_path, "source", "raw_sessions", "source_path LIKE ?", ("browser-capture-origin-copy-forward/%",))
        == []
    )


@pytest.mark.parametrize("mutation", ["frontier", "pointer", "membership"])
def test_browser_capture_origin_repair_rejects_underproven_semantic_canonical_head(
    tmp_path: Path, mutation: str
) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, mismatched_raw_id)
    if mutation == "frontier":
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "UPDATE raw_revision_heads SET accepted_frontier_kind = 'byte' WHERE accepted_raw_id = ?",
                (semantic_raw_id,),
            )
    elif mutation == "pointer":
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "UPDATE sessions SET raw_id = ? WHERE session_id = 'chatgpt-export:browser-origin-one'",
                (semantic_raw_id,),
            )
    else:
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                "UPDATE raw_session_memberships SET decision = 'ambiguous', decided_at_ms = 5 WHERE raw_id = ?",
                (semantic_raw_id,),
            )

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])

    assert report.ineligible_count == 1


@pytest.mark.parametrize("mutation", ["envelope", "membership", "application"])
def test_browser_capture_origin_repair_rejects_underproven_canonical_head(tmp_path: Path, mutation: str) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    canonical_raw_id = _seed_equivalent_canonical_head(tmp_path, mismatched_raw_id)
    if mutation == "envelope":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                "UPDATE raw_sessions SET revision_authority = 'quarantined' WHERE raw_id = ?", (canonical_raw_id,)
            )
    elif mutation == "membership":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                "UPDATE raw_session_memberships SET decision = 'ambiguous' WHERE raw_id = ?", (canonical_raw_id,)
            )
    else:
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "DELETE FROM raw_revision_applications WHERE raw_id = ? AND decision = 'selected_baseline'",
                (canonical_raw_id,),
            )

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])

    assert report.ineligible_count == 1


def test_browser_capture_origin_repair_rejects_missing_canonical_blob(tmp_path: Path) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    canonical_raw_id = _seed_equivalent_canonical_head(tmp_path, mismatched_raw_id)
    with sqlite3.connect(tmp_path / "source.db") as source:
        blob_hash = source.execute(
            "SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (canonical_raw_id,)
        ).fetchone()[0]
    BlobStore(tmp_path / "blob").blob_path(blob_hash.lower()).unlink()

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])

    assert report.ineligible_count == 1


def test_browser_capture_origin_copy_forward_reproves_before_source_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.storage.repair as repair_module

    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    original_lock = repair_module._lock_browser_origin_receipt

    def mutate_after_receipt(
        path: Path,
        items: list[repair_module.BrowserCaptureOriginRepairItem],
    ) -> repair_module._BrowserOriginReceipt:
        receipt = original_lock(path, items)
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute("UPDATE raw_sessions SET blob_size = blob_size + 1 WHERE raw_id = ?", (raw_id,))
        return receipt

    monkeypatch.setattr(repair_module, "_lock_browser_origin_receipt", mutate_after_receipt)
    with pytest.raises(RuntimeError, match="source evidence changed"):
        repair_browser_capture_origin_mismatches(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=tmp_path / "source-race.jsonl",
            proof_digest=dry_run.proof_digest,
        )
    assert (
        _rows(tmp_path, "source", "raw_sessions", "source_path LIKE ?", ("browser-capture-origin-copy-forward/%",))
        == []
    )


def test_browser_capture_origin_copy_forward_resumes_after_source_stage_interrupt(tmp_path: Path) -> None:
    import polylogue.storage.repair as repair_module

    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    item = dry_run.items[0]
    assert item.repair_strategy == "copy_forward"
    with sqlite3.connect(tmp_path / "source.db") as source:
        repair_module._stage_browser_origin_copy_forward_source(source, item)

    resumed = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=tmp_path / "resume-receipt.jsonl",
        proof_digest=dry_run.proof_digest,
    )

    assert resumed.repaired_count == 1
    assert resumed.items[0].status == "already_repaired"
    assert resumed.items[0].copy_forward_source_complete is True


def test_browser_capture_origin_copy_forward_refuses_incomplete_blob_reference(tmp_path: Path) -> None:
    import polylogue.storage.repair as repair_module

    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    item = dry_run.items[0]
    assert item.copy_forward_raw_id is not None
    with sqlite3.connect(tmp_path / "source.db") as source:
        repair_module._stage_browser_origin_copy_forward_source(source, item)
        source.execute("DELETE FROM blob_refs WHERE ref_id = ?", (item.copy_forward_raw_id,))

    with pytest.raises(RuntimeError, match="ineligible"):
        repair_browser_capture_origin_mismatches(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=tmp_path / "incomplete-copy-receipt.jsonl",
            proof_digest=dry_run.proof_digest,
        )
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute(
            "SELECT raw_id FROM sessions WHERE session_id = 'chatgpt-export:browser-origin-one'"
        ).fetchone() == (raw_id,)


def test_browser_capture_origin_live_membership_ignores_quarantined_conflict(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=tmp_path / "live-membership-repair.jsonl",
        proof_digest=dry_run.proof_digest,
    )
    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert copy_raw_id is not None
    session = _parse_one(Provider.CHATGPT, _browser_payload(), "browser-capture/chatgpt/one.json")[0]
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (),
        cursor=cursor,
        parser_fingerprint="browser-origin-test",
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        parsed_by_raw_id = {
            raw_id: session,
            copy_raw_id: session,
        }
        projections = {raw_id: session_revision_projection(session), copy_raw_id: session_revision_projection(session)}
        # Production guard reproduction: an unresolved quarantined member
        # chosen over the existing byte head raises at the real archive writer.
        with pytest.raises(RuntimeError, match="membership replay cannot retire"):
            archive.apply_raw_membership_classification(
                "chatgpt:browser-origin-one",
                MembershipClassification((raw_id,), (), ()),
                parsed_by_raw_id,
                projections,
                acquired_at_ms=5,
            )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        processor._apply_membership_sessions(archive, copy_raw_id, [session], acquired_at_ms=6)
        assert archive.raw_revision_head_raw_id("chatgpt:browser-origin-one") == copy_raw_id


def test_browser_capture_origin_live_membership_defers_all_quarantined_rows(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=tmp_path / "all-quarantined-repair.jsonl",
        proof_digest=dry_run.proof_digest,
    )
    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert copy_raw_id is not None
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            "UPDATE raw_session_memberships SET revision_authority = 'quarantined' "
            "WHERE logical_source_key = 'chatgpt:browser-origin-one'"
        )
    session = _parse_one(Provider.CHATGPT, _browser_payload(), "browser-capture/chatgpt/one.json")[0]
    cursor = CursorStore(tmp_path / "all-quarantined-cursor.sqlite")
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (),
        cursor=cursor,
        parser_fingerprint="browser-origin-test",
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        processor._apply_membership_sessions(archive, copy_raw_id, [session], acquired_at_ms=7)
        assert archive.raw_revision_head_raw_id("chatgpt:browser-origin-one") == copy_raw_id
