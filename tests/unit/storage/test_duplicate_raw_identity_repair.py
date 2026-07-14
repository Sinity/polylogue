"""Tests for polylogue-t0dy: reconcile the pre-#2729 duplicate-raw scheme.

PR #2729 aligned the one-shot importer and the live daemon watcher on one
deterministic raw-id scheme (no ``native_id``) so *new* ingests of a grouped/
split-session file converge on one raw row. It explicitly does not
retroactively repair pairs that already duplicated under the OLD,
native_id-inclusive scheme before that fix landed. ``repair_duplicate_raw_identity``
is the typed, receipted, CAS-gated actuator that performs that one-time
reconciliation without hand-writing a raw UPDATE.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.config import Config
from polylogue.core.enums import Provider, Role
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base_models import ParsedMessage, ParsedSession
from polylogue.storage.repair import repair_duplicate_raw_identity
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    RevisionApplicationReceipt,
    record_revision_application_sync,
)
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")


def _session() -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="carryover-session",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
    )


def _seed_duplicate_raw_pair(root: Path, *, legacy_native_id: str = "legacy-native-id-1") -> tuple[str, str, str, str]:
    """Seed the exact pre-#2729 duplicate shape.

    One raw ("canonical") is written the way the current, post-#2729 daemon
    watcher writes it: ``native_id`` NULL, deterministic id computed from
    (origin, source_path, source_index, blob_hash) alone. A second raw
    ("stale") clones its byte-identical content under the OLD, native_id-
    inclusive scheme and is the one actually bound as the accepted head/
    session pointer -- reproducing the exact incongruity #2729 prevents
    recurring but does not retroactively fix.
    """
    initialize_active_archive_root(root)
    payload = json.dumps({"marker": "duplicate-raw-fixture", "legacy_native_id": legacy_native_id}).encode()
    source_path = "codex-session/carryover.jsonl"
    session = _session()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        canonical_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX, payload=payload, source_path=source_path, acquired_at_ms=2
        )
        source_conn = archive._ensure_source_conn()
        row = source_conn.execute(
            """
            SELECT origin, capture_mode, source_path, source_index, blob_hash, blob_size
            FROM raw_sessions WHERE raw_id = ?
            """,
            (canonical_raw_id,),
        ).fetchone()
        origin, capture_mode, stored_source_path, source_index, blob_hash, blob_size = row
        stale_raw_id = deterministic_raw_session_id(
            str(origin), str(stored_source_path), int(source_index), bytes(blob_hash), native_id=legacy_native_id
        )
        blob_hash_hex = bytes(blob_hash).hex()
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, capture_mode, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms,
                revision_kind, source_revision, baseline_raw_id, acquisition_generation, revision_authority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'full', ?, ?, 0, 'byte_proven')
            """,
            (
                stale_raw_id,
                origin,
                capture_mode,
                legacy_native_id,
                stored_source_path,
                source_index,
                blob_hash,
                blob_size,
                1,
                blob_hash_hex,
                stale_raw_id,
            ),
        )
        source_conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, ?, ?)
            """,
            (blob_hash, stale_raw_id, stored_source_path, blob_size, 1),
        )
        source_conn.commit()
        _stored, session_id = archive.write_parsed_for_retained_raw(
            session,
            raw_id=stale_raw_id,
            source_path=source_path,
            acquired_at_ms=1,
            revision_authoritative=True,
        )
        logical_source_key = "codex:carryover-session"
        blob_hash_hex = bytes(blob_hash).hex()
        accepted_hash = bytes.fromhex(session_content_hash(session))
        record_revision_application_sync(
            archive._conn,
            RevisionApplicationReceipt(
                raw_id=stale_raw_id,
                session_id=session_id,
                logical_source_key=logical_source_key,
                source_revision=blob_hash_hex,
                acquisition_generation=0,
                decision=ApplicationDecision.SELECTED_BASELINE,
                accepted_raw_id=stale_raw_id,
                accepted_source_revision=blob_hash_hex,
                accepted_content_hash=accepted_hash,
                accepted_frontier_kind="byte",
                accepted_frontier=blob_size,
                baseline_raw_id=stale_raw_id,
                detail="pre-#2729 duplicate-raw fixture",
            ),
            decided_at_ms=1,
        )
        archive.commit()
    return stale_raw_id, canonical_raw_id, session_id, logical_source_key


def _rows(root: Path, tier: str, table: str, where: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(root / f"{tier}.db")) as conn:
        return sorted(conn.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchall())


def test_dry_run_proves_eligible_pair_without_mutating(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, session_id, _key = _seed_duplicate_raw_pair(tmp_path)
    before = {
        (tier, table): _rows(tmp_path, tier, table, "1 = 1", ())
        for tier, tables in {
            "source": ("raw_sessions", "blob_refs"),
            "index": ("sessions", "raw_revision_heads", "raw_revision_applications"),
        }.items()
        for table in tables
    }

    report = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])

    assert report.mode == "dry-run"
    assert report.requested_count == 1
    assert report.eligible_count == 1
    assert report.ineligible_count == 0
    assert report.repaired_count == 0
    item = report.items[0]
    assert item.status == "eligible"
    assert item.session_id == session_id
    assert item.logical_source_key == "codex:carryover-session"
    assert item.proof_digest is not None
    for key, rows in before.items():
        assert _rows(tmp_path, *key, "1 = 1", ()) == rows


def test_apply_repoints_head_and_session_preserving_stale_raw(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, session_id, key = _seed_duplicate_raw_pair(tmp_path)
    stale_before = _rows(tmp_path, "source", "raw_sessions", "raw_id = ?", (stale_raw_id,))

    dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    receipt = tmp_path / "t0dy-repair.json"
    applied = repair_duplicate_raw_identity(
        _config(tmp_path),
        [(stale_raw_id, canonical_raw_id)],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )

    assert applied.mode == "apply"
    assert applied.repaired_count == 1
    assert applied.items[0].repaired is True
    assert receipt.exists()

    # The stale raw's own row is byte-for-byte unchanged -- durable raw
    # evidence is never mutated or deleted, only its authority.
    assert _rows(tmp_path, "source", "raw_sessions", "raw_id = ?", (stale_raw_id,)) == stale_before

    with closing(sqlite3.connect(tmp_path / "index.db")) as index_conn:
        head = index_conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?", (key,)
        ).fetchone()
        assert head[0] == canonical_raw_id
        session_raw = index_conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        assert session_raw[0] == canonical_raw_id
        # The stale raw carries BOTH its original ``selected_baseline`` receipt
        # (from before repair) and the new ``superseded`` receipt (from
        # repair) -- immutable application receipts are append-only, never
        # overwritten -- so assert presence of the new one specifically
        # rather than assuming there is only one row.
        stale_decisions = {
            row[0]
            for row in index_conn.execute(
                "SELECT decision FROM raw_revision_applications WHERE raw_id = ? AND logical_source_key = ?",
                (stale_raw_id, key),
            ).fetchall()
        }
        assert stale_decisions == {"selected_baseline", "superseded"}
        stale_superseded_target = index_conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_applications "
            "WHERE raw_id = ? AND logical_source_key = ? AND decision = 'superseded'",
            (stale_raw_id, key),
        ).fetchone()
        assert stale_superseded_target[0] == canonical_raw_id
        canonical_decision = index_conn.execute(
            "SELECT decision, accepted_raw_id FROM raw_revision_applications WHERE raw_id = ? AND logical_source_key = ?",
            (canonical_raw_id, key),
        ).fetchone()
        assert canonical_decision == ("selected_baseline", canonical_raw_id)


def test_reapply_after_success_is_idempotent_already_repaired(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, _session_id, _key = _seed_duplicate_raw_pair(tmp_path)
    dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    repair_duplicate_raw_identity(
        _config(tmp_path),
        [(stale_raw_id, canonical_raw_id)],
        apply=True,
        receipt_path=tmp_path / "first.json",
        proof_digest=dry_run.proof_digest,
    )

    again = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])

    assert again.mode == "dry-run"
    assert again.already_repaired_count == 1
    assert again.eligible_count == 0
    assert again.ineligible_count == 0
    assert again.items[0].status == "already_repaired"


@pytest.mark.parametrize(
    "mutation",
    ("canonical_already_accepted", "stale_not_accepted", "content_differs", "same_scheme_both_null"),
)
def test_fails_closed_for_ineligible_shapes(tmp_path: Path, mutation: str) -> None:
    stale_raw_id, canonical_raw_id, session_id, key = _seed_duplicate_raw_pair(tmp_path)
    if mutation == "canonical_already_accepted":
        with closing(sqlite3.connect(tmp_path / "index.db")) as index_conn, index_conn:
            index_conn.execute(
                "INSERT INTO raw_revision_heads (logical_source_key, session_id, accepted_raw_id, "
                "accepted_source_revision, accepted_content_hash, accepted_frontier_kind, accepted_frontier, "
                "acquisition_generation, decided_at_ms) "
                "SELECT 'codex:other-session', session_id, ?, accepted_source_revision, accepted_content_hash, "
                "accepted_frontier_kind, accepted_frontier, acquisition_generation, decided_at_ms "
                "FROM raw_revision_heads WHERE logical_source_key = ?",
                (canonical_raw_id, key),
            )
    elif mutation == "stale_not_accepted":
        with closing(sqlite3.connect(tmp_path / "index.db")) as index_conn, index_conn:
            index_conn.execute(
                "UPDATE raw_revision_heads SET accepted_raw_id = ? WHERE logical_source_key = ?",
                (canonical_raw_id, key),
            )
    elif mutation == "content_differs":
        with closing(sqlite3.connect(tmp_path / "source.db")) as source_conn, source_conn:
            source_conn.execute(
                "UPDATE raw_sessions SET blob_size = blob_size + 1 WHERE raw_id = ?", (canonical_raw_id,)
            )
    else:
        with closing(sqlite3.connect(tmp_path / "source.db")) as source_conn, source_conn:
            source_conn.execute("UPDATE raw_sessions SET native_id = NULL WHERE raw_id = ?", (stale_raw_id,))

    report = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])

    assert report.ineligible_count == 1
    assert report.items[0].status == "ineligible"
    with pytest.raises(RuntimeError, match="ineligible"):
        repair_duplicate_raw_identity(
            _config(tmp_path),
            [(stale_raw_id, canonical_raw_id)],
            apply=True,
            receipt_path=tmp_path / "should-not-write.json",
            proof_digest=report.proof_digest,
        )


def test_rejects_duplicate_pairs_and_malformed_ids(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, _session_id, _key = _seed_duplicate_raw_pair(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        repair_duplicate_raw_identity(
            _config(tmp_path), [(stale_raw_id, canonical_raw_id), (stale_raw_id, canonical_raw_id)]
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        repair_duplicate_raw_identity(_config(tmp_path), [("not-a-raw-id", canonical_raw_id)])
    with pytest.raises(ValueError, match="1..100 entries"):
        repair_duplicate_raw_identity(_config(tmp_path), [])


def test_apply_refuses_stale_proof_digest(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, _session_id, key = _seed_duplicate_raw_pair(tmp_path)
    dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    with closing(sqlite3.connect(tmp_path / "index.db")) as index_conn, index_conn:
        index_conn.execute(
            "UPDATE raw_revision_heads SET decided_at_ms = decided_at_ms + 1 WHERE logical_source_key = ?", (key,)
        )

    with pytest.raises(RuntimeError, match="proof digest does not match"):
        repair_duplicate_raw_identity(
            _config(tmp_path),
            [(stale_raw_id, canonical_raw_id)],
            apply=True,
            receipt_path=tmp_path / "stale-digest.json",
            proof_digest=dry_run.proof_digest,
        )


def test_apply_requires_receipt_path(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, _session_id, _key = _seed_duplicate_raw_pair(tmp_path)
    dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    with pytest.raises(ValueError, match="explicit operator repair receipt path"):
        repair_duplicate_raw_identity(
            _config(tmp_path), [(stale_raw_id, canonical_raw_id)], apply=True, proof_digest=dry_run.proof_digest
        )
