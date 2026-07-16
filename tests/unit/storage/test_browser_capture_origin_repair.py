from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.archive.session_revision_membership import MembershipClassification
from polylogue.config import Config
from polylogue.core.enums import AssertionStatus, Provider
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.revision_backfill import _parse_one, backfill_historical_revision_evidence
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import resolve_raw_authority_blocker
from polylogue.storage.raw_reconciler import (
    RawAuthorityActuator,
    RawAuthorityFrontierState,
    inspect_raw_authority_frontier,
)
from polylogue.storage.repair import (
    inspect_browser_canonical_authority_conflicts,
    record_browser_canonical_authority_conflict_blockers,
    repair_browser_capture_origin_mismatches,
    repair_byte_proven_browser_capture_null_native_ids,
    repair_legacy_browser_capture_missing_native_ids,
    repair_quarantined_accepted_raws,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    RevisionApplicationReceipt,
    record_revision_application_sync,
)
from polylogue.storage.sqlite.archive_tiers.user_write import mark_assertion_status


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


def _seed_mismatched_browser_head(root: Path, native_id: str = "browser-origin-one") -> str:
    initialize_active_archive_root(root)
    source_path = (
        "browser-capture/chatgpt/one.json"
        if native_id == "browser-origin-one"
        else f"browser-capture/chatgpt/{native_id}.json"
    )
    payload = _browser_payload(native_id)
    session = _parse_one(Provider.CHATGPT, payload, source_path)[0]
    accepted_hash = bytes.fromhex(session_content_hash(session))
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.UNKNOWN,
            payload=payload,
            source_path=source_path,
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
            source_path=source_path,
            acquired_at_ms=1,
            revision_authoritative=True,
        )
        record_revision_application_sync(
            archive._conn,
            RevisionApplicationReceipt(
                raw_id=raw_id,
                session_id=session_id,
                logical_source_key=f"unknown:{native_id}",
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
    with closing(sqlite3.connect(root / "source.db")) as source, source:
        source.execute(
            """
            UPDATE raw_sessions
            SET origin = 'unknown-export',
                native_id = ?, logical_source_key = ?, revision_kind = 'full',
                source_revision = lower(hex(blob_hash)), acquisition_generation = 0,
                revision_authority = 'quarantined'
            WHERE raw_id = ?
            """,
            (native_id, f"unknown:{native_id}", raw_id),
        )
        source.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, 1, 0,
                      'quarantined', NULL, NULL)
            """,
            (raw_id, f"chatgpt:{native_id}", native_id, accepted_hash.hex(), accepted_hash),
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


def _seed_legacy_browser_head_without_native_id(root: Path) -> str:
    raw_id = _seed_mismatched_browser_head(root)
    with closing(sqlite3.connect(root / "source.db")) as source, source:
        source.execute("UPDATE raw_sessions SET native_id = NULL WHERE raw_id = ?", (raw_id,))
    return raw_id


def _seed_byte_proven_browser_head_without_native_id(root: Path, native_id: str = "browser-origin-one") -> str:
    raw_id = _seed_mismatched_browser_head(root, native_id)
    with closing(sqlite3.connect(root / "source.db")) as source, source:
        source.execute(
            """
            UPDATE raw_sessions
            SET native_id = NULL, baseline_raw_id = raw_id, revision_authority = 'byte_proven'
            WHERE raw_id = ?
            """,
            (raw_id,),
        )
        source.execute("DELETE FROM raw_session_memberships WHERE raw_id = ?", (raw_id,))
        source.execute("DELETE FROM raw_membership_census WHERE raw_id = ?", (raw_id,))
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
    with closing(sqlite3.connect(root / "source.db")) as source, source:
        source.execute(
            """
            UPDATE raw_sessions
            SET native_id = 'browser-origin-one', logical_source_key = 'chatgpt:browser-origin-one', revision_kind = 'full',
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


def _diverging_browser_payload(native_id: str = "browser-origin-one") -> bytes:
    """A same-shape browser capture whose *content* genuinely differs from ``_browser_payload``.

    Two messages instead of one, and the first message's text differs from
    ``_browser_payload``'s -- so a competing byte-proven head seeded from
    this payload has a real, provably-different message-hash sequence rather
    than the identical content ``_seed_equivalent_canonical_head`` seeds.
    """
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
            "title": "Diverging canonical capture",
            "create_time": 1_700_000_100,
            "update_time": 1_700_000_200,
            "current_node": "node-2",
            "mapping": {
                "node-1": {
                    "id": "node-1",
                    "parent": None,
                    "children": ["node-2"],
                    "message": {
                        "id": "message-1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["diverging text"]},
                        "create_time": 1_700_000_100,
                    },
                },
                "node-2": {
                    "id": "node-2",
                    "parent": "node-1",
                    "children": [],
                    "message": {
                        "id": "message-2",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["assistant reply"]},
                        "create_time": 1_700_000_101,
                    },
                },
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


def _seed_diverging_canonical_byte_head(root: Path, mismatched_raw_id: str) -> str:
    """Seed a competing byte-proven head with genuinely different content.

    Unlike ``_seed_equivalent_canonical_head`` (identical payload, so the
    competing content hash MATCHES and the witness takes the
    "not a hash divergence" branch), this seeds a competing head whose
    content is provably different -- exercising
    ``_browser_canonical_authority_conflict_witness``'s byte-frontier
    message-diff branch (the shape of the real production conflict
    ``88aefc84...``: incompatible canonical byte head diverging from
    message 0).
    """
    payload = _diverging_browser_payload()
    session = _parse_one(Provider.CHATGPT, payload, "browser-capture/chatgpt/diverging-canonical.json")[0]
    accepted_hash = bytes.fromhex(session_content_hash(session))
    projection = session_revision_projection(session)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="browser-capture/chatgpt/diverging-canonical.json",
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
            source_path="browser-capture/chatgpt/diverging-canonical.json",
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
                detail="diverging canonical head",
            ),
            decided_at_ms=4,
        )
        archive._conn.execute(
            "UPDATE sessions SET raw_id = ? WHERE session_id = ?",
            (mismatched_raw_id, session_id),
        )
        archive.commit()
    with closing(sqlite3.connect(root / "source.db")) as source, source:
        source.execute(
            """
            UPDATE raw_sessions
            SET native_id = 'browser-origin-one', logical_source_key = 'chatgpt:browser-origin-one', revision_kind = 'full',
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
            ) VALUES (?, 'canonical-parser', 'complete', 1, 4, 'diverging canonical evidence')
            """,
            (raw_id,),
        )
    return raw_id


def _seed_semantic_canonical_head(root: Path, mismatched_raw_id: str) -> str:
    raw_id = _seed_equivalent_canonical_head(root, mismatched_raw_id)
    with (
        closing(sqlite3.connect(root / "source.db")) as source,
        source,
        closing(sqlite3.connect(root / "index.db")) as index,
        index,
    ):
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
    with closing(sqlite3.connect(root / "source.db")) as source, source:
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
    with closing(sqlite3.connect(root / "index.db")) as index, index:
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


def _stage_active_index_generation(root: Path) -> Path:
    active_dir = root / ".index-generations" / "gen-v35"
    active_dir.mkdir(parents=True)
    active_index = active_dir / "index.db"
    index_link = root / "index.db"
    index_link.rename(active_index)
    index_link.symlink_to(active_index)
    return active_index


def _rows(root: Path, tier: str, table: str, where: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
    # sqlite3.Connection's context manager commits/rolls back, but does not
    # close the handle.  Legacy repair deliberately changes journal posture,
    # which requires every reader to have released its SQLite lock first.
    with closing(sqlite3.connect(root / f"{tier}.db")) as conn:
        return [tuple(row) for row in conn.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY 1", params)]


def _journal_modes(root: Path) -> dict[str, str]:
    modes: dict[str, str] = {}
    for name in ("source", "index"):
        with closing(sqlite3.connect(root / f"{name}.db")) as conn:
            modes[name] = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    return modes


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

    census = inspect_raw_authority_frontier(_config(tmp_path))
    planned = next(item for item in census.items if item.raw_id == raw_id)
    assert planned.state is RawAuthorityFrontierState.SAFELY_REKEYABLE
    assert planned.actuator is RawAuthorityActuator.COPY_FORWARD_ORIGIN
    assert planned.evidence_ref is not None

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


def test_browser_origin_repair_resumes_prelegacy_planned_receipt(tmp_path: Path) -> None:
    import polylogue.storage.repair as repair_module

    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    historical_target = repair_module._browser_origin_item_payload(dry_run.items[0])
    # Emulate the exact v1 ordinary target shape written before the legacy
    # actuator added its provenance-only fields.
    historical_target.pop("legacy_null_native_id", None)
    historical_target.pop("parser_derived_native_id", None)
    target_hash = hashlib.sha256(
        json.dumps([historical_target], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = tmp_path / "prelegacy-planned.jsonl"
    receipt.write_text(
        json.dumps(
            {
                "schema": "polylogue.browser-capture-origin-copy-forward.v1",
                "state": "planned",
                "target_hash": target_hash,
                "targets": [historical_target],
                "planned_at_ms": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
    )

    assert applied.repaired_count == 1
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned", "applied"]


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
    "mutation",
    [
        "blob",
        "head",
        "origin",
        "application",
        "generation",
        "authority",
        "native_id",
        "source_index",
        "blob_ref_path",
        "predecessor_source",
        "predecessor_raw",
        "append_start",
        "append_end",
        "capture_mode",
        "canonical_head",
    ],
)
def test_browser_capture_origin_copy_forward_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    with (
        closing(sqlite3.connect(tmp_path / "source.db")) as source,
        closing(sqlite3.connect(tmp_path / "index.db")) as index,
    ):
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
        elif mutation == "native_id":
            source.execute("UPDATE raw_sessions SET native_id = 'wrong-native-id' WHERE raw_id = ?", (raw_id,))
        elif mutation == "source_index":
            source.execute("UPDATE raw_sessions SET source_index = 1 WHERE raw_id = ?", (raw_id,))
        elif mutation == "blob_ref_path":
            source.execute(
                "UPDATE blob_refs SET source_path = 'browser-capture/wrong.json' WHERE ref_id = ?", (raw_id,)
            )
        elif mutation == "predecessor_source":
            source.execute(
                "UPDATE raw_sessions SET predecessor_source_revision = ? WHERE raw_id = ?", ("0" * 64, raw_id)
            )
        elif mutation == "predecessor_raw":
            source.execute("UPDATE raw_sessions SET predecessor_raw_id = ? WHERE raw_id = ?", ("f" * 64, raw_id))
        elif mutation == "append_start":
            source.execute("UPDATE raw_sessions SET append_start_offset = 0 WHERE raw_id = ?", (raw_id,))
        elif mutation == "append_end":
            source.execute("UPDATE raw_sessions SET append_end_offset = 1 WHERE raw_id = ?", (raw_id,))
        elif mutation == "capture_mode":
            source.execute("UPDATE raw_sessions SET capture_mode = 'chatgpt' WHERE raw_id = ?", (raw_id,))
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
        source.commit()
        index.commit()

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


def test_browser_capture_origin_rejects_legacy_raw_without_native_id(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute("UPDATE raw_sessions SET native_id = NULL WHERE raw_id = ?", (raw_id,))

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert report.items[0].reason == "source envelope does not exactly bind the normalized session"


def test_legacy_browser_native_id_copy_forward_preserves_evidence_and_is_idempotent(tmp_path: Path) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    ordinary = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    assert ordinary.ineligible_count == 1
    with (
        closing(sqlite3.connect(tmp_path / "source.db")) as source,
        closing(sqlite3.connect(tmp_path / "index.db")) as index,
    ):
        old_source = source.execute("SELECT * FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
        old_refs = source.execute("SELECT * FROM blob_refs WHERE ref_id = ? ORDER BY ref_type", (raw_id,)).fetchall()
        old_memberships = source.execute(
            "SELECT * FROM raw_session_memberships WHERE raw_id = ? ORDER BY logical_source_key", (raw_id,)
        ).fetchall()
        old_head = index.execute(
            "SELECT * FROM raw_revision_heads WHERE logical_source_key = 'unknown:browser-origin-one'"
        ).fetchone()
        old_applications = index.execute(
            "SELECT * FROM raw_revision_applications WHERE raw_id = ? ORDER BY decision_id", (raw_id,)
        ).fetchall()
    dry_run = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])
    item = dry_run.items[0]
    assert item.status == "eligible"
    assert item.legacy_null_native_id is True
    assert item.parser_derived_native_id == "browser-origin-one"
    receipt = tmp_path / "legacy-native-id.jsonl"
    applied = repair_legacy_browser_capture_missing_native_ids(
        _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
    )
    assert applied.repaired_count == 1
    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert copy_raw_id is not None
    lines = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert lines[0]["schema"] == "polylogue.browser-capture-legacy-native-id-copy-forward.v1"
    assert lines[0]["transaction_protocol"] == "rollback-superjournal-v1"
    assert lines[0]["targets"][0]["legacy_null_native_id"] is True
    assert lines[0]["targets"][0]["parser_derived_native_id"] == "browser-origin-one"
    assert lines[-1]["legacy_native_witness_bindings"] == [
        {"raw_id": raw_id, "legacy_null_native_id": True, "parser_derived_native_id": "browser-origin-one"}
    ]
    assert lines[-1]["transaction_protocol"] == "rollback-superjournal-v1"
    assert lines[-1]["legacy_journal_modes"]["after_restore"] == {"source": "wal", "index": "wal"}
    assert _journal_modes(tmp_path) == {"source": "wal", "index": "wal"}
    with (
        closing(sqlite3.connect(tmp_path / "source.db")) as source,
        closing(sqlite3.connect(tmp_path / "index.db")) as index,
    ):
        assert source.execute("SELECT * FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == old_source
        assert (
            source.execute("SELECT * FROM blob_refs WHERE ref_id = ? ORDER BY ref_type", (raw_id,)).fetchall()
            == old_refs
        )
        assert (
            source.execute(
                "SELECT * FROM raw_session_memberships WHERE raw_id = ? ORDER BY logical_source_key", (raw_id,)
            ).fetchall()
            == old_memberships
        )
        assert (
            index.execute(
                "SELECT * FROM raw_revision_heads WHERE logical_source_key = 'unknown:browser-origin-one'"
            ).fetchone()
            == old_head
        )
        assert (
            index.execute(
                "SELECT * FROM raw_revision_applications WHERE raw_id = ? ORDER BY decision_id", (raw_id,)
            ).fetchall()
            == old_applications
        )
        assert source.execute(
            "SELECT origin, native_id FROM raw_sessions WHERE raw_id = ?", (copy_raw_id,)
        ).fetchone() == (
            "chatgpt-export",
            "browser-origin-one",
        )
    reapplied = repair_legacy_browser_capture_missing_native_ids(
        _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
    )
    assert reapplied.already_repaired_count == 1


@pytest.mark.parametrize("checkpoint", ["before_commit", "after_commit"])
def test_legacy_browser_native_id_crash_boundary_recovers_planned_receipt(tmp_path: Path, checkpoint: str) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    dry_run = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])
    before = {
        (tier, table): _rows(tmp_path, tier, table, "1 = 1", ())
        for tier, tables in {
            "source": ("raw_sessions", "blob_refs", "raw_session_memberships", "raw_membership_census"),
            "index": ("sessions", "raw_revision_heads", "raw_revision_applications"),
        }.items()
        for table in tables
    }

    receipt = tmp_path / f"legacy-crash-{checkpoint}.jsonl"
    program = f"""
import os
from pathlib import Path

from polylogue.config import Config
import polylogue.storage.repair as repair

root = Path({str(tmp_path)!r})

def crash_at_selected_boundary(stage: str) -> None:
    if stage == {checkpoint!r}:
        os._exit(87)

repair._legacy_browser_copy_forward_checkpoint = crash_at_selected_boundary
repair.repair_legacy_browser_capture_missing_native_ids(
    Config(archive_root=root, render_root=root / 'render', sources=[], db_path=root / 'index.db'),
    [{raw_id!r}],
    apply=True,
    receipt_path=Path({str(receipt)!r}),
    proof_digest={dry_run.proof_digest!r},
)
"""
    child = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 87, child.stderr
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned"]
    assert _journal_modes(tmp_path) == {"source": "delete", "index": "delete"}
    if checkpoint == "before_commit":
        for (tier, table), rows in before.items():
            assert _rows(tmp_path, tier, table, "1 = 1", ()) == rows
    else:
        assert _rows(tmp_path, "source", "raw_sessions", "raw_id != ?", (raw_id,))
        assert _rows(tmp_path, "index", "raw_revision_heads", "logical_source_key LIKE ?", ("chatgpt:%",))

    resumed = repair_legacy_browser_capture_missing_native_ids(
        _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
    )

    assert resumed.already_repaired_count == 1
    assert _journal_modes(tmp_path) == {"source": "wal", "index": "wal"}
    lines = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [line["state"] for line in lines] == ["planned", "applied"]
    assert lines[-1]["transaction_protocol"] == "rollback-superjournal-v1"


@pytest.mark.parametrize("mutation", ["native", "path", "blob_ref", "census", "head", "application"])
def test_legacy_browser_native_id_copy_forward_rejects_any_witness_drift(tmp_path: Path, mutation: str) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    if mutation == "native":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute("UPDATE raw_sessions SET native_id = 'wrong-native' WHERE raw_id = ?", (raw_id,))
    elif mutation == "path":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                "UPDATE raw_sessions SET source_path = 'browser-capture/chatgpt/wrong.json' WHERE raw_id = ?", (raw_id,)
            )
    elif mutation == "blob_ref":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute("DELETE FROM blob_refs WHERE ref_id = ?", (raw_id,))
    elif mutation == "census":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute("UPDATE raw_membership_census SET member_count = 2 WHERE raw_id = ?", (raw_id,))
    elif mutation == "head":
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "UPDATE raw_revision_heads SET accepted_frontier = accepted_frontier + 1 WHERE logical_source_key = 'unknown:browser-origin-one'"
            )
    else:
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute("UPDATE raw_revision_applications SET decision = 'superseded' WHERE raw_id = ?", (raw_id,))

    report = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1


def test_legacy_browser_native_id_copy_forward_accepts_source_v7(tmp_path: Path) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        source.execute("ALTER TABLE raw_sessions DROP COLUMN capture_mode")
        source.execute("PRAGMA user_version = 7")
    dry_run = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])
    applied = repair_legacy_browser_capture_missing_native_ids(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=tmp_path / "legacy-source-v7.jsonl",
        proof_digest=dry_run.proof_digest,
    )
    assert applied.repaired_count == 1


def test_legacy_browser_native_id_copy_forward_rejects_stale_proof_before_writing(tmp_path: Path) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        before = source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
    with pytest.raises(RuntimeError, match="proof digest"):
        repair_legacy_browser_capture_missing_native_ids(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=tmp_path / "stale-proof.jsonl",
            proof_digest="0" * 64,
        )
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == before


def test_legacy_browser_native_id_copy_forward_refuses_preexisting_canonical_head(tmp_path: Path) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    _seed_equivalent_canonical_head(tmp_path, raw_id)

    report = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert report.items[0].reason == "legacy-native-id copy-forward refuses pre-existing canonical head authority"


def test_legacy_browser_native_id_copy_forwards_from_semantic_canonical_witness(tmp_path: Path) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, raw_id)
    sibling_raw_id = _seed_semantic_superseded_sibling(tmp_path, semantic_raw_id)
    active_index = _stage_active_index_generation(tmp_path)
    historical_before = _rows(
        tmp_path,
        "source",
        "raw_sessions",
        "raw_id IN (?, ?)",
        (semantic_raw_id, sibling_raw_id),
    )
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        source_count = source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]

    dry_run = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])

    assert dry_run.eligible_count == 1, dry_run.items[0].reason
    assert dry_run.items[0].legacy_null_native_id is True
    assert dry_run.items[0].repair_strategy == "copy_forward"
    assert dry_run.items[0].semantic_canonical_raw_id == semantic_raw_id
    assert dry_run.items[0].semantic_historical_raw_ids == (sibling_raw_id,)
    assert dry_run.items[0].semantic_witness_digest is not None
    assert (tmp_path / "index.db").resolve() == active_index
    receipt = tmp_path / "legacy-semantic-copy-receipt.jsonl"
    applied = repair_legacy_browser_capture_missing_native_ids(
        _config(tmp_path),
        [raw_id],
        apply=True,
        receipt_path=receipt,
        proof_digest=dry_run.proof_digest,
    )

    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert copy_raw_id is not None
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (source_count + 1,)
        assert source.execute("SELECT native_id FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone() == (None,)
    assert (
        _rows(tmp_path, "source", "raw_sessions", "raw_id IN (?, ?)", (semantic_raw_id, sibling_raw_id))
        == historical_before
    )
    lines = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert lines[-1]["transaction_protocol"] == "rollback-superjournal-v1"
    assert lines[-1]["semantic_witness_bindings"][0]["semantic_canonical_raw_id"] == semantic_raw_id
    assert lines[-1]["semantic_witness_bindings"][0]["semantic_historical_raw_ids"] == [sibling_raw_id]
    with closing(sqlite3.connect(tmp_path / "index.db")) as index:
        assert index.execute(
            "SELECT raw_id FROM sessions WHERE session_id = 'chatgpt-export:browser-origin-one'"
        ).fetchone() == (copy_raw_id,)


def test_legacy_browser_native_id_rejects_changed_semantic_witness_before_writing(tmp_path: Path) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, raw_id)
    dry_run = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])
    with closing(sqlite3.connect(tmp_path / "index.db")) as index, index:
        index.execute(
            "UPDATE raw_revision_heads SET accepted_frontier = accepted_frontier + 1 WHERE accepted_raw_id = ?",
            (semantic_raw_id,),
        )

    with pytest.raises(RuntimeError, match="proof digest"):
        repair_legacy_browser_capture_missing_native_ids(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=tmp_path / "legacy-semantic-stale-proof.jsonl",
            proof_digest=dry_run.proof_digest,
        )
    assert (
        _rows(
            tmp_path,
            "source",
            "raw_sessions",
            "source_path LIKE ?",
            ("browser-capture-origin-copy-forward/%",),
        )
        == []
    )


def test_legacy_browser_native_id_copy_forward_requires_single_membership_key(tmp_path: Path) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            )
            SELECT raw_id, 'chatgpt:competing-membership', provider_session_id,
                   source_revision, normalized_content_hash, message_count,
                   acquisition_generation, revision_authority, decision, decided_at_ms
            FROM raw_session_memberships
            WHERE raw_id = ?
            """,
            (raw_id,),
        )

    report = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert report.items[0].reason == "membership census does not exactly reproduce the accepted session"


def test_legacy_browser_native_id_copy_forward_requires_single_payload_blob_ref(tmp_path: Path) -> None:
    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            SELECT x'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
                   ref_id, ref_type, source_path, size_bytes, acquired_at_ms
            FROM blob_refs WHERE ref_id = ? AND ref_type = 'raw_payload'
            """,
            (raw_id,),
        )

    report = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert report.items[0].reason == "source envelope does not exactly bind the normalized session"


def test_legacy_browser_native_id_atomic_apply_rolls_back_after_source_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.storage.repair as repair_module

    raw_id = _seed_legacy_browser_head_without_native_id(tmp_path)
    dry_run = repair_legacy_browser_capture_missing_native_ids(_config(tmp_path), [raw_id])
    tables = {
        "source": ("raw_sessions", "blob_refs", "raw_session_memberships", "raw_membership_census"),
        "index": ("sessions", "raw_revision_heads", "raw_revision_applications"),
    }
    before = {
        (tier, table): _rows(tmp_path, tier, table, "1 = 1", ())
        for tier, tier_tables in tables.items()
        for table in tier_tables
    }

    def fail_after_source_stage(conn: sqlite3.Connection, item: object) -> None:
        raise RuntimeError("injected index transition failure")

    monkeypatch.setattr(repair_module, "_apply_browser_origin_repair_item", fail_after_source_stage)
    receipt = tmp_path / "legacy-atomic-rollback.jsonl"
    with pytest.raises(RuntimeError, match="injected index transition failure"):
        repair_legacy_browser_capture_missing_native_ids(
            _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
        )

    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned"]
    for (tier, table), rows in before.items():
        assert _rows(tmp_path, tier, table, "1 = 1", ()) == rows


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


@pytest.mark.parametrize(
    "mutation",
    [
        "head_frontier",
        "head_revision",
        "receipt_id",
        "receipt_time",
        "copy_predecessor_revision",
        "copy_predecessor_raw",
        "copy_append_start",
        "copy_append_end",
        "copy_capture_mode",
    ],
)
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
        elif mutation == "receipt_time":
            index.execute(
                "UPDATE raw_revision_applications SET decided_at_ms = decided_at_ms + 1 WHERE raw_id = ?",
                (copy_raw_id,),
            )
        else:
            with sqlite3.connect(tmp_path / "source.db") as source:
                if mutation == "copy_predecessor_revision":
                    source.execute(
                        "UPDATE raw_sessions SET predecessor_source_revision = ? WHERE raw_id = ?",
                        ("0" * 64, copy_raw_id),
                    )
                elif mutation == "copy_predecessor_raw":
                    source.execute(
                        "UPDATE raw_sessions SET predecessor_raw_id = ? WHERE raw_id = ?",
                        ("f" * 64, copy_raw_id),
                    )
                elif mutation == "copy_append_start":
                    source.execute("UPDATE raw_sessions SET append_start_offset = 0 WHERE raw_id = ?", (copy_raw_id,))
                elif mutation == "copy_append_end":
                    source.execute("UPDATE raw_sessions SET append_end_offset = 1 WHERE raw_id = ?", (copy_raw_id,))
                else:
                    source.execute("UPDATE raw_sessions SET capture_mode = 'unknown' WHERE raw_id = ?", (copy_raw_id,))

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


def test_browser_origin_repair_refuses_competing_terminal_copy_application(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    receipt = tmp_path / "competing-copy-application.jsonl"
    applied = repair_browser_capture_origin_mismatches(
        _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
    )
    item = applied.items[0]
    assert item.copy_forward_raw_id is not None
    assert item.session_id is not None
    assert item.canonical_logical_source_key is not None
    assert item.blob_hash is not None
    with sqlite3.connect(tmp_path / "source.db") as source:
        source_count = source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
    with sqlite3.connect(tmp_path / "index.db") as index:
        record_revision_application_sync(
            index,
            RevisionApplicationReceipt(
                raw_id=item.copy_forward_raw_id,
                session_id=item.session_id,
                logical_source_key=item.canonical_logical_source_key,
                source_revision=item.blob_hash,
                acquisition_generation=0,
                decision=ApplicationDecision.DEFERRED,
                accepted_raw_id=None,
                accepted_source_revision=None,
                accepted_content_hash=None,
                detail="competing copy receipt",
            ),
            decided_at_ms=99,
        )

    with pytest.raises(RuntimeError, match="ineligible"):
        repair_browser_capture_origin_mismatches(
            _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
        )
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (source_count,)
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned", "applied"]


@pytest.mark.parametrize(
    "decision", [ApplicationDecision.SUPERSEDED, ApplicationDecision.DEFERRED, ApplicationDecision.AMBIGUOUS]
)
def test_browser_origin_repair_refuses_extra_old_key_receipt(tmp_path: Path, decision: ApplicationDecision) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        blob_hash = source.execute(
            "SELECT lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone()[0]
    with sqlite3.connect(tmp_path / "index.db") as index:
        accepted_hash = bytes(
            index.execute(
                "SELECT content_hash FROM sessions WHERE session_id = 'chatgpt-export:browser-origin-one'"
            ).fetchone()[0]
        )
        record_revision_application_sync(
            index,
            RevisionApplicationReceipt(
                raw_id=raw_id,
                session_id="chatgpt-export:browser-origin-one",
                logical_source_key="unknown:browser-origin-one",
                source_revision=blob_hash,
                acquisition_generation=0,
                decision=decision,
                accepted_raw_id=raw_id if decision is ApplicationDecision.SUPERSEDED else None,
                accepted_source_revision=blob_hash if decision is ApplicationDecision.SUPERSEDED else None,
                accepted_content_hash=accepted_hash if decision is ApplicationDecision.SUPERSEDED else None,
                detail=f"unexpected {decision.value}",
            ),
            decided_at_ms=98,
        )

    assert repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id]).ineligible_count == 1


@pytest.mark.parametrize("head_shape", ["old", "semantic"])
def test_browser_origin_repair_refuses_head_receipt_timestamp_drift(tmp_path: Path, head_shape: str) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    accepted_raw_id = raw_id if head_shape == "old" else _seed_semantic_canonical_head(tmp_path, raw_id)
    with sqlite3.connect(tmp_path / "index.db") as index:
        index.execute(
            "UPDATE raw_revision_heads SET decided_at_ms = decided_at_ms + 1 WHERE accepted_raw_id = ?",
            (accepted_raw_id,),
        )

    assert repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id]).ineligible_count == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "receipt",
        "revision",
        "membership",
        "selected",
        "blob",
        "native_id",
        "source_index",
        "blob_ref_path",
        "predecessor_source",
        "predecessor_raw",
        "append_start",
        "append_end",
        "capture_mode",
        "negative_decided_at",
    ],
)
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
    elif mutation == "blob":
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
    elif mutation == "negative_decided_at":
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute("PRAGMA ignore_check_constraints = ON")
            index.execute(
                "UPDATE raw_revision_applications SET decided_at_ms = -1 WHERE raw_id = ?",
                (sibling_raw_id,),
            )
    else:
        updates = {
            "native_id": ("native_id = 'wrong-native-id'", ()),
            "source_index": ("source_index = 1", ()),
            "blob_ref_path": (None, ()),
            "predecessor_source": ("predecessor_source_revision = ?", ("0" * 64,)),
            "predecessor_raw": ("predecessor_raw_id = ?", ("f" * 64,)),
            "append_start": ("append_start_offset = 0", ()),
            "append_end": ("append_end_offset = 1", ()),
            "capture_mode": ("capture_mode = 'unknown'", ()),
        }
        assignment, params = updates[mutation]
        with sqlite3.connect(tmp_path / "source.db") as source:
            if assignment is None:
                source.execute(
                    "UPDATE blob_refs SET source_path = 'browser-capture/wrong.json' WHERE ref_id = ?",
                    (sibling_raw_id,),
                )
            else:
                source.execute(f"UPDATE raw_sessions SET {assignment} WHERE raw_id = ?", (*params, sibling_raw_id))

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


@pytest.mark.parametrize(
    "mutation",
    [
        "frontier",
        "pointer",
        "membership",
        "native_id",
        "source_index",
        "blob_ref_path",
        "predecessor_source",
        "predecessor_raw",
        "append_start",
        "append_end",
        "capture_mode",
    ],
)
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
    elif mutation == "membership":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                "UPDATE raw_session_memberships SET decision = 'ambiguous', decided_at_ms = 5 WHERE raw_id = ?",
                (semantic_raw_id,),
            )
    else:
        updates = {
            "native_id": ("native_id = 'wrong-native-id'", ()),
            "source_index": ("source_index = 1", ()),
            "blob_ref_path": (None, ()),
            "predecessor_source": ("predecessor_source_revision = ?", ("0" * 64,)),
            "predecessor_raw": ("predecessor_raw_id = ?", ("f" * 64,)),
            "append_start": ("append_start_offset = 0", ()),
            "append_end": ("append_end_offset = 1", ()),
            "capture_mode": ("capture_mode = 'unknown'", ()),
        }
        assignment, params = updates[mutation]
        with sqlite3.connect(tmp_path / "source.db") as source:
            if assignment is None:
                source.execute(
                    "UPDATE blob_refs SET source_path = 'browser-capture/wrong.json' WHERE ref_id = ?",
                    (semantic_raw_id,),
                )
            else:
                source.execute(f"UPDATE raw_sessions SET {assignment} WHERE raw_id = ?", (*params, semantic_raw_id))

    report = repair_browser_capture_origin_mismatches(_config(tmp_path), [mismatched_raw_id])

    assert report.ineligible_count == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "envelope",
        "membership",
        "application",
        "native_id",
        "source_index",
        "blob_ref_path",
        "predecessor_source",
        "predecessor_raw",
        "append_start",
        "append_end",
        "capture_mode",
    ],
)
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
    elif mutation == "application":
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "DELETE FROM raw_revision_applications WHERE raw_id = ? AND decision = 'selected_baseline'",
                (canonical_raw_id,),
            )
    else:
        updates = {
            "native_id": ("native_id = 'wrong-native-id'", ()),
            "source_index": ("source_index = 1", ()),
            "blob_ref_path": (None, ()),
            "predecessor_source": ("predecessor_source_revision = ?", ("0" * 64,)),
            "predecessor_raw": ("predecessor_raw_id = ?", ("f" * 64,)),
            "append_start": ("append_start_offset = 0", ()),
            "append_end": ("append_end_offset = 1", ()),
            "capture_mode": ("capture_mode = 'unknown'", ()),
        }
        assignment, params = updates[mutation]
        with sqlite3.connect(tmp_path / "source.db") as source:
            if assignment is None:
                source.execute(
                    "UPDATE blob_refs SET source_path = 'browser-capture/wrong.json' WHERE ref_id = ?",
                    (canonical_raw_id,),
                )
            else:
                source.execute(f"UPDATE raw_sessions SET {assignment} WHERE raw_id = ?", (*params, canonical_raw_id))

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


@pytest.mark.parametrize(
    "mutation",
    [
        "blob_size",
        "native_id",
        "source_index",
        "blob_ref_path",
        "predecessor_source",
        "predecessor_raw",
        "append_start",
        "append_end",
        "capture_mode",
    ],
)
def test_browser_capture_origin_copy_forward_reproves_before_source_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    import polylogue.storage.repair as repair_module

    raw_id = _seed_mismatched_browser_head(tmp_path)
    dry_run = repair_browser_capture_origin_mismatches(_config(tmp_path), [raw_id])
    original_verify = repair_module._verify_browser_origin_copy_forward_source_stage

    def mutate_at_locked_source_stage(
        archive_root: Path,
        source: sqlite3.Connection,
        item: repair_module.BrowserCaptureOriginRepairItem,
    ) -> None:
        if mutation == "blob_size":
            source.execute("UPDATE raw_sessions SET blob_size = blob_size + 1 WHERE raw_id = ?", (raw_id,))
        elif mutation == "native_id":
            source.execute("UPDATE raw_sessions SET native_id = 'wrong-native-id' WHERE raw_id = ?", (raw_id,))
        elif mutation == "source_index":
            source.execute("UPDATE raw_sessions SET source_index = 1 WHERE raw_id = ?", (raw_id,))
        elif mutation == "blob_ref_path":
            source.execute(
                "UPDATE blob_refs SET source_path = 'browser-capture/wrong.json' WHERE ref_id = ?", (raw_id,)
            )
        elif mutation == "predecessor_source":
            source.execute(
                "UPDATE raw_sessions SET predecessor_source_revision = ? WHERE raw_id = ?", ("0" * 64, raw_id)
            )
        elif mutation == "predecessor_raw":
            source.execute("UPDATE raw_sessions SET predecessor_raw_id = ? WHERE raw_id = ?", ("f" * 64, raw_id))
        elif mutation == "append_start":
            source.execute("UPDATE raw_sessions SET append_start_offset = 0 WHERE raw_id = ?", (raw_id,))
        elif mutation == "append_end":
            source.execute("UPDATE raw_sessions SET append_end_offset = 1 WHERE raw_id = ?", (raw_id,))
        else:
            source.execute("UPDATE raw_sessions SET capture_mode = 'chatgpt' WHERE raw_id = ?", (raw_id,))
        original_verify(archive_root, source, item)

    monkeypatch.setattr(
        repair_module, "_verify_browser_origin_copy_forward_source_stage", mutate_at_locked_source_stage
    )
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


@pytest.mark.parametrize(
    "native_id",
    (
        "no-canonical-773bbbf1",
        "no-canonical-bd47782e",
        "no-canonical-f43a203a",
    ),
)
def test_byte_proven_browser_rekey_copy_forwards_each_no_head_shape(tmp_path: Path, native_id: str) -> None:
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path, native_id)
    if native_id == "no-canonical-773bbbf1":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute("ALTER TABLE raw_sessions DROP COLUMN capture_mode")
            source.execute("PRAGMA user_version = 7")
    active_index = _stage_active_index_generation(tmp_path)
    old_evidence = {
        ("source", table): _rows(
            tmp_path, "source", table, "raw_id = ?" if table != "blob_refs" else "ref_id = ?", (raw_id,)
        )
        for table in ("raw_sessions", "blob_refs", "raw_session_memberships", "raw_membership_census")
    } | {
        ("index", "raw_revision_heads"): _rows(
            tmp_path, "index", "raw_revision_heads", "accepted_raw_id = ?", (raw_id,)
        ),
        ("index", "raw_revision_applications"): _rows(
            tmp_path, "index", "raw_revision_applications", "raw_id = ?", (raw_id,)
        ),
    }

    dry_run = repair_byte_proven_browser_capture_null_native_ids(_config(tmp_path), [raw_id])

    assert dry_run.eligible_count == 1, dry_run.items[0].reason
    assert dry_run.items[0].byte_proven_null_native_id_rekey is True
    assert dry_run.items[0].parsed_message_count == 1
    assert (tmp_path / "index.db").resolve() == active_index
    receipt = tmp_path / f"byte-rekey-{native_id}.jsonl"
    applied = repair_byte_proven_browser_capture_null_native_ids(
        _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
    )

    copy_raw_id = applied.items[0].copy_forward_raw_id
    assert applied.repaired_count == 1
    assert copy_raw_id is not None
    for (tier, table), before in old_evidence.items():
        key = "ref_id" if table == "blob_refs" else "accepted_raw_id" if table == "raw_revision_heads" else "raw_id"
        assert _rows(tmp_path, tier, table, f"{key} = ?", (raw_id,)) == before
    with closing(sqlite3.connect(tmp_path / "source.db")) as source:
        assert source.execute(
            "SELECT origin, native_id, logical_source_key, revision_authority FROM raw_sessions WHERE raw_id = ?",
            (copy_raw_id,),
        ).fetchone() == ("chatgpt-export", native_id, f"chatgpt:{native_id}", "byte_proven")
        assert source.execute(
            "SELECT provider_session_id, message_count, revision_authority, decision FROM raw_session_memberships "
            "WHERE raw_id = ?",
            (copy_raw_id,),
        ).fetchone() == (native_id, 1, "byte_proven", "applied")
    lines = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [line["state"] for line in lines] == ["planned", "applied"]
    assert lines[-1]["transaction_protocol"] == "rollback-superjournal-v1"
    reapplied = repair_byte_proven_browser_capture_null_native_ids(
        _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
    )
    assert reapplied.repaired_count == 0
    assert reapplied.already_repaired_count == 1


@pytest.mark.parametrize("case", ("semantic-2af730ea", "semantic-27527c15"))
def test_byte_proven_browser_rekey_preserves_exact_semantic_witness(tmp_path: Path, case: str) -> None:
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, raw_id)
    sibling_raw_id = _seed_semantic_superseded_sibling(tmp_path, semantic_raw_id)
    historical_before = _rows(tmp_path, "source", "raw_sessions", "raw_id IN (?, ?)", (semantic_raw_id, sibling_raw_id))

    dry_run = repair_byte_proven_browser_capture_null_native_ids(_config(tmp_path), [raw_id])

    assert dry_run.eligible_count == 1, dry_run.items[0].reason
    assert dry_run.items[0].semantic_canonical_raw_id == semantic_raw_id
    assert dry_run.items[0].semantic_historical_raw_ids == (sibling_raw_id,)
    receipt = tmp_path / f"byte-rekey-semantic-{case}.jsonl"
    applied = repair_byte_proven_browser_capture_null_native_ids(
        _config(tmp_path), [raw_id], apply=True, receipt_path=receipt, proof_digest=dry_run.proof_digest
    )

    assert applied.repaired_count == 1
    assert (
        _rows(tmp_path, "source", "raw_sessions", "raw_id IN (?, ?)", (semantic_raw_id, sibling_raw_id))
        == historical_before
    )
    assert applied.items[0].semantic_canonical_raw_id == semantic_raw_id
    assert applied.items[0].semantic_historical_raw_ids == (sibling_raw_id,)


@pytest.mark.parametrize(
    "conflict",
    ("canonical_hash_conflict", "superseded_equivalent_membership", "canonical_byte_head", "current_reparse_drift"),
)
def test_byte_proven_browser_rekey_fails_closed_for_observed_conflicts(tmp_path: Path, conflict: str) -> None:
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    if conflict == "canonical_hash_conflict":
        semantic_raw_id = _seed_semantic_canonical_head(tmp_path, raw_id)
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "UPDATE raw_revision_heads SET accepted_content_hash = x'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF' "
                "WHERE accepted_raw_id = ?",
                (semantic_raw_id,),
            )
    elif conflict == "superseded_equivalent_membership":
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                """
                INSERT INTO raw_session_memberships (
                    raw_id, logical_source_key, provider_session_id, source_revision,
                    normalized_content_hash, message_count, acquisition_generation,
                    revision_authority, decision, decided_at_ms
                ) SELECT raw_id, logical_source_key, 'browser-origin-one', source_revision,
                         x'0000000000000000000000000000000000000000000000000000000000000000', 1, 1,
                         'quarantined', 'superseded_equivalent', 3
                  FROM raw_sessions WHERE raw_id = ?
                """,
                (raw_id,),
            )
    elif conflict == "canonical_byte_head":
        _seed_equivalent_canonical_head(tmp_path, raw_id)
    else:
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute("UPDATE raw_sessions SET source_revision = ? WHERE raw_id = ?", ("0" * 64, raw_id))

    report = repair_byte_proven_browser_capture_null_native_ids(_config(tmp_path), [raw_id])

    assert report.ineligible_count == 1
    assert report.items[0].status == "ineligible"


def test_byte_proven_browser_rekey_refuses_stale_proof_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.storage.repair as repair_module

    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    dry_run = repair_byte_proven_browser_capture_null_native_ids(_config(tmp_path), [raw_id])
    with sqlite3.connect(tmp_path / "index.db") as index:
        index.execute(
            "UPDATE raw_revision_heads SET accepted_frontier = accepted_frontier + 1 WHERE accepted_raw_id = ?",
            (raw_id,),
        )
    with pytest.raises(RuntimeError, match="one or more targets are ineligible"):
        repair_byte_proven_browser_capture_null_native_ids(
            _config(tmp_path),
            [raw_id],
            apply=True,
            receipt_path=tmp_path / "stale.jsonl",
            proof_digest=dry_run.proof_digest,
        )

    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path / "rollback")
    dry_run = repair_byte_proven_browser_capture_null_native_ids(_config(tmp_path / "rollback"), [raw_id])
    before = {
        (tier, table): _rows(tmp_path / "rollback", tier, table, "1 = 1", ())
        for tier, tables in {
            "source": ("raw_sessions", "blob_refs", "raw_session_memberships", "raw_membership_census"),
            "index": ("sessions", "raw_revision_heads", "raw_revision_applications"),
        }.items()
        for table in tables
    }

    def fail_after_source_stage(conn: sqlite3.Connection, item: object) -> None:
        raise RuntimeError("injected byte rekey index failure")

    monkeypatch.setattr(repair_module, "_apply_browser_origin_repair_item", fail_after_source_stage)
    receipt = tmp_path / "rollback" / "rollback.jsonl"
    with pytest.raises(RuntimeError, match="injected byte rekey index failure"):
        repair_byte_proven_browser_capture_null_native_ids(
            _config(tmp_path / "rollback"),
            [raw_id],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned"]
    for (tier, table), rows in before.items():
        assert _rows(tmp_path / "rollback", tier, table, "1 = 1", ()) == rows


# --- polylogue-lkrc.3: evidence packets + durable blockers for unresolved conflicts ---


def test_inspect_conflicts_reports_resolved_when_actuator_would_succeed(tmp_path: Path) -> None:
    """A raw the ordinary rekey actuator can already repair is not a conflict."""
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)

    report = inspect_browser_canonical_authority_conflicts(_config(tmp_path), [raw_id])

    assert report.requested_count == 1
    assert report.resolved_count == 1
    assert report.conflict_count == 0
    assert report.items == ()


def test_inspect_conflicts_semantic_hash_divergence_evidence(tmp_path: Path) -> None:
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, raw_id)
    with sqlite3.connect(tmp_path / "index.db") as index:
        index.execute(
            "UPDATE raw_revision_heads SET accepted_content_hash = x'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF' "
            "WHERE accepted_raw_id = ?",
            (semantic_raw_id,),
        )

    report = inspect_browser_canonical_authority_conflicts(_config(tmp_path), [raw_id])

    assert report.conflict_count == 1
    assert report.resolved_count == 0
    item = report.items[0]
    assert item.raw_id == raw_id
    assert item.session_id == "chatgpt-export:browser-origin-one"
    assert item.canonical_logical_source_key == "chatgpt:browser-origin-one"
    assert item.competing_raw_id == semantic_raw_id
    assert item.competing_content_hash == "ff" * 32
    assert item.competing_frontier_kind == "semantic"
    assert item.unknown_raw_content_hash is not None
    assert item.unknown_raw_content_hash != item.competing_content_hash
    # Semantic frontiers are not a single retained raw's bytes, so a
    # message-level diff is deliberately not fabricated for them.
    assert item.divergent_message_index is None
    assert item.divergence_note is not None and "semantic frontier" in item.divergence_note
    assert item.evidence_digest is not None


def test_unified_frontier_conflict_requires_accepted_judgment_and_stays_visible(tmp_path: Path) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    _seed_diverging_canonical_byte_head(tmp_path, mismatched_raw_id)

    census = inspect_raw_authority_frontier(_config(tmp_path))

    conflict = next(item for item in census.items if item.raw_id == mismatched_raw_id)
    assert conflict.state is RawAuthorityFrontierState.CONFLICTING_AUTHORITY_NEEDS_JUDGMENT
    assert conflict.actuator is RawAuthorityActuator.REQUEST_JUDGMENT
    assert conflict.executable is False
    with sqlite3.connect(tmp_path / "source.db") as source:
        blocker_id, observed_json = source.execute(
            "SELECT blocker_id, observed_json FROM raw_authority_blockers WHERE plan_id = ? AND resolved_at_ms IS NULL",
            (conflict.plan_id,),
        ).fetchone()
    assertion_id = json.loads(observed_json)["judgment_assertion_id"]
    with sqlite3.connect(tmp_path / "user.db") as user:
        assert user.execute("SELECT status FROM assertions WHERE assertion_id = ?", (assertion_id,)).fetchone() == (
            AssertionStatus.CANDIDATE.value,
        )
    with pytest.raises(RuntimeError, match="explicitly accepted"):
        resolve_raw_authority_blocker(
            tmp_path,
            blocker_id,
            resolution="retain both authorities pending a future evidence change",
            assertion_id=assertion_id,
        )
    with sqlite3.connect(tmp_path / "user.db") as user, user:
        assert mark_assertion_status(user, assertion_id, AssertionStatus.ACCEPTED)
    resolved = resolve_raw_authority_blocker(
        tmp_path,
        blocker_id,
        resolution="retain both authorities pending a future evidence change",
        assertion_id=assertion_id,
    )
    assert resolved["operator_assertion_id"] == assertion_id

    repeated = inspect_raw_authority_frontier(_config(tmp_path))
    repeated_conflict = next(item for item in repeated.items if item.raw_id == mismatched_raw_id)
    assert repeated_conflict.plan_id == conflict.plan_id
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT COUNT(*) FROM raw_authority_blockers WHERE plan_id = ? AND resolved_at_ms IS NULL",
            (conflict.plan_id,),
        ).fetchone() == (0,)


def test_inspect_conflicts_membership_precondition_evidence(tmp_path: Path) -> None:
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority, decision, decided_at_ms
            ) SELECT raw_id, logical_source_key, 'browser-origin-one', source_revision,
                     x'0000000000000000000000000000000000000000000000000000000000000000', 1, 1,
                     'quarantined', 'superseded_equivalent', 3
              FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id,),
        )

    report = inspect_browser_canonical_authority_conflicts(_config(tmp_path), [raw_id])

    assert report.conflict_count == 1
    item = report.items[0]
    assert item.competing_raw_id is None
    assert item.divergence_note is not None
    assert "membership row" in item.divergence_note
    assert "superseded_equivalent" in item.divergence_note


def test_inspect_conflicts_matching_hash_is_membership_shaped_not_a_divergence(tmp_path: Path) -> None:
    """Equal content hashes mean the actuator's block is precondition-shaped, not authority conflict."""
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    canonical_raw_id = _seed_equivalent_canonical_head(tmp_path, raw_id)

    report = inspect_browser_canonical_authority_conflicts(_config(tmp_path), [raw_id])

    assert report.conflict_count == 1
    item = report.items[0]
    assert item.competing_raw_id == canonical_raw_id
    assert item.competing_content_hash == item.unknown_raw_content_hash
    assert item.divergent_message_index is None
    assert item.divergence_note is not None and "not a hash divergence" in item.divergence_note


def test_inspect_conflicts_byte_frontier_message_divergence_evidence(tmp_path: Path) -> None:
    """The untested byte-frontier competing-head diff path: real message-level divergence.

    Unlike ``test_inspect_conflicts_matching_hash_is_membership_shaped_not_a_divergence``
    (identical content, "not a hash divergence" branch) and
    ``test_inspect_conflicts_semantic_hash_divergence_evidence`` (semantic
    frontier, no message diff computed), this seeds a competing head that is
    both byte-frontier AND genuinely content-divergent -- the
    ``elif competing_frontier_kind == "byte" and competing_raw_id != raw_id``
    branch in ``_browser_canonical_authority_conflict_witness`` that re-reads
    the competing raw's own retained blob, reparses it, and computes
    ``divergent_message_index`` by zipping two message-hash projections. This
    is the shape of the real production conflict this evidence-builder
    exists to explain (an incompatible canonical byte head diverging from
    message 0). Deleting or breaking this branch (e.g. reverting to the
    semantic-frontier "no diff available" note, or leaving
    ``divergent_message_index`` unset) makes this test fail.
    """
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    competing_raw_id = _seed_diverging_canonical_byte_head(tmp_path, raw_id)

    report = inspect_browser_canonical_authority_conflicts(_config(tmp_path), [raw_id])

    assert report.conflict_count == 1
    item = report.items[0]
    assert item.competing_raw_id == competing_raw_id
    assert item.competing_frontier_kind == "byte"
    assert item.competing_content_hash is not None
    assert item.competing_content_hash != item.unknown_raw_content_hash
    # The competing head's own retained bytes were re-parsed (2 messages),
    # proving this is a real re-derivation, not a stub/None fallback.
    assert item.competing_message_count == 2
    assert item.unknown_raw_message_count == 1
    assert item.divergent_message_index == 0
    assert item.divergence_note is None
    assert item.evidence_digest is not None


def test_inspect_conflicts_rejects_duplicate_and_malformed_ids(tmp_path: Path) -> None:
    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        inspect_browser_canonical_authority_conflicts(_config(tmp_path), [raw_id, raw_id])
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        inspect_browser_canonical_authority_conflicts(_config(tmp_path), ["not-a-raw-id"])
    with pytest.raises(ValueError, match="1..100 entries"):
        inspect_browser_canonical_authority_conflicts(_config(tmp_path), [])


def test_record_conflict_blockers_persists_durable_candidate_assertions(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope

    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, raw_id)
    with sqlite3.connect(tmp_path / "index.db") as index:
        index.execute(
            "UPDATE raw_revision_heads SET accepted_content_hash = x'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF' "
            "WHERE accepted_raw_id = ?",
            (semantic_raw_id,),
        )

    report, assertion_ids = record_browser_canonical_authority_conflict_blockers(_config(tmp_path), [raw_id])

    assert report.conflict_count == 1
    assert len(assertion_ids) == 1
    with closing(sqlite3.connect(tmp_path / "user.db")) as user_conn:
        user_conn.row_factory = sqlite3.Row
        envelope = read_assertion_envelope(user_conn, assertion_ids[0])
        assert envelope is not None
        assert envelope.kind.value == "blocker"
        # Automated writers can never self-promote (upsert_assertion's chokepoint,
        # 37t.15): the row is a candidate awaiting explicit operator judgment even
        # though this function requested status=candidate/visibility=private itself.
        assert envelope.status.value == "candidate"
        assert envelope.context_policy["inject"] is False
        assert envelope.target_ref == "session:chatgpt-export:browser-origin-one"
        assert isinstance(envelope.value, dict)
        assert envelope.value["raw_id"] == raw_id
        assert envelope.value["competing_raw_id"] == semantic_raw_id

    # Re-running over identical evidence is idempotent: same assertion id, not a
    # duplicate row.
    _report_again, assertion_ids_again = record_browser_canonical_authority_conflict_blockers(
        _config(tmp_path), [raw_id]
    )
    assert assertion_ids_again == assertion_ids
    with closing(sqlite3.connect(tmp_path / "user.db")) as user_conn:
        count = user_conn.execute(
            "SELECT COUNT(*) FROM assertions WHERE kind = 'blocker' AND target_ref = ?",
            ("session:chatgpt-export:browser-origin-one",),
        ).fetchone()[0]
        assert count == 1


def test_record_conflict_blockers_never_clobbers_an_operator_judged_row(tmp_path: Path) -> None:
    """A judged blocker (accepted/rejected/deferred/superseded) must survive a re-run.

    Mirrors ``upsert_pathology_findings_as_assertions``'s terminal-judgment
    chokepoint (37t.15): once an operator has judged a candidate this
    detector produced, a later automated re-run over the SAME evidence must
    not resurrect or mutate the judged row's status/value -- only
    ``upsert_assertion``'s own ON CONFLICT DO UPDATE would otherwise
    overwrite the display fields (value/body_text/evidence_refs) on every
    call, since only ``status`` itself is protected by that function's
    chokepoint. Deleting the ``read_assertion_envelope``/``existing.status``
    guard in ``record_browser_canonical_authority_conflict_blockers`` makes
    this test fail: the accepted row's value would silently be overwritten
    back to the detector's regenerated evidence payload.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        judge_assertion_candidate,
        read_assertion_envelope,
    )

    raw_id = _seed_byte_proven_browser_head_without_native_id(tmp_path)
    semantic_raw_id = _seed_semantic_canonical_head(tmp_path, raw_id)
    with sqlite3.connect(tmp_path / "index.db") as index:
        index.execute(
            "UPDATE raw_revision_heads SET accepted_content_hash = x'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF' "
            "WHERE accepted_raw_id = ?",
            (semantic_raw_id,),
        )

    _report, assertion_ids = record_browser_canonical_authority_conflict_blockers(_config(tmp_path), [raw_id])
    assert len(assertion_ids) == 1
    assertion_ref = assertion_ids[0]

    with closing(sqlite3.connect(tmp_path / "user.db")) as user_conn:
        judge_assertion_candidate(
            user_conn,
            candidate_ref=assertion_ref,
            decision="accept",
            reason="operator confirmed this conflict is real",
        )
        user_conn.commit()
        judged = read_assertion_envelope(user_conn, assertion_ref)
        assert judged is not None
        assert judged.status.value == "accepted"
        judged_value = judged.value
        judged_updated_at_ms = judged.updated_at_ms

    # Re-running the detector over the identical evidence must leave the
    # judged row's status and value exactly as the operator left it. Passing
    # a distinct now_ms is the genuine negative control: value/status are
    # deterministically identical across both runs regardless of whether the
    # guard fires (same evidence -> same computed value, and
    # upsert_assertion's own terminal-judgment chokepoint separately protects
    # status), so those two fields alone cannot distinguish "guard
    # short-circuited before writing" from "guard removed, wrote anyway".
    # updated_at_ms can: upsert_assertion's ON CONFLICT always sets
    # updated_at_ms = excluded.updated_at_ms unconditionally, so if the guard
    # in record_browser_canonical_authority_conflict_blockers were removed
    # (or its read_assertion_envelope check bypassed), this second call would
    # still invoke upsert_assertion for the judged row and updated_at_ms
    # would move to the new now_ms -- an observable difference the guard
    # must prevent entirely by never calling upsert_assertion at all.
    _report_again, assertion_ids_again = record_browser_canonical_authority_conflict_blockers(
        _config(tmp_path), [raw_id], now_ms=judged_updated_at_ms + 999_000
    )
    assert assertion_ids_again == assertion_ids
    with closing(sqlite3.connect(tmp_path / "user.db")) as user_conn:
        still_judged = read_assertion_envelope(user_conn, assertion_ref)
        assert still_judged is not None
        assert still_judged.status.value == "accepted"
        assert still_judged.value == judged_value
        assert still_judged.updated_at_ms == judged_updated_at_ms
        # ``judge_assertion_candidate(decision="accept")`` legitimately leaves
        # TWO ``blocker`` rows for this target: the original candidate
        # (mutated in place to ``status=accepted`` by ``mark_assertion_status``,
        # same assertion id as ``assertion_ref``) plus a separate promoted
        # "resulting assertion" row that ``_promote_candidate_assertion``
        # inserts under a distinct id. The guard under test only has to leave
        # the ORIGINAL judged row alone -- it must not grow a THIRD row (which
        # would mean the re-run resurrected or duplicated the candidate).
        count = user_conn.execute(
            "SELECT COUNT(*) FROM assertions WHERE kind = 'blocker' AND target_ref = ?",
            ("session:chatgpt-export:browser-origin-one",),
        ).fetchone()[0]
        assert count == 2
