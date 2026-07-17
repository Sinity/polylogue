from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.config import Config
from polylogue.core.enums import AssertionStatus, Provider
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.revision_backfill import _parse_one
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import resolve_raw_authority_blocker
from polylogue.storage.raw_reconciler import (
    RawAuthorityActuator,
    RawAuthorityFrontierState,
    apply_raw_authority_frontier,
    inspect_raw_authority_frontier,
)
from polylogue.storage.repair import (
    inspect_browser_canonical_authority_conflicts,
    record_browser_canonical_authority_conflict_blockers,
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


def test_unified_frontier_applies_browser_origin_without_incident_receipt(tmp_path: Path) -> None:
    raw_id = _seed_mismatched_browser_head(tmp_path)
    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == raw_id)

    report = apply_raw_authority_frontier(
        _config(tmp_path),
        preview_census_id=preview.census_id,
        selected_plan_ids=(selected.plan_id,),
    )

    assert report.executed_plan_count == 1
    assert report.retryable_plan_count == 0
    postflight = inspect_raw_authority_frontier(_config(tmp_path))
    assert all(item.raw_id != raw_id or item.state is RawAuthorityFrontierState.SUPERSEDED for item in postflight.items)
    assert set(postflight.state_counts) <= {
        RawAuthorityFrontierState.PROVEN_CURRENT.value,
        RawAuthorityFrontierState.SUPERSEDED.value,
    }
    assert not (tmp_path / "recovery").exists()


def test_unified_frontier_restores_equivalent_canonical_browser_head(tmp_path: Path) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    canonical_raw_id = _seed_equivalent_canonical_head(tmp_path, mismatched_raw_id)
    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == mismatched_raw_id)
    strategy_item = cast(dict[str, Any], selected.strategy_witness["item"])
    assert strategy_item["repair_strategy"] == "restore_canonical_head"

    report = apply_raw_authority_frontier(
        _config(tmp_path),
        preview_census_id=preview.census_id,
        selected_plan_ids=(selected.plan_id,),
    )

    assert report.executed_plan_count == 1
    assert report.retryable_plan_count == 0
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute(
            "SELECT raw_id FROM sessions WHERE session_id = 'chatgpt-export:browser-origin-one'"
        ).fetchone() == (canonical_raw_id,)
    postflight = inspect_raw_authority_frontier(_config(tmp_path))
    assert set(postflight.state_counts) <= {
        RawAuthorityFrontierState.PROVEN_CURRENT.value,
        RawAuthorityFrontierState.SUPERSEDED.value,
    }


@pytest.mark.parametrize("authority", ["quarantined", "byte_proven"])
def test_unified_frontier_admits_historical_null_native_id_strategies(tmp_path: Path, authority: str) -> None:
    raw_id = (
        _seed_legacy_browser_head_without_native_id(tmp_path)
        if authority == "quarantined"
        else _seed_byte_proven_browser_head_without_native_id(tmp_path)
    )

    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == raw_id)
    strategy = cast(dict[str, object], selected.strategy_witness["item"])

    assert selected.state is RawAuthorityFrontierState.SAFELY_REKEYABLE
    assert selected.actuator is RawAuthorityActuator.COPY_FORWARD_ORIGIN
    assert strategy["legacy_null_native_id"] is (authority == "quarantined")
    assert strategy["byte_proven_null_native_id_rekey"] is (authority == "byte_proven")

    report = apply_raw_authority_frontier(
        _config(tmp_path),
        preview_census_id=preview.census_id,
        selected_plan_ids=(selected.plan_id,),
    )

    assert report.executed_plan_count == 1
    assert report.retryable_plan_count == 0


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


def test_unified_frontier_conflict_requires_typed_judgment_then_resumes_same_evidence(tmp_path: Path) -> None:
    mismatched_raw_id = _seed_mismatched_browser_head(tmp_path)
    canonical_raw_id = _seed_diverging_canonical_byte_head(tmp_path, mismatched_raw_id)

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
    with pytest.raises(RuntimeError, match="disposition=retain_canonical_authority"):
        resolve_raw_authority_blocker(
            tmp_path,
            blocker_id,
            resolution="retain the typed canonical authority",
            assertion_id=assertion_id,
        )
    resolved = resolve_raw_authority_blocker(
        tmp_path,
        blocker_id,
        resolution="retain the typed canonical authority",
        assertion_id=assertion_id,
        judgment_disposition="retain_canonical_authority",
    )
    assert resolved["operator_assertion_id"] == assertion_id
    assert resolved["judgment_disposition"] == "retain_canonical_authority"

    repeated = inspect_raw_authority_frontier(_config(tmp_path))
    successor = next(item for item in repeated.items if item.raw_id == mismatched_raw_id)
    assert successor.plan_id != conflict.plan_id
    assert successor.state is RawAuthorityFrontierState.SAFELY_REKEYABLE
    assert successor.actuator is RawAuthorityActuator.RESOLVE_CONFLICT
    applied = apply_raw_authority_frontier(
        _config(tmp_path),
        preview_census_id=repeated.census_id,
        selected_plan_ids=(successor.plan_id,),
    )
    assert applied.executed_plan_count == 1
    assert applied.retryable_plan_count == 0
    postflight = inspect_raw_authority_frontier(_config(tmp_path))
    assert set(postflight.state_counts) <= {
        RawAuthorityFrontierState.PROVEN_CURRENT.value,
        RawAuthorityFrontierState.SUPERSEDED.value,
    }
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute(
            "SELECT raw_id FROM sessions WHERE session_id = 'chatgpt-export:browser-origin-one'"
        ).fetchone() == (canonical_raw_id,)
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
