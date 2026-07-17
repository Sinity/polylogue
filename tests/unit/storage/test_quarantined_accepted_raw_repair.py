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
from polylogue.storage.raw_reconciler import (
    RawAuthorityFrontierState,
    apply_raw_authority_frontier,
    inspect_raw_authority_frontier,
)
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
def test_unified_frontier_applies_quarantine_refinement_without_incident_receipt(
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
            ) VALUES ('unified-artifact-witness', ?, 'chatgpt-export', 'repair-one.json', 0,
                      'session', 'supported_parseable', 'witness', 1, 1, 0, 1, 2)
            """,
            (raw_id,),
        )
    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == raw_id)

    report = apply_raw_authority_frontier(
        _config(tmp_path),
        preview_census_id=preview.census_id,
        selected_plan_ids=(selected.plan_id,),
    )

    assert report.executed_plan_count == 1
    assert report.retryable_plan_count == 0
    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT revision_authority, baseline_raw_id FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == ("byte_proven", raw_id)
    assert not (tmp_path / "recovery").exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing_blob", "blob_ref", "frontier", "session_hash", "application", "membership", "envelope"],
)
def test_unified_quarantine_strategy_rejects_mutated_authority_witness(tmp_path: Path, mutation: str) -> None:
    raw_id = _seed_invalid_head(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source, sqlite3.connect(tmp_path / "index.db") as index:
        if mutation == "missing_blob":
            blob_hash = str(
                source.execute("SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
            )
            BlobStore(tmp_path / "blob").blob_path(blob_hash.lower()).unlink()
        elif mutation == "blob_ref":
            source.execute("UPDATE blob_refs SET size_bytes = size_bytes + 1 WHERE ref_id = ?", (raw_id,))
        elif mutation == "frontier":
            index.execute("UPDATE raw_revision_heads SET accepted_frontier = accepted_frontier + 1")
        elif mutation == "session_hash":
            index.execute("UPDATE sessions SET content_hash = zeroblob(32)")
        elif mutation == "application":
            index.execute("UPDATE raw_revision_applications SET accepted_raw_id = ?", ("1" * 64,))
        elif mutation == "membership":
            source.execute("UPDATE raw_membership_census SET status = 'failed', member_count = 0")
        elif mutation == "envelope":
            source.execute("UPDATE raw_sessions SET logical_source_key = 'partial' WHERE raw_id = ?", (raw_id,))
        source.commit()
        index.commit()
    before = _logical_state(tmp_path, raw_id)

    census = inspect_raw_authority_frontier(_config(tmp_path))
    item = next(item for item in census.items if item.raw_id == raw_id)

    assert item.state is not RawAuthorityFrontierState.SAFELY_REKEYABLE
    assert _logical_state(tmp_path, raw_id) == before
