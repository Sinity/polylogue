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
    RawAuthorityActuator,
    RawAuthorityFrontierState,
    apply_raw_authority_frontier,
    inspect_raw_authority_frontier,
)
from polylogue.storage.repair import inspect_quarantined_accepted_raws
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


def _seed_quarantined_raw_fanout(root: Path) -> tuple[str, tuple[tuple[str, str], ...]]:
    """One quarantined raw accepted as the head by TWO different sessions.

    Mirrors the fan-out shape in ``test_duplicate_raw_identity_repair.py``'s
    ``_seed_duplicate_raw_fanout`` (forked/subagent/resumed sessions can
    physically replay the identical parent evidence, so the exact same
    ``raw_id`` can legitimately be the accepted head of more than one
    logical source key/session at once, polylogue-ihc8) -- but for
    quarantine-refinement specifically, the raw's own bytes can only ever
    re-parse into ONE genuine session's content
    (``_inspect_quarantined_accepted_raw`` requires exactly one normalized
    session and an exact content-hash match). So unlike duplicate-alias
    fan-out, at most ONE sibling here can ever be genuinely eligible;
    ``session_a`` is that one (its accepted content-hash/session-id
    genuinely match what the raw re-parses to). ``session_b`` represents a
    second, stale accepted-head row for the SAME raw_id whose own
    identity/content does not match -- exactly the shape observed live
    (a physically shared raw whose accepted-head rows for OTHER sessions
    can never re-verify against it). Before the fix, inspecting EITHER
    session crashed with "expected one accepted head, found 2" regardless
    of this distinction; after the fix, session_a is provably eligible and
    session_b is provably (and permanently) ineligible via the real
    content-mismatch check -- neither crashes the census.
    """
    initialize_active_archive_root(root)
    records = [_chatgpt_session("fanout-quarantine", "proof text")]
    payload = json.dumps(records[0], sort_keys=True).encode()
    source_path = "fanout-quarantine.json"
    parsed = _parse_one(Provider.CHATGPT, payload, source_path)
    assert parsed
    session_a = parsed[0]
    source_revision = hashlib.sha256(payload).hexdigest()
    content_hash_a = bytes.fromhex(session_content_hash(session_a))
    key_a = "chatgpt:fanout-quarantine"

    # Same origin (chatgpt-export) as session_a throughout -- a genuine
    # origin mismatch is a DIFFERENT, unrelated repair path (browser-origin
    # conflict), not the quarantine-refinement fan-out this test targets.
    stale_records = [_chatgpt_session("fanout-quarantine-stale-sibling", "unrelated stale content")]
    stale_payload = json.dumps(stale_records[0], sort_keys=True).encode()
    stale_parsed = _parse_one(Provider.CHATGPT, stale_payload, "fanout-quarantine-stale.json")
    assert stale_parsed
    stale_session = stale_parsed[0]
    content_hash_b = bytes.fromhex(session_content_hash(stale_session))
    key_b = "chatgpt:fanout-quarantine-stale-sibling"

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT, payload=payload, source_path=source_path, acquired_at_ms=1
        )
        _raw_id, session_id_a = archive.write_parsed_for_retained_raw(
            session_a, raw_id=raw_id, source_path=source_path, acquired_at_ms=1, revision_authoritative=True
        )
        assert session_id_a == "chatgpt-export:fanout-quarantine"
        # session_b's own row is written to a DISTINCT raw first (it needs a
        # real, materialized session/index row to exist), then retargeted
        # onto the SAME shared raw_id -- reproducing "this session's own
        # accepted head points at a raw it doesn't actually match" without
        # needing a second real parseable payload.
        stale_raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=stale_payload,
            source_path="fanout-quarantine-stale.json",
            acquired_at_ms=1,
        )
        _stale_raw_id, session_id_b = archive.write_parsed_for_retained_raw(
            stale_session,
            raw_id=stale_raw_id,
            source_path="fanout-quarantine-stale.json",
            acquired_at_ms=1,
            revision_authoritative=True,
        )
        for session_id, logical_source_key, content_hash in (
            (session_id_a, key_a, content_hash_a),
            (session_id_b, key_b, content_hash_b),
        ):
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
                    detail=f"pre-quarantine fanout fixture ({logical_source_key})",
                ),
                decided_at_ms=2,
            )
        archive.commit()
    with sqlite3.connect(root / "index.db") as index:
        # Retarget session_b's own materialized session + accepted head onto
        # the shared raw_id (it was written against stale_raw_id above only
        # to get a real row to retarget).
        index.execute("UPDATE sessions SET raw_id = ? WHERE session_id = ?", (raw_id, session_id_b))
        index.execute("UPDATE raw_revision_heads SET accepted_raw_id = ? WHERE session_id = ?", (raw_id, session_id_b))
        index.commit()
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, acquisition_generation,
                revision_authority
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 'quarantined')
            """,
            (raw_id, key_a, "fanout-quarantine", content_hash_a.hex(), content_hash_a, len(session_a.messages)),
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
    with sqlite3.connect(root / "source.db") as source:
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
        source.commit()
    return raw_id, ((session_id_a, key_a), (session_id_b, key_b))


def test_quarantine_refinement_fanout_scopes_by_logical_source_key(tmp_path: Path) -> None:
    """polylogue-zaiz regression: fan-out siblings must not share one proof.

    Before the fix, ``_inspect_quarantined_accepted_raw`` looked up the
    accepted-head row by ``accepted_raw_id`` alone, requiring exactly one
    match system-wide. For this fan-out shape (two sessions sharing one
    quarantined raw), that unconditionally raised "expected one accepted
    head, found 2" for BOTH sessions -- crashing the whole census before
    ever reaching the real, content-based eligibility question. After the
    fix, each session gets an independent, correctly-scoped classification:
    ``session_a`` (whose accepted content genuinely matches what the raw
    re-parses to) is provably eligible; ``session_b`` (a stale sibling
    accepted-head row sharing the same raw_id, whose own content does not
    match) is provably -- and permanently -- ineligible via the real
    content-mismatch check, not a crash.
    """
    raw_id, heads = _seed_quarantined_raw_fanout(tmp_path)
    (session_a, key_a), (session_b, key_b) = heads

    census = inspect_raw_authority_frontier(_config(tmp_path))
    by_key = {item.logical_source_key: item for item in census.items if item.raw_id == raw_id}

    assert set(by_key) == {key_a, key_b}
    assert by_key[key_a].state is RawAuthorityFrontierState.SAFELY_REKEYABLE
    assert by_key[key_a].actuator is RawAuthorityActuator.REFINE_QUARANTINE
    assert by_key[key_a].executable

    assert by_key[key_b].state is RawAuthorityFrontierState.UNRESOLVED_PROVENANCE
    assert by_key[key_b].actuator is RawAuthorityActuator.REFINE_QUARANTINE
    assert not by_key[key_b].executable

    # The deep proof itself (not just the census-level fallback reason)
    # must show the real content-mismatch cause, not the old scoping crash.
    inspected = inspect_quarantined_accepted_raws(_config(tmp_path), [(raw_id, key_b)])
    assert inspected[0].status == "ineligible"
    assert "expected one accepted head" not in inspected[0].reason
    assert "differs from the accepted session" in inspected[0].reason


def test_quarantine_refinement_applies_for_the_matching_sibling_only(tmp_path: Path) -> None:
    """polylogue-zaiz regression: the genuinely-matching sibling refines cleanly.

    Unlike duplicate-alias fan-out (where eligibility is a race -- any
    sibling looks eligible until a canonical twin is claimed), quarantine-
    refinement eligibility is a fixed content-match fact: only
    ``session_a`` (whose accepted content genuinely matches the raw) is
    ever ``SAFELY_REKEYABLE``. Applying its plan must succeed end-to-end
    without disturbing ``session_b``'s own (permanently ineligible,
    unaffected) accepted-head row.
    """
    raw_id, heads = _seed_quarantined_raw_fanout(tmp_path)
    (session_a, key_a), (session_b, key_b) = heads

    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(
        item
        for item in preview.items
        if item.raw_id == raw_id
        and item.logical_source_key == key_a
        and item.state is RawAuthorityFrontierState.SAFELY_REKEYABLE
    )

    report = apply_raw_authority_frontier(
        _config(tmp_path),
        preview_census_id=preview.census_id,
        selected_plan_ids=(selected.plan_id,),
    )

    assert report.executed_plan_count == 1
    assert report.retryable_plan_count == 0
    assert report.success

    with sqlite3.connect(tmp_path / "source.db") as source:
        assert source.execute(
            "SELECT revision_authority, baseline_raw_id FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone() == ("byte_proven", raw_id)
    with sqlite3.connect(tmp_path / "index.db") as index:
        # session_b's own head still points at the shared raw_id -- the
        # refinement is scoped to session_a only, not a global side effect.
        assert index.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE session_id = ?", (session_b,)
        ).fetchone() == (raw_id,)
