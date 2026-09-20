"""Tests for polylogue-t0dy: reconcile the pre-#2729 duplicate-raw scheme.

PR #2729 aligned the one-shot importer and live watcher on one deterministic
raw-id scheme. These tests prove the shared raw-authority reconciler discovers
and classifies the resulting historical duplicate-alias state.
"""

from __future__ import annotations

import hashlib
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
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.raw_reconciler import (
    RawAuthorityActuator,
    RawAuthorityFrontierState,
    inspect_raw_authority_frontier,
)
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


def test_unified_frontier_census_plans_duplicate_alias_with_stable_evidence(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, _session_id, _logical_key = _seed_duplicate_raw_pair(tmp_path)

    first = inspect_raw_authority_frontier(_config(tmp_path))
    second = inspect_raw_authority_frontier(_config(tmp_path))

    duplicate = next(item for item in first.items if item.raw_id == stale_raw_id)
    duplicate_again = next(item for item in second.items if item.raw_id == stale_raw_id)
    assert duplicate.state is RawAuthorityFrontierState.DUPLICATE_ALIAS
    assert duplicate.actuator is RawAuthorityActuator.FOLD_DUPLICATE_ALIAS
    assert duplicate.input_raw_ids == tuple(sorted((stale_raw_id, canonical_raw_id)))
    assert duplicate.plan_id == duplicate_again.plan_id
    assert duplicate.evidence_digest == duplicate_again.evidence_digest
    # An executable duplicate alias is not a blocking obligation, so no durable
    # blocker is published for it and its evidence travels inline on the item.
    assert duplicate.evidence_ref is None
    assert first.state_counts[RawAuthorityFrontierState.DUPLICATE_ALIAS.value] == 1
    assert first.executable_plan_count == 1
    # Two passes over an unchanged frontier name the same content-addressed
    # pass and write no rows: the inspection itself is not durable any more.
    assert first.pass_id == second.pass_id
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers").fetchone()[0] == 0

    # The plan's evidence now travels on the item itself rather than behind a
    # census-detail handle: it is exactly the payload the surface serializes.
    assert duplicate.input_raw_ids == tuple(sorted((stale_raw_id, canonical_raw_id)))
    assert duplicate.strategy_witness["canonical_raw_id"] == canonical_raw_id
    assert duplicate.source_preconditions["blob_hash"]
    assert duplicate.index_preconditions["accepted_content_hash"]


def test_duplicate_alias_census_uses_active_generation_not_shadow_index(tmp_path: Path) -> None:
    stale_raw_id, _canonical_raw_id, _session_id, _logical_key = _seed_duplicate_raw_pair(tmp_path)
    active_index = tmp_path / "generations" / "active" / "index.db"
    active_index.parent.mkdir(parents=True)
    (tmp_path / "index.db").replace(active_index)
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")
    (tmp_path / "index.db").write_bytes(b"not a sqlite database")

    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
    )
    census = inspect_raw_authority_frontier(config)

    duplicate = next(item for item in census.items if item.raw_id == stale_raw_id)
    assert duplicate.state is RawAuthorityFrontierState.DUPLICATE_ALIAS
    assert duplicate.actuator is RawAuthorityActuator.FOLD_DUPLICATE_ALIAS


def test_unified_frontier_census_prioritizes_missing_bytes_over_safe_actuation(tmp_path: Path) -> None:
    stale_raw_id, _canonical_raw_id, _session_id, _logical_key = _seed_duplicate_raw_pair(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        blob_hash = bytes(
            conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (stale_raw_id,)).fetchone()[0]
        )
    blob_path = tmp_path / "blob" / blob_hash.hex()[:2] / blob_hash.hex()[2:]
    blob_path.unlink()

    census = inspect_raw_authority_frontier(_config(tmp_path))

    missing = next(item for item in census.items if item.raw_id == stale_raw_id)
    assert missing.state is RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE
    assert missing.actuator is RawAuthorityActuator.REACQUIRE
    assert missing.executable is False
    with sqlite3.connect(tmp_path / "source.db") as conn:
        obligation = conn.execute(
            """
            SELECT b.reason, b.resolved_at_ms,
                   json_extract(b.expected_json, '$.authority_witness')
            FROM raw_authority_blockers AS b
            WHERE json_extract(b.expected_json, '$.plan_id') = ?
            """,
            (missing.plan_id,),
        ).fetchone()
    assert obligation is not None
    assert obligation[1] is None
    assert json.loads(obligation[2])["state"] == RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE.value
    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_blocker_count"] == 1
    assert raw_materialization_ready(readiness) is False
    refs = readiness["raw_authority_frontier_remediation_refs"]
    assert isinstance(refs, list) and refs[0]["plan_id"] == missing.plan_id
    assert refs[0]["observed_pass_id"] == census.pass_id
    assert missing.evidence_ref == refs[0]["blocker_id"]

    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(b"wrong bytes at the expected content-addressed path")
    still_missing = inspect_raw_authority_frontier(_config(tmp_path))
    wrong_bytes = next(item for item in still_missing.items if item.raw_id == stale_raw_id)
    assert wrong_bytes.state is RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE
    assert "do not prove" in wrong_bytes.reason

    reacquired = json.dumps({"marker": "duplicate-raw-fixture", "legacy_native_id": "legacy-native-id-1"}).encode()
    assert hashlib.sha256(reacquired).digest() == blob_hash
    blob_path.write_bytes(reacquired)
    advanced = inspect_raw_authority_frontier(_config(tmp_path))
    assert (
        next(item for item in advanced.items if item.raw_id == stale_raw_id).state
        is RawAuthorityFrontierState.DUPLICATE_ALIAS
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_authority_blockers WHERE json_extract(expected_json, '$.plan_id') = ? AND resolved_at_ms IS NULL",
            (missing.plan_id,),
        ).fetchone() == (0,)


def test_unified_frontier_first_census_rejects_replaced_blob_bytes(tmp_path: Path) -> None:
    stale_raw_id, _canonical_raw_id, _session_id, _logical_key = _seed_duplicate_raw_pair(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        blob_hash = bytes(
            conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (stale_raw_id,)).fetchone()[0]
        )
    blob_path = tmp_path / "blob" / blob_hash.hex()[:2] / blob_hash.hex()[2:]
    blob_path.write_bytes(b"replacement bytes present before the first census")

    census = inspect_raw_authority_frontier(_config(tmp_path))

    replaced = next(item for item in census.items if item.raw_id == stale_raw_id)
    assert replaced.state is RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE
    assert replaced.actuator is RawAuthorityActuator.REACQUIRE


def _rows(root: Path, tier: str, table: str, where: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(root / f"{tier}.db")) as conn:
        return sorted(conn.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchall())


def _seed_duplicate_raw_fanout(root: Path) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Seed one stale raw shared as the accepted head across TWO sessions.

    Reproduces polylogue-ihc8: forked/subagent/resumed sessions can physically
    replay the identical parent evidence, so the exact same pre-#2729
    native-id-inclusive ``raw_id`` can legitimately be the accepted head of
    *more than one* logical source key/session at once. Only one canonical
    (native_id=NULL) duplicate raw exists for the shared content, so at most
    one of the sessions can ever be folded onto it -- the other must be
    provably distinguished from "already repaired", not silently
    misclassified as if it were the one that got folded.
    """
    initialize_active_archive_root(root)
    payload = json.dumps({"marker": "duplicate-raw-fanout-fixture"}).encode()
    source_path = "codex-session/carryover-fanout.jsonl"
    legacy_native_id = "legacy-native-id-fanout"
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

        heads: list[tuple[str, str]] = []
        for suffix in ("a", "b"):
            session = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=f"fanout-session-{suffix}",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text=f"hello-{suffix}")],
            )
            _stored, session_id = archive.write_parsed_for_retained_raw(
                session,
                raw_id=stale_raw_id,
                source_path=source_path,
                acquired_at_ms=1,
                revision_authoritative=True,
            )
            logical_source_key = f"codex:fanout-session-{suffix}"
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
                    detail=f"pre-#2729 duplicate-raw fanout fixture ({suffix})",
                ),
                decided_at_ms=1,
            )
            heads.append((session_id, logical_source_key))
        archive.commit()
        return stale_raw_id, canonical_raw_id, tuple(heads)


def test_duplicate_alias_witness_is_scoped_to_its_own_session_not_a_fanout_sibling(tmp_path: Path) -> None:
    """polylogue-ihc8 regression: classification must not cross-contaminate siblings.

    Before the fix, ``_inspect_duplicate_raw_identity`` looked up the stale
    raw's accepted-head row by ``accepted_raw_id`` alone. When that raw_id was
    the accepted head of two different logical source keys (the fanout
    shape), ``fetchone()`` picked one arbitrarily -- so the frontier item
    classified for session A could carry a strategy witness describing
    session B's head instead of its own.
    """
    stale_raw_id, canonical_raw_id, heads = _seed_duplicate_raw_fanout(tmp_path)
    (session_a, key_a), (session_b, key_b) = heads

    census = inspect_raw_authority_frontier(_config(tmp_path))
    duplicate_items = [
        item
        for item in census.items
        if item.raw_id == stale_raw_id and item.state is RawAuthorityFrontierState.DUPLICATE_ALIAS
    ]
    assert len(duplicate_items) == 2
    by_key = {item.logical_source_key: item for item in duplicate_items}
    assert set(by_key) == {key_a, key_b}
    for logical_source_key, session_id in ((key_a, session_a), (key_b, session_b)):
        item = by_key[logical_source_key]
        assert item.actuator is RawAuthorityActuator.FOLD_DUPLICATE_ALIAS
        assert item.session_id == session_id
        # The strategy witness bound to THIS item must describe THIS item's
        # own session/key -- not whichever sibling an unscoped lookup happened
        # to find first.
        assert item.strategy_witness["session_id"] == session_id
        assert item.strategy_witness["logical_source_key"] == logical_source_key
        assert item.strategy_witness["canonical_raw_id"] == canonical_raw_id
    assert by_key[key_a].plan_id != by_key[key_b].plan_id


def test_duplicate_alias_ineligible_proof_does_not_crash_the_whole_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-dmvo regression: an "ineligible" duplicate-alias proof must not crash the census.

    ``_inspect_duplicate_raw_identity`` returns ``status="ineligible"`` (never
    raises) for several legitimate, expected N:1 fan-out terminal states --
    e.g. a sibling session sharing the same stale raw as its accepted head,
    once the single available canonical twin has already been claimed by a
    *different* sibling's fold (real reasons observed live: "canonical raw is
    already an accepted head", "stale raw is not the currently accepted head
    of this logical source key"). Before the fix, ``_classify_frontier``
    treated ANY status outside {eligible, already_repaired} as a fatal proof
    violation and raised -- which crashed the *entire* frontier census, not
    just this one raw's classification. Observed live: this held the
    daemon's sole writer lock for 9+ minutes before failing the whole pass,
    with 10 other queued daemon actors starved behind it, repeating every
    retry cycle. This directly exercises ``_classify_frontier``'s own
    ineligible-status branch (monkeypatching the proof helper it calls,
    ``_inspect_duplicate_raw_identity``) rather than trying to reproduce the
    live census/apply ordering that produces "ineligible" naturally --
    reconstructing that exact multi-pass state was not reliably
    reproducible in a single-pass fixture, but the fix's own behavior is
    fully exercised regardless of which upstream condition triggers it.
    """
    stale_raw_id, canonical_raw_id, heads = _seed_duplicate_raw_fanout(tmp_path)
    (session_a, key_a), (session_b, key_b) = heads

    import polylogue.storage.raw_convergence as raw_convergence_module
    from polylogue.storage.raw_convergence import DuplicateRawIdentityRepairItem

    real_inspect = raw_convergence_module._inspect_duplicate_raw_identity

    def fake_inspect(
        conn: object, archive_root: object, stale: str, canonical: str, logical_source_key: str
    ) -> DuplicateRawIdentityRepairItem:
        if logical_source_key == key_b:
            return DuplicateRawIdentityRepairItem(
                stale_raw_id=stale,
                canonical_raw_id=canonical,
                status="ineligible",
                reason="canonical raw is already an accepted head; not a dangling duplicate",
            )
        return real_inspect(conn, archive_root, stale, canonical, logical_source_key)  # type: ignore[arg-type]

    monkeypatch.setattr(raw_convergence_module, "_inspect_duplicate_raw_identity", fake_inspect)

    # The regression: this must not raise, and must still classify session A
    # (the genuinely eligible sibling) correctly.
    census = inspect_raw_authority_frontier(_config(tmp_path))

    by_key = {item.logical_source_key: item for item in census.items if item.raw_id == stale_raw_id}
    assert by_key[key_a].state is RawAuthorityFrontierState.DUPLICATE_ALIAS
    assert by_key[key_a].actuator is RawAuthorityActuator.FOLD_DUPLICATE_ALIAS
    assert by_key[key_a].executable

    sibling_item = by_key[key_b]
    assert sibling_item.state is RawAuthorityFrontierState.UNRESOLVED_PROVENANCE
    assert sibling_item.actuator is RawAuthorityActuator.NONE
    assert not sibling_item.executable
    assert "already an accepted head" in sibling_item.reason
