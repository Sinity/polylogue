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
from polylogue.storage.raw_authority import record_raw_replay_outcome
from polylogue.storage.raw_reconciler import (
    RawAuthorityActuator,
    RawAuthorityFrontierState,
    apply_raw_authority_frontier,
    inspect_raw_authority_frontier,
    recover_interrupted_raw_authority_frontier,
)
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
    assert duplicate.evidence_ref is not None
    assert first.state_counts[RawAuthorityFrontierState.DUPLICATE_ALIAS.value] == 1
    assert first.executable_plan_count == 1

    with sqlite3.connect(tmp_path / "source.db") as conn:
        persisted = conn.execute(
            """
            SELECT p.input_raw_ids_json, p.authority_witness_json,
                   p.source_preconditions_json, p.index_preconditions_json
            FROM raw_authority_census_plans AS cp
            JOIN raw_authority_plans AS p ON p.plan_id = cp.plan_id
            WHERE cp.census_id = ? AND p.plan_id = ?
            """,
            (first.census_id, duplicate.plan_id),
        ).fetchone()
    assert persisted is not None
    assert json.loads(persisted[0]) == sorted((stale_raw_id, canonical_raw_id))
    assert json.loads(persisted[1])["actuator"] == RawAuthorityActuator.FOLD_DUPLICATE_ALIAS.value
    assert json.loads(persisted[2])["blob_hash"]
    assert json.loads(persisted[3])["accepted_content_hash"]


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
            SELECT b.reason, b.resolved_at_ms, p.authority_witness_json
            FROM raw_authority_blockers AS b
            JOIN raw_authority_plans AS p ON p.plan_id = b.plan_id
            WHERE b.plan_id = ?
            """,
            (missing.plan_id,),
        ).fetchone()
    assert obligation is not None
    assert obligation[1] is None
    assert json.loads(obligation[2])["state"] == RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE.value
    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_frontier_blocking_count"] == 1
    assert raw_materialization_ready(readiness) is False
    refs = readiness["raw_authority_frontier_remediation_refs"]
    assert isinstance(refs, list) and refs[0]["plan_id"] == missing.plan_id

    reacquired = json.dumps({"marker": "duplicate-raw-fixture", "legacy_native_id": "legacy-native-id-1"}).encode()
    assert hashlib.sha256(reacquired).digest() == blob_hash
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(reacquired)
    advanced = inspect_raw_authority_frontier(_config(tmp_path))
    assert (
        next(item for item in advanced.items if item.raw_id == stale_raw_id).state
        is RawAuthorityFrontierState.DUPLICATE_ALIAS
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_authority_blockers WHERE plan_id = ? AND resolved_at_ms IS NULL",
            (missing.plan_id,),
        ).fetchone() == (0,)


def test_unified_frontier_apply_drives_duplicate_strategy_and_postflight(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, session_id, logical_key = _seed_duplicate_raw_pair(tmp_path)
    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == stale_raw_id)

    report = apply_raw_authority_frontier(
        _config(tmp_path),
        preview_census_id=preview.census_id,
        selected_plan_ids=(selected.plan_id,),
        receipt_dir=tmp_path / "unified-receipts",
    )

    assert report.success is True
    assert report.selected_plan_count == report.executed_plan_count == 1
    assert report.retryable_plan_count == 0
    assert len(report.outcome_refs) == 1
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
            (logical_key,),
        ).fetchone() == (canonical_raw_id,)
        assert conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone() == (
            canonical_raw_id,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            "SELECT lifecycle_status FROM raw_authority_censuses WHERE census_id = ?",
            (report.census_id,),
        ).fetchone()
        assert row == ("completed",)
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (stale_raw_id,)).fetchone() == (1,)
    postflight = inspect_raw_authority_frontier(_config(tmp_path))
    assert any(
        item.raw_id == stale_raw_id and item.state is RawAuthorityFrontierState.SUPERSEDED for item in postflight.items
    )


def test_unified_frontier_recovers_crash_after_strategy_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_raw_id, canonical_raw_id, _session_id, logical_key = _seed_duplicate_raw_pair(tmp_path)
    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == stale_raw_id)

    def crash_before_outcome(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected crash after strategy commit")

    monkeypatch.setattr("polylogue.storage.raw_reconciler.record_raw_replay_outcome", crash_before_outcome)
    with pytest.raises(RuntimeError, match="injected crash"):
        apply_raw_authority_frontier(
            _config(tmp_path),
            preview_census_id=preview.census_id,
            selected_plan_ids=(selected.plan_id,),
            receipt_dir=tmp_path / "crash-receipts",
        )
    monkeypatch.setattr("polylogue.storage.raw_reconciler.record_raw_replay_outcome", record_raw_replay_outcome)

    recovered = recover_interrupted_raw_authority_frontier(_config(tmp_path))

    assert recovered == (selected.plan_id,)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute(
            "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?",
            (logical_key,),
        ).fetchone() == (canonical_raw_id,)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT lifecycle_status FROM raw_authority_censuses WHERE mode = 'apply' ORDER BY sequence_no DESC LIMIT 1"
        ).fetchone() == ("interrupted",)


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


def test_apply_serializes_one_receipt_inode_against_a_concurrent_apply(tmp_path: Path) -> None:
    """A second ``--apply`` against the same receipt path must fail closed.

    Exercises ``_lock_duplicate_raw_identity_repair_receipt`` directly: hold
    its flock open (as a genuinely concurrent apply would), then prove the
    real ``repair_duplicate_raw_identity`` entrypoint refuses to proceed and
    never mutates the durable authority underneath the held lock. Deleting
    the flock acquisition from the receipt lock (regressing to the old bare
    ``receipt_path.exists()`` TOCTOU check) makes this test fail: the second
    call would instead race straight into the transaction.
    """
    stale_raw_id, canonical_raw_id, _session_id, key = _seed_duplicate_raw_pair(tmp_path)
    dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    receipt_path = tmp_path / "locked.jsonl"
    from polylogue.storage import repair as repair_module

    locked = repair_module._lock_duplicate_raw_identity_repair_receipt(receipt_path, list(dry_run.items))
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            repair_duplicate_raw_identity(
                _config(tmp_path),
                [(stale_raw_id, canonical_raw_id)],
                apply=True,
                receipt_path=receipt_path,
                proof_digest=dry_run.proof_digest,
            )
        with closing(sqlite3.connect(tmp_path / "index.db")) as index_conn:
            head = index_conn.execute(
                "SELECT accepted_raw_id FROM raw_revision_heads WHERE logical_source_key = ?", (key,)
            ).fetchone()
            assert head[0] == stale_raw_id
    finally:
        locked.close()


def test_apply_resumes_planned_receipt_after_a_crash_before_the_terminal_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash between commit and the terminal receipt append must be resumable and auditable.

    Injects a failure into ``_finish_duplicate_raw_identity_repair_receipt``
    (the real production function that appends the fsynced ``applied``
    record) after the index-tier transaction has already committed. The
    receipt on disk must show only the ``planned`` record -- proving the
    planned phase is durably written *before* mutation, unlike the old
    single unlocked ``write_text`` -- and re-invoking apply against the SAME
    receipt path must resume to a terminal ``applied`` record without
    re-mutating anything (idempotent recovery), once the operator re-proves
    current authority with a fresh dry-run digest (the accepted head's own
    ``decided_at_ms`` legitimately changed under repair, so reusing the
    original pre-repair digest is correctly refused by the top-level CAS
    gate -- this mirrors exactly how an operator would recover in practice).
    Removing the planned-phase write or the crash-safe resume path makes
    this test fail.
    """
    stale_raw_id, canonical_raw_id, _session_id, _key = _seed_duplicate_raw_pair(tmp_path)
    dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    receipt = tmp_path / "planned-resume.jsonl"
    from polylogue.storage import repair as repair_module

    original_finish = repair_module._finish_duplicate_raw_identity_repair_receipt
    monkeypatch.setattr(
        repair_module,
        "_finish_duplicate_raw_identity_repair_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected terminal append crash")),
    )
    with pytest.raises(RuntimeError, match="terminal append crash"):
        repair_duplicate_raw_identity(
            _config(tmp_path),
            [(stale_raw_id, canonical_raw_id)],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    assert [json.loads(line)["state"] for line in receipt.read_text().splitlines()] == ["planned"]
    with closing(sqlite3.connect(tmp_path / "index.db")) as index_conn:
        session_raw = index_conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (_session_id,)).fetchone()
        assert session_raw[0] == canonical_raw_id

    monkeypatch.setattr(repair_module, "_finish_duplicate_raw_identity_repair_receipt", original_finish)
    post_crash_dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    assert post_crash_dry_run.already_repaired_count == 1
    resumed = repair_duplicate_raw_identity(
        _config(tmp_path),
        [(stale_raw_id, canonical_raw_id)],
        apply=True,
        receipt_path=receipt,
        proof_digest=post_crash_dry_run.proof_digest,
    )
    assert resumed.already_repaired_count == 1
    assert resumed.repaired_count == 0
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [record["state"] for record in records] == ["planned", "applied"]
    assert records[1]["repaired_stale_raw_ids"] == []


def test_receipt_path_must_not_be_a_symlink(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, _session_id, _key = _seed_duplicate_raw_pair(tmp_path)
    dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    target = tmp_path / "outside-target.jsonl"
    receipt = tmp_path / "receipt-symlink.jsonl"
    receipt.symlink_to(target)

    with pytest.raises(RuntimeError, match="symbolic link"):
        repair_duplicate_raw_identity(
            _config(tmp_path),
            [(stale_raw_id, canonical_raw_id)],
            apply=True,
            receipt_path=receipt,
            proof_digest=dry_run.proof_digest,
        )
    assert not target.exists()


def test_apply_requires_receipt_path(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, _session_id, _key = _seed_duplicate_raw_pair(tmp_path)
    dry_run = repair_duplicate_raw_identity(_config(tmp_path), [(stale_raw_id, canonical_raw_id)])
    with pytest.raises(ValueError, match="explicit operator repair receipt path"):
        repair_duplicate_raw_identity(
            _config(tmp_path), [(stale_raw_id, canonical_raw_id)], apply=True, proof_digest=dry_run.proof_digest
        )
