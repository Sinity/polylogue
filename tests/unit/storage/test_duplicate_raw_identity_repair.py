"""Tests for polylogue-t0dy: reconcile the pre-#2729 duplicate-raw scheme.

PR #2729 aligned the one-shot importer and live watcher on one deterministic
raw-id scheme. These tests prove the shared raw-authority reconciler discovers,
applies, and crash-recovers the resulting historical duplicate-alias state.
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
from polylogue.storage.raw_authority import (
    finalize_raw_authority_census,
    read_raw_authority_detail,
    record_raw_replay_outcome,
)
from polylogue.storage.raw_reconciler import (
    RawAuthorityActuator,
    RawAuthorityFrontierState,
    apply_raw_authority_frontier,
    inspect_raw_authority_frontier,
    recover_interrupted_raw_authority_frontier,
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
    detail = read_raw_authority_detail(tmp_path, str(refs[0]["detail_query_handle"]))
    assert missing.plan_id in str(detail["chunk"])

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
            "SELECT COUNT(*) FROM raw_authority_blockers WHERE plan_id = ? AND resolved_at_ms IS NULL",
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


def test_unified_frontier_apply_obeys_offline_daemon_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_raw_id, _canonical_raw_id, _session_id, _logical_key = _seed_duplicate_raw_pair(tmp_path)
    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == stale_raw_id)
    monkeypatch.setattr(
        "polylogue.maintenance.offline_guard.offline_maintenance_block_reason",
        lambda *_args, **_kwargs: "daemon owns the archive write lease",
    )

    with pytest.raises(RuntimeError, match="daemon owns"):
        apply_raw_authority_frontier(
            _config(tmp_path),
            preview_census_id=preview.census_id,
            selected_plan_ids=(selected.plan_id,),
        )


def test_unified_frontier_apply_drives_duplicate_strategy_and_postflight(tmp_path: Path) -> None:
    stale_raw_id, canonical_raw_id, session_id, logical_key = _seed_duplicate_raw_pair(tmp_path)
    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == stale_raw_id)

    report = apply_raw_authority_frontier(
        _config(tmp_path),
        preview_census_id=preview.census_id,
        selected_plan_ids=(selected.plan_id,),
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
        assert conn.execute(
            "SELECT outcome_status FROM raw_authority_census_plans WHERE selected = 1 ORDER BY ordinal DESC LIMIT 1"
        ).fetchone() == ("rejected_stale",)
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone() == (
            1,
        )


def test_unified_frontier_recovers_crash_after_outcome_before_postflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_raw_id, _canonical_raw_id, _session_id, _logical_key = _seed_duplicate_raw_pair(tmp_path)
    preview = inspect_raw_authority_frontier(_config(tmp_path))
    selected = next(item for item in preview.items if item.raw_id == stale_raw_id)

    def crash_before_postflight(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected crash after durable outcome")

    monkeypatch.setattr("polylogue.storage.raw_reconciler.finalize_raw_authority_census", crash_before_postflight)
    with pytest.raises(RuntimeError, match="injected crash"):
        apply_raw_authority_frontier(
            _config(tmp_path),
            preview_census_id=preview.census_id,
            selected_plan_ids=(selected.plan_id,),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            """
            SELECT lifecycle_status, outcome_recorded
            FROM raw_authority_censuses AS c
            JOIN raw_authority_census_plans AS cp USING (census_id)
            WHERE c.mode = 'apply' ORDER BY c.sequence_no DESC LIMIT 1
            """
        ).fetchone() == ("planned", 1)

    monkeypatch.setattr(
        "polylogue.storage.raw_reconciler.finalize_raw_authority_census",
        finalize_raw_authority_census,
    )
    recovered = recover_interrupted_raw_authority_frontier(_config(tmp_path))

    assert recovered == ()
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT lifecycle_status FROM raw_authority_censuses WHERE mode = 'apply' ORDER BY sequence_no DESC LIMIT 1"
        ).fetchone() == ("interrupted",)


def _rows(root: Path, tier: str, table: str, where: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(root / f"{tier}.db")) as conn:
        return sorted(conn.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchall())
