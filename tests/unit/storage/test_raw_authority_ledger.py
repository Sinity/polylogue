from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, json_document
from polylogue.maintenance.models import MaintenanceCategory
from polylogue.sources.revision_backfill import census_historical_revision_evidence
from polylogue.storage import raw_authority as raw_authority_mod
from polylogue.storage import repair as repair_mod
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot
from polylogue.storage.raw_authority import (
    RawReplayPlan,
    RawReplayPlanOutcome,
    RawReplayPlanStatus,
    build_raw_replay_plans,
    finalize_raw_authority_census,
    read_raw_authority_census,
    record_raw_authority_census,
    record_raw_replay_outcome,
    reject_stale_raw_replay_plan,
    resolve_raw_authority_blocker,
    validate_raw_replay_plan,
)
from polylogue.storage.repair import RepairResult, repair_raw_materialization
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "archive.db")


def _write_codex_raw(
    root: Path,
    *,
    native_id: str,
    source_path: str,
    acquired_at_ms: int,
    text: str = "",
) -> str:
    payload = (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'
        f'{{"type":"response_item","payload":{{"type":"message","id":"m-{acquired_at_ms}",'
        f'"role":"user","content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
        )


def test_moved_path_census_stabilizes_preview_and_apply_plan_identity(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    old_raw = _write_codex_raw(
        tmp_path,
        native_id="moved-session",
        source_path="old/location.jsonl",
        acquired_at_ms=2,
        text="old",
    )
    new_raw = _write_codex_raw(
        tmp_path,
        native_id="moved-session",
        source_path="new/location.jsonl",
        acquired_at_ms=1,
        text="new",
    )
    # Prior history already knows the logical key at another path.  The new
    # raw begins as an uncensused singleton and must discover that history
    # before an immutable plan is assigned.
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[old_raw])

    preview = repair_raw_materialization(_config(tmp_path), dry_run=True, raw_artifact_limit=1)
    applied = repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)

    assert len(preview.plan_outcomes) == len(applied.plan_outcomes) == 1
    assert preview.plan_outcomes[0].plan_id == applied.plan_outcomes[0].plan_id
    assert set(preview.plan_outcomes[0].input_raw_ids) == {old_raw, new_raw}
    assert set(applied.plan_outcomes[0].input_raw_ids) == {old_raw, new_raw}


def test_census_ledger_conserves_unselected_plan_and_application_receipt(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _write_codex_raw(tmp_path, native_id="first", source_path="first.jsonl", acquired_at_ms=1)
    _write_codex_raw(tmp_path, native_id="second", source_path="second.jsonl", acquired_at_ms=2)

    result = repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)

    assert result.census_receipt is not None
    assert result.census_receipt.plan_count == 2
    assert result.census_receipt.executable_plan_count == 2
    assert result.census_receipt.residual_plan_count == 0
    assert result.census_receipt.lifecycle_status == "completed"
    assert result.metrics["raw_materialization_plan_outcome_count"] == 2.0
    assert result.metrics["raw_materialization_plan_carried_forward_count"] == 1.0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        rows = conn.execute(
            """
            SELECT selected, outcome_status, application_receipt_json
            FROM raw_authority_census_plans
            WHERE census_id = ? ORDER BY ordinal
            """,
            (result.census_receipt.census_id,),
        ).fetchall()
        plan_row = conn.execute(
            """
            SELECT input_raw_ids_json, logical_keys_json, authority_witness_json,
                   source_preconditions_json, index_preconditions_json
            FROM raw_authority_plans
            WHERE plan_id = (
                SELECT plan_id FROM raw_authority_census_plans
                WHERE census_id = ? AND selected = 1
            )
            """,
            (result.census_receipt.census_id,),
        ).fetchone()
    assert {row[1] for row in rows} == {"executed", "carried_forward"}
    executed = next(row for row in rows if row[1] == "executed")
    assert executed[0] == 1
    assert '"application_rows"' in executed[2]
    assert '"membership_rows"' in executed[2]
    application_receipt = json.loads(executed[2])
    assert application_receipt["application_rows"]
    assert application_receipt["head_rows"]
    assert application_receipt["session_rows"]
    assert plan_row is not None
    assert all(value not in (None, "", "[]", "{}") for value in plan_row)
    readiness = raw_materialization_readiness_snapshot(tmp_path)
    census_status = cast(dict[str, object], readiness["raw_authority_census"])
    assert census_status["census_id"] == result.census_receipt.census_id
    assert census_status["inventory_digest"] == result.census_receipt.inventory_digest
    assert census_status["residual_digest"] == result.census_receipt.residual_digest
    assert census_status["plan_count"] == 2
    assert census_status["executable_plan_count"] == 2
    assert census_status["residual_plan_count"] == 0
    assert census_status["lifecycle_status"] == "completed"
    assert census_status["query_handle"] == result.census_receipt.query_handle
    first_page = read_raw_authority_census(tmp_path, result.census_receipt.query_handle, limit=1)
    assert first_page["returned_count"] == 1
    assert first_page["next_query_handle"] is not None
    second_page = read_raw_authority_census(tmp_path, cast(str, first_page["next_query_handle"]), limit=1)
    assert second_page["returned_count"] == 1
    assert second_page["next_query_handle"] is None
    assert {
        cast(dict[str, object], item)["outcome_status"]
        for item in (*cast(list[object], first_page["plans"]), *cast(list[object], second_page["plans"]))
    } == {"executed", "carried_forward"}


def test_two_successive_quiescent_censuses_are_required_for_fixed_point(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _write_codex_raw(tmp_path, native_id="fixed", source_path="fixed.jsonl", acquired_at_ms=1)
    assert repair_raw_materialization(_config(tmp_path)).repaired_count == 1

    first_empty = repair_raw_materialization(_config(tmp_path), dry_run=True)
    second_empty = repair_raw_materialization(_config(tmp_path), dry_run=True)

    assert first_empty.census_receipt is not None
    assert second_empty.census_receipt is not None
    assert first_empty.census_receipt.fixed_point is False
    assert second_empty.census_receipt.fixed_point is True
    assert first_empty.census_receipt.inventory_digest == second_empty.census_receipt.inventory_digest
    assert first_empty.census_receipt.residual_digest == second_empty.census_receipt.residual_digest


def test_stale_plan_persists_blocker_before_automatic_replay_refuses_work(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="stale", source_path="stale.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    census = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids={plan.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "stale"},
        residual={},
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = 'moved-after-plan.jsonl' WHERE raw_id = ?", (raw_id,))
        conn.commit()

    valid, observed = validate_raw_replay_plan(tmp_path, plan)
    assert valid is False
    outcome = reject_stale_raw_replay_plan(tmp_path, census.census_id, plan, observed)

    assert outcome.status.value == "rejected_stale"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 1
        )
        assert (
            conn.execute(
                "SELECT outcome_status FROM raw_authority_census_plans WHERE census_id = ? AND plan_id = ?",
                (census.census_id, plan.plan_id),
            ).fetchone()[0]
            == "rejected_stale"
        )
    refused = repair_raw_materialization(_config(tmp_path))
    assert refused.success is False
    assert refused.metrics["raw_materialization_unresolved_blocker_count"] == 1.0


def test_interrupted_census_has_no_partial_plan_visibility_and_retries_once(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    first = _write_codex_raw(tmp_path, native_id="atomic-first", source_path="atomic-first.jsonl", acquired_at_ms=1)
    second = _write_codex_raw(
        tmp_path,
        native_id="atomic-second",
        source_path="atomic-second.jsonl",
        acquired_at_ms=2,
    )
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[first, second])
    plans = build_raw_replay_plans(tmp_path, ((first,), (second,)))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            CREATE TRIGGER abort_second_census_plan
            BEFORE INSERT ON raw_authority_census_plans
            WHEN NEW.ordinal = 1
            BEGIN
                SELECT RAISE(ABORT, 'synthetic census interruption');
            END
            """
        )
        conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="synthetic census interruption"):
        record_raw_authority_census(
            tmp_path,
            plans,
            selected_plan_ids={plan.plan_id for plan in plans},
            mode="apply",
            quiescent=True,
            scope={"test": "interruption"},
            residual={},
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_plans").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_census_plans").fetchone()[0] == 0
        conn.execute("DROP TRIGGER abort_second_census_plan")
        conn.commit()

    receipt = record_raw_authority_census(
        tmp_path,
        plans,
        selected_plan_ids={plan.plan_id for plan in plans},
        mode="apply",
        quiescent=True,
        scope={"test": "interruption"},
        residual={},
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM raw_authority_census_plans WHERE census_id = ?",
                (receipt.census_id,),
            ).fetchone()[0]
            == 2
        )
    for plan in plans:
        record_raw_replay_outcome(
            tmp_path,
            receipt.census_id,
            RawReplayPlanOutcome(
                plan.plan_id,
                plan.input_raw_ids,
                RawReplayPlanStatus.RETRYABLE,
                "test interruption recovered",
                "retry",
            ),
        )
    finalized = finalize_raw_authority_census(tmp_path, receipt.census_id, interrupted=True)
    assert finalized.lifecycle_status == "interrupted"


def test_global_census_quiesces_moved_component_before_any_plan_is_published(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    first = _write_codex_raw(
        tmp_path,
        native_id="merged",
        source_path="merged-old.jsonl",
        acquired_at_ms=1,
        text="old",
    )
    second = _write_codex_raw(
        tmp_path,
        native_id="merged",
        source_path="merged-new.jsonl",
        acquired_at_ms=2,
        text="new",
    )
    third = _write_codex_raw(
        tmp_path,
        native_id="independent",
        source_path="independent.jsonl",
        acquired_at_ms=3,
    )

    preview = repair_raw_materialization(_config(tmp_path), dry_run=True, raw_artifact_limit=1)

    assert preview.census_receipt is not None
    assert preview.census_receipt.quiescent is True
    ledger = read_raw_authority_census(tmp_path, preview.census_receipt.query_handle)
    raw_sets = {
        frozenset(cast(list[str], cast(dict[str, object], cast(dict[str, object], item)["plan"])["input_raw_ids"]))
        for item in cast(list[object], ledger["plans"])
    }
    assert raw_sets == {frozenset((first, second)), frozenset((third,))}


def test_interrupted_apply_recovers_exact_durable_postconditions(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _write_codex_raw(tmp_path, native_id="crash", source_path="crash.jsonl", acquired_at_ms=1)

    with patch.object(repair_mod, "raw_replay_application_receipt", side_effect=RuntimeError("synthetic crash")):
        with pytest.raises(RuntimeError, match="synthetic crash"):
            repair_raw_materialization(_config(tmp_path))

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_censuses WHERE lifecycle_status = 'planned'").fetchone()[0]
            == 1
        )
    recovered = repair_raw_materialization(_config(tmp_path))
    assert recovered.metrics["raw_materialization_recovered_census_count"] == 1.0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """
            SELECT c.lifecycle_status, cp.outcome_status
            FROM raw_authority_censuses AS c
            JOIN raw_authority_census_plans AS cp ON cp.census_id = c.census_id
            WHERE c.lifecycle_status = 'interrupted'
            """
        ).fetchone()
    assert row == ("interrupted", "executed")


def test_parsed_timestamp_without_exact_application_receipt_fails_closed(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _write_codex_raw(tmp_path, native_id="receipt", source_path="receipt.jsonl", acquired_at_ms=1)
    real_receipt = raw_authority_mod.raw_replay_application_receipt

    def incomplete_receipt(root: Path, plan: RawReplayPlan) -> JSONDocument:
        payload = dict(real_receipt(root, plan))
        payload["head_rows"] = []
        return json_document(payload)

    with patch.object(repair_mod, "raw_replay_application_receipt", side_effect=incomplete_receipt):
        result = repair_raw_materialization(_config(tmp_path))

    assert result.plan_outcomes[0].status is RawReplayPlanStatus.REJECTED_STALE
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 1
        )


def test_stale_blocker_resolution_replans_current_evidence_and_resumes(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="resume", source_path="resume.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    census = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids={plan.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "resolve"},
        residual={},
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = 'resume-moved.jsonl' WHERE raw_id = ?", (raw_id,))
        conn.commit()
    valid, observed = validate_raw_replay_plan(tmp_path, plan)
    assert valid is False
    rejected = reject_stale_raw_replay_plan(tmp_path, census.census_id, plan, observed)
    blocker_id = cast(str, cast(dict[str, object], rejected.application_receipt)["blocker_id"])

    resolution = resolve_raw_authority_blocker(tmp_path, blocker_id, resolution="current path is authoritative")
    resumed = repair_raw_materialization(_config(tmp_path))

    assert resolution["blocker_id"] == blocker_id
    assert resumed.metrics.get("raw_materialization_unresolved_blocker_count", 0.0) == 0.0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 0
        )


def test_fixed_point_compares_residual_identity_and_parser_fingerprint(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    first = record_raw_authority_census(
        tmp_path,
        (),
        selected_plan_ids=set(),
        executable_plan_ids=set(),
        mode="dry_run",
        quiescent=True,
        scope={"test": "fixed-point"},
        residual={"missing_blob_raw_ids": ["a"]},
    )
    second = record_raw_authority_census(
        tmp_path,
        (),
        selected_plan_ids=set(),
        executable_plan_ids=set(),
        mode="dry_run",
        quiescent=True,
        scope={"test": "fixed-point"},
        residual={"missing_blob_raw_ids": ["b"]},
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_authority_censuses SET parser_fingerprint = 'stale-parser' WHERE census_id = ?",
            (second.census_id,),
        )
        conn.commit()
    third = record_raw_authority_census(
        tmp_path,
        (),
        selected_plan_ids=set(),
        executable_plan_ids=set(),
        mode="dry_run",
        quiescent=True,
        scope={"test": "fixed-point"},
        residual={"missing_blob_raw_ids": ["b"]},
    )
    fourth = record_raw_authority_census(
        tmp_path,
        (),
        selected_plan_ids=set(),
        executable_plan_ids=set(),
        mode="dry_run",
        quiescent=True,
        scope={"test": "fixed-point"},
        residual={"missing_blob_raw_ids": ["b"]},
    )
    assert first.fixed_point is False
    assert second.fixed_point is False
    assert third.fixed_point is False
    assert fourth.fixed_point is True


def test_stale_per_raw_parser_fingerprint_is_recensused_before_planning(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="parser-drift", source_path="parser-drift.jsonl", acquired_at_ms=1)
    first = repair_raw_materialization(_config(tmp_path), dry_run=True)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_authority_parser_census SET parser_fingerprint = 'old-parser' WHERE raw_id = ?",
            (raw_id,),
        )
        conn.commit()

    second = repair_raw_materialization(_config(tmp_path), dry_run=True)

    assert first.plan_outcomes[0].plan_id == second.plan_outcomes[0].plan_id
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT parser_fingerprint FROM raw_authority_parser_census WHERE raw_id = ?",
                (raw_id,),
            ).fetchone()[0]
            == "revision-membership-v1"
        )


def test_repair_result_bounds_public_plan_outcomes() -> None:
    outcomes = tuple(
        RawReplayPlanOutcome(
            f"plan-{index}",
            (f"raw-{index}",),
            RawReplayPlanStatus.RETRYABLE,
            "test",
            "retry",
        )
        for index in range(10)
    )
    result = RepairResult(
        "raw_materialization",
        MaintenanceCategory.DERIVED_REPAIR,
        False,
        0,
        False,
        plan_outcomes=outcomes,
    ).to_dict()
    assert result["plan_outcome_count"] == 10
    assert len(cast(list[object], result["plan_outcomes"])) == 8
    assert result["plan_outcomes_truncated"] is True
