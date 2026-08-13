from __future__ import annotations

import inspect
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
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
from polylogue.storage import raw_reconciler as raw_reconciler_mod
from polylogue.storage import repair as repair_mod
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import (
    AUTO_STALE_PLAN_RESOLUTION,
    RAW_AUTHORITY_PARSER_FINGERPRINT,
    RawReplayPlan,
    RawReplayPlanOutcome,
    RawReplayPlanStatus,
    auto_resolve_stale_plan_blockers,
    build_raw_replay_plans,
    finalize_raw_authority_census,
    read_raw_authority_census,
    read_raw_authority_detail,
    record_raw_authority_census,
    record_raw_replay_outcome,
    reject_stale_raw_replay_plan,
    resolve_raw_authority_blocker,
    validate_raw_replay_plan,
)
from polylogue.storage.raw_reconciler import RawAuthorityFrontierState, inspect_raw_authority_frontier
from polylogue.storage.repair import RepairResult, repair_raw_materialization
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "archive.db")


def _read_detail_document(root: Path, query_handle: str, *, chunk_chars: int = 256) -> dict[str, object]:
    chunks: list[str] = []
    handle: str | None = query_handle
    digest: str | None = None
    for _page in range(10_000):
        assert handle is not None
        page = read_raw_authority_detail(root, handle, chunk_chars=chunk_chars)
        chunk = cast(str, page["chunk"])
        assert len(chunk) <= chunk_chars
        chunks.append(chunk)
        page_digest = cast(str, page["document_sha256"])
        digest = digest or page_digest
        assert page_digest == digest
        handle = cast(str | None, page["next_query_handle"])
        if handle is None:
            break
    else:
        raise AssertionError("raw authority detail pagination did not terminate")
    document = json.loads("".join(chunks))
    assert isinstance(document, dict)
    return cast(dict[str, object], document)


def _write_codex_raw(
    root: Path,
    *,
    native_id: str,
    source_path: str,
    acquired_at_ms: int,
    text: str = "authored content",
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

    incomplete = repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)
    assert incomplete.census_receipt is not None
    assert incomplete.census_receipt.quiescent is False
    assert incomplete.census_receipt.plan_count == 0

    result = repair_raw_materialization(_config(tmp_path), raw_artifact_limit=1)

    assert result.census_receipt is not None
    assert result.census_receipt.plan_count == 2
    assert result.census_receipt.executable_plan_count == 2
    assert result.census_receipt.residual_plan_count == 0
    assert result.census_receipt.post_plan_count == 1
    assert result.census_receipt.post_inventory_digest is not None
    assert result.census_receipt.post_inventory_digest != result.census_receipt.inventory_digest
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
    first_item = cast(dict[str, object], cast(list[object], first_page["plans"])[0])
    assert "application_receipt" not in first_item
    assert "input_raw_ids" not in cast(dict[str, object], first_item["plan"])
    detail = _read_detail_document(tmp_path, cast(str, first_item["detail_query_handle"]))
    assert cast(dict[str, object], detail["plan"])["input_raw_ids"]
    assert cast(dict[str, object], first_page["census"])["post_plan_count"] == 1
    assert len(cast(list[object], first_page["post_plans"])) == 1


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
    assert raw_materialization_ready(raw_materialization_readiness_snapshot(tmp_path)) is False


def test_auto_resolve_stale_plan_blockers_unblocks_materialization_unattended(tmp_path: Path) -> None:
    """polylogue-d7im: one stale-plan blocker halts repair_materialization
    archive-wide (unresolved_raw_replay_blockers counts it), even though
    resolving it requires no operator judgment -- it only recomputes the
    plan from current evidence, exactly as an unattended crash-recovery pass
    already does elsewhere. auto_resolve_stale_plan_blockers must clear it
    without any human-supplied resolution text, and repair must proceed
    on the very next call."""
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="stale2", source_path="stale2.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    census = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids={plan.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "stale-auto-resolve"},
        residual={},
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = 'moved-after-plan-2.jsonl' WHERE raw_id = ?", (raw_id,))
        conn.commit()
    valid, observed = validate_raw_replay_plan(tmp_path, plan)
    assert valid is False
    reject_stale_raw_replay_plan(tmp_path, census.census_id, plan, observed)

    refused = repair_raw_materialization(_config(tmp_path))
    assert refused.success is False

    resolved_count = auto_resolve_stale_plan_blockers(tmp_path)

    assert resolved_count == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 0
        )
        resolution = conn.execute(
            "SELECT json_extract(resolution, '$.operator_resolution') FROM raw_authority_blockers"
        ).fetchone()[0]
        assert resolution == AUTO_STALE_PLAN_RESOLUTION

    proceeds = repair_raw_materialization(_config(tmp_path))
    assert proceeds.metrics.get("raw_materialization_unresolved_blocker_count", 0.0) == 0.0

    # Idempotent: nothing left to clear on a second call.
    assert auto_resolve_stale_plan_blockers(tmp_path) == 0


def test_auto_resolve_stale_plan_blockers_never_touches_frontier_judgment_blockers(tmp_path: Path) -> None:
    """Frontier-judgment blockers require an accepted assertion id +
    disposition (enforced inside resolve_raw_authority_blocker itself);
    auto_resolve_stale_plan_blockers must only ever query non-frontier
    (stale_plan) blockers, never attempt -- and therefore never accidentally
    satisfy -- a judgment-gated one."""
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="judgment", source_path="judgment.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    base_plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    frontier_plan = RawReplayPlan(
        plan_id=base_plan.plan_id,
        input_digest=base_plan.input_digest,
        input_raw_ids=base_plan.input_raw_ids,
        logical_keys=base_plan.logical_keys,
        authority_witness=json_document({"schema": "polylogue.raw-authority-frontier-plan.v1"}),
        source_preconditions=base_plan.source_preconditions,
        index_preconditions=base_plan.index_preconditions,
    )
    census = record_raw_authority_census(
        tmp_path,
        (frontier_plan,),
        selected_plan_ids={frontier_plan.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "frontier-judgment"},
        residual={},
    )
    reject_stale_raw_replay_plan(
        tmp_path,
        census.census_id,
        frontier_plan,
        json_document({"judgment_assertion_id": "assertion:unaccepted"}),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        blocker_id = conn.execute(
            "SELECT blocker_id FROM raw_authority_blockers WHERE resolved_at_ms IS NULL"
        ).fetchone()[0]

    resolved_count = auto_resolve_stale_plan_blockers(tmp_path)

    assert resolved_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM raw_authority_blockers WHERE blocker_id = ? AND resolved_at_ms IS NULL",
                (blocker_id,),
            ).fetchone()[0]
            == 1
        )


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
    finalized = finalize_raw_authority_census(
        tmp_path,
        receipt.census_id,
        post_plans=plans,
        post_residual={},
        interrupted=True,
    )
    assert finalized.lifecycle_status == "interrupted"


def test_postflight_allows_carried_plan_superseded_by_selected_logical_source(tmp_path: Path) -> None:
    """A selected authority change may retire another raw for its logical source.

    Anti-vacuity: treating raw-id disjointness as the only relationship makes
    finalization reject this valid postflight because ``retired`` is omitted.
    """
    initialize_active_archive_root(tmp_path)
    selected = RawReplayPlan(
        plan_id="selected",
        input_digest="a" * 64,
        input_raw_ids=("selected-raw",),
        logical_keys=("chatgpt:shared",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    retired = RawReplayPlan(
        plan_id="retired",
        input_digest="b" * 64,
        input_raw_ids=("retired-raw",),
        logical_keys=("chatgpt:shared",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    independent = RawReplayPlan(
        plan_id="independent",
        input_digest="c" * 64,
        input_raw_ids=("independent-raw",),
        logical_keys=("chatgpt:independent",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    receipt = record_raw_authority_census(
        tmp_path,
        (selected, retired, independent),
        selected_plan_ids={selected.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "logical-supersession"},
        residual={},
    )
    record_raw_replay_outcome(
        tmp_path,
        receipt.census_id,
        RawReplayPlanOutcome(
            selected.plan_id,
            selected.input_raw_ids,
            RawReplayPlanStatus.EXECUTED,
            "selected authority reached terminal state",
            "none",
        ),
    )

    finalized = finalize_raw_authority_census(
        tmp_path,
        receipt.census_id,
        post_plans=(independent,),
        post_residual={},
    )

    assert finalized.lifecycle_status == "completed"
    assert finalized.post_plan_count == 1


def test_postflight_rejects_missing_independent_carried_plan(tmp_path: Path) -> None:
    """Logical-source supersession must not weaken independent-plan protection."""
    initialize_active_archive_root(tmp_path)
    selected = RawReplayPlan(
        plan_id="selected",
        input_digest="a" * 64,
        input_raw_ids=("selected-raw",),
        logical_keys=("chatgpt:selected",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    independent = RawReplayPlan(
        plan_id="independent",
        input_digest="c" * 64,
        input_raw_ids=("independent-raw",),
        logical_keys=("chatgpt:independent",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    receipt = record_raw_authority_census(
        tmp_path,
        (selected, independent),
        selected_plan_ids={selected.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "independent-preservation"},
        residual={},
    )
    record_raw_replay_outcome(
        tmp_path,
        receipt.census_id,
        RawReplayPlanOutcome(
            selected.plan_id,
            selected.input_raw_ids,
            RawReplayPlanStatus.EXECUTED,
            "selected authority reached terminal state",
            "none",
        ),
    )

    with pytest.raises(RuntimeError, match="postflight changed a retryable/carried-forward plan"):
        finalize_raw_authority_census(
            tmp_path,
            receipt.census_id,
            post_plans=(),
            post_residual={},
        )


def test_postflight_rejects_carried_plan_shared_with_retryable_selection(tmp_path: Path) -> None:
    """Retryable work cannot retire a different raw for the same logical source."""
    initialize_active_archive_root(tmp_path)
    retryable = RawReplayPlan(
        plan_id="retryable",
        input_digest="a" * 64,
        input_raw_ids=("retryable-raw",),
        logical_keys=("chatgpt:shared",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    carried = RawReplayPlan(
        plan_id="carried",
        input_digest="b" * 64,
        input_raw_ids=("carried-raw",),
        logical_keys=("chatgpt:shared",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    receipt = record_raw_authority_census(
        tmp_path,
        (retryable, carried),
        selected_plan_ids={retryable.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "retryable-does-not-supersede"},
        residual={},
    )
    record_raw_replay_outcome(
        tmp_path,
        receipt.census_id,
        RawReplayPlanOutcome(
            retryable.plan_id,
            retryable.input_raw_ids,
            RawReplayPlanStatus.RETRYABLE,
            "selected authority remains executable",
            "retry after the current writer pass",
        ),
    )

    with pytest.raises(RuntimeError, match="postflight changed a retryable/carried-forward plan"):
        finalize_raw_authority_census(
            tmp_path,
            receipt.census_id,
            post_plans=(),
            post_residual={},
        )


def test_interrupted_finalize_tolerates_a_reclassified_carried_plan(tmp_path: Path) -> None:
    """polylogue-ewfp regression: crash recovery must tolerate legitimate reclassification.

    A "planned" census can sit unfinalized across an arbitrary gap (a daemon
    restart, days of subsequent live activity) before crash recovery
    (``interrupted=True``) finalizes it. During that gap, a retryable/
    carried-forward plan's own evidence can legitimately shift for reasons
    unrelated to what THIS census selected -- e.g. a raw getting quarantined
    by an unrelated safety mechanism, or (as observed live) a duplicate-alias
    fan-out sibling's canonical getting claimed by another sibling. Before
    the fix, ``finalize_raw_authority_census`` applied the same strict
    "every retryable/carried-forward plan must survive unchanged" check
    regardless of ``interrupted``, so a census stuck at this exact shape
    could NEVER finalize -- every recovery attempt re-selected and
    re-failed the identical stale plan_ids forever (observed live: a
    200+ second writer-lock hold on every daemon restart). The identical
    scenario must still raise for a normal, uninterrupted apply (see
    ``test_postflight_rejects_carried_plan_shared_with_retryable_selection``
    immediately above) -- only crash recovery gets the tolerance.
    """
    initialize_active_archive_root(tmp_path)
    retryable = RawReplayPlan(
        plan_id="retryable",
        input_digest="a" * 64,
        input_raw_ids=("retryable-raw",),
        logical_keys=("chatgpt:shared",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    carried = RawReplayPlan(
        plan_id="carried",
        input_digest="b" * 64,
        input_raw_ids=("carried-raw",),
        logical_keys=("chatgpt:shared",),
        authority_witness=json_document({}),
        source_preconditions=json_document({}),
        index_preconditions=json_document({}),
    )
    receipt = record_raw_authority_census(
        tmp_path,
        (retryable, carried),
        selected_plan_ids={retryable.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "interrupted-tolerates-reclassification"},
        residual={},
    )
    record_raw_replay_outcome(
        tmp_path,
        receipt.census_id,
        RawReplayPlanOutcome(
            retryable.plan_id,
            retryable.input_raw_ids,
            RawReplayPlanStatus.RETRYABLE,
            "selected authority remains executable",
            "retry after the current writer pass",
        ),
    )

    # The regression: this must not raise, even though `carried`'s plan_id
    # (recorded in the census as retryable/carried-forward, sharing
    # logical_keys with the selected plan) is absent from post_plans --
    # exactly the shape the non-interrupted test above asserts DOES raise.
    finalized = finalize_raw_authority_census(
        tmp_path,
        receipt.census_id,
        post_plans=(),
        post_residual={},
        interrupted=True,
    )

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

    incomplete_receipts = []
    for _expected_pass in range(2):
        incomplete = repair_raw_materialization(_config(tmp_path), dry_run=True, raw_artifact_limit=1)
        assert incomplete.census_receipt is not None
        assert incomplete.census_receipt.quiescent is False
        assert incomplete.census_receipt.plan_count == 0
        assert incomplete.metrics["raw_materialization_census_component_limit"] == 1.0
        assert incomplete.metrics["raw_materialization_census_components_attempted"] == 1.0
        incomplete_ledger = read_raw_authority_census(tmp_path, incomplete.census_receipt.query_handle)
        assert incomplete_ledger["plans"] == []
        census_detail = _read_detail_document(
            tmp_path,
            cast(str, cast(dict[str, object], incomplete_ledger["census"])["detail_query_handle"]),
        )
        pending_residual = cast(dict[str, object], census_detail["residual"])
        assert cast(int, pending_residual["census_pending_raw_count"]) >= 1
        assert len(cast(str, pending_residual["census_pending_raw_digest"])) == 64
        assert "census_pending_raw_ids" not in pending_residual
        incomplete_receipts.append(incomplete.census_receipt.census_id)
    assert len(set(incomplete_receipts)) == 2

    preview = repair_raw_materialization(_config(tmp_path), dry_run=True, raw_artifact_limit=1)

    assert preview.census_receipt is not None
    assert preview.census_receipt.quiescent is True
    ledger = read_raw_authority_census(tmp_path, preview.census_receipt.query_handle)
    raw_sets = {
        frozenset(
            cast(
                list[str],
                cast(
                    dict[str, object],
                    _read_detail_document(
                        tmp_path,
                        cast(str, cast(dict[str, object], item)["detail_query_handle"]),
                    )["plan"],
                )["input_raw_ids"],
            )
        )
        for item in cast(list[object], ledger["plans"])
    }
    assert raw_sets == {frozenset((first, second)), frozenset((third,))}


def test_census_page_bounds_one_oversized_plan_and_detail_chunks_reconstruct_it(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    raw_ids = tuple(f"raw-{index:05d}" for index in range(2_000))
    plan = RawReplayPlan(
        plan_id="raw-replay:oversized",
        input_digest="a" * 64,
        input_raw_ids=raw_ids,
        logical_keys=tuple(f"codex:key-{index:05d}" for index in range(2_000)),
        authority_witness=json_document({"rows": [{"raw_id": raw_id} for raw_id in raw_ids]}),
        source_preconditions=json_document({"rows": [{"raw_id": raw_id} for raw_id in raw_ids]}),
        index_preconditions=json_document({"rows": [{"raw_id": raw_id} for raw_id in raw_ids]}),
    )
    receipt = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids=set(),
        executable_plan_ids={plan.plan_id},
        mode="dry_run",
        quiescent=True,
        scope={"test": "oversized"},
        residual={},
    )

    page = read_raw_authority_census(tmp_path, receipt.query_handle, limit=1)

    assert len(json.dumps(page)) < 8_000
    item = cast(dict[str, object], cast(list[object], page["plans"])[0])
    summary = cast(dict[str, object], item["plan"])
    assert summary["input_raw_count"] == 2_000
    assert "input_raw_ids" not in summary
    detail = _read_detail_document(tmp_path, cast(str, item["detail_query_handle"]))
    detail_plan = cast(dict[str, object], detail["plan"])
    assert detail_plan["input_raw_ids"] == list(raw_ids)
    assert len(cast(list[object], cast(dict[str, object], detail_plan["authority_witness"])["rows"])) == 2_000


def test_interrupted_apply_recovers_exact_durable_postconditions(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(
        tmp_path,
        native_id="crash",
        source_path="crash.jsonl",
        acquired_at_ms=1,
        text="interrupted-authority-fts-needle",
    )

    with patch.object(repair_mod, "raw_replay_application_receipt", side_effect=RuntimeError("synthetic crash")):
        with pytest.raises(RuntimeError, match="synthetic crash"):
            repair_raw_materialization(_config(tmp_path))

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_censuses WHERE lifecycle_status = 'planned'").fetchone()[0]
            == 1
        )

    # The injected failure is deliberately after the production replay writer
    # has committed its source/index work but before the immutable plan outcome
    # is receipted.  Resume must preserve that already accepted authority and
    # its trigger-maintained FTS projection; it may only finish durable ledger
    # accounting for the interrupted census.
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_before_resume = source_conn.execute(
            """
            SELECT raw_id, logical_source_key, source_revision, revision_kind,
                   revision_authority, parsed_at_ms
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchall()
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        heads_before_resume = index_conn.execute(
            """
            SELECT logical_source_key, session_id, accepted_raw_id,
                   accepted_source_revision, hex(accepted_content_hash),
                   accepted_frontier_kind, accepted_frontier
            FROM raw_revision_heads ORDER BY logical_source_key
            """
        ).fetchall()
        sessions_before_resume = index_conn.execute(
            "SELECT session_id, raw_id, hex(content_hash) FROM sessions ORDER BY session_id"
        ).fetchall()
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        fts_before_resume = dict(archive.index_status())
        fts_hits_before_resume = archive.search_blocks("interrupted-authority-fts-needle")

    assert source_before_resume
    assert heads_before_resume
    assert heads_before_resume[0][2] == raw_id
    assert sessions_before_resume
    assert fts_before_resume["exists"] is True
    assert fts_hits_before_resume

    recovered = repair_raw_materialization(_config(tmp_path))
    assert recovered.metrics["raw_materialization_recovered_census_count"] == 1.0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """
            SELECT c.lifecycle_status, cp.outcome_status, cp.application_receipt_json
            FROM raw_authority_censuses AS c
            JOIN raw_authority_census_plans AS cp ON cp.census_id = c.census_id
            WHERE c.lifecycle_status = 'interrupted'
            """
        ).fetchone()
    assert row is not None
    assert row[:2] == ("interrupted", "executed")
    recovered_receipt = json.loads(row[2])
    assert isinstance(recovered_receipt["application_rows"], list)
    assert isinstance(recovered_receipt["membership_rows"], list)
    assert recovered_receipt["application_rows"] or recovered_receipt["membership_rows"]
    assert recovered_receipt["head_rows"]
    assert recovered_receipt["session_rows"]
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_after_resume = source_conn.execute(
            """
            SELECT raw_id, logical_source_key, source_revision, revision_kind,
                   revision_authority, parsed_at_ms
            FROM raw_sessions WHERE raw_id = ?
            """,
            (raw_id,),
        ).fetchall()
    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        heads_after_resume = index_conn.execute(
            """
            SELECT logical_source_key, session_id, accepted_raw_id,
                   accepted_source_revision, hex(accepted_content_hash),
                   accepted_frontier_kind, accepted_frontier
            FROM raw_revision_heads ORDER BY logical_source_key
            """
        ).fetchall()
        sessions_after_resume = index_conn.execute(
            "SELECT session_id, raw_id, hex(content_hash) FROM sessions ORDER BY session_id"
        ).fetchall()
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        fts_after_resume = dict(archive.index_status())
        fts_hits_after_resume = archive.search_blocks("interrupted-authority-fts-needle")

    assert source_after_resume == source_before_resume
    assert heads_after_resume == heads_before_resume
    assert sessions_after_resume == sessions_before_resume
    assert fts_after_resume == fts_before_resume
    assert fts_hits_after_resume == fts_hits_before_resume


def test_parsed_timestamp_without_exact_application_receipt_fails_closed(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _write_codex_raw(tmp_path, native_id="receipt", source_path="receipt.jsonl", acquired_at_ms=1)
    real_receipt = raw_authority_mod.raw_replay_application_receipt

    def incomplete_receipt(
        root: Path,
        plan: RawReplayPlan,
        *,
        index_db_path: Path | None = None,
    ) -> JSONDocument:
        payload = dict(real_receipt(root, plan, index_db_path=index_db_path))
        payload["head_rows"] = []
        return json_document(payload)

    with patch.object(repair_mod, "raw_replay_application_receipt", side_effect=incomplete_receipt):
        result = repair_raw_materialization(_config(tmp_path))

    assert result.plan_outcomes[0].status is RawReplayPlanStatus.REJECTED_STALE
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 1
        )


def test_application_receipt_reads_the_active_generation_not_shadow_index(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="active-receipt", source_path="active.jsonl", acquired_at_ms=1)
    assert repair_raw_materialization(_config(tmp_path)).success is True
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    (tmp_path / ".index-active-pointer").write_text(str(active_index), encoding="utf-8")

    receipt = raw_authority_mod.raw_replay_application_receipt(tmp_path, plan)

    assert receipt["index_db_path"] == str(active_index)
    assert receipt["application_rows"] == []


@pytest.mark.parametrize("field", ["session_id", "accepted_raw_id", "accepted_content_hash"])
def test_application_receipt_requires_exact_application_authority(tmp_path: Path, field: str) -> None:
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id=f"exact-{field}", source_path=f"{field}.jsonl", acquired_at_ms=1)
    assert repair_raw_materialization(_config(tmp_path)).success is True
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    receipt = dict(raw_authority_mod.raw_replay_application_receipt(tmp_path, plan))
    application_rows = cast(list[dict[str, object]], receipt["application_rows"])
    assert application_rows
    application_rows[0][field] = f"wrong-{field}"

    valid, problems = raw_authority_mod.validate_raw_replay_application_receipt(plan, receipt)

    assert valid is False
    assert any("no application accepted authority matches" in problem for problem in problems)


def test_recovery_rejects_partial_expanded_membership_postconditions(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    _write_codex_raw(
        tmp_path,
        native_id="partial-component",
        source_path="partial-old.jsonl",
        acquired_at_ms=1,
        text="old",
    )
    second = _write_codex_raw(
        tmp_path,
        native_id="partial-component",
        source_path="partial-new.jsonl",
        acquired_at_ms=2,
        text="new",
    )

    with patch.object(repair_mod, "raw_replay_application_receipt", side_effect=RuntimeError("synthetic crash")):
        with pytest.raises(RuntimeError, match="synthetic crash"):
            repair_raw_materialization(_config(tmp_path))

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("DELETE FROM raw_session_memberships WHERE raw_id = ?", (second,))
        conn.commit()

    recovered = repair_raw_materialization(_config(tmp_path))

    assert recovered.success is False
    assert recovered.metrics["raw_materialization_unresolved_blocker_count"] == 1.0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """
            SELECT cp.outcome_status
            FROM raw_authority_census_plans AS cp
            JOIN raw_authority_censuses AS c ON c.census_id = cp.census_id
            WHERE c.lifecycle_status = 'interrupted' AND cp.selected = 1
            """
        ).fetchone()
        assert row == ("rejected_stale",)
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

    census_page = read_raw_authority_census(tmp_path, census.query_handle)
    plan_summary = cast(dict[str, object], cast(list[object], census_page["plans"])[0])
    first_detail_page = read_raw_authority_detail(
        tmp_path,
        cast(str, plan_summary["detail_query_handle"]),
        chunk_chars=256,
    )
    stale_continuation = cast(str, first_detail_page["next_query_handle"])

    resolution = resolve_raw_authority_blocker(tmp_path, blocker_id, resolution="current path is authoritative")
    with pytest.raises(RuntimeError, match="raw authority detail changed"):
        read_raw_authority_detail(tmp_path, stale_continuation, chunk_chars=256)
    current_detail = _read_detail_document(tmp_path, cast(str, resolution["detail_query_handle"]))
    resumed = repair_raw_materialization(_config(tmp_path))

    assert resolution["blocker_id"] == blocker_id
    resolution_plan = cast(dict[str, object], resolution["current_plan"])
    assert resolution_plan["input_raw_count"] == 1
    assert "input_raw_ids" not in resolution_plan
    stored_resolution = cast(
        dict[str, object],
        cast(dict[str, object], cast(list[object], current_detail["blockers"])[0])["resolution"],
    )
    assert cast(dict[str, object], stored_resolution["current_plan"])["input_raw_ids"] == [raw_id]
    assert resumed.success is True
    assert resumed.repaired_count == 1
    assert "raw_materialization_unresolved_blocker_count" not in resumed.metrics
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 0
        )


def test_identical_stale_rejection_after_resolution_creates_new_open_blocker(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="repeat", source_path="repeat.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET source_path = 'repeat-moved.jsonl' WHERE raw_id = ?", (raw_id,))
        conn.commit()
    valid, observed = validate_raw_replay_plan(tmp_path, plan)
    assert valid is False

    first_census = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids={plan.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "repeat-1"},
        residual={},
    )
    first = reject_stale_raw_replay_plan(tmp_path, first_census.census_id, plan, observed)
    first_blocker = cast(str, cast(dict[str, object], first.application_receipt)["blocker_id"])
    resolve_raw_authority_blocker(tmp_path, first_blocker, resolution="acknowledge first occurrence")

    second_census = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids={plan.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "repeat-2"},
        residual={},
    )
    second = reject_stale_raw_replay_plan(tmp_path, second_census.census_id, plan, observed)
    second_blocker = cast(str, cast(dict[str, object], second.application_receipt)["blocker_id"])

    assert second_blocker != first_blocker
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 1
        )
    page = read_raw_authority_census(tmp_path, second_census.query_handle)
    assert page["blocker_count"] == 1
    item = cast(dict[str, object], cast(list[object], page["plans"])[0])
    detail = _read_detail_document(tmp_path, cast(str, item["detail_query_handle"]))
    blockers = cast(list[object], detail["blockers"])
    assert len(blockers) == 1
    assert cast(dict[str, object], blockers[0])["blocker_id"] == second_blocker


def test_recovery_returns_planned_census_after_all_outcomes_are_recorded(tmp_path: Path) -> None:
    """A crash after outcome commit finalizes the planned census on restart."""
    initialize_active_archive_root(tmp_path)
    plan = RawReplayPlan(
        plan_id="raw-replay:outcome-recorded",
        input_digest="a" * 64,
        input_raw_ids=("raw-outcome-recorded",),
        logical_keys=("codex:outcome-recorded",),
        authority_witness={},
        source_preconditions={},
        index_preconditions={},
    )
    census = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids={plan.plan_id},
        mode="apply",
        quiescent=True,
        scope={"test": "outcome-recorded"},
        residual={},
    )
    record_raw_replay_outcome(
        tmp_path,
        census.census_id,
        RawReplayPlanOutcome(plan.plan_id, plan.input_raw_ids, RawReplayPlanStatus.EXECUTED, "done", "none"),
    )

    recovered = repair_raw_materialization(_config(tmp_path))

    assert recovered.metrics["raw_materialization_recovered_census_count"] == 1.0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute(
            "SELECT lifecycle_status FROM raw_authority_censuses WHERE census_id = ?", (census.census_id,)
        ).fetchone() == ("interrupted",)
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_authority_censuses WHERE lifecycle_status = 'planned'"
        ).fetchone() == (0,)


def test_frontier_preview_cannot_claim_one_plan_twice(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    plan = RawReplayPlan(
        plan_id="raw-authority-frontier:" + "a" * 64,
        input_digest="b" * 64,
        input_raw_ids=("raw-once",),
        logical_keys=("codex:once",),
        authority_witness={"schema": "polylogue.raw-authority-frontier-plan.v1"},
        source_preconditions={},
        index_preconditions={},
    )
    scope = {"schema": "polylogue.raw-authority-frontier-scope.v1", "preview_census_id": "preview:once"}
    record_raw_authority_census(
        tmp_path, (plan,), selected_plan_ids={plan.plan_id}, mode="apply", quiescent=True, scope=scope, residual={}
    )

    with pytest.raises(RuntimeError, match="already claimed"):
        record_raw_authority_census(
            tmp_path, (plan,), selected_plan_ids={plan.plan_id}, mode="apply", quiescent=True, scope=scope, residual={}
        )


def test_concurrent_frontier_claims_serialize_before_apply_census(tmp_path: Path) -> None:
    """One preview plan has one durable apply census, even under concurrent callers."""
    initialize_active_archive_root(tmp_path)
    plan = RawReplayPlan(
        plan_id="raw-authority-frontier:" + "c" * 64,
        input_digest="d" * 64,
        input_raw_ids=("raw-concurrent",),
        logical_keys=("codex:concurrent",),
        authority_witness={"schema": "polylogue.raw-authority-frontier-plan.v1"},
        source_preconditions={},
        index_preconditions={},
    )
    scope = {"schema": "polylogue.raw-authority-frontier-scope.v1", "preview_census_id": "preview:concurrent"}
    barrier = threading.Barrier(2)

    def claim() -> str:
        barrier.wait()
        try:
            record_raw_authority_census(
                tmp_path,
                (plan,),
                selected_plan_ids={plan.plan_id},
                mode="apply",
                quiescent=True,
                scope=scope,
                residual={},
            )
        except RuntimeError as exc:
            assert "already claimed" in str(exc)
            return "rejected"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: claim(), range(2)))

    assert sorted(outcomes) == ["claimed", "rejected"]
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_censuses WHERE mode = 'apply'").fetchone() == (1,)


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
            == RAW_AUTHORITY_PARSER_FINGERPRINT
        )


def _seed_ambiguous_membership_component(
    tmp_path: Path,
    *,
    native_id: str,
    parser_fingerprint: str | None,
) -> tuple[str, RawReplayPlanOutcome]:
    """Seed one raw whose membership decision is durably 'ambiguous'.

    ``parser_fingerprint`` controls what (if anything) the per-raw
    ``raw_authority_parser_census`` row records: the CURRENT fingerprint (the
    ambiguous verdict should still be terminal), a fingerprint listed in
    ``SUPERSEDED_MEMBERSHIP_FINGERPRINTS`` (the verdict is stale and must be
    replayable), or ``None`` (no census row at all -- absent evidence must
    stay conservative and remain terminal).
    """
    raw_id = _write_codex_raw(tmp_path, native_id=native_id, source_path=f"{native_id}.jsonl", acquired_at_ms=1)
    logical_source_key = f"codex-session:{native_id}"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, decision, decided_at_ms
            ) VALUES (?, ?, ?, ?, ?, 1, 'ambiguous', 1)
            """,
            (raw_id, logical_source_key, native_id, "rev-1", bytes(32)),
        )
        if parser_fingerprint is not None:
            conn.execute(
                """
                INSERT INTO raw_authority_parser_census (
                    raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms
                ) VALUES (?, ?, 'complete', ?, 'test-seeded', 0)
                """,
                (raw_id, parser_fingerprint, json.dumps([logical_source_key])),
            )
        conn.commit()
    (plan,) = build_raw_replay_plans(tmp_path, [(raw_id,)])
    empty_remaining = repair_mod.RawMaterializationCandidates([], 0, 0)
    (outcome,) = repair_mod._raw_replay_plan_outcomes(
        tmp_path,
        resolve_active_index_path(tmp_path),
        [plan],
        remaining=empty_remaining,
    )
    return raw_id, outcome


def test_ambiguous_verdict_under_current_fingerprint_stays_terminal(tmp_path: Path) -> None:
    """polylogue-9dxn: an 'ambiguous' decision recorded under the CURRENT
    classifier fingerprint is still authoritative -- it must not be
    replayed without new evidence.
    """
    initialize_active_archive_root(tmp_path)
    _raw_id, outcome = _seed_ambiguous_membership_component(
        tmp_path, native_id="current-ambiguous", parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT
    )
    assert outcome.status is RawReplayPlanStatus.TERMINAL
    assert "ambiguous" in outcome.reason.lower()


@pytest.mark.parametrize("superseded_fingerprint", ["revision-membership-v1", "revision-membership-v2"])
def test_ambiguous_verdict_under_superseded_fingerprint_is_replayable(
    tmp_path: Path, superseded_fingerprint: str
) -> None:
    """polylogue-9dxn: an 'ambiguous' decision recorded under a fingerprint
    listed in SUPERSEDED_MEMBERSHIP_FINGERPRINTS is stale -- a corrected
    classifier deserves a chance to re-derive it, so it must not be
    terminal.

    Anti-vacuity: this exercises the real production route
    ``repair._raw_replay_plan_outcome`` (via the public
    ``build_raw_replay_plans``/``_raw_replay_plan_outcomes`` pair used by
    ``repair_raw_materialization``, the daemon's live raw-materialization
    repair entrypoint). Reverting the fingerprint-gating clause added to the
    terminal query in ``storage/repair.py`` (the ``LEFT JOIN
    raw_authority_parser_census`` + ``NOT COALESCE(... IN (SELECT value FROM
    json_each(?)) ...)`` guard) makes this test fail by re-classifying the
    plan as TERMINAL.
    """
    initialize_active_archive_root(tmp_path)
    assert superseded_fingerprint in raw_authority_mod.SUPERSEDED_MEMBERSHIP_FINGERPRINTS
    _raw_id, outcome = _seed_ambiguous_membership_component(
        tmp_path, native_id="superseded-ambiguous", parser_fingerprint=superseded_fingerprint
    )
    assert outcome.status is not RawReplayPlanStatus.TERMINAL


def test_ambiguous_verdict_with_no_census_row_stays_terminal(tmp_path: Path) -> None:
    """polylogue-9dxn: absent census evidence must default to conservative
    (terminal), not to "assume the classifier fix already applies".
    """
    initialize_active_archive_root(tmp_path)
    _raw_id, outcome = _seed_ambiguous_membership_component(
        tmp_path, native_id="uncensused-ambiguous", parser_fingerprint=None
    )
    assert outcome.status is RawReplayPlanStatus.TERMINAL


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


def test_repair_result_omits_unbounded_receipt_rows_from_outcome_sample() -> None:
    outcome = RawReplayPlanOutcome(
        "plan-with-receipt",
        tuple(f"raw-{index}" for index in range(100)),
        RawReplayPlanStatus.EXECUTED,
        "done",
        "none",
        json_document({"application_rows": [{"row": index} for index in range(1000)]}),
    )
    result = RepairResult(
        "raw_materialization",
        MaintenanceCategory.DERIVED_REPAIR,
        False,
        1,
        True,
        plan_outcomes=(outcome,),
    ).to_dict()
    sample = cast(list[dict[str, object]], result["plan_outcomes"])[0]
    assert sample["has_application_receipt"] is True
    assert "application_receipt" not in sample
    assert sample["input_raw_count"] == 100
    assert len(cast(list[object], sample["input_raw_id_sample"])) == 8
    assert sample["input_raw_id_sample_truncated"] is True
    assert "input_raw_ids" not in sample


def test_frontier_classifies_dangling_head_session_as_corrupt(tmp_path: Path) -> None:
    """polylogue-lkrc AC1/AC6/AC7: the CORRUPT terminal state had zero
    regression coverage anywhere in the suite even though it is one of the
    eight mutually exclusive frontier states the reconciler declares and
    persists as a durable blocker.

    This reproduces the first of ``_classify_frontier``'s three CORRUPT
    triggers (``polylogue/storage/raw_reconciler.py``): ``raw_revision_heads``
    still names a ``session_id`` but the materialized session row it points at
    is gone (a torn write, an interrupted rebuild, or manual tampering with
    the rebuildable index tier). Proven-current accepted heads must never
    silently read as healthy in this shape.
    """
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="dangling-session", source_path="dangling.jsonl", acquired_at_ms=1)
    assert repair_raw_materialization(_config(tmp_path)).repaired_count == 1

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        session_id = index_conn.execute(
            "SELECT session_id FROM raw_revision_heads WHERE accepted_raw_id = ?", (raw_id,)
        ).fetchone()[0]
        # Simulate the accepted head surviving while its materialized session
        # vanishes underneath it -- the index tier is rebuildable and this is
        # exactly the kind of partial state a crash mid-rebuild can leave.
        index_conn.execute("PRAGMA foreign_keys = OFF")
        index_conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        index_conn.commit()

    census = inspect_raw_authority_frontier(_config(tmp_path))

    item = next(entry for entry in census.items if entry.raw_id == raw_id)
    assert item.state is RawAuthorityFrontierState.CORRUPT
    assert item.reason == "accepted head has no matching materialized session"
    assert item.executable is False
    assert census.state_counts[RawAuthorityFrontierState.CORRUPT.value] == 1

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        blocker = source_conn.execute(
            "SELECT reason, resolved_at_ms FROM raw_authority_blockers WHERE plan_id = ?",
            (item.plan_id,),
        ).fetchone()
    assert blocker is not None
    assert blocker[1] is None

    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_frontier_blocking_count"] == 1
    assert raw_materialization_ready(readiness) is False
    refs = cast(list[dict[str, object]], readiness["raw_authority_frontier_remediation_refs"])
    assert item.plan_id in {ref["plan_id"] for ref in refs}


def test_frontier_classifies_head_session_raw_mismatch_as_corrupt(tmp_path: Path) -> None:
    """polylogue-lkrc AC1/AC6/AC7: reproduces the second reachable CORRUPT
    trigger -- the accepted head names one raw as authoritative
    (``accepted_raw_id``) while the materialized session it points at was
    actually built from a *different* raw. This is the torn-write shape the
    reconciler's own comment describes ("accepted revision head and
    materialized session select different raw authority"): a genuine
    disagreement between two derived-tier tables that a read-only census
    must surface as a durable blocker rather than silently trust the head.
    """
    initialize_active_archive_root(tmp_path)
    accepted_raw_id = _write_codex_raw(
        tmp_path, native_id="mismatch-one", source_path="mismatch.jsonl", acquired_at_ms=1
    )
    assert repair_raw_materialization(_config(tmp_path)).repaired_count == 1
    # An independent, never-materialized raw acquisition -- stands in for the
    # "wrong" raw a corrupted head could point at.
    phantom_raw_id = _write_codex_raw(tmp_path, native_id="phantom-only", source_path="phantom.jsonl", acquired_at_ms=2)

    with sqlite3.connect(tmp_path / "index.db") as index_conn:
        logical_source_key = index_conn.execute(
            "SELECT logical_source_key FROM raw_revision_heads WHERE accepted_raw_id = ?", (accepted_raw_id,)
        ).fetchone()[0]
        index_conn.execute(
            "UPDATE raw_revision_heads SET accepted_raw_id = ? WHERE logical_source_key = ?",
            (phantom_raw_id, logical_source_key),
        )
        index_conn.commit()

    census = inspect_raw_authority_frontier(_config(tmp_path))

    # The session itself was materialized from accepted_raw_id, so the
    # classifier resolves the raw row by the SESSION's own raw_id, not the
    # (now wrong) value stashed on the head row -- the surfaced item is still
    # keyed by the real session raw, with the head's disagreement in the
    # reason/evidence.
    item = next(entry for entry in census.items if entry.raw_id == accepted_raw_id)
    assert item.state is RawAuthorityFrontierState.CORRUPT
    assert item.reason == "accepted revision head and materialized session select different raw authority"
    assert item.executable is False
    assert item.index_preconditions["head_accepted_raw_id"] == phantom_raw_id
    assert item.index_preconditions["accepted_raw_id"] == accepted_raw_id
    assert census.state_counts[RawAuthorityFrontierState.CORRUPT.value] == 1

    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_frontier_blocking_count"] == 1
    assert raw_materialization_ready(readiness) is False


def test_verified_blob_receipt_invalidates_when_blob_bytes_change_underneath_it(tmp_path: Path) -> None:
    """polylogue-byw3y: the safety-critical half of the receipt cache.

    A verification receipt is a durable HINT, never an authority: it may only
    ever be trusted for the exact on-disk fingerprint (dev/inode/size/mtime/
    ctime) it was recorded against. If a blob's bytes are corrupted/mutated
    in place -- the file at the content-addressed path no longer matches its
    own filename hash -- the next census MUST re-verify from scratch and
    reclassify the frontier item as unproven, never silently keep trusting a
    stale receipt. This is the regression this bead's whole design exists to
    prevent: a performance win here would be worthless (and actively unsafe)
    if it could paper over real corruption.
    """
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(
        tmp_path, native_id="tamper-target", source_path="tamper.jsonl", acquired_at_ms=1, text="hello"
    )
    assert repair_raw_materialization(_config(tmp_path)).repaired_count == 1

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        blob_hash_hex = str(
            source_conn.execute("SELECT hex(blob_hash) FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        ).lower()

    # First census: proves the blob, records a durable receipt.
    census = inspect_raw_authority_frontier(_config(tmp_path))
    item = next(entry for entry in census.items if entry.raw_id == raw_id)
    assert item.state is RawAuthorityFrontierState.PROVEN_CURRENT

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        receipt = source_conn.execute(
            "SELECT st_size FROM verified_blob_receipts WHERE blob_hash = ?",
            (bytes.fromhex(blob_hash_hex),),
        ).fetchone()
    assert receipt is not None, "first census must persist a verification receipt"

    # Corrupt the blob bytes IN PLACE -- same content-addressed filename,
    # different content. This is the exact shape a stale-but-trusted receipt
    # would silently paper over: the fingerprint's st_size differs, so the
    # census must force a real re-hash rather than trust the old receipt.
    blob_path = BlobStore(tmp_path / "blob").blob_path(blob_hash_hex)
    blob_path.write_bytes(blob_path.read_bytes() + b"tampered-bytes")

    census2 = inspect_raw_authority_frontier(_config(tmp_path))
    item2 = next(entry for entry in census2.items if entry.raw_id == raw_id)
    assert item2.state is RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE
    assert item2.reason == "accepted head raw bytes do not prove the expected content-addressed digest"
    assert census2.state_counts[RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE.value] == 1


def test_verified_blob_receipt_skips_rehash_on_unchanged_blob_across_census_passes(tmp_path: Path) -> None:
    """polylogue-byw3y: the performance half -- a blob verified once and left
    untouched must not be re-hashed by a second census pass. Counts actual
    ``BlobStore.verify`` invocations (the real content re-hash) rather than
    trusting a wall-clock or state proxy, so the assertion fails honestly if
    the receipt cache regresses back to re-verifying every restart.
    """
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(
        tmp_path, native_id="unchanged-target", source_path="unchanged.jsonl", acquired_at_ms=1, text="hello"
    )
    assert repair_raw_materialization(_config(tmp_path)).repaired_count == 1

    verify_calls: list[str] = []
    real_verify = BlobStore.verify

    def _counting_verify(self: BlobStore, hash_hex: str) -> bool:
        verify_calls.append(hash_hex)
        return real_verify(self, hash_hex)

    with patch.object(BlobStore, "verify", _counting_verify):
        census1 = inspect_raw_authority_frontier(_config(tmp_path))
        assert len(verify_calls) == 1, "first census must hash the blob at least once"
        item1 = next(entry for entry in census1.items if entry.raw_id == raw_id)
        assert item1.state is RawAuthorityFrontierState.PROVEN_CURRENT

        verify_calls.clear()
        census2 = inspect_raw_authority_frontier(_config(tmp_path))
        assert verify_calls == [], "second census over an unchanged blob must reuse the durable receipt"
        item2 = next(entry for entry in census2.items if entry.raw_id == raw_id)
        assert item2.state is RawAuthorityFrontierState.PROVEN_CURRENT


def test_ineligible_quarantined_raw_gets_a_terminal_actuator_not_refine_quarantine(tmp_path: Path) -> None:
    """polylogue-u19l: reproduces the absorbing-state defect end to end.

    Live evidence (source.db, read-only, 2026-07-31): 4,147 open
    ``raw_authority_blockers`` rows all read "accepted raw authority remains
    quarantined pending exact refinement proof" with actuator
    REFINE_QUARANTINE, 15,205/17,384 frontier plans residual, fixed_point=0
    on all 256 retained censuses, and the gap count only ever grew
    (16,874 -> 17,384). Root cause: REFINE_QUARANTINE has a real apply()
    dispatch branch (``raw_reconciler.py``, the ``item.actuator is
    RawAuthorityActuator.REFINE_QUARANTINE`` block), but the executability
    gate (``_EXECUTABLE_STATES``) only ever admits SAFELY_REKEYABLE /
    DUPLICATE_ALIAS states -- so once ``inspect_quarantined_accepted_raws``
    proves a quarantined raw's refinement is "ineligible" (a permanent
    structural fact, not a transient one -- see the reasons enumerated in
    ``_inspect_quarantined_accepted_raw``), the census silently promised an
    actuator that neither the daemon nor the operator break-glass path
    could ever select.

    Force a raw into exactly that "ineligible" shape by accepting it
    normally, then flipping only its ``revision_authority`` to
    'quarantined' out from under an otherwise byte-proven envelope -- this
    fails ``_inspect_quarantined_accepted_raw``'s typed-envelope check
    (source/predecessor/baseline columns don't match any of the three
    admitted envelopes), which is exactly the "source raw has an
    incompatible typed authority envelope" ineligibility reason observed
    live. The frontier item must come back non-executable AND with an
    honest NONE actuator -- never REFINE_QUARANTINE -- while remaining
    countable (state_counts) and operator-visible (raw_authority_blockers).
    """
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(
        tmp_path, native_id="quarantine-ineligible", source_path="quarantine.jsonl", acquired_at_ms=1
    )
    assert repair_raw_materialization(_config(tmp_path)).repaired_count == 1

    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        source_conn.execute(
            "UPDATE raw_sessions SET revision_authority = 'quarantined' WHERE raw_id = ?",
            (raw_id,),
        )
        source_conn.commit()

    census = inspect_raw_authority_frontier(_config(tmp_path))

    item = next(entry for entry in census.items if entry.raw_id == raw_id)
    assert item.state is RawAuthorityFrontierState.UNRESOLVED_PROVENANCE
    assert item.actuator is raw_reconciler_mod.RawAuthorityActuator.NONE
    assert item.executable is False
    assert "ineligible" in item.reason
    assert census.state_counts[RawAuthorityFrontierState.UNRESOLVED_PROVENANCE.value] == 1

    # Terminal, countable, operator-visible: still tracked as an open
    # blocker (an operator can find it), just no longer misrepresented as
    # "an automatic actuator will resolve this".
    with sqlite3.connect(tmp_path / "source.db") as source_conn:
        blocker = source_conn.execute(
            "SELECT reason, resolved_at_ms FROM raw_authority_blockers WHERE plan_id = ?",
            (item.plan_id,),
        ).fetchone()
    assert blocker is not None
    assert blocker[1] is None
    assert "ineligible" in blocker[0]

    readiness = raw_materialization_readiness_snapshot(tmp_path)
    assert readiness["raw_authority_frontier_blocking_count"] == 1


def test_frontier_item_construction_rejects_unreachable_actuator_state_pairs() -> None:
    """polylogue-w32w: the invariant made structurally enforceable.

    ``RawAuthorityFrontierItem.__post_init__`` must reject any combination
    of a dispatch-handled actuator (one with a real apply() branch) paired
    with a state the executability gate does not admit -- the exact defect
    class polylogue-u19l fixed for REFINE_QUARANTINE. This proves the guard
    fires at construction, not merely that today's call sites happen to
    comply.
    """
    row: dict[str, object] = {
        "accepted_raw_id": "raw-1",
        "logical_source_key": "codex:native-1",
        "session_id": "codex-session:native-1",
    }
    with pytest.raises(ValueError, match="unreachable"):
        raw_reconciler_mod._item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            actuator=raw_reconciler_mod.RawAuthorityActuator.REFINE_QUARANTINE,
            row=row,
            reason="an actuator with a real apply() handler must only pair with an executable state",
        )
    # The dual is fine: an executable state may pair with a dispatched actuator.
    executable_item = raw_reconciler_mod._item(
        state=RawAuthorityFrontierState.SAFELY_REKEYABLE,
        actuator=raw_reconciler_mod.RawAuthorityActuator.REFINE_QUARANTINE,
        row=row,
        reason="eligible",
    )
    assert executable_item.executable is True
    # And a non-dispatched actuator (REQUEST_JUDGMENT, REACQUIRE, NONE) may
    # legitimately pair with a non-executable state -- those resolve
    # out-of-band (operator judgment, ordinary re-acquisition), not through
    # this apply dispatcher.
    judgment_item = raw_reconciler_mod._item(
        state=RawAuthorityFrontierState.CONFLICTING_AUTHORITY_NEEDS_JUDGMENT,
        actuator=raw_reconciler_mod.RawAuthorityActuator.REQUEST_JUDGMENT,
        row=row,
        reason="needs an operator disposition",
    )
    assert judgment_item.executable is False


def test_apply_dispatched_actuators_match_apply_branches() -> None:
    """polylogue-w32w: keep ``_APPLY_DISPATCHED_ACTUATORS`` from drifting out
    of sync with ``apply_raw_authority_frontier``'s actual dispatch
    branches. If a future actuator gets a real ``if item.actuator is
    RawAuthorityActuator.<X>:`` handler without also being added to
    ``_APPLY_DISPATCHED_ACTUATORS``, the new handler is silently exempt
    from the ``RawAuthorityFrontierItem.__post_init__`` reachability
    invariant -- exactly the kind of drift that let polylogue-u19l happen
    undetected. Parses the actual dispatch branches out of the module
    source rather than hand-duplicating the list, so this fails the moment
    the two go out of sync in either direction.
    """
    source = inspect.getsource(raw_reconciler_mod)
    member_names = re.findall(r"item\.actuator is RawAuthorityActuator\.(\w+)", source)
    dispatched_in_source = {raw_reconciler_mod.RawAuthorityActuator[member_name] for member_name in member_names}
    assert dispatched_in_source == raw_reconciler_mod._APPLY_DISPATCHED_ACTUATORS


def _census_row_counts(root: Path) -> tuple[int, int, int]:
    with sqlite3.connect(root / "source.db") as conn:
        return (
            int(conn.execute("SELECT COUNT(*) FROM raw_authority_censuses").fetchone()[0]),
            int(conn.execute("SELECT COUNT(*) FROM raw_authority_census_plans").fetchone()[0]),
            int(conn.execute("SELECT COUNT(*) FROM raw_authority_census_post_plans").fetchone()[0]),
        )


def test_census_plan_rows_are_bounded_by_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue-wkc6: every census re-records its ENTIRE pending plan set, so an
    archive whose plans do not drain grows without bound. On the live archive that
    reached 6,039 MB across two tables and seven indexes -- 89% of a durable tier
    whose actual evidence was 52 MB -- with 99.98% of rows carrying the
    ``carried_forward`` outcome, i.e. recording that nothing happened, and no DELETE
    against these tables existing anywhere in the tree.

    Retention must bound the per-plan rows at the point of growth. Deleting
    ``prune_raw_authority_census_history``'s call in ``record_raw_authority_census``
    makes this fail with row counts growing linearly in census count.
    """
    monkeypatch.setattr(raw_authority_mod, "RAW_AUTHORITY_CENSUS_PLAN_RETENTION", 3)
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="retain", source_path="retain.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]

    for index in range(9):
        record_raw_authority_census(
            tmp_path,
            (plan,),
            selected_plan_ids=set(),
            mode="dry_run",
            quiescent=True,
            scope={"test": f"retention-{index}"},
            residual={},
        )

    headers, plan_rows, post_plan_rows = _census_row_counts(tmp_path)

    assert headers == 9, "headers keep their own, much larger window"
    assert plan_rows == 3, f"plan rows must be bounded by retention, got {plan_rows}"
    assert post_plan_rows == 3


def test_census_retention_keeps_the_newest_censuses_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retention must drop the OLDEST censuses, never the newest: the per-census
    inspection pager (``WHERE census_id = ? ORDER BY ordinal``) is the only reader
    of these rows, and it is always pointed at recent history."""
    monkeypatch.setattr(raw_authority_mod, "RAW_AUTHORITY_CENSUS_PLAN_RETENTION", 2)
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="newest", source_path="newest.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]

    receipts = [
        record_raw_authority_census(
            tmp_path,
            (plan,),
            selected_plan_ids=set(),
            mode="dry_run",
            quiescent=True,
            scope={"test": f"newest-{index}"},
            residual={},
        )
        for index in range(5)
    ]

    with sqlite3.connect(tmp_path / "source.db") as conn:
        surviving = {str(row[0]) for row in conn.execute("SELECT DISTINCT census_id FROM raw_authority_census_plans")}

    assert surviving == {receipts[-1].census_id, receipts[-2].census_id}


def test_census_header_retention_preserves_predecessor_fk_and_prunes_safe_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production census route compacts lineage without breaking its FK.

    The production route below builds a retention boundary, keeps the newest
    retained child-to-parent edge, cuts the oldest retained header's
    predecessor to NULL, and proves that disconnected history is pruned while
    foreign-key enforcement remains enabled.
    """
    monkeypatch.setattr(raw_authority_mod, "RAW_AUTHORITY_CENSUS_PLAN_RETENTION", 2)
    monkeypatch.setattr(raw_authority_mod, "RAW_AUTHORITY_CENSUS_HEADER_RETENTION", 2)
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="predecessor", source_path="predecessor.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]

    receipts = [
        record_raw_authority_census(
            tmp_path,
            (plan,),
            selected_plan_ids=set(),
            mode="dry_run",
            quiescent=True,
            scope={"test": f"predecessor-{index}"},
            residual={},
        )
        for index in range(3)
    ]

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        child_predecessor = conn.execute(
            "SELECT predecessor_census_id FROM raw_authority_censuses WHERE census_id = ?",
            (receipts[2].census_id,),
        ).fetchone()[0]
        boundary_predecessor = conn.execute(
            "SELECT predecessor_census_id FROM raw_authority_censuses WHERE census_id = ?",
            (receipts[1].census_id,),
        ).fetchone()[0]
        retained_headers = {str(row[0]) for row in conn.execute("SELECT census_id FROM raw_authority_censuses")}
        assert child_predecessor == receipts[1].census_id
        assert boundary_predecessor is None
        assert retained_headers == {receipt.census_id for receipt in receipts[1:]}
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

    # A disconnected older header is safe to retire on the next production
    # prune cycle. It exercises the bounded-delete path without severing the
    # predecessor chain under test above.
    safe_root = tmp_path / "safe-prune"
    initialize_active_archive_root(safe_root)
    with sqlite3.connect(safe_root / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO raw_authority_censuses (
                census_id, sequence_no, scope_json, residual_json,
                parser_fingerprint, mode, lifecycle_status, quiescent,
                inventory_digest, residual_digest, plan_count,
                post_inventory_digest, post_residual_json, post_residual_digest,
                post_plan_count, postflight_at_ms, executable_plan_count,
                residual_plan_count, predecessor_census_id, fixed_point,
                created_at_ms, completed_at_ms
            ) VALUES (?, ?, '{}', '{}', 'test', 'dry_run', 'completed', 1,
                      ?, ?, 0, ?, '{}', ?, 0, 1, 0, 0, NULL, 0, 1, 1)
            """,
            (
                ("census:detached:1", 1, "a" * 64, "b" * 64, "a" * 64, "b" * 64),
                ("census:detached:2", 2, "c" * 64, "d" * 64, "c" * 64, "d" * 64),
            ),
        )
        conn.commit()

    later = record_raw_authority_census(
        safe_root,
        (),
        selected_plan_ids=set(),
        mode="dry_run",
        quiescent=True,
        scope={"test": "predecessor-safe-prune"},
        residual={},
    )

    with sqlite3.connect(safe_root / "source.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        surviving = {str(row[0]) for row in conn.execute("SELECT census_id FROM raw_authority_censuses")}
        assert "census:detached:1" not in surviving
        assert "census:detached:2" in surviving
        assert later.census_id in surviving
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_census_header_retention_bounds_a_long_contiguous_production_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A continuous census chain stays bounded at an explicit NULL boundary."""
    retention = 4
    monkeypatch.setattr(raw_authority_mod, "RAW_AUTHORITY_CENSUS_HEADER_RETENTION", retention)
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="long-chain", source_path="long-chain.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]
    census_count = retention + 12

    receipts = [
        record_raw_authority_census(
            tmp_path,
            (plan,),
            selected_plan_ids=set(),
            mode="dry_run",
            quiescent=True,
            scope={"test": f"long-chain-{index}"},
            residual={},
        )
        for index in range(census_count)
    ]

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        rows = conn.execute(
            """
            SELECT sequence_no, census_id, predecessor_census_id
            FROM raw_authority_censuses
            ORDER BY sequence_no
            """
        ).fetchall()
        assert len(rows) == retention
        assert [int(row[0]) for row in rows] == list(range(census_count - retention + 1, census_count + 1))
        assert rows[0][1] == receipts[census_count - retention].census_id
        assert rows[0][2] is None
        assert [row[2] for row in rows[1:]] == [rows[index][1] for index in range(retention - 1)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_census_plan_rows_prune_past_retention_even_with_an_unresolved_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-z7ko/f4z9: the obligation guard used to pin an ENTIRE
    census's plan-row snapshot for as long as ANY blocker on it stayed
    unresolved, however far outside the retention window it fell. Live
    evidence (source.db, 2026-07-31): ``RAW_AUTHORITY_CENSUS_PLAN_RETENTION``
    is 8, yet 41 of the most recent ~112 censuses (sequence 820-932) still
    held plan rows, because a structurally permanent obligation -- e.g. a
    quarantined raw with no logical source key to refine against -- never
    resolves, and continuous live ingestion keeps minting fresh ones, each
    anchoring whichever census first observed it. Result: 709,264 rows and
    ~870 MiB in a *durable* tier, growing ~100k rows/hour, with
    ``fixed_point`` never reaching 1 on any of 256 censuses ever run.

    The fix: a blocker's own ``expected_json``/``observed_json`` columns
    already carry a complete, self-contained snapshot of the plan and its
    evidence (``_reconcile_frontier_obligations``), and
    ``resolve_raw_authority_blocker`` reads that plus the census-independent
    ``raw_authority_plans`` table -- never
    ``raw_authority_census_plans``/``_post_plans``. So plan-row DETAIL no
    longer needs to survive for an unresolved blocker to stay resolvable;
    only the census HEADER does (a real FK: ``raw_authority_blockers``
    references ``raw_authority_censuses(census_id)``).

    Simulates the permanent-obligation shape directly: one durable,
    unresolved blocker planted on the very first of nine census passes.
    Proves plan-row detail for that first census prunes on the ordinary
    count-based window like any other, the blocker itself is never
    silently marked resolved, and its census HEADER survives (the FK it
    actually needs).
    """
    monkeypatch.setattr(raw_authority_mod, "RAW_AUTHORITY_CENSUS_PLAN_RETENTION", 2)
    initialize_active_archive_root(tmp_path)
    raw_id = _write_codex_raw(tmp_path, native_id="pinned", source_path="pinned.jsonl", acquired_at_ms=1)
    census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    plan = build_raw_replay_plans(tmp_path, ((raw_id,),))[0]

    first_receipt = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids=set(),
        mode="dry_run",
        quiescent=True,
        scope={"test": "pin-origin"},
        residual={},
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_authority_blockers (
                blocker_id, plan_id, census_id, reason, expected_json,
                observed_json, created_at_ms
            ) VALUES (?, ?, ?, ?, '{}', '{}', 1)
            """,
            (
                "raw-authority-blocker:permanently-pinned",
                plan.plan_id,
                first_receipt.census_id,
                "quarantined-raw refinement strategy proved this raw ineligible: a structurally permanent fact",
            ),
        )
        conn.commit()

    for index in range(8):
        record_raw_authority_census(
            tmp_path,
            (plan,),
            selected_plan_ids=set(),
            mode="dry_run",
            quiescent=True,
            scope={"test": f"pin-followup-{index}"},
            residual={},
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        surviving_plan_censuses = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT census_id FROM raw_authority_census_plans")
        }
        blocker_resolved_at = conn.execute(
            "SELECT resolved_at_ms FROM raw_authority_blockers WHERE blocker_id = ?",
            ("raw-authority-blocker:permanently-pinned",),
        ).fetchone()[0]
        header_row = conn.execute(
            "SELECT 1 FROM raw_authority_censuses WHERE census_id = ?", (first_receipt.census_id,)
        ).fetchone()

    # Never silently cleared -- this is the failure mode the old guard was
    # (over-)protecting against, and it must still not happen.
    assert blocker_resolved_at is None
    # Its census HEADER survives -- the real FK obligation.
    assert header_row is not None
    # But the plan-row DETAIL for that census prunes like everyone else's,
    # bounded to the retention window regardless of the still-open blocker.
    assert first_receipt.census_id not in surviving_plan_censuses
    assert len(surviving_plan_censuses) == 2
