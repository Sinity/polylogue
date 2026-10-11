"""Separate frozen Source census from the active replay acknowledgement."""

from __future__ import annotations

import json
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.enums import Provider
from polylogue.operations.raw_frontier_inspection import frontier_coverage_for_archive
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
from polylogue.sources import revision_backfill
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.frontier_inspection import (
    declared_frontier_triggers,
    inspect_prepared_raw_authority_frontier,
    read_frontier_inspection_mark,
    read_frontier_journal_state,
)
from polylogue.storage.raw_reconciler import (
    _FRONTIER_INDEX_INPUT_SQL,
    _FRONTIER_SOURCE_INPUT_SQL,
    _classify_frontier_row,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.prepared_replay import run_on_convergence_owner
from tests.infra.raw_owner_routes import replay_retained_raws
from tests.infra.retained_parser_payloads import _chatgpt_session


def _snapshot(root: Path, index_path: Path, raw_id: str) -> dict[str, Any]:
    """Read committed predicates and call the inspection's production classifier."""
    import sqlite3

    with (
        closing(open_readonly_connection(root / "source.db")) as source,
        closing(open_readonly_connection(index_path)) as index,
        closing(open_readonly_connection(root / "ops.db")) as ops,
    ):
        source.row_factory = index.row_factory = sqlite3.Row
        journal = read_frontier_journal_state(
            source=source,
            index=index,
            ops=ops,
            source_path=root / "source.db",
            index_path=index_path,
            ops_path=root / "ops.db",
            source_triggers=declared_frontier_triggers(ARCHIVE_DDL_BY_TIER[ArchiveTier.SOURCE]),
            index_triggers=declared_frontier_triggers(INDEX_DDL),
            cursor_triggers=declared_frontier_triggers(OPS_DDL),
        )
        membership = dict(
            source.execute(
                "SELECT m.logical_source_key,m.decision,m.revision_authority,c.status,c.member_count,"
                "c.parser_fingerprint FROM raw_session_memberships m JOIN raw_membership_census c USING(raw_id) "
                "WHERE m.raw_id=?",
                (raw_id,),
            ).fetchone()
        )
        parser = dict(
            source.execute(
                "SELECT status,parser_fingerprint,logical_keys_json FROM raw_authority_parser_census WHERE raw_id=?",
                (raw_id,),
            ).fetchone()
        )
        head = index.execute(_FRONTIER_INDEX_INPUT_SQL, (membership["logical_source_key"],)).fetchone()
        frontier = None
        if head is not None:
            row = dict(head)
            row.update(
                dict(
                    source.execute(
                        _FRONTIER_SOURCE_INPUT_SQL,
                        (membership["logical_source_key"], membership["logical_source_key"], raw_id),
                    ).fetchone()
                )
            )
            verified = BlobStore(root / "blob").verify(str(row["blob_hash"]).lower())
            assert verified
            item = _classify_frontier_row(row, verified_bytes=verified)
            frontier = item.to_dict()
        mark = read_frontier_inspection_mark(ops)
        return {
            "membership": membership,
            "parser": parser,
            "sessions": index.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "frontier": frontier,
            "journal": asdict(journal),
            "mark": None if mark is None else asdict(mark),
        }


def test_source_decision_waits_for_active_replay_after_owned_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: Any,
) -> None:
    """Equal retained bytes become proven only after the actual Source commit.

    Calling the Source publisher during inactive replay, treating a complete
    census as applied authority, or skipping active acknowledgement turns red.
    """
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    payload = json.dumps([_chatgpt_session("phase-session", "neutral")]).encode()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=payload,
            source_path="neutral.json",
            canonical_source_path="neutral.json",
            acquired_at_ms=1,
        )

    # Canonical compute commits its required Source census in place. Closing
    # the resulting preparation leaves its Index publication unconsumed.
    def census_only(compute: Any) -> None:
        adapter = make_raw_observation_derivation(root, compute_adapter=compute)
        prepared = adapter.compute(raw_observation_frame(root, raw_ids=(raw_id,)), raw_id, replay_current=True)
        try:
            assert prepared.prepared_membership_plans
        finally:
            prepared.close()

    run_on_convergence_owner(root, "fixture.phase.census", census_only)
    initial = _snapshot(root, root / "index.db", raw_id)
    source_before = (root / "source.db").read_bytes()
    with write_lease("fixture.phase.begin", archive_root=root):
        cold = ColdBuildGeneration.begin(
            root,
            reason="neutral-phase",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent"),)),
        )
    register_cold_build_generation(cold)
    executions: list[dict[str, Any]] = []
    publications: list[dict[str, Any]] = []
    real_apply = revision_backfill.apply_prepared_revision_replay
    real_publish = revision_governance.publish_prepared_revision_source
    publishing_replay = False

    def observed_apply(*args: Any, **kwargs: Any) -> Any:
        nonlocal publishing_replay
        seal = kwargs["reference_seal"]
        executions.append(
            {
                "destination": None if seal.destination is None else seal.destination.kind,
                "prepared_decisions": {
                    key: {raw: decision.value for raw, decision in plan.source_decisions.items()}
                    for key, plan in kwargs["prepared_membership_plans"].items()
                    if plan.source_decisions is not None
                },
            }
        )
        publishing_replay = True
        try:
            return real_apply(*args, **kwargs)
        finally:
            publishing_replay = False

    def observed_publish(seal: Any, permit: Any) -> None:
        if not publishing_replay:
            real_publish(seal, permit)
            return
        publications.append({"before": _snapshot(root, root / "index.db", raw_id)})
        real_publish(seal, permit)
        publications[-1]["after"] = _snapshot(root, root / "index.db", raw_id)

    monkeypatch.setattr(revision_backfill, "apply_prepared_revision_replay", observed_apply)
    monkeypatch.setattr(revision_governance, "publish_prepared_revision_source", observed_publish)
    try:
        inactive_receipts = replay_retained_raws(root, (raw_id,))
        assert sum(row.replayed_logical_sources for row in inactive_receipts) == 1
        inactive = _snapshot(root, Path(cold.generation.index_path), raw_id)
        assert not publications
        assert (root / "source.db").read_bytes() == source_before
        with write_lease("fixture.phase.readiness", archive_root=root):
            cold.prepare_promotion_candidate()
        ready = _snapshot(root, Path(cold.generation.index_path), raw_id)
        assert (root / "source.db").read_bytes() == source_before
        with cold.prepare_promotion_proof() as proof:
            with write_lease("fixture.phase.promote", archive_root=root):
                promoted = cold.promote_prepared(proof)
        clear_cold_build_generation()
        after_promotion = _snapshot(root, Path(promoted.index_path), raw_id)
        assert (root / "source.db").read_bytes() == source_before

        def inspect(compute: Any) -> Any:
            return inspect_prepared_raw_authority_frontier(
                root,
                input_demand=compute.amend_current_input_demand,
            )

        blocked_inspection = run_on_convergence_owner(root, "fixture.phase.frontier.blocked", inspect)
        blocked_coverage = frontier_coverage_for_archive(root)
        blocked = _snapshot(root, Path(promoted.index_path), raw_id)
        assert not blocked_inspection.healthy and blocked_inspection.blocking_head_checks == 1
        assert blocked_coverage["available"] and blocked_coverage["current"] and not blocked_coverage["healthy"]
        active_receipts = replay_retained_raws(root, (raw_id,))
        assert sum(row.replayed_logical_sources for row in active_receipts) == 1
        active = _snapshot(root, Path(promoted.index_path), raw_id)
        assert not frontier_coverage_for_archive(root)["current"]
        healthy_inspection = run_on_convergence_owner(root, "fixture.phase.frontier.healthy", inspect)
        healthy_coverage = frontier_coverage_for_archive(root)
        healthy = _snapshot(root, Path(promoted.index_path), raw_id)
        assert healthy_inspection.healthy and healthy_inspection.blocking_head_checks == 0
        assert healthy_coverage["current"] and healthy_coverage["healthy"]

        for phase in (initial, inactive, ready, after_promotion, active):
            assert phase["membership"]["status"] == phase["parser"]["status"] == "complete"
            assert phase["membership"]["member_count"] == 1
        assert initial["sessions"] == 0
        for phase in (inactive, ready, after_promotion):
            assert phase["sessions"] == 1
            assert phase["membership"]["decision"] is None
            assert phase["frontier"]["state"] == "unresolved_provenance"
        assert active["membership"]["decision"] == "applied"
        assert active["membership"]["revision_authority"] == "byte_proven"
        assert active["frontier"]["state"] == "proven_current"
        assert len(executions) == 2 and executions[0]["destination"] == "owned_inactive"
        assert executions[0]["prepared_decisions"] == executions[1]["prepared_decisions"]
        assert len(publications) == 1
        assert publications[0]["before"]["frontier"]["state"] == "unresolved_provenance"
        assert publications[0]["after"]["frontier"]["state"] == "proven_current"
        assert publications[0]["after"]["journal"]["source_high"] > publications[0]["before"]["journal"]["source_high"]
        assert blocked["mark"]["state"] == "blocked"
        assert healthy["mark"]["state"] == "healthy"
    finally:
        clear_cold_build_generation()
        if not cold.promoted:
            with write_lease("fixture.phase.discard", archive_root=root):
                cold.discard()
