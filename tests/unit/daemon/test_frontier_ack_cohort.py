"""Real frontier publication across staggered Source acknowledgements."""

from __future__ import annotations

import json
import traceback
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon import cli as daemon_cli
from polylogue.operations import raw_frontier_inspection
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
from polylogue.sources import revision_backfill
from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.storage import frontier_inspection as inspection_substrate
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.frontier_inspection import (
    declared_frontier_triggers,
    read_frontier_inspection_mark,
    read_frontier_journal_state,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
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


def _snapshot(root: Path, index_path: Path | None = None) -> dict[str, Any]:
    index_path = resolve_active_index_path(root) if index_path is None else index_path
    with (
        closing(open_readonly_connection(root / "source.db")) as source,
        closing(open_readonly_connection(index_path)) as index,
        closing(open_readonly_connection(root / "ops.db")) as ops,
    ):
        state = read_frontier_journal_state(
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
        mark = read_frontier_inspection_mark(ops)
        return {
            "frame": asdict(state),
            "mark": None if mark is None else asdict(mark),
            "memberships": [
                dict(zip(("raw_id", "decision", "authority", "census", "members", "parser"), row, strict=True))
                for row in source.execute(
                    "SELECT m.raw_id,m.decision,m.revision_authority,c.status,c.member_count,p.status "
                    "FROM raw_session_memberships m JOIN raw_membership_census c USING(raw_id) "
                    "JOIN raw_authority_parser_census p USING(raw_id) ORDER BY m.raw_id"
                )
            ],
            "sessions": index.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "messages": index.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "heads": index.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone()[0],
            "debt": [
                dict(zip(("status", "attempts", "deadline", "error"), row, strict=True))
                for row in ops.execute(
                    "SELECT status,attempts,next_retry_at,last_error FROM convergence_debt "
                    "WHERE stage='raw_frontier_inspection' ORDER BY target_id"
                )
            ],
        }


@pytest.mark.frozen_clock_modules(
    "polylogue.daemon.cli",
    "polylogue.sources.live.cursor",
    "polylogue.sources.live.convergence_debt_retry",
)
def test_partial_acknowledgements_publish_current_frontier_and_suppress_unchanged_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: Any
) -> None:
    """Partial progress publishes current evidence; unchanged waits do no work.

    Source permits, replay, promotion, currency, inspection and debt execution
    remain real. Only the stage catalog is narrowed to the concern under test.
    """
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    report: dict[str, Any] = {"cohort_size": 6, "phases": [], "inspection_bodies": [], "replay_bodies": []}
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for number in range(6):
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=json.dumps([_chatgpt_session(f"cohort-{number}", f"neutral-{number}")]).encode(),
                    source_path=f"neutral-{number}.json",
                    canonical_source_path=f"neutral-{number}.json",
                    acquired_at_ms=number + 1,
                )
            )

    def census_only(compute: Any) -> None:
        adapter = make_raw_observation_derivation(root, compute_adapter=compute)
        for raw_id in raw_ids:
            prepared = adapter.compute(raw_observation_frame(root, raw_ids=(raw_id,)), raw_id, replay_current=True)
            try:
                assert prepared.prepared_membership_plans
            finally:
                prepared.close()

    run_on_convergence_owner(root, "fixture.cohort.census", census_only)
    source_before = (root / "source.db").read_bytes()
    report["phases"].append({"phase": "complete_census", **_snapshot(root)})
    real_apply = revision_backfill.apply_prepared_revision_replay

    def observed_apply(*args: Any, **kwargs: Any) -> Any:
        seal = kwargs["reference_seal"]
        body = {
            "t": frozen_clock.time() - frozen_clock.start,
            "destination": None if seal.destination is None else seal.destination.kind,
            "prepared_decisions": {
                key: {raw: decision.value for raw, decision in plan.source_decisions.items()}
                for key, plan in kwargs["prepared_membership_plans"].items()
                if plan.source_decisions is not None
            },
        }
        result = real_apply(*args, **kwargs)
        body["result"] = asdict(result)
        report["replay_bodies"].append(body)
        return result

    monkeypatch.setattr(revision_backfill, "apply_prepared_revision_replay", observed_apply)
    with write_lease("fixture.cohort.begin", archive_root=root):
        cold = ColdBuildGeneration.begin(
            root,
            reason="neutral-cohort",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent"),)),
        )
    register_cold_build_generation(cold)
    try:
        inactive = replay_retained_raws(root, tuple(raw_ids))
        assert sum(row.replayed_logical_sources for row in inactive) == 6
        assert (root / "source.db").read_bytes() == source_before
        report["phases"].append({"phase": "inactive_replay", **_snapshot(root, Path(cold.generation.index_path))})
        with write_lease("fixture.cohort.readiness", archive_root=root):
            cold.prepare_promotion_candidate()
        with cold.prepare_promotion_proof() as proof:
            with write_lease("fixture.cohort.promote", archive_root=root):
                cold.promote_prepared(proof)
        clear_cold_build_generation()
        assert (root / "source.db").read_bytes() == source_before
        promoted = _snapshot(root)
        assert promoted["heads"] == promoted["sessions"] == promoted["messages"] == 6
        assert all(row["decision"] is None for row in promoted["memberships"])
        assert all(row["census"] == row["parser"] == "complete" for row in promoted["memberships"])
        report["phases"].append({"phase": "promotion", **promoted})

        real_inspect = inspection_substrate.inspect_prepared_raw_authority_frontier
        real_mode = inspection_substrate.frontier_inspection_mode
        real_proofs = inspection_substrate._iter_selected_frontier_proofs
        current_attempt: dict[str, Any] | None = None

        def observed_mode(*args: Any, **kwargs: Any) -> Any:
            mode = real_mode(*args, **kwargs)
            if current_attempt is not None:
                current_attempt["selected_mode"] = mode
            return mode

        def observed_proofs(*args: Any, **kwargs: Any) -> Any:
            for item, chain in real_proofs(*args, **kwargs):
                if current_attempt is not None:
                    current_attempt["actual_head_proofs"] += 1
                    current_attempt["proof_states"].append(item.state.value)
                yield item, chain

        def observed_inspect(*args: Any, **kwargs: Any) -> Any:
            nonlocal current_attempt
            body: dict[str, Any] = {
                "t": frozen_clock.time() - frozen_clock.start,
                "actual_head_proofs": 0,
                "proof_states": [],
            }
            report["inspection_bodies"].append(body)
            current_attempt = body
            try:
                outcome = real_inspect(*args, **kwargs)
                body.update(asdict(outcome))
                body["settlement"] = "published"
                return outcome
            except BaseException as failure:
                body.update(settlement="refused", error_type=type(failure).__name__, error=str(failure))
                body["traceback"] = traceback.format_exc()
                raise
            finally:
                current_attempt = None

        monkeypatch.setattr(raw_frontier_inspection, "inspect_prepared_raw_authority_frontier", observed_inspect)
        monkeypatch.setattr(inspection_substrate, "frontier_inspection_mode", observed_mode)
        monkeypatch.setattr(inspection_substrate, "_iter_selected_frontier_proofs", observed_proofs)
        monkeypatch.setattr(
            "polylogue.daemon.convergence_stages.make_default_convergence_stages",
            lambda db, *, compute_adapter: (
                raw_frontier_inspection.make_raw_frontier_inspection_stage(db, compute_adapter=compute_adapter),
            ),
        )

        def drain(phase: str) -> dict[str, Any]:
            before = len(report["inspection_bodies"])
            run_on_convergence_owner(
                root,
                "fixture.cohort.debt",
                lambda compute: daemon_cli._drain_convergence_debt_and_archive(
                    root / "index.db", compute_adapter=compute
                ),
            )
            snapshot = {
                "phase": phase,
                "t": frozen_clock.time() - frozen_clock.start,
                "new_inspection_bodies": len(report["inspection_bodies"]) - before,
                "coverage": raw_frontier_inspection.frontier_coverage_for_archive(root),
                **_snapshot(root),
            }
            # Findings are already recorded on the canonical mark.
            snapshot["coverage"].pop("findings", None)
            report["phases"].append(snapshot)
            return snapshot

        blocked = drain("first_blocked")
        assert blocked["coverage"]["current"] and not blocked["coverage"]["healthy"]
        deadline = blocked["debt"][0]["deadline"]
        assert datetime.fromisoformat(deadline).timestamp() == frozen_clock.time() + 60
        unchanged = drain("unchanged_t0")
        assert unchanged["new_inspection_bodies"] == 0 and unchanged["debt"] == blocked["debt"]
        for start in (0, 2, 4):
            frozen_clock.advance(10)
            receipts = replay_retained_raws(root, tuple(raw_ids[start : start + 2]))
            assert sum(row.replayed_logical_sources for row in receipts) == 2
            before = _snapshot(root)
            assert sum(row["decision"] == "applied" for row in before["memberships"]) == start + 2
            assert not raw_frontier_inspection.frontier_coverage_for_archive(root)["current"]
            report["phases"].append(
                {"phase": "active_acknowledgement", "t": frozen_clock.time() - frozen_clock.start, **before}
            )
            measured = drain(f"after_pair_{start // 2 + 1}")
            assert measured["coverage"]["current"]
            assert measured["coverage"]["healthy"] is (start == 4)
            assert report["inspection_bodies"][-1]["settlement"] == "published"
            unchanged = drain(f"unchanged_pair_{start // 2 + 1}")
            assert unchanged["new_inspection_bodies"] == 0 and unchanged["debt"] == measured["debt"]

        final = _snapshot(root)
        assert final["sessions"] == final["messages"] == final["heads"] == 6
        assert all(row["decision"] == "applied" and row["authority"] == "byte_proven" for row in final["memberships"])
        assert raw_frontier_inspection.frontier_coverage_for_archive(root)["healthy"]
        assert not final["debt"]
    finally:
        clear_cold_build_generation()
        if not cold.promoted:
            with write_lease("fixture.cohort.discard", archive_root=root):
                cold.discard()
