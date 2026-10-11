"""Real frontier publication across staggered Source acknowledgements."""

from __future__ import annotations

import hashlib
import json
import traceback
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Any, cast

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.core.stage_admission import admit_stage_write
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
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection, owned_daemon_connection
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.prepared_replay import run_on_convergence_owner
from tests.infra.raw_owner_routes import replay_retained_raws
from tests.infra.retained_parser_payloads import _CLAUDE_USER_RECORD, _chatgpt_session


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
            "pending": [
                tuple(row)
                for row in ops.execute("SELECT * FROM raw_frontier_pending_heads ORDER BY logical_source_key")
            ],
            "pending_raws": list(ops.execute("SELECT * FROM raw_frontier_pending_raws ORDER BY raw_id")),
            "membership_inputs": list(
                ops.execute("SELECT * FROM raw_frontier_membership_inputs ORDER BY raw_id,logical_source_key")
            ),
            "checkpoint_bytes": ops.execute(
                "SELECT coalesce(sum(length(inputs_json)+length(contribution_json)),0) FROM raw_frontier_pending_heads"
            ).fetchone()[0]
            + ops.execute(
                "SELECT coalesce(sum(length(structural_digest)),0) FROM raw_frontier_pending_raws"
            ).fetchone()[0]
            + ops.execute(
                "SELECT coalesce(sum(length(disposition_json)),0) FROM raw_frontier_membership_inputs"
            ).fetchone()[0],
            "debt": [
                dict(zip(("status", "attempts", "deadline", "error"), row, strict=True))
                for row in ops.execute(
                    "SELECT status,attempts,next_retry_at,last_error FROM convergence_debt "
                    "WHERE stage='raw_frontier_inspection' ORDER BY target_id"
                )
            ],
        }


@pytest.mark.parametrize("cohort_size", [3, 6])
@pytest.mark.frozen_clock_modules(
    "polylogue.daemon.cli",
    "polylogue.sources.live.cursor",
    "polylogue.sources.live.convergence_debt_retry",
)
def test_partial_acknowledgements_publish_current_frontier_and_suppress_unchanged_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: Any, cohort_size: int, pytestconfig: pytest.Config
) -> None:
    """Partial progress publishes current evidence; unchanged waits do no work.

    Source permits, replay, promotion, currency, inspection and debt execution
    remain real. Only the stage catalog is narrowed to the concern under test.
    """
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    report: dict[str, Any] = {"cohort_size": cohort_size, "phases": [], "inspection_bodies": [], "replay_bodies": []}
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for number in range(cohort_size):
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=json.dumps(
                        [
                            _chatgpt_session(f"cohort-{number}", f"neutral-{number}"),
                            *(
                                [
                                    _chatgpt_session(f"shared-{member}", f"shared-neutral-{member}")
                                    for member in range(cohort_size - 1)
                                ]
                                if number == 0
                                else []
                            ),
                        ]
                    ).encode(),
                    source_path=f"neutral-{number}.json",
                    canonical_source_path=f"neutral-{number}.json",
                    acquired_at_ms=number + 1,
                )
            )

        byte_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=_CLAUDE_USER_RECORD,
            source_path="byte.jsonl",
            canonical_source_path="byte.jsonl",
            acquired_at_ms=20,
            native_id="strict-replay",
            revision=RawRevisionEnvelope(
                "claude-code-session:strict-replay",
                RawRevisionKind.FULL,
                hashlib.sha256(_CLAUDE_USER_RECORD).hexdigest(),
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
    all_raws = (*raw_ids, byte_id)

    def census_only(compute: Any) -> None:
        adapter = make_raw_observation_derivation(root, compute_adapter=compute)
        for raw_id in all_raws:
            prepared = adapter.compute(raw_observation_frame(root, raw_ids=(raw_id,)), raw_id, replay_current=True)
            try:
                if raw_id != byte_id:
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
        inactive = replay_retained_raws(root, all_raws)
        assert sum(row.replayed_logical_sources for row in inactive) == 2 * cohort_size
        assert (root / "source.db").read_bytes() == source_before
        report["phases"].append({"phase": "inactive_replay", **_snapshot(root, Path(cold.generation.index_path))})
        with write_lease("fixture.cohort.readiness", archive_root=root):
            cold.prepare_promotion_candidate()
        with cold.prepare_promotion_proof() as proof:
            with write_lease("fixture.cohort.promote", archive_root=root):
                cold.promote_prepared(proof)
        clear_cold_build_generation()
        assert (root / "source.db").read_bytes() == source_before
        with write_lease("fixture.cohort.cursors", archive_root=root):
            with owned_daemon_connection(root / "ops.db", archive_root=root) as ops, ops:
                ops.executemany(
                    "INSERT INTO ingest_cursor(source_path,canonical_source_path,byte_offset,updated_at_ms) VALUES(?,?,0,1)",
                    [(f"neutral-{number}.json", f"neutral-{number}.json") for number in range(cohort_size)]
                    + [("byte.jsonl", "byte.jsonl")],
                )
                ops.execute(
                    "INSERT INTO ingest_cursor(source_path,byte_offset,deferred_end_offset,updated_at_ms) VALUES('deferred.jsonl',0,10,1)"
                )
        promoted = _snapshot(root)
        assert promoted["heads"] == promoted["sessions"] == promoted["messages"] == 2 * cohort_size
        assert all(row["decision"] is None for row in promoted["memberships"])
        assert all(row["census"] == row["parser"] == "complete" for row in promoted["memberships"])
        report["phases"].append({"phase": "promotion", **promoted})

        real_inspect = inspection_substrate.inspect_prepared_raw_authority_frontier
        real_mode = inspection_substrate.frontier_inspection_mode
        real_proofs = inspection_substrate._iter_selected_frontier_proofs
        real_checkpoint = inspection_substrate._raw_checkpoint
        real_checkpoint_rows = inspection_substrate._source_checkpoint_rows
        real_publish_checkpoint = inspection_substrate._publish_frontier_checkpoint
        real_terminal_paths = inspection_substrate.PreparedFrontierSource.terminal_artifact_paths

        def observed_terminal_paths(source: Any, paths: set[str]) -> Any:
            if current_attempt is not None:
                current_attempt["terminal_proof_paths"].extend(sorted(paths))
            return real_terminal_paths(source, paths)

        class ObservedCheckpointWriter:
            """Forward real SQL and count its actual mutations and bound bytes."""

            def __init__(self, writer: Any) -> None:
                self.writer = writer

            def record(self, sql: str, page: tuple[tuple[Any, ...], ...], cursor: Any) -> None:
                assert current_attempt is not None
                current_attempt["checkpoint_mutated_rows"] += cursor.rowcount
                current_attempt["checkpoint_bound_bytes"] += sum(
                    len(cell) if isinstance(cell, bytes) else len(str(cell).encode()) for row in page for cell in row
                )
                if "raw_frontier_pending_heads" in sql and cursor.rowcount:
                    current_attempt["checkpoint_head_keys"].extend(str(row[0]) for row in page)
                current_attempt["checkpoint_sql"].append(
                    {"sql": sql, "rows": cursor.rowcount, "parameter_rows": len(page)}
                )

            def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
                cursor = self.writer.execute(sql, parameters)
                self.record(sql, (parameters,) if parameters else (), cursor)
                return cursor

            def executemany(self, sql: str, page: Any) -> Any:
                retained = tuple(page)
                cursor = self.writer.executemany(sql, retained)
                self.record(sql, retained, cursor)
                return cursor

        def observed_publish_checkpoint(writer: Any, selection: Any, *, delta: bool) -> None:
            assert current_attempt is not None
            current_attempt["checkpoint_delta"] = delta
            real_publish_checkpoint(cast(Connection, ObservedCheckpointWriter(writer)), selection, delta=delta)

        def observed_checkpoint(*args: Any, **kwargs: Any) -> Any:
            if current_attempt is not None:
                current_attempt["raw_snapshots"] += 1
            return real_checkpoint(*args, **kwargs)

        def observed_checkpoint_rows(*args: Any, **kwargs: Any) -> Any:
            for row in real_checkpoint_rows(*args, **kwargs):
                if current_attempt is not None:
                    current_attempt["snapshot_source_rows"] += 1
                    current_attempt["snapshot_source_bytes"] += len(json.dumps(row, sort_keys=True))
                yield row

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
                "raw_snapshots": 0,
                "snapshot_source_rows": 0,
                "snapshot_source_bytes": 0,
                "proof_states": [],
                "checkpoint_mutated_rows": 0,
                "checkpoint_bound_bytes": 0,
                "checkpoint_head_keys": [],
                "checkpoint_sql": [],
                "terminal_proof_paths": [],
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
        monkeypatch.setattr(inspection_substrate, "_raw_checkpoint", observed_checkpoint)
        monkeypatch.setattr(inspection_substrate, "_source_checkpoint_rows", observed_checkpoint_rows)
        monkeypatch.setattr(inspection_substrate, "_publish_frontier_checkpoint", observed_publish_checkpoint)
        monkeypatch.setattr(
            inspection_substrate.PreparedFrontierSource, "terminal_artifact_paths", observed_terminal_paths
        )
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
        assert len(blocked["pending_raws"]) == cohort_size
        assert len(blocked["membership_inputs"]) == 2 * cohort_size - 1
        assert report["inspection_bodies"][-1]["raw_snapshots"] == cohort_size
        deadline = blocked["debt"][0]["deadline"]
        assert datetime.fromisoformat(deadline).timestamp() == frozen_clock.time() + 60
        unchanged = drain("unchanged_t0")
        assert unchanged["new_inspection_bodies"] == 0 and unchanged["debt"] == blocked["debt"]

        def acknowledge_member(key: str) -> None:
            # Prepare the actual membership classification against the current
            # retained bytes and accepted head, then publish exactly one member
            # through its canonical Source producer and original permit.
            def body(compute: Any) -> None:
                adapter = make_raw_observation_derivation(root, compute_adapter=compute)
                prepared = adapter.compute(
                    raw_observation_frame(root, raw_ids=(raw_ids[0],)), raw_ids[0], replay_current=True
                )
                try:
                    assert prepared.prepared_membership_plans is not None
                    plan = prepared.prepared_membership_plans[key]
                    assert plan.head_plan is not None
                    decisions = revision_governance.membership_decisions_for_head_plan(
                        plan.classification, plan.head_plan, suppressed=False
                    )
                    with PreparedIndexMutation(resolve_active_index_path(root), archive_root=root) as seal:
                        with seal.original_read_snapshot(), seal.source_producer():
                            revision_governance.prepare_membership_classification_source(
                                seal,
                                key,
                                plan.classification,
                                decisions=decisions,
                                decided_at_ms=int(frozen_clock.time() * 1000),
                                projections=plan.projections,
                                head_plan=plan.head_plan,
                            )
                        permit = seal.prepare_source_mutation()
                        admit_stage_write(
                            "fixture.cohort.member-ack",
                            lambda: revision_governance.publish_prepared_revision_source(seal, permit),
                        )
                finally:
                    prepared.close()

            run_on_convergence_owner(root, "fixture.cohort.prepare-member-ack", body)

        with closing(open_readonly_connection(root / "source.db")) as source:
            shared_keys = tuple(
                row[0]
                for row in source.execute(
                    "SELECT logical_source_key FROM raw_session_memberships WHERE raw_id=? ORDER BY logical_source_key",
                    (raw_ids[0],),
                )
            )
        assert len(shared_keys) == cohort_size
        steps = [(raw_ids[0], key) for key in shared_keys] + [(raw, None) for raw in raw_ids[1:]]
        for start, (raw, key) in enumerate(steps):
            frozen_clock.advance(10)
            if key is not None:
                acknowledge_member(key)
            else:
                receipts = replay_retained_raws(root, (raw,))
                assert sum(row.replayed_logical_sources for row in receipts) == 1
            before = _snapshot(root)
            assert sum(row["decision"] == "applied" for row in before["memberships"]) == start + 1
            assert not raw_frontier_inspection.frontier_coverage_for_archive(root)["current"]
            report["phases"].append(
                {"phase": "active_acknowledgement", "t": frozen_clock.time() - frozen_clock.start, **before}
            )
            if start == 0:
                prior = _snapshot(root)
                actual_publish = revision_governance.publish_prepared_revision_source

                def refuse_source(*args: Any, **kwargs: Any) -> None:
                    raise ReferenceSealStaleError("injected refusal before actual Source acceptance")

                monkeypatch.setattr(revision_governance, "publish_prepared_revision_source", refuse_source)
                with pytest.raises(ReferenceSealStaleError):
                    run_on_convergence_owner(
                        root,
                        "fixture.cohort.refusal",
                        lambda compute: observed_inspect(root, input_demand=compute.amend_current_input_demand),
                    )
                assert report["inspection_bodies"][-1]["checkpoint_mutated_rows"] == 0
                refused = _snapshot(root)
                for field in ("mark", "pending", "pending_raws", "membership_inputs"):
                    assert refused[field] == prior[field]
                monkeypatch.setattr(revision_governance, "publish_prepared_revision_source", actual_publish)
            measured = drain(f"after_ack_{start + 1}")
            assert measured["coverage"]["current"]
            assert measured["coverage"]["healthy"] is (start == len(steps) - 1)
            body = report["inspection_bodies"][-1]
            assert body["settlement"] == "published" and body["mode"] == "delta"
            assert body["actual_head_proofs"] == (cohort_size if key is not None else 1)
            assert body["raw_snapshots"] <= 2
            assert body["checkpoint_delta"]
            assert body["terminal_proof_paths"] == ["deferred.jsonl"]
            assert len(body["checkpoint_head_keys"]) == 1
            if key is not None:
                assert body["checkpoint_head_keys"] == [key]
                assert body["checkpoint_mutated_rows"] == (cohort_size + 2 if start == cohort_size - 1 else 2)
            else:
                assert body["checkpoint_mutated_rows"] == 3
                assert body["checkpoint_bound_bytes"] < 256
            assert body["blocking_head_checks"] == len(steps) - start - 1
            assert body["eligible_cursor_checks"] == cohort_size + 2
            assert body["comparable_cursor_checks"] == 1
            assert len(measured["pending"]) == len(steps) - start - 1
            unchanged = drain(f"unchanged_ack_{start + 1}")
            assert unchanged["new_inspection_bodies"] == 0 and unchanged["debt"] == measured["debt"]
            if start == 0:
                frozen_clock.advance(1)
                with ArchiveStore.open_existing(root, read_only=False) as archive:
                    archive.mark_raw_parse_succeeded(raw_ids[-1], provider=Provider.CHATGPT)
                unsupported = run_on_convergence_owner(
                    root,
                    "fixture.cohort.unsupported",
                    lambda compute: observed_inspect(root, input_demand=compute.amend_current_input_demand),
                )
                report["inspection_bodies"][-1]["purpose"] = "unsupported_non_ack"
                assert unsupported.mode == "full" and unsupported.accepted_head_checks == 2 * cohort_size
                assert len(_snapshot(root)["pending"]) == 2 * cohort_size - 2

        final = _snapshot(root)
        assert final["sessions"] == final["messages"] == final["heads"] == 2 * cohort_size
        assert all(row["decision"] == "applied" and row["authority"] == "byte_proven" for row in final["memberships"])
        assert raw_frontier_inspection.frontier_coverage_for_archive(root)["healthy"]
        assert not final["debt"] and not final["pending"]
        assert report["inspection_bodies"][0]["actual_head_proofs"] == 2 * cohort_size
        assert (
            sum(
                body["actual_head_proofs"]
                for body in report["inspection_bodies"]
                if body["settlement"] == "published" and "purpose" not in body
            )
            == cohort_size * cohort_size + 3 * cohort_size - 1
        )
        diagnostic = json.dumps(report, sort_keys=True, indent=2, default=bytes.hex)
        (tmp_path / "proof-counts.json").write_text(diagnostic)
        # Successful managed runs dispose their fixture tree. Keep neutral
        # measurements beside their declared retained report, when supplied.
        retained_report = pytestconfig.getoption("--polylogue-report-file", default=None)
        if retained_report:
            Path(retained_report).with_name(f"frontier-ack-cohort-{cohort_size}.json").write_text(diagnostic)

    finally:
        clear_cold_build_generation()
        if not cold.promoted:
            with write_lease("fixture.cohort.discard", archive_root=root):
                cold.discard()
