"""The raw-authority census ledger is gone from fresh archives.

Four source-tier tables and two indexes recorded one row per pending replay
plan on every inspection pass. polylogue-gen6d rejected that shape and the
2026-09-15 ruling on polylogue-6kur retires it, so a fresh source generation
must not declare them and the surviving frontier inspection must keep
publishing the durable obligations readiness still depends on.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.archive_readiness import (
    raw_materialization_readiness_snapshot,
    raw_materialization_ready,
)
from polylogue.storage.frontier_inspection import (
    inspect_prepared_raw_authority_frontier,
    prepared_frontier_blocker_acknowledgement,
)
from polylogue.storage.raw_reconciler import RawAuthorityFrontierState
from polylogue.storage.sqlite.archive_tiers.schema_inventory import canonical_schema_objects
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner

#: Every object the census ledger declared.
RETIRED_CENSUS_OBJECTS = (
    "table:raw_authority_censuses",
    "table:raw_authority_plans",
    "table:raw_authority_census_plans",
    "table:raw_authority_census_post_plans",
    "index:idx_raw_authority_census_plans_status",
    "index:idx_raw_authority_census_plans_attempts",
)

#: Durable raw-authority relations the ruling explicitly retains. They are the
#: anti-vacuity partner of every assertion below: a change that deleted the
#: whole raw-authority family rather than the census ledger fails here.
RETAINED_AUTHORITY_TABLES = (
    "raw_authority_blockers",
    "raw_authority_parser_census",
    "raw_authority_verdicts",
    "raw_membership_census",
)


@pytest.mark.parametrize("object_ref", RETIRED_CENSUS_OBJECTS)
def test_fresh_source_ddl_omits_every_census_ledger_object(object_ref: str) -> None:
    """A fresh source generation declares none of the census ledger objects."""
    declared = {obj.object_ref.split(":", 1)[1] for obj in canonical_schema_objects(ArchiveTier.SOURCE)}
    assert object_ref not in declared


@pytest.mark.parametrize("table", RETAINED_AUTHORITY_TABLES)
def test_retained_raw_authority_tables_are_still_declared(table: str) -> None:
    """The retained durable evidence is untouched by the retirement."""
    declared = {obj.table_name for obj in canonical_schema_objects(ArchiveTier.SOURCE)}
    assert table in declared


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_a_fresh_archive_builds_without_the_census_ledger(tmp_path: Path) -> None:
    """Bootstrapping creates the retained tables and none of the retired ones."""
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        live = {
            str(name)
            for (name,) in conn.execute("SELECT name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'").fetchall()
        }
    assert live.issuperset(RETAINED_AUTHORITY_TABLES)
    assert live.isdisjoint({ref.split(":", 1)[1] for ref in RETIRED_CENSUS_OBJECTS})


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_frontier_inspection_writes_no_source_ledger_when_unchanged(tmp_path: Path) -> None:
    """Inspection measures Ops coverage without resurrecting a Source pass ledger."""
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    source_before = (tmp_path / "source.db").read_bytes()
    async with prepared_live_convergence_owner(tmp_path) as owner:
        first = await owner.run_convergence_sync(
            "fixture.retirement.inspect",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
        second = await owner.run_convergence_sync(
            "fixture.retirement.inspect",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
    assert first.mode == "full" and first.healthy
    assert first.pass_id is not None
    assert second.mode == "current" and second.healthy and second.pass_id is None
    assert (tmp_path / "source.db").read_bytes() == source_before
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers").fetchone()[0] == 0
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_frontier_inspection").fetchone()[0] == 1


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_readiness_still_reads_blockers_after_the_census_tables_are_gone(tmp_path: Path) -> None:
    """A blocking frontier refutes readiness through the durable blocker row.

    This is the regression the retirement could have caused silently: the old
    projection returned a zeroed payload -- including a zero blocker count --
    whenever ``raw_authority_censuses`` was absent, so dropping the table
    would have reported a blocked archive as clean.

    Anti-vacuity: resolving the one blocker flips both assertions back.
    """

    def seed_blocker() -> None:
        bootstrap_archive_root(tmp_path)
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute(
                """
                INSERT INTO raw_authority_blockers (
                    blocker_id, plan_input_digest, observed_pass_id, reason,
                    expected_json, observed_json, created_at_ms
                ) VALUES (
                    'raw-authority-blocker:retirement-test', ?, 'raw-authority-frontier-pass:test',
                    'missing bytes require reacquisition', ?,
                    '{"state": "missing_bytes_reacquire"}', 1000
                )
                """,
                (
                    "d" * 64,
                    json.dumps(
                        {
                            "plan_id": "raw-replay:retirement-test",
                            "input_digest": "d" * 64,
                            "input_raw_ids": [],
                            "logical_keys": [],
                            "authority_witness": {"schema": "polylogue.raw-authority-frontier-plan.v1"},
                            "source_preconditions": {},
                            "index_preconditions": {},
                        }
                    ),
                ),
            )
            conn.commit()

    await run_archive_fixture_write(tmp_path, seed_blocker)

    blocked = raw_materialization_readiness_snapshot(tmp_path)
    assert blocked["raw_authority_blocker_count"] == 1
    assert raw_materialization_ready(blocked) is False
    refs = blocked["raw_authority_frontier_remediation_refs"]
    assert isinstance(refs, list)
    assert refs[0]["blocker_id"] == "raw-authority-blocker:retirement-test"
    assert refs[0]["observed_pass_id"] == "raw-authority-frontier-pass:test"

    from polylogue.core.stage_admission import admit_stage_write

    async with prepared_live_convergence_owner(tmp_path) as owner:

        def resolve_blocker() -> None:
            with prepared_frontier_blocker_acknowledgement(
                tmp_path,
                "raw-authority-blocker:retirement-test",
                resolution="reacquired the missing bytes",
                input_demand=owner._compute_adapter.amend_current_input_demand,
            ) as prepared:
                assert prepared.found
                admit_stage_write("fixture.retirement.resolve", prepared.publish)

        await owner.run_convergence_sync("fixture.retirement.acknowledge", resolve_blocker)

    cleared = raw_materialization_readiness_snapshot(tmp_path)
    assert cleared["raw_authority_blocker_count"] == 0
    assert cleared["raw_authority_frontier_remediation_refs"] == []


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_frontier_obligation_states_are_the_only_blocker_writers_left(tmp_path: Path) -> None:
    """Prepared inspection retains the obligation vocabulary after census retirement."""
    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    async with prepared_live_convergence_owner(tmp_path) as owner:
        outcome = await owner.run_convergence_sync(
            "fixture.retirement.obligations",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
    assert outcome.healthy
    with sqlite3.connect(tmp_path / "source.db") as conn:
        origins = {
            str(origin)
            for (origin,) in conn.execute(
                "SELECT json_extract(observed_json, '$.blocker_origin') FROM raw_authority_blockers"
            ).fetchall()
        }
    assert origins <= {"frontier_obligation"}
    assert {state.value for state in RawAuthorityFrontierState} >= {"missing_bytes_reacquire", "corrupt"}


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_terminal_supersessions_remain_readable_without_creating_obligations(tmp_path: Path) -> None:
    """Resident replay retains the losing snapshot and its terminal application evidence."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    def acquire() -> tuple[str, str]:
        bootstrap_archive_root(tmp_path)
        records = [
            {
                "type": "user",
                "uuid": f"retirement-user-{index}",
                "sessionId": "retirement-session",
                "timestamp": f"2026-07-20T10:00:0{index}.000Z",
                "message": {"role": "user", "content": f"neutral message {index}"},
            }
            for index in range(2)
        ]
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_ids = []
            for count in (1, 2):
                raw_ids.append(
                    archive.write_raw_payload(
                        provider=Provider.CLAUDE_CODE,
                        payload=("".join(json.dumps(record) + "\n" for record in records[:count])).encode(),
                        source_path="neutral/retirement-session.jsonl",
                        canonical_source_path="neutral/retirement-session.jsonl",
                        acquired_at_ms=count,
                    )
                )
            archive.commit()
        return raw_ids[0], raw_ids[1]

    old_raw, new_raw = await run_archive_fixture_write(tmp_path, acquire)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((old_raw, new_raw))).require_complete()
        assert receipts
        outcome = await owner.run_convergence_sync(
            "fixture.retirement.terminal",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id IN (?,?)", (old_raw, new_raw)).fetchone()[0]
            == 2
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 0
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert (
            conn.execute("SELECT decision FROM raw_revision_applications WHERE raw_id=?", (old_raw,)).fetchone()[0]
            == "superseded"
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_revision_heads WHERE accepted_raw_id=?", (old_raw,)).fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM raw_revision_heads WHERE accepted_raw_id=?", (new_raw,)).fetchone()[0]
            > 0
        )
    assert outcome.healthy and outcome.blocking_head_checks == 0
