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

from polylogue.config import Config
from polylogue.storage.archive_readiness import (
    raw_materialization_readiness_snapshot,
    raw_materialization_ready,
)
from polylogue.storage.raw_authority import resolve_raw_authority_blocker
from polylogue.storage.raw_reconciler import RawAuthorityFrontierState, inspect_raw_authority_frontier
from polylogue.storage.sqlite.archive_tiers.schema_inventory import canonical_schema_objects
from polylogue.storage.sqlite.archive_tiers.source import RETIRED_SOURCE_SCHEMA_OBJECTS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root

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


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")


@pytest.mark.parametrize("object_ref", RETIRED_CENSUS_OBJECTS)
def test_fresh_source_ddl_omits_every_census_ledger_object(object_ref: str) -> None:
    """A fresh source generation declares none of the census ledger objects."""
    declared = {obj.object_ref.split(":", 1)[1] for obj in canonical_schema_objects(ArchiveTier.SOURCE)}
    assert object_ref not in declared
    assert object_ref in RETIRED_SOURCE_SCHEMA_OBJECTS


@pytest.mark.parametrize("table", RETAINED_AUTHORITY_TABLES)
def test_retained_raw_authority_tables_are_still_declared(table: str) -> None:
    """The retained durable evidence is untouched by the retirement."""
    declared = {obj.table_name for obj in canonical_schema_objects(ArchiveTier.SOURCE)}
    assert table in declared
    assert f"table:{table}" not in RETIRED_SOURCE_SCHEMA_OBJECTS


def test_a_fresh_archive_builds_without_the_census_ledger(tmp_path: Path) -> None:
    """Bootstrapping creates the retained tables and none of the retired ones."""
    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        live = {
            str(name)
            for (name,) in conn.execute("SELECT name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'").fetchall()
        }
    assert live.issuperset(RETAINED_AUTHORITY_TABLES)
    assert live.isdisjoint({ref.split(":", 1)[1] for ref in RETIRED_CENSUS_OBJECTS})


def test_frontier_inspection_is_content_addressed_and_writes_nothing_when_unchanged(tmp_path: Path) -> None:
    """Two passes over an unchanged, clean frontier agree and record no rows.

    Anti-vacuity: the census ledger's whole purpose was recording one row set
    per pass. If any per-pass ledger were reintroduced, the second pass would
    either write rows here or mint a different ``pass_id``.
    """
    bootstrap_archive_root(tmp_path)

    first = inspect_raw_authority_frontier(_config(tmp_path))
    second = inspect_raw_authority_frontier(_config(tmp_path))

    assert first.pass_id == second.pass_id
    assert first.pass_id.startswith("raw-authority-frontier-pass:")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers").fetchone()[0] == 0


def test_readiness_still_reads_blockers_after_the_census_tables_are_gone(tmp_path: Path) -> None:
    """A blocking frontier refutes readiness through the durable blocker row.

    This is the regression the retirement could have caused silently: the old
    projection returned a zeroed payload -- including a zero blocker count --
    whenever ``raw_authority_censuses`` was absent, so dropping the table
    would have reported a blocked archive as clean.

    Anti-vacuity: resolving the one blocker flips both assertions back.
    """
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

    blocked = raw_materialization_readiness_snapshot(tmp_path)
    assert blocked["raw_authority_blocker_count"] == 1
    assert raw_materialization_ready(blocked) is False
    refs = blocked["raw_authority_frontier_remediation_refs"]
    assert isinstance(refs, list)
    assert refs[0]["blocker_id"] == "raw-authority-blocker:retirement-test"
    assert refs[0]["observed_pass_id"] == "raw-authority-frontier-pass:test"

    resolve_raw_authority_blocker(
        tmp_path,
        "raw-authority-blocker:retirement-test",
        resolution="reacquired the missing bytes",
    )

    cleared = raw_materialization_readiness_snapshot(tmp_path)
    assert cleared["raw_authority_blocker_count"] == 0
    assert cleared["raw_authority_frontier_remediation_refs"] == []


def test_frontier_obligation_states_are_the_only_blocker_writers_left(tmp_path: Path) -> None:
    """Every published blocker names an obligation state, not a plan outcome.

    The retired stale-plan and invalid-application blocker writers updated
    ``raw_authority_census_plans`` in the same transaction as their blocker
    insert, so they could not survive the drop. This pins what remains.
    """
    bootstrap_archive_root(tmp_path)
    inspect_raw_authority_frontier(_config(tmp_path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        origins = {
            str(origin)
            for (origin,) in conn.execute(
                "SELECT json_extract(observed_json, '$.blocker_origin') FROM raw_authority_blockers"
            ).fetchall()
        }
    assert origins <= {"frontier_obligation"}
    assert {state.value for state in RawAuthorityFrontierState} >= {"missing_bytes_reacquire", "corrupt"}


def _seed_blocking_frontier_head(root: Path) -> None:
    """Record an accepted head whose raw authority is absent from source.db.

    ``_classify_frontier`` reads that as ``missing_bytes_reacquire`` -- one of
    the three obligation states -- so the next pass publishes a durable blocker.
    """
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_revision_heads (
                logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                acquisition_generation, append_end_offset, decided_at_ms
            ) VALUES ('logical:absent', 'codex-session:absent', 'raw:absent', 'rev-1',
                      ?, 'byte', 0, 0, NULL, 1000)
            """,
            (b"\x11" * 32,),
        )
        conn.commit()


def _open_blocker_ids(root: Path) -> list[str]:
    with sqlite3.connect(root / "source.db") as conn:
        return [
            str(blocker_id)
            for (blocker_id,) in conn.execute(
                "SELECT blocker_id FROM raw_authority_blockers WHERE resolved_at_ms IS NULL ORDER BY blocker_id"
            ).fetchall()
        ]


def test_acknowledging_an_undischarged_obligation_does_not_survive_the_next_pass(tmp_path: Path) -> None:
    """Resolution acknowledges evidence; only changed evidence discharges it.

    ``pass_id`` is a content address over the frontier inventory, so a pass over
    unchanged blocking evidence reconstructs the identical ``blocker_id``. With
    an insert-only reconciliation the tombstone survived that collision
    forever: ``raw_authority_blocker_count`` stayed zero and readiness reported
    a clean archive while the same missing bytes were still missing.

    Anti-vacuity: removing the reopening UPDATE in
    ``_reconcile_frontier_obligations`` leaves the second pass with no open
    blocker and fails the reopen assertion. The disproof half below pins the
    other direction, so a blanket "always reopen" cannot pass either.
    """
    bootstrap_archive_root(tmp_path)
    _seed_blocking_frontier_head(tmp_path)

    inspect_raw_authority_frontier(_config(tmp_path))
    published = _open_blocker_ids(tmp_path)
    assert len(published) == 1
    blocker_id = published[0]

    assert raw_materialization_ready(raw_materialization_readiness_snapshot(tmp_path)) is False

    # Readiness reads the open-blocker count, so an acknowledgement alone makes
    # the archive look clean. That is the window the reopen has to close.
    resolve_raw_authority_blocker(tmp_path, blocker_id, resolution="acknowledged without reacquiring the bytes")
    assert _open_blocker_ids(tmp_path) == []

    # The evidence has not changed, so the obligation is not discharged.
    inspect_raw_authority_frontier(_config(tmp_path))
    assert _open_blocker_ids(tmp_path) == [blocker_id]
    assert raw_materialization_ready(raw_materialization_readiness_snapshot(tmp_path)) is False

    # Opposite direction: evidence that no longer blocks stays closed.
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM raw_revision_heads WHERE logical_source_key = 'logical:absent'")
        conn.commit()
    inspect_raw_authority_frontier(_config(tmp_path))
    assert _open_blocker_ids(tmp_path) == []
    inspect_raw_authority_frontier(_config(tmp_path))
    assert _open_blocker_ids(tmp_path) == []
