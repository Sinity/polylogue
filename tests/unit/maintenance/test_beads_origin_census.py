"""Production-route tests for the read-only retired Beads-origin census."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.config import ResolvedRuntimeConfig
from polylogue.maintenance.beads_origin_census import BeadsOriginCensusError, write_census_receipt


def _runtime(root: Path, *raw_roots: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(archive_root=root, index_db=root / "index.db"),
        source_paths=SimpleNamespace(explicit=tuple(raw_roots), beads=()),
        sources=(),
    )


def test_census_distinguishes_zero_unavailable_and_populated_roots(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    raw_root = tmp_path / "repo"
    (raw_root / ".beads").mkdir(parents=True)
    (raw_root / ".beads" / "interactions.jsonl").write_text('{"type":"field_change"}\n', encoding="utf-8")
    missing = tmp_path / "missing"
    receipt = tmp_path / "receipt.json"

    payload = write_census_receipt(cast(ResolvedRuntimeConfig, _runtime(archive, raw_root, missing)), receipt)
    states = {item["name"]: item["state"] for item in payload["surfaces"]}

    assert states["archive"] == "zero"
    assert states[f"source:{raw_root}"] == "populated"
    assert states[f"source:{missing}"] == "unavailable"
    assert payload["production_mutation_performed"] is False
    assert payload["plan"]["no_apply_in_this_operation"] is True
    assert receipt.stat().st_mode & 0o222 == 0


def test_census_records_db_zero_and_failed_surfaces(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "source.db").write_bytes(b"not sqlite")
    (archive / "index.db").write_bytes(b"not sqlite")

    payload = write_census_receipt(cast(ResolvedRuntimeConfig, _runtime(archive)), tmp_path / "receipt.json")
    db_states = {item["name"]: item["state"] for item in payload["surfaces"]}
    assert db_states["source.db"] == "failed"
    assert db_states["index.db"] == "failed"
    assert "raw_sessions.origin" in payload["affected_tables"]["source.db"]
    assert "derived_rebuild" in payload["plan"]


def test_receipt_is_immutable_and_contains_exact_plan_digest(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    receipt = tmp_path / "receipt.json"
    payload = write_census_receipt(cast(ResolvedRuntimeConfig, _runtime(archive)), receipt)
    assert json.loads(receipt.read_text(encoding="utf-8"))["plan_digest"] == payload["plan_digest"]
    with pytest.raises(BeadsOriginCensusError, match="immutable"):
        write_census_receipt(cast(ResolvedRuntimeConfig, _runtime(archive)), receipt)


def test_the_plan_prescribes_no_durable_migration(tmp_path: Path) -> None:
    """The plan must describe the constraint surface that actually exists.

    It used to prescribe "apply the additive source migration that narrows
    retired-origin checks" and to state that every ``origin`` column carried
    an enum CHECK admitting the token. The durable half of that is false: the
    source tier's origin columns carry only a non-empty check, because
    vocabulary membership there is validated at the write boundary by
    ``require_vocabulary``. An operator following the old plan would go
    looking for a migration target that does not exist.

    Anti-vacuity: restore a plan step that applies a migration, or drop
    ``no_durable_migration_required``, and this goes red.
    """
    archive = tmp_path / "archive"
    archive.mkdir()

    plan = write_census_receipt(cast(ResolvedRuntimeConfig, _runtime(archive)), tmp_path / "receipt.json")["plan"]

    assert plan["no_durable_migration_required"] is True
    assert not any("migration" in step for step in plan["steps"])
    assert any("write boundary" in detail for detail in plan["constraints"].values())
    assert any("re-created from the narrowed enum" in detail for detail in plan["constraints"].values())


def test_the_plan_matches_the_ddl_a_fresh_archive_actually_creates(tmp_path: Path) -> None:
    """Ground the durable/derived asymmetry in real DDL, not in the prose.

    The plan is only honest while the durable tier stays free of an
    enum-generated origin CHECK and the derived tier keeps generating one.
    This reads the freshly created schema of both.

    Anti-vacuity: add ``CHECK(origin IN (...))`` to ``raw_sessions`` and the
    durable assertion goes red -- which is exactly the signal that the removal
    plan needs a durable migration step again. Stop generating the derived
    CHECK from the enum and the second assertion goes red instead.
    """
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "fresh"
    with ArchiveStore(root) as facade:
        index_sql = {
            str(row[0]): str(row[1])
            for row in facade._conn.execute("SELECT name, sql FROM sqlite_master WHERE sql NOT NULL")
        }
        with sqlite3.connect(facade.source_db_path) as source:
            source_sql = {
                str(row[0]): str(row[1])
                for row in source.execute("SELECT name, sql FROM sqlite_master WHERE sql NOT NULL")
            }

    assert not [name for name, sql in source_sql.items() if "beads-issue" in sql]
    assert "beads-issue" in index_sql["sessions"]
    assert "beads-issue" in index_sql["session_links"]


def test_the_plan_protects_the_other_beads_issue_vocabulary(tmp_path: Path) -> None:
    """``beads-issue`` spells one string across two unrelated vocabularies.

    ``Origin.BEADS_ISSUE`` is the retired session origin. ``ObjectRef(kind=
    "beads-issue")`` is a live work-effect object kind that
    ``analysis/work_effects.py`` mints from a ``.beads/interactions.jsonl``
    ledger. A grep-driven removal would delete both.

    Anti-vacuity: drop ``non_targets`` from the plan, or stop naming
    ``work_effects.py``, and this goes red.
    """
    archive = tmp_path / "archive"
    archive.mkdir()

    plan = write_census_receipt(cast(ResolvedRuntimeConfig, _runtime(archive)), tmp_path / "receipt.json")["plan"]

    assert "polylogue/analysis/work_effects.py" in plan["non_targets"]
    assert all(detail.strip() for detail in plan["non_targets"].values())
