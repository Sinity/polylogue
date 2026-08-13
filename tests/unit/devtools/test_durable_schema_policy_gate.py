"""Merge-time policy coverage for durable migration slot ownership."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify, verify_schema_upgrade_lane
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS, durable_migration_claim_for_sql


def _healthy_delta_report() -> dict[str, object]:
    return {
        "ok": True,
        "compatibility_floor": 1,
        "missing_versions": (),
        "duplicate_versions": (),
        "invalid_versions": (),
    }


def test_checked_in_durable_migrations_have_unique_contention_keys() -> None:
    claims = verify_schema_upgrade_lane._durable_migration_claims_on_disk()
    report = verify_schema_upgrade_lane.durable_migration_collision_report(claims)
    assert report["ok"] is True
    assert report["collisions"] == []
    assert {claim.tier for claim in claims} == DURABLE_MIGRATION_TIERS


def test_schema_policy_accepts_only_canonical_train_sidecar_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    source = migrations / "source"
    source.mkdir(parents=True)
    (source / "027_future.sql").write_text("SELECT 1;\n", encoding="utf-8")
    (source / "027.train.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(verify_schema_upgrade_lane, "MIGRATIONS_DIR", migrations)

    assert verify_schema_upgrade_lane._invalid_migration_paths() == []

    (source / "027_future.bad.json").write_text("{}\n", encoding="utf-8")
    assert verify_schema_upgrade_lane._invalid_migration_paths() == [source / "027_future.bad.json"]


def test_policy_json_names_every_duplicate_owner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        Path("polylogue/storage/sqlite/migrations/source/008_first.sql"),
        "-- migration-safety: additive-no-backup\nCREATE TABLE first (id INTEGER);\n",
        owner_ref="owner:first",
    )
    second = durable_migration_claim_for_sql(
        ArchiveTier.SOURCE,
        Path("polylogue/storage/sqlite/migrations/source/008_second.sql"),
        "-- migration-safety: additive-no-backup\nCREATE TABLE second (id INTEGER);\n",
        owner_ref="owner:second",
    )
    monkeypatch.setattr(verify_schema_upgrade_lane, "_collect_upgrade_helpers", lambda: [])
    monkeypatch.setattr(verify_schema_upgrade_lane, "_invalid_migration_paths", lambda: [])
    monkeypatch.setattr(
        verify_schema_upgrade_lane,
        "_durable_migration_claims_on_disk",
        lambda: (first, second),
    )
    monkeypatch.setattr(
        verify_schema_upgrade_lane,
        "index_delta_declaration_report",
        lambda _version: _healthy_delta_report(),
    )

    assert verify_schema_upgrade_lane.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    collision_report = payload["durable_migration_collisions"]
    assert collision_report["ok"] is False
    assert collision_report["collisions"][0]["contention_key"] == ["source", 8, 8]
    serialized = json.dumps(collision_report)
    assert "008_first.sql" in serialized
    assert "008_second.sql" in serialized
    assert "owner:first" in serialized
    assert "owner:second" in serialized
    assert "rebase and renumber" in serialized


def test_schema_versioning_policy_runs_exactly_once_in_every_noncommit_fast_gate() -> None:
    for quick, lab in ((True, False), (True, True), (False, False), (False, True)):
        labels = [
            label
            for label, _command in verify.build_verify_steps(
                quick=quick,
                lab=lab,
                skip_slow=True,
            )
        ]
        assert labels.count("lab policy schema-versioning") == 1

    commit_labels = [
        label
        for label, _command in verify.build_verify_steps(
            quick=True,
            lab=False,
            skip_slow=True,
            commit=True,
        )
    ]
    assert "lab policy schema-versioning" not in commit_labels
