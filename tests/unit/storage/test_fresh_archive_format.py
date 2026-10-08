"""Every tier is born at one before declared durable trains advance it."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_BASELINE_VERSION_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive_plan import ARCHIVE_FORMAT_LINEAGE
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import durable_train_manifest_paths
from polylogue.storage.sqlite.migration_runner import capture_durable_schema_inventory, durable_migration_claims

_CURRENT_OBJECTS = {
    ArchiveTier.SOURCE: ("table", "accepted_marker_inputs"),
    ArchiveTier.INDEX: ("table", "sessions"),
    ArchiveTier.EMBEDDINGS: ("table", "message_embeddings_meta"),
    ArchiveTier.USER: ("table", "assertions"),
    ArchiveTier.OPS: ("table", "convergence_debt"),
    ArchiveTier.AUDIT: ("index", "idx_machine_requests_operation"),
}


#: Durable schema inventories (``capture_durable_schema_inventory``) of the
#: schemas the predecessor lineage reached by replaying its numbered chains:
#: Source baseline plus slots 002-006, User and Audit baselines, as rendered
#: at 761a652f84. The v6 lineage folded those chains into its v1 baselines, so
#: a fresh v1 tier must reproduce them exactly; a later durable change is a
#: numbered migration, never an edit of this floor.
_FOLDED_CHAIN_SCHEMA_INVENTORY = {
    ArchiveTier.SOURCE: "57f22c8ea0396efa6b9f09849a5560890223b545e64726a6b1cb18417494109c",
    ArchiveTier.USER: "7d7e5af469600cd9b920114bb734fb35ab644722872773d980370b1605f31530",
    ArchiveTier.AUDIT: "471c789089ef9d8f61198a1a26b9780bae4ee19531bf4f13f758b1e990189700",
}


def test_fresh_archive_is_born_at_one_with_the_folded_durable_schema(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    marker = json.loads((tmp_path / ".polylogue-format.json").read_text(encoding="utf-8"))
    assert ARCHIVE_FORMAT_LINEAGE == "polylogue.archive-format.v6"
    assert marker["format"] == ARCHIVE_FORMAT_LINEAGE
    assert marker["floor_version"] == 1
    assert marker["tier_versions"] == dict.fromkeys((tier.value for tier in ArchiveTier), 1)
    assert durable_train_manifest_paths(tmp_path / ".maintenance-state" / "durable-change-trains") == ()
    for tier, spec in ARCHIVE_TIER_SPECS.items():
        assert ARCHIVE_BASELINE_VERSION_BY_TIER[tier] == 1
        with closing(sqlite3.connect(tmp_path / spec.filename)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[tier]
            assert conn.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = ? AND name = ?",
                _CURRENT_OBJECTS[tier],
            ).fetchone() == (1,)
            if tier in _FOLDED_CHAIN_SCHEMA_INVENTORY:
                assert ARCHIVE_VERSION_BY_TIER[tier] == 1
                assert durable_migration_claims(tier) == ()
                assert capture_durable_schema_inventory(conn).sha256 == _FOLDED_CHAIN_SCHEMA_INVENTORY[tier]


def test_old_marker_with_valid_digest_is_refused_before_tier_writes(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    marker_path = tmp_path / ".polylogue-format.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["format"] = "polylogue.archive-format.v2"
    payload = {key: value for key, value in marker.items() if key != "digest"}
    marker["digest"] = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    before = {tier: (tmp_path / spec.filename).read_bytes() for tier, spec in ARCHIVE_TIER_SPECS.items()}

    with pytest.raises(RuntimeError, match="does not identify polylogue.archive-format.v6"):
        initialize_active_archive_root(tmp_path)

    assert {tier: (tmp_path / spec.filename).read_bytes() for tier, spec in ARCHIVE_TIER_SPECS.items()} == before
