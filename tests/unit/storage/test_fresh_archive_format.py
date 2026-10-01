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

_CURRENT_OBJECTS = {
    ArchiveTier.SOURCE: ("table", "accepted_marker_inputs"),
    ArchiveTier.INDEX: ("table", "sessions"),
    ArchiveTier.EMBEDDINGS: ("table", "message_embeddings_meta"),
    ArchiveTier.USER: ("table", "assertions"),
    ArchiveTier.OPS: ("table", "convergence_debt"),
    ArchiveTier.AUDIT: ("index", "idx_machine_requests_operation"),
}


def test_fresh_archive_is_born_at_one_then_applies_declared_runtime_trains(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)

    marker = json.loads((tmp_path / ".polylogue-format.json").read_text(encoding="utf-8"))
    assert ARCHIVE_FORMAT_LINEAGE == "polylogue.archive-format.v5"
    assert marker["format"] == ARCHIVE_FORMAT_LINEAGE
    assert marker["floor_version"] == 1
    assert marker["tier_versions"] == dict.fromkeys((tier.value for tier in ArchiveTier), 1)
    for tier, spec in ARCHIVE_TIER_SPECS.items():
        assert ARCHIVE_BASELINE_VERSION_BY_TIER[tier] == 1
        with closing(sqlite3.connect(tmp_path / spec.filename)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_VERSION_BY_TIER[tier]
            assert conn.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = ? AND name = ?",
                _CURRENT_OBJECTS[tier],
            ).fetchone() == (1,)


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

    with pytest.raises(RuntimeError, match="does not identify polylogue.archive-format.v5"):
        initialize_active_archive_root(tmp_path)

    assert {tier: (tmp_path / spec.filename).read_bytes() for tier, spec in ARCHIVE_TIER_SPECS.items()} == before
