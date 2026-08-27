"""Production-route tests for the read-only retired Beads-origin census."""

from __future__ import annotations

import json
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
