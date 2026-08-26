from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from devtools.physical_identity_census import build_report, classify_collision


def test_confidence_labels_do_not_call_missing_family_proof(tmp_path: Path) -> None:
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE raw_sessions (origin TEXT, native_id TEXT, capture_mode TEXT, detected_provider TEXT, raw_id TEXT)"
        )
        conn.executemany(
            "INSERT INTO raw_sessions VALUES (?, ?, ?, ?, ?)",
            [
                ("aistudio-drive", "same", "gemini", None, "one"),
                ("aistudio-drive", "same", "drive", None, "two"),
                ("aistudio-drive", "unknown", None, None, "three"),
                ("aistudio-drive", "unknown", None, None, "four"),
            ],
        )
    report = build_report(db)
    assert report["summary"] == {"candidate_groups": 2, "high": 1, "medium": 0, "low": 1}
    assert {row["confidence"] for row in report["collisions"]} == {"high", "low"}
    assert classify_collision(set(), 2) == "low"  # anti-vacuity: family-free duplicates are not high
    assert classify_collision({"gemini"}, 2) == "medium"


def test_two_source_families_project_one_public_ref_without_physical_collision() -> None:
    fixture = json.loads(Path("tests/fixtures/physical_identity/two-families.json").read_text())
    members = fixture["members"]
    physical_keys = [[member["source_family"], fixture["native_id"]] for member in members]
    public_refs = {(fixture["public_origin"], fixture["native_id"])}
    assert physical_keys == fixture["expected"]["physical_keys"]
    assert len(physical_keys) == 2
    assert len(public_refs) == 1
    assert fixture["expected"]["resolution"] == "ambiguous"
