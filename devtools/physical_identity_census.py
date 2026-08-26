"""Census raw evidence for sessions hidden by public-origin collapse.

This is deliberately read-only and does not propose a schema migration.  The
source tier is the only place where acquisition-family evidence survives the
current ``origin:native_id`` index key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from polylogue.config import get_config


def classify_collision(families: set[str], raw_count: int) -> str:
    """Label evidence strength without pretending missing provenance is proof."""
    if len(families) > 1:
        return "high"
    if raw_count > 1:
        return "medium" if families else "low"
    return "low"


def build_report(source_db: Path) -> dict[str, Any]:
    """Return a privacy-safe collision census from a source database."""
    source_db = source_db.expanduser().resolve()
    uri = f"file:{source_db}?mode=ro"
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(raw_sessions)")}
        detected_provider = ", detected_provider" if "detected_provider" in columns else ", NULL AS detected_provider"
        rows = conn.execute(
            f"""
            SELECT origin, native_id, capture_mode{detected_provider}, raw_id
            FROM raw_sessions
            WHERE native_id IS NOT NULL
            ORDER BY origin, native_id, raw_id
            """
        )
        for row in rows:
            groups[(str(row["origin"]), str(row["native_id"]))].append(
                {
                    "raw_id": str(row["raw_id"]),
                    "family": row["capture_mode"] or row["detected_provider"],
                }
            )
    collisions = []
    for (origin, native_id), members in groups.items():
        families = {str(item["family"]) for item in members if item["family"]}
        if len(members) < 2:
            continue
        collisions.append(
            {
                "origin": origin,
                "native_id_sha256": hashlib.sha256(native_id.encode()).hexdigest(),
                "raw_count": len(members),
                "families": sorted(families),
                "confidence": classify_collision(families, len(members)),
                "raw_ids_sha256": sorted(hashlib.sha256(item["raw_id"].encode()).hexdigest() for item in members),
            }
        )
    return {
        "artifact": "physical-session-identity-census",
        "version": 1,
        "source_db": str(source_db),
        "evidence": "source.raw_sessions; native_id and acquisition-family provenance",
        "confidence_labels": {
            "high": "same public origin/native_id and two or more distinct family hints",
            "medium": "same public origin/native_id across raw rows, but family hints do not distinguish them",
            "low": "duplicate candidate with no usable family hint; requires raw-byte review",
        },
        "summary": {
            "candidate_groups": len(collisions),
            "high": sum(item["confidence"] == "high" for item in collisions),
            "medium": sum(item["confidence"] == "medium" for item in collisions),
            "low": sum(item["confidence"] == "low" for item in collisions),
        },
        "collisions": collisions,
        "interpretation": "A high-confidence row proves source evidence was collapsed by the current key; it does not prove the historical bytes are independently splittable.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="devtools workspace physical-identity-census")
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_db = args.source_db or ((args.archive_root or get_config().archive_root) / "source.db")
    report = build_report(source_db)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("physical-identity-census: " + " ".join(f"{k}={v}" for k, v in report["summary"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
