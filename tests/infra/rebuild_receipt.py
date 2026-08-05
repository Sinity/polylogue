"""Targeted schema-inference receipts for real rebuild-route tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from polylogue.maintenance import schema_inference_gate as gate
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import source_revision_snapshot
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def write_valid_rebuild_receipt(
    archive_root: Path,
    receipt_path: Path,
    *,
    generated_at: datetime | None = None,
) -> Path:
    """Write a minimally complete, identity-bound PASS for a test archive."""

    root = archive_root.absolute()
    location = ArchiveLocation.resolve(root)
    identity = ArchiveIdentity.resolve_location(location)
    with sqlite3.connect(root / "source.db") as source:
        origins = {str(row[0]) for row in source.execute("SELECT DISTINCT origin FROM raw_sessions ORDER BY origin")}
        ground_truth_origins: dict[str, object] = {}
        for origin in sorted(origins):
            declared = gate.GROUND_TRUTH_INPUTS.get(origin, {"exempt": False})
            if bool(declared.get("exempt")):
                ground_truth_origins[origin] = {
                    "exempt": True,
                    "reason": declared.get("reason"),
                    "raw_external_mapping": gate._raw_external_mapping(
                        source, origin=origin, inventory=(), exempt=True
                    ),
                }
                continue
            external_root = root.parent / f"{root.name}-{origin}-ground-truth"
            external_root.mkdir(parents=True, exist_ok=True)
            raw_rows = source.execute(
                "SELECT raw_id, blob_hash FROM raw_sessions WHERE origin = ? ORDER BY raw_id", (origin,)
            ).fetchall()
            for raw_id, blob_hash in raw_rows:
                (external_root / f"{raw_id}.bin").write_bytes(BlobStore(root / "blob").read_all(bytes(blob_hash).hex()))
            inventory = gate._external_inventory((external_root,))
            mapping = gate._raw_external_mapping(source, origin=origin, inventory=inventory, exempt=False)
            ground_truth_origins[origin] = {
                "exempt": False,
                "declared_roots": [str(external_root)],
                "external_inventory": inventory,
                "raw_external_mapping": mapping,
                "passed": all(item["disposition"] == "matched-external" for item in mapping),
            }

    ground_truth = {
        "passed": True,
        "origins": ground_truth_origins,
    }
    digest = gate._canonical_external_ground_truth_digest(cast(dict[str, object], ground_truth_origins))
    ground_truth["external_ground_truth_digest"] = digest
    query_results = {
        gate_id: {"gate": gate_id, "passed": True, "count": 0}
        for gate_id in (*gate._HARD_GATE_SQL, "zero-unexplained-byte-duplicates")
    }
    source_entry = {
        "expected_user_version": ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].version,
        "actual_user_version": ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].version,
        "matches_expected": True,
    }
    payload: dict[str, object] = {
        "schema": gate.RECEIPT_SCHEMA,
        "gate_version": gate.GATE_VERSION,
        "generated_at": (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "verdict": "PASS",
        "archive_root": str(root),
        "archive_identity": gate._archive_receipt_identity(location),
        "source_identity": {
            "durable_id": identity.durable_id,
            "source_tier": identity.tier("source").as_dict(),
        },
        "source_snapshot": source_revision_snapshot(root),
        "external_ground_truth_digest": digest,
        "source_schema_identity": source_entry,
        "query_results": query_results,
        "ground_truth_inputs": ground_truth,
        "corpus_fidelity": {"passed": True},
        "full_blob_hash_verification": {"passed": True},
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt_path


__all__ = ["write_valid_rebuild_receipt"]
