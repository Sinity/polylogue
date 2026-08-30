"""Read-only authority census for durable hook-event payloads.

The logical event row carries the replayable JSON value.  Its blob and carrier
rows are independent durability/provenance evidence and must agree with that
value; a missing blob is not evidence that no bytes are required.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HookEventAuthorityCensus:
    """Complete classification of the rows visible in one source database."""

    row_count: int
    carrier_role_counts: dict[str, int]
    event_type_counts: dict[str, int]
    dispositions: dict[str, int]
    issues: tuple[str, ...]

    @property
    def clean_count(self) -> int:
        return self.dispositions.get("clean", 0)

    @property
    def blocked_count(self) -> int:
        return self.row_count - self.clean_count

    @property
    def source_sealable(self) -> bool:
        return self.blocked_count == 0 and not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "carrier_role_counts": dict(self.carrier_role_counts),
            "event_type_counts": dict(self.event_type_counts),
            "dispositions": dict(self.dispositions),
            "clean_count": self.clean_count,
            "blocked_count": self.blocked_count,
            "source_sealable": self.source_sealable,
            "issues": list(self.issues),
        }


def census_hook_event_authority(conn: sqlite3.Connection) -> HookEventAuthorityCensus:
    """Classify every hook row and prove its durable payload relationships.

    This function never reads or writes the filesystem and never repairs rows.
    It is therefore safe to run against a live source tier while preparing a
    clean-generation seal.
    """

    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing_tables = sorted({"raw_hook_events", "blob_refs", "hook_event_carriers"} - tables)
        if missing_tables:
            count = (
                int(conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone()[0])
                if "raw_hook_events" in tables
                else 0
            )
            return HookEventAuthorityCensus(
                count,
                {},
                {},
                {"schema-unavailable": count},
                (f"missing source table(s): {', '.join(missing_tables)}",),
            )
        rows = conn.execute(
            "SELECT hook_event_id, event_type, payload_json, blob_hash FROM raw_hook_events ORDER BY hook_event_id"
        ).fetchall()
    except sqlite3.Error as exc:
        return HookEventAuthorityCensus(0, {}, {}, {"schema-unavailable": 0}, (f"raw_hook_events unreadable: {exc}",))

    carrier_roles: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    issues: list[str] = []
    for event_id, event_type, payload_json, blob_hash in rows:
        event_id = str(event_id)
        event_types[str(event_type)] += 1
        row_issues: list[str] = []
        try:
            payload = json.loads(str(payload_json))
        except (TypeError, json.JSONDecodeError):
            payload = None
            row_issues.append("malformed-inline-payload")
        if not isinstance(payload, dict):
            row_issues.append("inline-payload-not-object")
            payload_digest = None
        else:
            payload_digest = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).digest()
        if blob_hash is None:
            row_issues.append("missing-blob-hash")
        else:
            ref = conn.execute(
                "SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'hook_payload' AND ref_id = ? AND blob_hash = ?",
                (event_id, blob_hash),
            ).fetchone()
            if ref is None or int(ref[0]) != 1:
                row_issues.append("blob-ref-disagreement")
        carriers = conn.execute(
            "SELECT blob_hash, payload_digest, carrier_role FROM hook_event_carriers "
            "WHERE hook_event_id = ? ORDER BY source_id, relative_path",
            (event_id,),
        ).fetchall()
        if not carriers:
            row_issues.append("missing-carrier")
        for carrier_hash, carrier_digest, carrier_role in carriers:
            carrier_roles[str(carrier_role)] += 1
            if blob_hash is None or bytes(carrier_hash) != bytes(blob_hash):
                row_issues.append("carrier-blob-disagreement")
            if payload_digest is None or bytes(carrier_digest) != payload_digest:
                row_issues.append("carrier-payload-disagreement")
        disposition = "clean" if not row_issues else "blocked"
        dispositions[disposition] += 1
        issues.extend(f"{event_id}:{issue}" for issue in sorted(set(row_issues)))
    return HookEventAuthorityCensus(
        row_count=len(rows),
        carrier_role_counts=dict(sorted(carrier_roles.items())),
        event_type_counts=dict(sorted(event_types.items())),
        dispositions=dict(sorted(dispositions.items())),
        issues=tuple(issues),
    )


__all__ = ["HookEventAuthorityCensus", "census_hook_event_authority"]
