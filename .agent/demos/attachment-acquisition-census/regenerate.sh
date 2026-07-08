#!/usr/bin/env bash
set -euo pipefail

repo="${POLYLOGUE_REPO:-/realm/project/polylogue}"
archive_root="${POLYLOGUE_ARCHIVE_ROOT:-/home/sinity/.local/share/polylogue}"
demo_root="$repo/.agent/demos/attachment-acquisition-census"
mkdir -p "$demo_root"
cd "$repo"

reconcile_json="$(POLYLOGUE_ARCHIVE_ROOT="$archive_root" POLYLOGUE_FORCE_PLAIN=1 \
  .venv/bin/polylogue ops maintenance attachment-acquisition-debt --output-format json)"
echo "$reconcile_json" > "$demo_root/reconcile-attachment-acquisition-debt.json"

python3 - "$demo_root" "$archive_root" "$reconcile_json" <<'PY'
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

demo_root = Path(sys.argv[1])
archive_root = Path(sys.argv[2])
reconcile = json.loads(sys.argv[3])
index_db = archive_root / "index.db"

conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

# One row per attachment with its origin (an attachment referenced from
# multiple sessions picks the lexicographically-first origin and that is
# noted, not hidden -- cross-origin fan-out is rare and worth flagging,
# not averaging away). LEFT JOIN throughout: an attachment with zero refs
# (ref_count fell to 0 after a ref delete, but the row persists) must still
# appear -- as origin=NULL -- or the census total silently undercounts
# against `attachment-acquisition-debt`, which counts every attachments row.
rows = conn.execute(
    """
    SELECT
        a.attachment_id AS attachment_id,
        a.acquisition_status AS acquisition_status,
        a.byte_count AS byte_count,
        a.blob_hash AS blob_hash,
        MIN(s.origin) AS origin,
        COUNT(DISTINCT s.origin) AS distinct_origin_count,
        GROUP_CONCAT(DISTINCT r.upload_origin) AS upload_origins
    FROM attachments a
    LEFT JOIN attachment_refs r ON r.attachment_id = a.attachment_id
    LEFT JOIN sessions s ON s.session_id = r.session_id
    GROUP BY a.attachment_id
    """
).fetchall()
conn.close()

from polylogue.storage.blob_store import get_blob_store  # noqa: E402

store = get_blob_store()

groups: dict[tuple[str, str], dict[str, object]] = {}
cross_origin_attachment_count = 0
for row in rows:
    origin = row["origin"] or "(unknown)"
    status = row["acquisition_status"]
    if row["distinct_origin_count"] > 1:
        cross_origin_attachment_count += 1
    key = (origin, status)
    bucket = groups.setdefault(
        key,
        {
            "attachment_count": 0,
            "declared_byte_sum": 0,
            "acquired_blob_count": 0,
            "acquired_blob_bytes_on_disk": 0,
            "missing_blob_ref_count": 0,
            "upload_origin_counts": defaultdict(int),
            "sample_attachment_ids": [],
        },
    )
    bucket["attachment_count"] += 1
    bucket["declared_byte_sum"] += int(row["byte_count"] or 0)
    for upload_origin in (row["upload_origins"] or "").split(","):
        if upload_origin:
            bucket["upload_origin_counts"][upload_origin] += 1
    if len(bucket["sample_attachment_ids"]) < 20:
        bucket["sample_attachment_ids"].append(row["attachment_id"])

    blob_hash = row["blob_hash"]
    if status == "acquired" and blob_hash is not None:
        hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash)
        if store.exists(hash_hex):
            bucket["acquired_blob_count"] += 1
            bucket["acquired_blob_bytes_on_disk"] += store.blob_path(hash_hex).stat().st_size
        else:
            bucket["missing_blob_ref_count"] += 1

census_rows = []
totals = {
    "attachment_count": 0,
    "declared_byte_sum": 0,
    "acquired_blob_count": 0,
    "acquired_blob_bytes_on_disk": 0,
    "missing_blob_ref_count": 0,
}
for (origin, status), bucket in sorted(groups.items()):
    census_rows.append(
        {
            "origin": origin,
            "acquisition_status": status,
            "attachment_count": bucket["attachment_count"],
            "declared_byte_sum": bucket["declared_byte_sum"],
            "acquired_blob_count": bucket["acquired_blob_count"],
            "acquired_blob_bytes_on_disk": bucket["acquired_blob_bytes_on_disk"],
            "missing_blob_ref_count": bucket["missing_blob_ref_count"],
            "upload_origin_counts": dict(bucket["upload_origin_counts"]),
            "sample_attachment_ids": bucket["sample_attachment_ids"],
        }
    )
    for k in totals:
        totals[k] += bucket[k] if k in bucket else 0

payload = {
    "archive_root": str(archive_root),
    "cross_origin_attachment_count": cross_origin_attachment_count,
    "totals": totals,
    "rows": census_rows,
    "reconciliation": {
        "attachment_acquisition_debt_command": reconcile,
        "totals_match": (
            totals["attachment_count"] == reconcile["total_attachments"]
            and totals["acquired_blob_count"] + reconcile["acquired_missing_blob_count"]
            == reconcile["acquired_count"]
            and totals["missing_blob_ref_count"] == reconcile["acquired_missing_blob_count"]
        ),
    },
}

(demo_root / "census.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Attachment Acquisition Census",
    "",
    f"Archive root: `{archive_root}`",
    "",
    "Read-only census over the active archive (polylogue-83u.6), grouped by"
    " (origin, acquisition_status). `unfetched` is the honest floor (bytes"
    " never fetched, e.g. source-deleted / pre-install / provider-expiry) --"
    " not a defect backlog. `missing_blob_ref_count` is the one genuinely"
    " actionable class: an `acquired` row whose blob file is absent.",
    "",
    "## Totals",
    "",
    f"- Attachments: {totals['attachment_count']:,}",
    f"- Declared bytes: {totals['declared_byte_sum']:,}",
    f"- Acquired blobs on disk: {totals['acquired_blob_count']:,} ({totals['acquired_blob_bytes_on_disk']:,} bytes)",
    f"- Missing blob refs (actionable debt): {totals['missing_blob_ref_count']:,}",
    f"- Cross-origin attachments (referenced from >1 origin): {cross_origin_attachment_count:,}",
    f"- Reconciles against `polylogue ops maintenance attachment-acquisition-debt`: "
    f"{payload['reconciliation']['totals_match']}",
    "",
    "## By origin / acquisition_status",
    "",
    "| Origin | Status | Count | Declared bytes | Acquired-on-disk | Missing blob refs |",
    "|---|---|---:|---:|---:|---:|",
]
for row in census_rows:
    lines.append(
        f"| {row['origin']} | {row['acquisition_status']} | {row['attachment_count']:,} | "
        f"{row['declared_byte_sum']:,} | {row['acquired_blob_count']:,} | {row['missing_blob_ref_count']:,} |"
    )
lines.append("")

(demo_root / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {demo_root / 'census.json'} and {demo_root / 'ANALYSIS.md'}")
PY
