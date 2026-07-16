---
created: "2026-07-03T08:25:00+02:00"
purpose: "Research synthesis for Beads issue polylogue-83u.4"
status: "complete"
project: "polylogue"
---

# Blob Reference Classification

## Context

Research lane for `polylogue-83u.4`, originally phrased around classifying
39,586 missing referenced blobs in a production backup.

## Current Active Archive Evidence

Active root: `/home/sinity/.local/share/polylogue`.

Read-only probes reported:

- `polylogue ops maintenance blob-reference-debt --output-format json`:
  source-tier debt is clean, with 37,552 source reference rows, 20,269
  distinct referenced blobs, and 0 missing distinct blobs.
- `polylogue ops diagnostics workload --blob-reference-debt --json`:
  `missing_referenced_blobs=0`, with source reference rows from
  `raw_sessions=16709` and `blob_refs=20032`.
- `blob-reference-restore-direct` dry-run and recovery plan found 0 candidates.

The historical 39,586 warning is therefore not reproducible against the current
active archive. Treat it as stale backup evidence or evidence from a different
archive root unless the original backup report/manifest is supplied.

## Attachment State

Current `index.db` attachment rows are shaped correctly:

- `attachments`: 7,226 rows.
- `acquired`: 958 rows, all with non-null blob hashes; filesystem check found
  0 missing files.
- `unfetched`: 6,268 rows, all with `blob_hash IS NULL`.

This is the expected post-v13 shape: unfetched attachments are not missing
referenced blobs. Acquired attachments with missing blob files would be a
separate attachment acquisition debt class, not source-tier backup blob debt.

## Relevant Code

- `polylogue/daemon/backup.py` uses `referenced_blob_hashes()` and
  `scan_blob_reference_debt()` for backup warning data.
- `polylogue/storage/sqlite/archive_tiers/source.py` defines source-tier
  references: `raw_sessions.blob_hash` and `blob_refs`.
- `polylogue/storage/blob_integrity.py` groups missing source references.
- `polylogue/storage/sqlite/archive_tiers/index.py` now makes
  `attachments.blob_hash` nullable and carries `acquisition_status`.
- `polylogue/storage/sqlite/archive_tiers/write.py` no longer fabricates
  attachment blob hashes for unfetched attachment refs.

## Implementation Direction

Make source-tier backup debt and index-tier attachment acquisition state
explicitly separate in diagnostics, backup warnings, docs, and tests.

Acceptance criteria:

- A read-only diagnostic reports source-tier backup blob debt separately from
  index-tier attachment acquisition state.
- Unfetched attachments with `blob_hash NULL` are not counted as missing
  referenced blobs.
- Acquired attachments whose blob file is missing are classified as attachment
  acquisition debt, not source backup debt.
- Backup warnings/docs say "source-tier referenced blobs" unless the backup
  command also emits an attachment section.
- If the original 39,586 backup artifact is found, classify it as stale archive
  root, pre-v13 synthetic attachment hashes, or source refs since restored.

Verification:

```bash
POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue POLYLOGUE_FORCE_PLAIN=1 \
  polylogue ops maintenance blob-reference-debt --output-format json --sample-limit 5 --group-limit 20

POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue POLYLOGUE_FORCE_PLAIN=1 \
  polylogue ops diagnostics workload --blob-reference-debt --json

devtools test tests/unit/storage/test_blob_integrity.py -k "blob_reference_debt or attachment"
devtools test tests/unit/cli/test_archive_maintenance_cli.py -k "blob_reference"
devtools test tests/unit/daemon/test_backup.py -k "missing_blob"
devtools verify --quick
```
