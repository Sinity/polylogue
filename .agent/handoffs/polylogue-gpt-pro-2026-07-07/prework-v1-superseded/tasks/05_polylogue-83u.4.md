# 05. polylogue-83u.4 — Separate source-tier referenced-blob debt from index attachment-acquisition debt

Priority: **P1**  
Lane: **blob-integrity**  
Readiness: **ready-now / diagnostic-code**

Depends on packet(s): polylogue-8jg9.4 + polylogue-8jg9.2

## Why this is urgent / critical-path

The system previously reported 39,586 missing referenced blobs. The bead notes now say current source-tier referenced blob debt appears clean, while many index attachment rows are simply unfetched with `blob_hash NULL`. Those are different states and must not be collapsed.

## Static diagnosis / likely mechanism

Likely mechanism: blob diagnostics count attachment rows without blob hashes as missing referenced blobs. But an unfetched attachment is acquisition debt, not a broken blob reference. A true missing referenced blob means a durable table points at a non-null hash whose file is absent.

## Implementation plan

Implementation shape:
1. Add/extend a diagnostics model with two top-level sections:
   - `source_reference_debt`: raw/source-tier blob reference rows with non-null hashes, present/missing/recoverable/accepted.
   - `attachment_acquisition_debt`: index attachment rows grouped by acquisition status: acquired+present, acquired+missing-file, unfetched-null-hash, unavailable, recoverable-local/zip/drive.
2. Change backup/workload warnings to say exactly which section is bad.
3. `blob_hash IS NULL` attachments must increment `unfetched`, not `missing_blob_ref`.
4. Add bounded samples by source/origin/reference type. Keep live archive probes read-only (`mode=ro`).
5. Feed the same primitives into the 83u.6 census.

## Test plan

Tests:
- clean source DB + index attachments with `blob_hash NULL` => source missing=0, unfetched=N, no “missing referenced blob” warning.
- acquired attachment with hash whose blob file is missing => attachment acquired_missing=N.
- raw/source reference to missing hash => source_reference_debt missing=N.
- diagnostic JSON has both sections and reconciles totals.

## Verification command / proof

`devtools test tests/unit/storage/test_blob_integrity.py tests/unit/operations/test_archive_debt.py -k 'blob_reference_debt or attachment_acquisition or missing_blob'`

## Pitfalls

Do not “fix” missing source references by inventing synthetic hashes or deleting attachment rows. This packet is classification and truthful diagnostics first; restoration is only for rows whose original bytes can be verified.

## Files/functions to inspect or touch

- `polylogue/storage/blob_integrity.py`
- `polylogue/operations/archive_debt.py`
- `polylogue/storage/sqlite/archive_tiers/index.py:attachments`
- `polylogue/storage/blob_store.py`
