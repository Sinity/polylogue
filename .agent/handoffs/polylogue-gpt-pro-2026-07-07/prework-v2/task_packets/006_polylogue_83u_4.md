# 006. polylogue-83u.4 — Classify the 39,586 missing referenced blobs in the production backup

Priority/type/status: **P1 / task / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Backup verifier warns 'referenced blobs missing: 39586'. Likely dominated by the pre-v13 synthetic attachment rows — classify via ops maintenance blob-reference-debt, split real acquisition debt from by-construction fakes, restore direct-file paths where SHA-verified. Gates trusting full_evidence backups. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Refine this from recovery-only into an executable classification/product issue. Current active archive evidence (POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue) shows source-tier referenced blob debt is clean: blob-reference-debt reports 37,552 source reference rows, 20,269 distinct referenced blobs, and 0 missing; diagnostics workload reports missing_referenced_blobs=0 with source reference_sources raw_sessions=16,709 and blob_refs=20,032. Direct restore dry-run and raw-backed recovery plan both report 0 candidates. The historical 39,586 warning is therefore not reproducible against the active archive and should be treated as stale/different-backup evidence unless the original backup directory/report is supplied. Current index attachments are: 7,226 total, 958 acquired with non-null hashes and 0 missing acquired blob files, 6,268 unfetched with blob_hash NULL. Product change should make source backup debt and index attachment acquisition debt explicit and separately reported.

## Acceptance criteria

Every one of the 39,586 missing referenced blobs classified by the blob-reference-debt classifier (table, ref type, origin, recoverability); direct-file-recoverable subset restored via blob-reference-restore-direct with SHA-256 verification; the irrecoverable remainder documented with counts and the recovery-vs-accept decision recorded.

## Static mechanism / likely defect

Production backup contains many missing blob references. The critical distinction is source-tier lost bytes vs index-tier unfetched attachments vs intentionally omitted/private material.

## Source anchors to inspect first

- `polylogue/browser_capture/models.py:22` — BrowserCaptureAttachment has metadata and possible data fields; acquisition policy must be explicit.
- `polylogue/browser_capture/server.py:259` — Capture POST writes payloads to spool after admission checks.
- `polylogue/archive/attachment/models.py` — Normalized attachment model should carry acquired/missing/recoverable state.
- `polylogue/storage/blob_store.py` — Byte storage API and hash validation live here.
- `polylogue/daemon/http.py:983` — _check_auth_logic uses direct equality and allows all when token is unset.
- `polylogue/daemon/http.py:1037` — _check_auth currently accepts query-string access_token broadly.
- `polylogue/daemon/http.py:1294` — do_GET dispatches without central Host/Origin admission.
- `polylogue/daemon/http.py:1301` — _check_cross_origin applies only to POST and allows absent Origin.
- `polylogue/browser_capture/receiver.py:45` — BrowserCaptureReceiverConfig defaults auth_token to None.
- `polylogue/browser_capture/server.py:54` — _origin_allowed accepts absent Origin.
- `polylogue/browser_capture/server.py:68` — _check_token accepts every request when auth_token is None and uses direct equality otherwise.
- `polylogue/browser_capture/server.py:47` — Only per-request max body exists; add spool file/count/bytes governor.
- `polylogue/storage/blob_store.py:352` — detect_orphans only compares disk hashes to caller-supplied referenced IDs.
- `polylogue/storage/blob_store.py:387` — cleanup_orphans deletes caller-supplied hashes and lacks lease/ref/generation checks.
- `polylogue/storage/blob_gc.py:163` — _has_active_lease exists in the safer GC path.
- `polylogue/storage/blob_gc.py:307` — run_blob_gc is the safer planner/executor path to route destructive cleanup through.
- `polylogue/storage/blob_gc.py:393` — GC generation/age gate exists in run_blob_gc.

## Implementation plan

1. Build a classifier that enumerates every blob reference by table, column/ref type, origin, hash, source material, and acquisition policy.
2. Group into: present, restorable from source, never-acquired metadata-only, intentionally omitted, private/redacted, irrecoverable.
3. Restore direct-file-recoverable blobs with SHA-256 verification.
4. Write a durable debt report and block public backup/attachment claims until classified.

## Tests to add

- Synthetic DB with present/missing/restorable/metadata-only refs classifies all buckets.
- Restore path refuses hash mismatch.
- Debt report totals equal raw referenced-hash count; no silent remainder.

## Verification commands

- ``devtools test tests/unit/storage/test_blob_integrity.py tests/unit/operations/test_archive_debt.py -k 'blob_reference_debt or attachment_acquisition or missing_blob'``

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
