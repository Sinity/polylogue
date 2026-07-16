# 037. polylogue-83u.6 — Attachment acquisition census by origin and byte volume

Priority/type/status: **P2 / task / open**. Lane: **01-blob-attachment-integrity**. Release: **B-storage-byte-integrity**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Post-v13: measure acquired/unfetched/unavailable by origin and byte volume on the live archive. Quantifies how much of the metadata-only gap has actually closed and what re-acquisition would recover — the sizing input for the acquisition beads in this program and the honesty check on any 'attachments preserved' claim.

## Existing design note

Read-only attachment-acquisition census over the active archive (promoted from the 2026-07-04 notes sidecar). Open source.db (raw_sessions/artifact_observations if needed), index.db (attachments/artifact_observations), and resolve blob paths via polylogue/storage/blob_store.py + archive-tier path helpers, all with SQLite URI mode=ro; never mutate the live archive. Group by (origin, acquisition_status) and emit attachment_count, declared_byte_sum, acquired_blob_count, acquired_blob_bytes_on_disk, unfetched_count, unavailable_count, missing_blob_ref_count, top source_ref classes, and a bounded (~20) sample of hashes/paths. Reuse the blob-reference-debt / diagnostics-workload primitives so totals reconcile. Persist a JSON + short markdown census under .agent/scratch/research/ (or a demo-shelf evidence entry). Baseline the current archive as 'before' (per 83u.4 evidence: 7,226 attachment rows = 958 acquired w/ non-null hash + 0 missing acquired blobs, 6,268 unfetched blob_hash NULL) and re-run as 'after' once 83u.2/83u.3 acquisition beads land. Pitfall: unfetched (NULL blob_hash) rows are honest-absent, not missing_blob_ref debt — keep those lanes distinct.

## Acceptance criteria

1. A committed census artifact (JSON + markdown under .agent/scratch/research/ or demo-shelf) reports attachments grouped by (origin, acquisition_status) with attachment_count, declared_byte_sum, acquired_blob_count/bytes-on-disk, unfetched_count, unavailable_count, and missing_blob_ref_count, distinguishing genuinely-unfetchable from re-acquirable. 2. A before/after pair is captured: baseline now, re-run after the acquisition beads; the delta is written back into parent 83u as the epic's before/after closing evidence. 3. Follow-up beads are filed only for actionable acquisition classes (live local source path, archive-member re-acquisition, genuinely unavailable), not per missing row. Verify: the census command runs read-only (mode=ro) against POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue and its totals reconcile against `polylogue ops diagnostics workload --blob-reference-debt --json`; no write connection is opened against the live archive.

## Static mechanism / likely defect

The bead already gives a strong design: read source.db/index.db/blob store read-only; group by origin and acquisition_status; reconcile against blob-reference-debt diagnostics. Baseline numbers in the bead: 7,226 attachment rows, 958 acquired with non-null hash and 0 missing acquired blobs, 6,268 unfetched with NULL hash.

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

1. Implementation shape:
2. 1. Add a devtools or ops diagnostic command that opens archive DBs using SQLite `mode=ro`.
3. 2. Resolve blob paths through `BlobStore` without mutating.
4. 3. Group rows by `(origin, acquisition_status)` and emit counts, declared byte sum, acquired blob count, bytes on disk, unfetched, unavailable, missing acquired blob refs, top source_ref classes, and bounded samples.
5. 4. Write JSON + Markdown under `.agent/scratch/research/` or a demo shelf.
6. 5. Store baseline now, then rerun after 83u.2/83u.3 to measure delta.

## Tests to add

- command refuses or is proven not to open write connections against a fixture/live path.
- fixture totals reconcile with blob-reference-debt primitive.
- missing acquired blob and unfetched NULL hash are separated.
- sample list is bounded.

## Verification commands

- ``devtools test tests/unit/operations/test_attachment_census*.py -k 'read_only or acquisition or census'` and one read-only run against a copy/live archive as operator evidence.`

## Pitfalls

- Do not delete or compress before byte references are classified and lease/ref safety is proven.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
