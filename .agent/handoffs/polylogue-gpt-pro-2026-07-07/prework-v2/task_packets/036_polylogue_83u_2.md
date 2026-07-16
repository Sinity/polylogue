# 036. polylogue-83u.2 — Attachment byte acquisition for non-inline sources (Drive/zip/local)

Priority/type/status: **P2 / feature / open**. Lane: **01-blob-attachment-integrity**. Release: **B-storage-byte-integrity**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Acquire bytes where the handle is live: Drive via DriveSourceClient.download_bytes inside the iterator scope (un-bypass download_assets); export-zip member resolution while the zipfile is open; local paths via transport-only local_source_path under a source-root allowlist + realpath-escape check. Deposit onto ParsedAttachment.inline_bytes; reuse the shipped true-hash write. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Un-bypass byte acquisition at each live-handle boundary and deposit bytes onto ParsedAttachment.inline_bytes, then reuse the shipped true-SHA-256 blob write (the _acquire_attachment_blob path). (a) Drive: call DriveSourceClient.download_bytes INSIDE the iterator scope so bytes are read before the source handle closes — restore the deleted download_assets path. (b) export-zip: resolve and read the member while the zipfile is still open. (c) local paths: a transport-only local_source_path resolved under a source-root allowlist, guarded by a realpath-escape check (canonicalize and assert the resolved path stays within an allowed root; reject symlink/`..` escapes). Non-live handles stay honest-unfetched with source_url/source_path preserved for later re-acquisition (never a synthetic hash). Pitfall: inline_bytes is transport-only and must not widen the content-hash surface; acquisition is idempotent by true content hash.

## Acceptance criteria

1. For each source class a seeded fixture ingests attachments whose bytes are live and asserts acquisition_status='acquired' with a blob file present at the true SHA-256 of the bytes: Drive (download_bytes inside iterator scope), export-zip member (read while zip open), and local path (allowlisted transport). Verify: `devtools test tests/unit/sources/` selection covering the three fixtures passes. 2. A local path outside the source-root allowlist is rejected by the realpath-escape check (test asserts the rejection; no read occurs). 3. Handles that are not live remain acquisition_status='unfetched' with source_url/source_path preserved and no synthetic blob_hash written. 4. The acquisition path does not alter sessions.content_hash for otherwise-identical content (idempotency test). Observable: a live re-ingest raises the acquired-attachment count as measured by the 83u.6 census.

## Static mechanism / likely defect

Bead design identifies three live-handle boundaries: Drive downloads inside iterator scope, export-zip members while zipfile is open, and local paths guarded by a source-root allowlist. Existing parser model already supports `ParsedAttachment.inline_bytes`.

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
2. 1. Drive: restore/download asset bytes with `DriveSourceClient.download_bytes` inside the iterator/lifetime of the source handle.
3. 2. Export ZIP: resolve attachment member and read it while the `ZipFile` is still open; attach as inline bytes.
4. 3. Local path: accept a transport-only `local_source_path`, canonicalize with `realpath`, require it to stay under declared source-root allowlist, reject symlink/`..` escapes before opening.
5. 4. Non-live handles remain `unfetched` with `source_url/source_path`; no synthetic hash.
6. 5. Assert acquired attachment blob hash equals true SHA-256 of bytes and session content hash remains stable for otherwise-identical content.

## Tests to add

- Drive fixture downloads inside iterator and stores true blob.
- ZIP fixture stores member bytes before close.
- local allowlisted file stores true blob.
- local path escape is rejected and no read happens.
- closed/non-live handle stays unfetched.
- ingest idempotency/content hash unchanged except attachment acquisition state.

## Verification commands

- ``devtools test tests/unit/sources/ -k 'attachment and (drive or zip or local or inline_bytes)'``

## Pitfalls

- Do not delete or compress before byte references are classified and lease/ref safety is proven.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
