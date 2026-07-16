# 137. polylogue-83u — Attachment & blob evidence integrity: bytes exist, are honest, and stay affordable

Priority/type/status: **P1 / epic / open**. Lane: **01-blob-attachment-integrity**. Release: **B-storage-byte-integrity**. Readiness: **epic-needs-child-closure**.

## What the bead says

Attachments are metadata-only by construction: 8,425 rows claim 8.4GB, 0 blobs exist, 56% zero-byte; blob_hash was synthetic until v13 made it honest-nullable with acquisition_status. This program makes attachment/blob evidence real end-to-end: acquire bytes where handles are live, classify what is genuinely unfetchable, keep the backup verifier trustworthy, and compress the store. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Acceptance criteria

REFRAMED (operator 2026-07-04): the goal is to CAPTURE attachment bytes going forward, not miss-then-account. (1) Forward capture is default at ingest/browser-capture: uploaded + inline bytes land in the blob store at acquisition time (83u.3, 83u.1). (2) Non-inline bytes that STILL EXIST at their source are re-acquired (83u.2) — 'we're not getting some that exist' is a bug, not acceptable loss. (3) A permanent unfetchable floor is NORMAL and expected (source deleted, pre-install history, provider expiry) — the census (83u.6) reports it as honest baseline accounting, never as a failure to fix. Terminal state: no attachment whose bytes were reachable at capture time is lost; the unfetchable floor is measured and explained; no synthetic hashes. Verify: a live-capture session with an upload stores the blob; the census separates reachable-but-missed (bug) from genuinely-unfetchable (normal).

## Static mechanism / likely defect

Issue description localizes the mechanism: Attachments are metadata-only by construction: 8,425 rows claim 8.4GB, 0 blobs exist, 56% zero-byte; blob_hash was synthetic until v13 made it honest-nullable with acquisition_status. This program makes attachment/blob evidence real end-to-end: acquire bytes where handles are live, classify what is genuinely unfetchable, keep the backup verifier trustworthy, and compress the store. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

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

1. Inventory open child beads and map them to the invariant named by the epic.
2. Add/verify a terminal acceptance checklist for the epic rather than landing broad code.
3. Close only after child beads are closed or explicitly split out with new blockers.

## Tests to add

- Acceptance proof: REFRAMED (operator 2026-07-04): the goal is to CAPTURE attachment bytes going forward, not miss-then-account.
- Acceptance proof: (1) Forward capture is default at ingest/browser-capture: uploaded + inline bytes land in the blob store at acquisition time (83u.3, 83u.1).
- Acceptance proof: (2) Non-inline bytes that STILL EXIST at their source are re-acquired (83u.2) — 'we're not getting some that exist' is a bug, not acceptable loss.
- Acceptance proof: (3) A permanent unfetchable floor is NORMAL and expected (source deleted, pre-install history, provider expiry) — the census (83u.6) reports it as honest baseline accounting, never as a failure to fix.
- Acceptance proof: Terminal state: no attachment whose bytes were reachable at capture time is lost
- Acceptance proof: the unfetchable floor is measured and explained
- Acceptance proof: no synthetic hashes.
- Acceptance proof: Verify: a live-capture session with an upload stores the blob

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not delete or compress before byte references are classified and lease/ref safety is proven.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
